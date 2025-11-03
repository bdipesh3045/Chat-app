import os
import warnings
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
import tempfile

# -----------------------------------------
# 🚀 Flask App Setup
# -----------------------------------------
app = Flask(__name__)

# Suppress TensorFlow + protobuf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# -----------------------------------------
# 🧠 ACL Risk Thresholds
# -----------------------------------------
KNEE_FLEXION_THRESHOLD = 30
KNEE_VALGUS_THRESHOLD = 10
TRUNK_FORWARD_THRESHOLD = 30
TRUNK_LATERAL_THRESHOLD = 10
SEVERE_STIFF_FLEXION = 15
SEVERE_VALGUS_THRESHOLD = 20
SEVERE_TRUNK_FORWARD = 15
SEVERE_TRUNK_LATERAL = 20

# -----------------------------------------
# 🧮 Helper Functions
# -----------------------------------------
def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))

def calc_acl_risk_score(L_flex, R_flex, L_valgus, R_valgus, trunk_forward, trunk_lateral):
    score = 0
    if L_flex < KNEE_FLEXION_THRESHOLD or R_flex < KNEE_FLEXION_THRESHOLD: score += 1
    if L_flex < SEVERE_STIFF_FLEXION or R_flex < SEVERE_STIFF_FLEXION: score += 1
    if L_valgus > KNEE_VALGUS_THRESHOLD or R_valgus > KNEE_VALGUS_THRESHOLD: score += 1
    if L_valgus > SEVERE_VALGUS_THRESHOLD or R_valgus > SEVERE_VALGUS_THRESHOLD: score += 1
    if trunk_forward < TRUNK_FORWARD_THRESHOLD: score += 1
    if trunk_forward < SEVERE_TRUNK_FORWARD: score += 1
    if trunk_lateral > TRUNK_LATERAL_THRESHOLD: score += 1
    if trunk_lateral > SEVERE_TRUNK_LATERAL: score += 1
    return score

def risk_label(avg):
    if avg < 3:
        return "No Risk"
    elif avg < 6:
        return "Moderate"
    else:
        return "High"

# -----------------------------------------
# 🧍 Load MediaPipe Pose Model Once
# -----------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------
# 🎥 Core Video Processing Logic
# -----------------------------------------
def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            scores.append(0)
            continue

        lm = res.pose_landmarks.landmark
        h, w, _ = frame.shape

        def P(i):
            return np.array([lm[i].x*w, lm[i].y*h, lm[i].z*w], dtype=np.float32)

        # Key joints
        LHIP, RHIP = P(23), P(24)
        LKNEE, RKNEE = P(25), P(26)
        LANK, RANK = P(27), P(28)
        LSH, RSH = P(11), P(12)
        MIDHIP, MIDSH = 0.5*(LHIP + RHIP), 0.5*(LSH + RSH)

        # Angles
        v_thigh_L, v_shank_L = LHIP - LKNEE, LANK - LKNEE
        v_thigh_R, v_shank_R = RHIP - RKNEE, RANK - RKNEE
        L_flex = 180 - angle_between(v_thigh_L, v_shank_L)
        R_flex = 180 - angle_between(v_thigh_R, v_shank_R)

        # Valgus (2D)
        v_thigh_L_xy = np.array([LHIP[0]-LKNEE[0], LHIP[1]-LKNEE[1]])
        v_shank_L_xy = np.array([LANK[0]-LKNEE[0], LANK[1]-LKNEE[1]])
        v_thigh_R_xy = np.array([RHIP[0]-RKNEE[0], RHIP[1]-RKNEE[1]])
        v_shank_R_xy = np.array([RANK[0]-RKNEE[0], RANK[1]-RKNEE[1]])
        L_valgus = max(0.0, 180 - angle_between(v_thigh_L_xy, v_shank_L_xy))
        R_valgus = max(0.0, 180 - angle_between(v_thigh_R_xy, v_shank_R_xy))

        # Trunk angles
        trunk_vec = MIDSH - MIDHIP
        vertical = np.array([0.0, -1.0, 0.0])
        trunk_yz = np.array([0.0, trunk_vec[1], trunk_vec[2]])
        trunk_xy = np.array([trunk_vec[0], trunk_vec[1], 0.0])

        trunk_forward = (
            np.degrees(np.arccos(np.clip(np.dot(trunk_yz/np.linalg.norm(trunk_yz), vertical), -1, 1)))
            if np.linalg.norm(trunk_yz) != 0 else 0
        )
        trunk_lateral = (
            np.degrees(np.arccos(np.clip(np.dot(trunk_xy/np.linalg.norm(trunk_xy), vertical), -1, 1)))
            if np.linalg.norm(trunk_xy) != 0 else 0
        )

        score = calc_acl_risk_score(L_flex, R_flex, L_valgus, R_valgus, trunk_forward, trunk_lateral)
        scores.append(score)

    cap.release()

    avg_score = float(np.mean(scores)) if scores else 0.0
    score_out_of_10 = min(round((avg_score / 8.0) * 10.0, 2), 10.0)  # 8 = max per-frame points
    return {
        "score_out_of_10": score_out_of_10,
        "risk_level": risk_label(avg_score),
        "avg_raw_score": round(avg_score, 2)
    }

# -----------------------------------------
# 🌐 API Endpoint
# -----------------------------------------
@app.route("/analyze-acl", methods=["POST"])
def analyze_acl():
    """Upload a video to analyze ACL risk."""
    if "file" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        result = process_video(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

# -----------------------------------------
# ▶️ Run Flask Server
# -----------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1010, debug=True)
