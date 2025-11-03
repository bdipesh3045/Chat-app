import os
import warnings
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from datetime import datetime

# ------------------ Environment Cleanup ------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow logs
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ------------------ Config ------------------
camera_index = 0           # camera ID (0 = default)
display_scale = 1.0        # adjust for smaller preview window
record_output = True       # save output video
output_video_path = 'acl_risk_live.mp4'
log_csv_path = 'acl_risk_events.csv'

# Thresholds (moderate risk)
KNEE_FLEXION_THRESHOLD = 30          # stiff if < 30° at contact
FLEXION_EXCURSION_THRESHOLD = 45     # excursion < 45° after landing
KNEE_VALGUS_THRESHOLD = 10           # >10° valgus
TRUNK_FORWARD_THRESHOLD = 30         # forward flexion < 30° = too upright
TRUNK_LATERAL_THRESHOLD = 10         # lateral lean > 10°
ASYM_CONTACT_FRAME_DIFF = 1          # foot contact stagger > 1 frame
ASYM_KNEE_DIFF_THRESHOLD = 15        # peak flex diff > 15°

# Severe thresholds (for "ACL INJURY" state)
SEVERE_STIFF_FLEXION = 15            # < 15° flexion at contact
SEVERE_VALGUS_THRESHOLD = 20         # > 20° valgus
SEVERE_TRUNK_FORWARD = 15            # < 15° forward flexion (very upright)
SEVERE_TRUNK_LATERAL = 20            # > 20° lateral lean
ASYM_CONTACT_SEVERE_DIFF = 3         # > 3 frames
ASYM_KNEE_DIFF_SEVERE = 30           # > 30° peak flex diff

# Velocity detection params (pixels/frame)
VEL_DOWN_THRESH = 1.0
VEL_STOP_THRESH = 0.5

# Temporal smoothing: number of frames to aggregate severity
SEVERITY_WINDOW = 10
INJURY_FRAMES_REQUIRED = 5  # within the window, require at least this many severe frames
RISK_AVG_THRESHOLD = 1.0    # avg score >=1 => ACL RISK
INJURY_AVG_THRESHOLD = 3.0  # avg score >=3 => ACL INJURY

# ------------------ Helpers ------------------
def angle_between(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))

def put_text(img, text, org, color=(255,255,255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

# ------------------ MediaPipe Init ------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Compatibility: define drawing specs for all MediaPipe versions
try:
    LANDMARK_STYLE = mp_styles.get_default_pose_landmarks_style()
    CONNECTION_STYLE = mp_styles.get_default_pose_connections_style()
except AttributeError:
    LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
    CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)

# ------------------ Camera Init ------------------
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise IOError(f"Cannot open camera index {camera_index}")

ret, frame = cap.read()
if not ret:
    raise IOError("Camera opened but no frames received.")
frame_height, frame_width = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

out = None
if record_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ------------------ State Variables ------------------
frame_idx = 0
violations_log = []
left_contact_frame = right_contact_frame = None
left_initial_flex = right_initial_flex = None
left_peak_flex = right_peak_flex = 0.0
left_peak_valgus = right_peak_valgus = 0.0
left_peak_flex_frame = right_peak_flex_frame = None
left_peak_valgus_frame = right_peak_valgus_frame = None
foot_down_left = foot_down_right = False
hist_len = 3
left_y_hist, right_y_hist = deque(maxlen=hist_len), deque(maxlen=hist_len)
prev_left_vel = prev_right_vel = 0.0

# severity buffer
severity_window = deque(maxlen=SEVERITY_WINDOW)

print("Press 'q' to quit, 'r' to reset tracking.\n")

# ------------------ Main Loop ------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    disp = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    status_text = "SAFE"
    status_color = (0, 200, 0)
    violations_this_frame = []

    # per-frame severity scoring
    frame_score = 0     # moderate = +1 each, severe = +2 each
    severe_hits = 0     # count how many “severe” triggers this frame

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        def P(i):
            return np.array([
                lm[i].x * frame_width,
                lm[i].y * frame_height,
                lm[i].z * frame_width
            ], dtype=np.float32)

        # Keypoints
        LHIP, RHIP = P(23), P(24)
        LKNEE, RKNEE = P(25), P(26)
        LANK, RANK = P(27), P(28)
        LTOE, RTOE = P(31), P(32)
        LSH, RSH = P(11), P(12)
        MIDHIP, MIDSH = 0.5*(LHIP+RHIP), 0.5*(LSH+RSH)

        # Angles
        v_thigh_L, v_shank_L = LHIP - LKNEE, LANK - LKNEE
        v_thigh_R, v_shank_R = RHIP - RKNEE, RANK - RKNEE
        raw_L = angle_between(v_thigh_L, v_shank_L)
        raw_R = angle_between(v_thigh_R, v_shank_R)
        L_flex, R_flex = 180 - raw_L, 180 - raw_R

        v_thigh_L_xy = np.array([LHIP[0]-LKNEE[0], LHIP[1]-LKNEE[1]])
        v_shank_L_xy = np.array([LANK[0]-LKNEE[0], LANK[1]-LKNEE[1]])
        v_thigh_R_xy = np.array([RHIP[0]-RKNEE[0], RHIP[1]-RKNEE[1]])
        v_shank_R_xy = np.array([RANK[0]-RKNEE[0], RANK[1]-RKNEE[1]])
        L_valgus = max(0.0, 180 - angle_between(v_thigh_L_xy, v_shank_L_xy))
        R_valgus = max(0.0, 180 - angle_between(v_thigh_R_xy, v_shank_R_xy))

        trunk_vec = MIDSH - MIDHIP
        vertical = np.array([0.0, -1.0, 0.0])
        trunk_yz = np.array([0.0, trunk_vec[1], trunk_vec[2]])
        trunk_xy = np.array([trunk_vec[0], trunk_vec[1], 0.0])
        trunk_forward = np.degrees(np.arccos(np.clip(np.dot(trunk_yz/np.linalg.norm(trunk_yz), vertical), -1, 1))) if np.linalg.norm(trunk_yz)!=0 else 0
        trunk_lateral = np.degrees(np.arccos(np.clip(np.dot(trunk_xy/np.linalg.norm(trunk_xy), vertical), -1, 1))) if np.linalg.norm(trunk_xy)!=0 else 0

        # Foot vertical velocities
        left_y_hist.append(LTOE[1]); right_y_hist.append(RTOE[1])
        L_vel = left_y_hist[-1]-left_y_hist[-2] if len(left_y_hist)>=2 else 0
        R_vel = right_y_hist[-1]-right_y_hist[-2] if len(right_y_hist)>=2 else 0

        # Detect contacts (initial landing)
        if (not foot_down_left) and prev_left_vel > VEL_DOWN_THRESH and L_vel <= VEL_STOP_THRESH:
            foot_down_left = True
            left_contact_frame, left_initial_flex = frame_idx, L_flex
            # Moderate/severe stiff landing
            if L_flex < KNEE_FLEXION_THRESHOLD:
                violations_this_frame.append(f"Stiff landing LEFT ({L_flex:.1f}° < {KNEE_FLEXION_THRESHOLD}°)")
                frame_score += 1
            if L_flex < SEVERE_STIFF_FLEXION:
                severe_hits += 1; frame_score += 1  # extra point for severe

            # If this is first contact, evaluate trunk at IC
            if right_contact_frame is None or frame_idx <= right_contact_frame:
                if trunk_forward < TRUNK_FORWARD_THRESHOLD:
                    violations_this_frame.append(f"Upright trunk ({trunk_forward:.1f}° < {TRUNK_FORWARD_THRESHOLD}°)")
                    frame_score += 1
                if trunk_forward < SEVERE_TRUNK_FORWARD:
                    severe_hits += 1; frame_score += 1
                if trunk_lateral > TRUNK_LATERAL_THRESHOLD:
                    violations_this_frame.append(f"Lateral trunk lean {trunk_lateral:.1f}° > {TRUNK_LATERAL_THRESHOLD}°")
                    frame_score += 1
                if trunk_lateral > SEVERE_TRUNK_LATERAL:
                    severe_hits += 1; frame_score += 1

        if (not foot_down_right) and prev_right_vel > VEL_DOWN_THRESH and R_vel <= VEL_STOP_THRESH:
            foot_down_right = True
            right_contact_frame, right_initial_flex = frame_idx, R_flex
            if R_flex < KNEE_FLEXION_THRESHOLD:
                violations_this_frame.append(f"Stiff landing RIGHT ({R_flex:.1f}° < {KNEE_FLEXION_THRESHOLD}°)")
                frame_score += 1
            if R_flex < SEVERE_STIFF_FLEXION:
                severe_hits += 1; frame_score += 1

            if left_contact_frame is None or frame_idx <= left_contact_frame:
                if trunk_forward < TRUNK_FORWARD_THRESHOLD:
                    violations_this_frame.append(f"Upright trunk ({trunk_forward:.1f}° < {TRUNK_FORWARD_THRESHOLD}°)")
                    frame_score += 1
                if trunk_forward < SEVERE_TRUNK_FORWARD:
                    severe_hits += 1; frame_score += 1
                if trunk_lateral > TRUNK_LATERAL_THRESHOLD:
                    violations_this_frame.append(f"Lateral trunk lean {trunk_lateral:.1f}° > {TRUNK_LATERAL_THRESHOLD}°")
                    frame_score += 1
                if trunk_lateral > SEVERE_TRUNK_LATERAL:
                    severe_hits += 1; frame_score += 1

        # Track peaks after contact + add severity
        if foot_down_left:
            if L_flex > left_peak_flex:
                left_peak_flex, left_peak_flex_frame = L_flex, frame_idx
            if L_valgus > left_peak_valgus:
                left_peak_valgus, left_peak_valgus_frame = L_valgus, frame_idx
            if L_valgus > KNEE_VALGUS_THRESHOLD:
                violations_this_frame.append(f"Knee valgus LEFT {L_valgus:.1f}° > {KNEE_VALGUS_THRESHOLD}°")
                frame_score += 1
            if L_valgus > SEVERE_VALGUS_THRESHOLD:
                severe_hits += 1; frame_score += 1

        if foot_down_right:
            if R_flex > right_peak_flex:
                right_peak_flex, right_peak_flex_frame = R_flex, frame_idx
            if R_valgus > right_peak_valgus:
                right_peak_valgus, right_peak_valgus_frame = R_valgus, frame_idx
            if R_valgus > KNEE_VALGUS_THRESHOLD:
                violations_this_frame.append(f"Knee valgus RIGHT {R_valgus:.1f}° > {KNEE_VALGUS_THRESHOLD}°")
                frame_score += 1
            if R_valgus > SEVERE_VALGUS_THRESHOLD:
                severe_hits += 1; frame_score += 1

        prev_left_vel, prev_right_vel = L_vel, R_vel

        # Draw landmarks
        mp_drawing.draw_landmarks(
            disp, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            LANDMARK_STYLE, CONNECTION_STYLE
        )

        # HUD
        put_text(disp, f"L flex {L_flex:.0f}° | valgus {L_valgus:.0f}°", (10, 30))
        put_text(disp, f"R flex {R_flex:.0f}° | valgus {R_valgus:.0f}°", (10, 60))
        put_text(disp, f"Trunk forward {trunk_forward:.0f}° | lateral {trunk_lateral:.0f}°", (10, 90))
    else:
        left_y_hist.clear(); right_y_hist.clear()
        prev_left_vel = prev_right_vel = 0.0

    # Update severity buffer
    # Weight severe_hits by +2 each internally (already did +1 extra per severe above)
    severity_window.append(frame_score)

    # Decide status with smoothing
    avg_score = np.mean(severity_window) if len(severity_window) > 0 else 0.0
    severe_count = sum(1 for s in severity_window if s >= INJURY_AVG_THRESHOLD)

    if severe_count >= INJURY_FRAMES_REQUIRED or avg_score >= INJURY_AVG_THRESHOLD:
        status_text = "ACL INJURY"
        status_color = (0, 0, 255)
    elif avg_score >= RISK_AVG_THRESHOLD or len(violations_this_frame) > 0:
        status_text = "ACL RISK"
        status_color = (0, 255, 255)
    else:
        status_text = "SAFE"
        status_color = (0, 200, 0)

    # Logging (only when not SAFE)
    if status_text != "SAFE":
        for v in (violations_this_frame or ["elevated risk indicators"]):
            violations_log.append({
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "frame": frame_idx,
                "status": status_text,
                "avg_score": float(avg_score),
                "message": v
            })

    # Banner + severity meter
    cv2.rectangle(disp, (0,0), (frame_width,40), (0,0,0), -1)
    put_text(disp, f"{status_text}", (10,28), status_color, scale=0.9, thick=2)
    put_text(disp, f"avg:{avg_score:.1f} window:{len(severity_window)}/{SEVERITY_WINDOW}", (200,28), (200,200,200), scale=0.6, thick=1)

    if display_scale != 1.0:
        disp = cv2.resize(disp, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("ACL Risk (Live)", disp)
    if out: out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'):
        print("Resetting tracking...")
        left_contact_frame = right_contact_frame = None
        left_initial_flex = right_initial_flex = None
        left_peak_flex = right_peak_flex = 0.0
        left_peak_valgus = right_peak_valgus = 0.0
        left_y_hist.clear(); right_y_hist.clear()
        foot_down_left = foot_down_right = False
        prev_left_vel = prev_right_vel = 0.0
        severity_window.clear()

    frame_idx += 1

# ------------------ Post-run Checks (summary) ------------------
post = []
if left_contact_frame and left_peak_flex:
    exc = left_peak_flex - left_initial_flex
    if exc < FLEXION_EXCURSION_THRESHOLD:
        post.append(f"Left flexion excursion {exc:.1f}° < {FLEXION_EXCURSION_THRESHOLD}°")
if right_contact_frame and right_peak_flex:
    exc = right_peak_flex - right_initial_flex
    if exc < FLEXION_EXCURSION_THRESHOLD:
        post.append(f"Right flexion excursion {exc:.1f}° < {FLEXION_EXCURSION_THRESHOLD}°")

if left_peak_valgus > KNEE_VALGUS_THRESHOLD:
    post.append(f"Peak LEFT valgus {left_peak_valgus:.1f}° frame {left_peak_valgus_frame}")
if right_peak_valgus > KNEE_VALGUS_THRESHOLD:
    post.append(f"Peak RIGHT valgus {right_peak_valgus:.1f}° frame {right_peak_valgus_frame}")

if left_contact_frame and right_contact_frame:
    diff = abs(left_contact_frame - right_contact_frame)
    if diff > ASYM_CONTACT_FRAME_DIFF:
        first = "LEFT" if left_contact_frame < right_contact_frame else "RIGHT"
        post.append(f"Staggered contact: {first} first, Δ={diff} frames")

if left_peak_flex and right_peak_flex:
    diff = abs(left_peak_flex - right_peak_flex)
    if diff > ASYM_KNEE_DIFF_THRESHOLD:
        dom = "LEFT" if left_peak_flex > right_peak_flex else "RIGHT"
        post.append(f"Unilateral loading: {dom} knee Δ={diff:.1f}° > {ASYM_KNEE_DIFF_THRESHOLD}°")

if post:
    for p in post:
        violations_log.append({"timestamp": datetime.now().isoformat(timespec='seconds'),
                               "frame": frame_idx, "status": "POST", "avg_score": None, "message": p})
    pd.DataFrame(violations_log).to_csv(log_csv_path, index=False)
    print(f"\n⚠️  Events logged to {log_csv_path}")
else:
    if violations_log:
        pd.DataFrame(violations_log).to_csv(log_csv_path, index=False)
        print(f"\n⚠️  Events logged to {log_csv_path}")
    else:
        print("\n✅ No high-risk events detected.")

cap.release()
if out: out.release()
pose.close()
cv2.destroyAllWindows()
