from flask import Flask, request, jsonify
import requests
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------
# 🔑 API Keys
# --------------------------
TOMTOM_API_KEY = "W1F9GRs9PpuOSJSrv5VPEImP2Nwso9z3"
GEMINI_API_KEY = "AIzaSyAE0F2sKMlI2DU6qHcsZtmKoxVpK8EAG2I"

# --------------------------
# 🚀 Initialize Flask & Gemini
# --------------------------
app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.4,
    google_api_key=GEMINI_API_KEY
)

# --------------------------
# 🧠 Summarization Function
# --------------------------
def summarize_json(data) -> str:
    """
    Summarize JSON traffic data using Gemini.
    """
    prompt = f"""
No Yapping — be fast.
You are an expert traffic data summarizer.

Summarize this traffic JSON clearly and briefly in human-readable form:
{data}
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# --------------------------
# 🌍 POST Route
# --------------------------
@app.route('/get_loc', methods=['POST'])
def get_location():
    """
    POST JSON body example:
    {
        "lat": 40.7580,
        "lon": -73.9855
    }
    """
    try:
        body = request.get_json()

        if not body or "lat" not in body or "lon" not in body:
            return jsonify({"error": "Please provide both 'lat' and 'lon' fields."}), 400

        lat = body["lat"]
        lon = body["lon"]

        print(f"Fetching traffic data for {lat}, {lon}")

        # Call TomTom Traffic API
        url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        params = {
            "key": TOMTOM_API_KEY,
            "point": f"{lat},{lon}",
            "unit": "KMPH",
            "style": "absolute"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200 or "flowSegmentData" not in data:
            return jsonify({"error": "Failed to fetch traffic data", "details": data}), 500

        flow = data["flowSegmentData"]

        flow_data = {
            "Current Speed": flow.get("currentSpeed"),
            "Free Flow Speed": flow.get("freeFlowSpeed"),
            "Current Travel Time (sec)": flow.get("currentTravelTime"),
            "Free Flow Travel Time (sec)": flow.get("freeFlowTravelTime"),
            "Confidence": flow.get("confidence")
        }

        # Get Gemini summary
        summary = summarize_json(flow_data)

        return jsonify({
            "latitude": lat,
            "longitude": lon,
            "traffic_data": flow_data,
            "ai_summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------
# 🏠 Root Route
# --------------------------
@app.route('/')
def home():
    return {"message": "Welcome to the Traffic Summary API! POST to /get_loc with lat/lon."}


# --------------------------
# 🏁 Run App
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
