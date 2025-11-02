from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="AI Warm-up Assistant API")

# --------------------------
# 🔐 Gemini API Key
# --------------------------
GEM_API_KEY = "AIzaSyAE0F2sKMlI2DU6qHcsZtmKoxVpK8EAG2I"

# --------------------------
# 💾 In-memory chat sessions per username
# --------------------------
user_sessions = {}

# --------------------------
# 🧠 Prompt builder
# --------------------------
def prompt_return(data, height, weight,gender, age, training_type, experience_level, limitations):
    if data == "exercise":
        print("Prompt works")
        return f"""
You are a certified strength and conditioning coach and physiotherapist who specializes in safe and effective warm-up design.

Your task is to suggest a personalized warm-up routine based on the user’s body parameters and training goals for the day.

The warm-up must:
- Prepare the body for the specified training (mobility, activation, heart rate increase, movement readiness)
- Be tailored to the user’s height, weight, age, experience level, and any injuries or limitations
- Be time-efficient (10–20 minutes max)
- Include sets/reps/time and brief coaching cues for each movement
- Explain why each section of the warm-up is included
- Prioritize safety, form, and progressive intensity

Height: {height} cm  
Weight: {weight} kg  
Age: {age}  
Gender : {gender}
Training Type: {training_type}  
Experience Level: {experience_level}  
Limitations: {limitations}

### 🧭 Output Format
1️⃣ **Warm-Up Summary**
   - 2–3 sentences summarizing the focus (mobility, activation, injury prevention, etc.)

2️⃣ **Warm-Up Routine (structured)**
   - General Activation (list 2–3 exercises with duration)
   - Mobility Work (list 3–4 movements)
   - Muscle Activation (list 2–3 exercises)
   - Specific Prep (list 1–2 drills)

3️⃣ **Coaching Notes**
   - Explain key points like breathing, pacing, and injury prevention.

4️⃣ **Optional Adjustments**
   - Add variations for users with pain, stiffness, or limited time.

5️⃣ **Motivational Message**
   - A short encouraging message for the athlete.
"""



# --------------------------
# ⚙️ User Session Manager
# --------------------------
def get_chain(username: str):
    """Get or create a chat session for a specific username."""
    if username not in user_sessions:
        print(user_sessions.items())
        user_sessions[username] = {
            "history": [],
            "llm": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                temperature=0.3,
                google_api_key=GEM_API_KEY
            )
        }
    print("Get chain")
    return user_sessions[username]


# --------------------------
# 💬 Chat Function
# --------------------------
def chat(username: str, user_input: str, input_params: Dict):
    session = get_chain(username)
    llm = session["llm"]
    history = session["history"]
    if input_params["data"]=="chat":
        output_param="You are a coach bot who will help people to guide about acl injury"
    else:
    # 🧠 Use dynamic user parameters
        print("else")
        output_param = prompt_return(**input_params)
    print(output_param,"hello")
    # Build the chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a friendly and knowledgeable AI assistant. Be concise but helpful.{output_param}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    print(1)
    prompt_value = prompt.invoke({
        "chat_history": history,
        "input": user_input
    })
    print(2)

    response = llm.invoke(prompt_value)
    print(3)
    # Store conversation history
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
    x=response.content
    print(x)
    print(type(x))
    return x


# --------------------------
# 📦 Request Schema
# --------------------------
class ChatRequest(BaseModel):
    username: str
    user_input: str
    input_params: Dict

# --------------------------
# 🚀 FastAPI Endpoints
# --------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for the warm-up assistant."""
    try:
        print(request.username, request.user_input, request.input_params)
        response = chat(request.username, request.user_input, request.input_params)
        print("Working")
        return {"username": request.username, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():

    return {"message": "Welcome to the AI Warm-up Assistant API!"}


def coach_mem(player_name, daily: Dict, weekly: Dict):
    data=f"""
    ### 🧍 Player Details
    **Name:** {player_name}
    ### 📅 Today's Metrics
    - Pain: {daily.get('knee_pain', 'N/A')}/10
    - Fatigue: {daily.get('leg_freshness', 'N/A')}/10
    - Stability: {daily.get('stability', 'N/A')}/10
    - Sleep: {daily.get('sleep_hours', 'N/A')}h
    - Warm-up: {'Done' if daily.get('warmup_done') else 'Skipped'}
    - Mobility: {'Stiff' if daily.get('mobility_stiffness') else 'Good'}

    ---

    ### 🗓️ 7-Day Weekly Trends
    - Avg Pain: {weekly.get('avg_pain', 'N/A')}
    - Avg Sleep: {weekly.get('avg_sleep', 'N/A')}
    - Avg Freshness: {weekly.get('avg_freshness', 'N/A')}
    - Avg Intensity: {weekly.get('avg_intensity', 'N/A')}
    - Warm-up Compliance: {weekly.get('warmup_rate', 0)*100:.0f}%
    - Days with Stiffness: {weekly.get('stiff_days', 'N/A')}

    ---

    """
    return data

def build_acl_summary_prompt(player_name, daily: Dict, weekly: Dict):
    """
    Build a complete ACL status summary prompt for the AI coach assistant.
    Uses daily and weekly data only.

    Args:
        player_name (str): Player name.
        daily (dict): Today's metrics.
        weekly (dict): 7-day averages/trends.

    Returns:
        str: formatted prompt string for the LLM.
    """

    # Debug log
    print("Daily data:", daily)
    print("Weekly data:", weekly)

    prompt = f"""
You are an assistant for a strength and conditioning coach.
Summarize this player's ACL-related status clearly and briefly.

---

### 🧍 Player Details
**Name:** {player_name}

---

### 📅 Today's Metrics
- Pain: {daily.get('knee_pain', 'N/A')}/10
- Fatigue: {daily.get('leg_freshness', 'N/A')}/10
- Stability: {daily.get('stability', 'N/A')}/10
- Sleep: {daily.get('sleep_hours', 'N/A')}h
- Warm-up: {'Done' if daily.get('warmup_done') else 'Skipped'}
- Mobility: {'Stiff' if daily.get('mobility_stiffness') else 'Good'}

---

### 🗓️ 7-Day Weekly Trends
- Avg Pain: {weekly.get('avg_pain', 'N/A')}
- Avg Sleep: {weekly.get('avg_sleep', 'N/A')}
- Avg Freshness: {weekly.get('avg_freshness', 'N/A')}
- Avg Intensity: {weekly.get('avg_intensity', 'N/A')}
- Warm-up Compliance: {weekly.get('warmup_rate', 0)*100:.0f}%
- Days with Stiffness: {weekly.get('stiff_days', 'N/A')}

---

### 🧠 What to Provide
Most important thing store everything in memory for future reference and make the summary as brief as possible no yapping
1️⃣ **ACL Risk Level** (Low / Moderate / High)  
2️⃣ **2 Key Observations** (patterns or risk signals)  
3️⃣ **2–3 Actionable Coaching Recommendations** (e.g., adjust load, mobility drills)  
4️⃣ **Weekly Trend Summary** — risk trend (Improving / Stable / Worsening)  
6️⃣ **ACL Susceptibility Score (1–100)** — rate current injury risk.  
7.Make a brief report -> No yapping 


    """
    print("Prompt built successfully ✅")
    return prompt





# --------------------------
# 📊 Coach ACL Summary Endpoint
# --------------------------
class CoachRequest(BaseModel):
    coach_name: str
    player_name: str
    daily: Dict
    weekly: Dict
    mode : str
    input:str

@app.post("/coach")
def coach_summary_endpoint(request: CoachRequest):
    """
    Generate an ACL summary report for a player (for coach view).
    """
    try:
        session = get_chain(request.coach_name)
        llm = session["llm"]
        if request.mode=="chat":
            prompt_text=f"{request.input} Player details: {coach_mem(
                player_name=request.player_name,
                daily=request.daily,
                weekly=request.weekly
            )}"
        # 🧠 Build AI prompt
        else:
            prompt_text = build_acl_summary_prompt(
                player_name=request.player_name,
                daily=request.daily,
                weekly=request.weekly
            )

        # 🔮 Call Gemini
        response = llm.invoke(prompt_text)
        summary = response.content

        # Optionally store in session history
        session["history"].append(HumanMessage(content=f"Summarize ACL for {request.player_name}"))
        session["history"].append(AIMessage(content=summary))

        return {
            "player_name": request.player_name,
            "coach": request.coach_name,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
