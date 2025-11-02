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
def prompt_return(data, height, weight, age, training_type, experience_level, limitations):
    if data == "exercise" or data!=None:
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
        output_param = prompt_return(**input_params)
    print(output_param)
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


