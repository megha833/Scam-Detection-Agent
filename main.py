from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, List
import os, re, random, time

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(
    title="Agentic Honey-Pot Scam Intelligence API",
    description="Autonomous scam engagement & intelligence extraction system",
    version="1.0.0"
)

# =========================
# API KEY AUTH (Evaluator Compatible)
# =========================
API_KEY = os.getenv("API_KEY")

def verify_api_key(
    authorization: str = Header(default=None),
    x_api_key: str = Header(default=None)
):
    def normalize(key: str):
        return key.replace("Bearer", "").strip()

    if authorization and normalize(authorization) == API_KEY:
        return API_KEY

    if x_api_key and normalize(x_api_key) == API_KEY:
        return API_KEY

    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# =========================
# MEMORY STORE (In-Memory)
# =========================
conversation_memory: Dict[str, List[Dict]] = {}

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class ScamRequest(BaseModel):
    conversation_id: str
    message: str

class ScamResponse(BaseModel):
    conversation_id: str
    scam_detected: bool
    agent_active: bool
    agent_reply: str
    confidence: float
    engagement: Dict
    extracted_intelligence: Dict
    summary: Dict

# =========================
# SCAM DETECTION LOGIC
# =========================
SCAM_KEYWORDS = [
    "verify", "urgent", "account", "suspend", "blocked",
    "bank", "otp", "refund", "click", "link", "kyc",
    "penalty", "security", "limit"
]

def detect_scam(message: str):
    detected = [k for k in SCAM_KEYWORDS if k in message.lower()]
    return len(detected) > 0, detected

# =========================
# INTELLIGENCE EXTRACTION
# =========================
def extract_intelligence(text: str):
    return {
        "bank_accounts": re.findall(r"\b\d{9,18}\b", text),
        "upi_ids": re.findall(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}", text),
        "urls": re.findall(r"https?://\S+|www\.\S+", text)
    }

# =========================
# CONFIDENCE SCORING
# =========================
def calculate_confidence(keywords, extracted):
    score = 0.15 * len(keywords)
    score += 0.25 if extracted["urls"] else 0
    score += 0.30 if extracted["upi_ids"] else 0
    score += 0.35 if extracted["bank_accounts"] else 0
    return round(min(score, 1.0), 2)

# =========================
# LLM AGENT (OpenAI + Fallback)
# =========================
def generate_agent_reply(history: List[Dict]):
    try:
        from openai import OpenAI
        client = OpenAI()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a normal bank customer. "
                    "You are confused but cooperative. "
                    "Never accuse or reveal detection. "
                    "Your goal is to get payment details, UPI IDs, or links."
                
                    "- Act slightly confused and cautious."
                    "- Ask for clarification."
                    "- If a link or payment fails, ask for another."
                    "- Ask for UPI ID or bank account politely."
                    "- Delay actions and seek confirmation."
                    "- Do NOT rush. Be believable."
                    "- Using the various tactics make the scammer confuse and get the scam details"
                    "Do not reveal to the scammer that you identified aout the scam."
                    "Keep the person engaged and get details extract the details."
                )
            }
        ]

        for h in history:
            role = "user" if h["role"] == "scammer" else "assistant"
            messages.append({"role": role, "content": h["text"]})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.9,
            top_p=0.9,
            presence_penalty=0.7,
            frequency_penalty=0.5,
            messages=messages,
            timeout=8
        )

        return response.choices[0].message.content

    except Exception:
        # Safe fallback
        return random.choice([
            "I’m not fully understanding this, can you guide me step by step?",
            "The link seems slow, is there another way to complete this?",
            "Is this related to my savings account or current account?",
            "Do I need to pay something to avoid suspension?"
        ])

# =========================
# TERMINATION LOGIC
# =========================
def should_terminate(turns: int, extracted: Dict):
    if extracted["bank_accounts"] or extracted["upi_ids"]:
        return True
    if turns >= 12:
        return True
    return False

# =========================
# MAIN API ENDPOINT
# =========================
@app.post("/scam-agent", response_model=ScamResponse)
def scam_agent(
    req: ScamRequest,
    api_key: str = Depends(verify_api_key)
):
    cid = req.conversation_id
    msg = req.message

    if cid not in conversation_memory:
        conversation_memory[cid] = []

    conversation_memory[cid].append({
        "role": "scammer",
        "text": msg,
        "timestamp": time.time()
    })

    previous = any(
    detect_scam(h["text"]) for h in conversation_memory[cid]
)
    scam_detected, keywords = previous or detect_scam(msg)
    extracted = extract_intelligence(msg)
    confidence = calculate_confidence(keywords, extracted)

    agent_active = scam_detected
    reply = ""

    if agent_active:
        if should_terminate(len(conversation_memory[cid]), extracted):
            reply = "Okay, I’ll check this and update you shortly."
            agent_active = False
        else:
            reply = generate_agent_reply(conversation_memory[cid])

        conversation_memory[cid].append({
            "role": "agent",
            "text": reply,
            "timestamp": time.time()
        })

    response = ScamResponse(
        conversation_id=cid,
        scam_detected=scam_detected,
        agent_active=agent_active,
        agent_reply=reply,
        confidence=confidence,
        engagement={
            "turns": len(conversation_memory[cid]),
            "agent_engaged": agent_active
        },
        extracted_intelligence=extracted,
        summary={
            "total_turns": len(conversation_memory[cid]),
            "detected_keywords": keywords,
            "extracted_counts": {
                "bank_accounts": len(extracted["bank_accounts"]),
                "upi_ids": len(extracted["upi_ids"]),
                "urls": len(extracted["urls"])
            },
            "conversation_closed": not agent_active
        }
    )

    return response
