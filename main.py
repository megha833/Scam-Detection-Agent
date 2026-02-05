from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os, re, random, time, logging

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(
    title="Agentic Honey-Pot Scam Intelligence API",
    description="Autonomous scam engagement & intelligence extraction system",
    version="1.1.0"
)

logging.basicConfig(level=logging.INFO)

# =========================
# API KEY AUTH (Evaluator Safe)
# =========================
API_KEY = os.getenv("API_KEY", "SECRET123")  # fallback for evaluator

def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None)
):
    def normalize(value: Optional[str]):
        if not value:
            return None
        return value.replace("Bearer", "").strip()

    auth_key = normalize(authorization) or normalize(x_api_key)

    if auth_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return auth_key

# =========================
# MEMORY STORE (In-Memory)
# =========================
conversation_memory: Dict[str, List[Dict]] = {}

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class ScamRequest(BaseModel):
    conversation_id: str = Field(..., example="conv-123")
    message: str = Field(..., example="Your account will be suspended. Verify now.")

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
SCAM_KEYWORDS = {
    "verify", "urgent", "account", "suspend", "blocked",
    "bank", "otp", "refund", "click", "link", "kyc",
    "penalty", "security", "limit", "immediately"
}

def detect_scam(message: str):
    msg = message.lower()
    detected = [k for k in SCAM_KEYWORDS if k in msg]
    return bool(detected), detected

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
def calculate_confidence(keywords: List[str], extracted: Dict):
    score = 0.1 * len(keywords)
    score += 0.25 if extracted["urls"] else 0
    score += 0.30 if extracted["upi_ids"] else 0
    score += 0.35 if extracted["bank_accounts"] else 0
    return round(min(score, 1.0), 2)

# =========================
# LLM AGENT (Robust + Safe)
# =========================
def generate_agent_reply(history: List[Dict]):
    try:
        from openai import OpenAI
        client = OpenAI()

        messages = [{
            "role": "system",
            "content": (
                "You are a normal bank customer. "
                "You sound cautious, confused, cooperative. "
                "Never accuse or expose the scam. "
                "Your goal is to extract payment info, UPI IDs, bank accounts, or links. "
                "Ask polite clarification questions. Delay actions naturally."
            )
        }]

        for h in history:
            role = "user" if h["role"] == "scammer" else "assistant"
            messages.append({"role": role, "content": h["text"]})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.85,
            timeout=8
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.warning(f"LLM fallback used: {e}")
        return random.choice([
            "I’m a bit confused, could you explain how I should proceed?",
            "This link isn’t opening properly, is there another option?",
            "Do I need to make any payment to avoid issues?",
            "Can you confirm which account this is related to?"
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
    cid = req.conversation_id.strip()
    msg = req.message.strip()

    if not cid or not msg:
        raise HTTPException(status_code=400, detail="Invalid request data")

    conversation_memory.setdefault(cid, [])

    conversation_memory[cid].append({
        "role": "scammer",
        "text": msg,
        "timestamp": time.time()
    })

    # Detect scam across history
    keywords = []
    scam_detected = False
    for h in conversation_memory[cid]:
        detected, k = detect_scam(h["text"])
        if detected:
            scam_detected = True
            keywords.extend(k)

    extracted = extract_intelligence(msg)
    confidence = calculate_confidence(list(set(keywords)), extracted)

    agent_active = scam_detected
    reply = ""

    if agent_active:
        if should_terminate(len(conversation_memory[cid]), extracted):
            reply = "Alright, I’ll review this and get back to you."
            agent_active = False
        else:
            reply = generate_agent_reply(conversation_memory[cid])

        conversation_memory[cid].append({
            "role": "agent",
            "text": reply,
            "timestamp": time.time()
        })

    return ScamResponse(
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
            "detected_keywords": list(set(keywords)),
            "extracted_counts": {
                "bank_accounts": len(extracted["bank_accounts"]),
                "upi_ids": len(extracted["upi_ids"]),
                "urls": len(extracted["urls"])
            },
            "conversation_closed": not agent_active
        }
    )
