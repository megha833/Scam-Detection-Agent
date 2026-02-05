from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os, re, random, time

# ======================
# APP INIT
# ======================
app = FastAPI(title="Agentic Honey-Pot Scam Intelligence API")

# ======================
# AUTH
# ======================
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "SECRET123")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# ======================
# MEMORY
# ======================
conversation_memory = {}

# ======================
# MODELS
# ======================
class ScamRequest(BaseModel):
    conversation_id: str
    message: str

class ScamResponse(BaseModel):
    conversation_id: str
    scam_detected: bool
    agent_active: bool
    agent_reply: str
    confidence: float
    engagement: dict
    extracted_intelligence: dict
    summary: dict

# ======================
# SCAM DETECTION
# ======================
SCAM_KEYWORDS = [
    "blocked", "verify", "urgent", "refund", "bank",
    "account", "otp", "click", "link", "kyc"
]

def detect_scam(message):
    return any(k in message.lower() for k in SCAM_KEYWORDS)

# ======================
# EXTRACTION
# ======================
def extract_intelligence(text):
    return {
        "bank_accounts": re.findall(r"\b\d{9,18}\b", text),
        "upi_ids": re.findall(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}", text),
        "urls": re.findall(r"https?://\S+|www\.\S+", text)
    }

# ======================
# CONFIDENCE
# ======================
def calculate_confidence(message, extracted):
    score = 0.2 * sum(k in message.lower() for k in SCAM_KEYWORDS)
    score += 0.3 if extracted["urls"] else 0
    score += 0.3 if extracted["upi_ids"] else 0
    score += 0.4 if extracted["bank_accounts"] else 0
    return min(score, 1.0)

# ======================
# AGENT (LLM + FALLBACK)
# ======================
def agent_reply(history):
    try:
        from openai import OpenAI
        client = OpenAI()

        messages = [
            {"role": "system", "content": "You are a confused but polite bank customer. Never accuse."}
        ]

        for h in history:
            role = "user" if h["role"] == "scammer" else "assistant"
            messages.append({"role": role, "content": h["text"]})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.85,
            top_p=0.9,
            presence_penalty=0.6,
            frequency_penalty=0.4,
            messages=messages,
            timeout=6
        )

        return response.choices[0].message.content

    except Exception:
        fallback = [
            "I’m not very familiar with this, can you explain?",
            "The link isn’t opening properly, is there another way?",
            "Which bank is this related to?",
            "Do I need to do this immediately?"
        ]
        return random.choice(fallback)

# ======================
# TERMINATION
# ======================
def should_terminate(turns, extracted):
    return turns >= 8 or extracted["upi_ids"] or extracted["bank_accounts"]

# ======================
# ROUTE
# ======================
@app.post("/scam-agent", response_model=ScamResponse)
def scam_agent(req: ScamRequest, api_key: str = Depends(verify_api_key)):

    cid = req.conversation_id
    msg = req.message

    if cid not in conversation_memory:
        conversation_memory[cid] = []

    conversation_memory[cid].append({"role": "scammer", "text": msg})

    scam_detected = detect_scam(msg)
    extracted = extract_intelligence(msg)
    confidence = calculate_confidence(msg, extracted)

    agent_active = scam_detected
    reply = ""

    if agent_active:
        if should_terminate(len(conversation_memory[cid]), extracted):
            reply = "Okay, thanks. I’ll check and get back to you."
            agent_active = False
        else:
            reply = agent_reply(conversation_memory[cid])

        conversation_memory[cid].append({"role": "agent", "text": reply})

    summary = {
        "total_turns": len(conversation_memory[cid]),
        "scam_type": "Financial / Phishing" if scam_detected else "Unknown",
        "conversation_closed": not agent_active
    }

    return ScamResponse(
        conversation_id=cid,
        scam_detected=scam_detected,
        agent_active=agent_active,
        agent_reply=reply,
        confidence=confidence,
        engagement={"turns": len(conversation_memory[cid])},
        extracted_intelligence=extracted,
        summary=summary
    )
