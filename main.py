from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os, re, random
from typing import Dict, List

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
# MEMORY (Per Conversation)
# ======================
conversation_memory: Dict[str, List[Dict]] = {}

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
    "account", "otp", "click", "link", "kyc",
    "suspend", "limited", "payment"
]

def detect_scam(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in SCAM_KEYWORDS)

# ======================
# INTELLIGENCE EXTRACTION
# ======================
def extract_intelligence(text: str) -> dict:
    return {
        "bank_accounts": re.findall(r"\b\d{9,18}\b", text),
        "upi_ids": re.findall(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}", text),
        "urls": re.findall(r"https?://\S+|www\.\S+", text)
    }

# ======================
# CONFIDENCE SCORING
# ======================
def calculate_confidence(message: str, extracted: dict) -> float:
    score = 0.0
    for k in SCAM_KEYWORDS:
        if k in message.lower():
            score += 0.08
    if extracted["urls"]:
        score += 0.3
    if extracted["upi_ids"]:
        score += 0.4
    if extracted["bank_accounts"]:
        score += 0.5
    return min(score, 1.0)

# ======================
# AGENT SYSTEM PROMPT
# ======================
SYSTEM_PROMPT = """
You are a normal bank customer.
You do NOT know this is a scam.
You must NEVER accuse, warn, or reveal detection.
Your goals:
- Keep the conversation going naturally
- Act slightly confused and cautious
- Ask for clarification
- If a link or payment fails, ask for another
- Ask for UPI ID or bank account politely
- Delay actions and seek confirmation
Do NOT rush. Be believable.
"""

# ======================
# AGENT RESPONSE (LLM + FALLBACK)
# ======================
def generate_agent_reply(history: List[Dict], extracted: dict) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Let LLM know what is already extracted
        messages.append({
            "role": "system",
            "content": f"Extracted so far: {extracted}. Try to obtain missing information naturally."
        })

        for h in history:
            role = "user" if h["role"] == "scammer" else "assistant"
            messages.append({"role": role, "content": h["text"]})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.9,
            top_p=0.9,
            presence_penalty=0.6,
            frequency_penalty=0.4,
            messages=messages,
            timeout=6
        )

        return response.choices[0].message.content.strip()

    except Exception:
        fallback = [
            "I tried the link but it didn’t open properly. Can you resend it?",
            "I’m getting an error in my UPI app. Is there another ID?",
            "Is there a bank transfer option instead?",
            "Can you explain the steps once more? I don’t want to make a mistake."
        ]
        return random.choice(fallback)

# ======================
# TERMINATION LOGIC
# ======================
def should_terminate(turns: int, extracted: dict) -> bool:
    if extracted["upi_ids"] and extracted["bank_accounts"]:
        return True
    if turns >= 10:
        return True
    return False

# ======================
# MAIN ROUTE
# ======================
@app.post("/scam-agent", response_model=ScamResponse)
def scam_agent(req: ScamRequest, api_key: str = Depends(verify_api_key)):

    cid = req.conversation_id
    msg = req.message

    if cid not in conversation_memory:
        conversation_memory[cid] = []

    conversation_memory[cid].append({"role": "scammer", "text": msg})

    previous = any(
    detect_scam(h["text"]) for h in conversation_memory[cid])
    scam_detected = previous or detect_scam(msg)

    extracted = extract_intelligence(msg)
    confidence = calculate_confidence(msg, extracted)

    agent_active = scam_detected
    reply = ""

    if agent_active:
        if should_terminate(len(conversation_memory[cid]), extracted):
            reply = "Okay, thanks. I’ll check this and get back to you."
            agent_active = False
        else:
            reply = generate_agent_reply(conversation_memory[cid], extracted)

        conversation_memory[cid].append({"role": "agent", "text": reply})

    summary = {
        "total_turns": len(conversation_memory[cid]),
        "detected_keywords": [k for k in SCAM_KEYWORDS if k in msg.lower()],
        "extracted_counts": {
            "bank_accounts": len(extracted["bank_accounts"]),
            "upi_ids": len(extracted["upi_ids"]),
            "urls": len(extracted["urls"])
        },
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
