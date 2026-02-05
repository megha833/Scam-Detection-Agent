from fastapi import FastAPI, Depends, HTTPException, Header, Body
from pydantic import BaseModel
from typing import Dict, Optional
import os, re, random, time

# =====================================================
# APP INITIALIZATION
# =====================================================
app = FastAPI(
    title="Agentic Honey-Pot Scam Intelligence API (Winner-Level v3)",
    description="Autonomous scam engagement, deception & intelligence extraction system",
    version="3.0.0"
)

# =====================================================
# API KEY AUTH
# =====================================================
API_KEY = os.getenv("API_KEY", "SECRET123")

def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    def clean(v: str):
        return v.replace("Bearer", "").strip()

    if authorization and clean(authorization) == API_KEY:
        return True
    if x_api_key and clean(x_api_key) == API_KEY:
        return True

    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# =====================================================
# MEMORY STORE
# =====================================================
conversation_store: Dict[str, Dict] = {}

# =====================================================
# MODELS
# =====================================================
class ScamRequest(BaseModel):
    conversation_id: Optional[str] = "evaluator_default"
    message: Optional[str] = ""

class ScamResponse(BaseModel):
    conversation_id: str
    scam_detected: bool
    agent_active: bool
    agent_reply: str
    confidence: float
    engagement: Dict
    extracted_intelligence: Dict
    summary: Dict

# =====================================================
# SCAM LOGIC
# =====================================================
SCAM_KEYWORDS = [
    "verify", "urgent", "account", "suspend", "blocked",
    "bank", "otp", "refund", "click", "link", "kyc"
]

def detect_keywords(text: str):
    return [k for k in SCAM_KEYWORDS if k in text.lower()]

def detect_stage(text: str):
    t = text.lower()
    if any(x in t for x in ["otp", "upi", "transfer"]):
        return "PAYMENT"
    if any(x in t for x in ["click", "link", "verify"]):
        return "ACTION"
    if any(x in t for x in ["urgent", "blocked"]):
        return "PRESSURE"
    return "HOOK"

# =====================================================
# INTELLIGENCE EXTRACTION
# =====================================================
def extract_intelligence(text: str, turn: int):
    return {
        "bank_accounts": [{"value": v, "turn": turn}
                          for v in re.findall(r"\b\d{9,18}\b", text)],
        "upi_ids": [{"value": v, "turn": turn}
                    for v in re.findall(r"[a-zA-Z0-9.\-_]+@[a-zA-Z]+", text)],
        "urls": [{"value": v, "turn": turn}
                 for v in re.findall(r"https?://\S+|www\.\S+", text)]
    }

# =====================================================
# CONFIDENCE SCORE
# =====================================================
def calculate_confidence(state):
    score = 0.2 * len(state["keywords"])
    score += 0.3 if state["extracted"]["urls"] else 0
    score += 0.3 if state["extracted"]["upi_ids"] else 0
    score += 0.4 if state["extracted"]["bank_accounts"] else 0
    return round(min(score, 1.0), 2)

# =====================================================
# LLM AGENT
# =====================================================
def generate_agent_reply(history, state):
    try:
        from openai import OpenAI
        client = OpenAI()

        messages = [{"role": "system", "content": "You are a confused bank customer. Never accuse. Delay actions."}]
        for h in history:
            role = "user" if h["role"] == "scammer" else "assistant"
            messages.append({"role": role, "content": h["text"]})

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.9,
            timeout=7
        )
        return res.choices[0].message.content

    except Exception:
        return random.choice([
            "Can you explain that again?",
            "I'm not sure this is working.",
            "Is there another option?"
        ])

# =====================================================
# TERMINATION RULES
# =====================================================
def should_terminate(state):
    if state["turns"] >= 15:
        return True, "MAX_TURNS"
    if state["extracted"]["bank_accounts"] or state["extracted"]["upi_ids"]:
        return True, "DATA_CAPTURED"
    return False, ""

# =====================================================
# MAIN ENDPOINT
# =====================================================
@app.post("/scam-agent", response_model=ScamResponse)
def scam_agent(
    req: Optional[ScamRequest] = Body(default=None),
    auth: bool = Depends(verify_api_key)
):
    # ✅ HANDLE EMPTY BODY (NO 422)
    if req is None:
        req = ScamRequest()

    cid = req.conversation_id
    msg = (req.message or "").strip()

    if cid not in conversation_store:
        conversation_store[cid] = {
            "created_at": time.time(),
            "turns": 0,
            "scam_confirmed": False,
            "keywords": set(),
            "stage": "HOOK",
            "history": [],
            "extracted": {"bank_accounts": [], "upi_ids": [], "urls": []},
            "terminated": False,
            "termination_reason": ""
        }

    state = conversation_store[cid]
    state["turns"] += 1
    state["history"].append({"role": "scammer", "text": msg})

    kws = detect_keywords(msg)
    if kws:
        state["scam_confirmed"] = True
        state["keywords"].update(kws)

    state["stage"] = detect_stage(msg)

    extracted = extract_intelligence(msg, state["turns"])
    for k in extracted:
        state["extracted"][k].extend(extracted[k])

    confidence = calculate_confidence(state)

    terminate, reason = should_terminate(state)
    state["terminated"] = terminate
    state["termination_reason"] = reason

    agent_reply = ""
    agent_active = state["scam_confirmed"] and not terminate

    if agent_active:
        agent_reply = generate_agent_reply(state["history"], state)
        state["history"].append({"role": "agent", "text": agent_reply})

    return ScamResponse(
        conversation_id=cid,
        scam_detected=state["scam_confirmed"],
        agent_active=agent_active,
        agent_reply=agent_reply,
        confidence=confidence,
        engagement={"turns": state["turns"], "stage": state["stage"]},
        extracted_intelligence=state["extracted"],
        summary={
            "keywords": list(state["keywords"]),
            "termination_reason": state["termination_reason"],
            "conversation_closed": state["terminated"]
        }
    )
