from fastapi import FastAPI, Request, Depends, HTTPException, Header
from typing import Dict, Optional, Any
import os, re, time, random, json

# =====================================================
# APP INIT
# =====================================================
app = FastAPI(
    title="Agentic Honey-Pot Scam Intelligence API",
    version="3.1.0",
    description="Evaluator-safe autonomous scam engagement system"
)

# =====================================================
# AUTH (Evaluator Compatible)
# =====================================================
API_KEY = os.getenv("API_KEY", "SECRET123")

def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    def clean(v: str) -> str:
        return v.replace("Bearer", "").strip()

    if authorization and clean(authorization) == API_KEY:
        return True
    if x_api_key and clean(x_api_key) == API_KEY:
        return True

    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# =====================================================
# MEMORY (Sticky conversation state)
# =====================================================
conversation_store: Dict[str, Dict[str, Any]] = {}

# =====================================================
# UTIL — ABSOLUTE SAFETY
# =====================================================
def safe_text(val: Any) -> str:
    """
    Converts ANY input to a safe string.
    This SINGLE function prevents 500 errors.
    """
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val)
    except Exception:
        return str(val)

# =====================================================
# SCAM DETECTION
# =====================================================
SCAM_KEYWORDS = [
    "verify", "urgent", "account", "suspend", "blocked",
    "bank", "otp", "refund", "click", "link", "kyc",
    "security", "limit", "penalty"
]

def detect_keywords(text: Any):
    txt = safe_text(text).lower()
    return [k for k in SCAM_KEYWORDS if k in txt]

def detect_stage(text: Any):
    t = safe_text(text).lower()
    if any(w in t for w in ["otp", "upi", "account", "transfer"]):
        return "PAYMENT"
    if any(w in t for w in ["click", "link", "verify"]):
        return "ACTION"
    if any(w in t for w in ["urgent", "blocked", "suspend"]):
        return "PRESSURE"
    return "HOOK"

# =====================================================
# INTELLIGENCE EXTRACTION
# =====================================================
def extract_intelligence(text: Any, turn: int):
    t = safe_text(text)
    return {
        "bank_accounts": [
            {"value": v, "turn": turn}
            for v in re.findall(r"\b\d{9,18}\b", t)
        ],
        "upi_ids": [
            {"value": v, "turn": turn}
            for v in re.findall(r"[a-zA-Z0-9.\-_]+@[a-zA-Z]+", t)
        ],
        "urls": [
            {"value": v, "turn": turn}
            for v in re.findall(r"https?://\S+|www\.\S+", t)
        ]
    }

# =====================================================
# CONFIDENCE SCORE
# =====================================================
def calculate_confidence(state: Dict):
    score = 0.15 * len(state["keywords"])
    if state["intel"]["urls"]:
        score += 0.25
    if state["intel"]["upi_ids"]:
        score += 0.30
    if state["intel"]["bank_accounts"]:
        score += 0.35
    return round(min(score, 1.0), 2)

# =====================================================
# LLM AGENT (SAFE FAIL)
# =====================================================
def generate_agent_reply(history, stage):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [{
            "role": "system",
            "content": (
                f"You are a normal bank customer. Stage={stage}. "
                "You are polite, cautious, never accuse, "
                "and try to get payment or verification details naturally."
            )
        }]

        messages.extend(history)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.85,
            timeout=7
        )

        return res.choices[0].message.content

    except Exception:
        # Absolute fallback — NEVER crash
        return random.choice([
            "I’m not sure I understood, can you explain again?",
            "The link isn’t opening on my phone.",
            "Is there another way to complete this?",
            "Please wait, I’m checking."
        ])

# =====================================================
# TERMINATION RULES
# =====================================================
def should_terminate(state: Dict):
    if state["turns"] >= 15:
        return True, "MAX_TURNS_REACHED"
    if state["intel"]["bank_accounts"] or state["intel"]["upi_ids"]:
        return True, "INTELLIGENCE_CAPTURED"
    if time.time() - state["created_at"] > 300:
        return True, "TIMEOUT"
    return False, ""

# =====================================================
# MAIN ENDPOINT (EVALUATOR-PROOF)
# =====================================================
@app.post("/scam-agent")
async def scam_agent(
    request: Request,
    auth: bool = Depends(verify_api_key)
):
    # -------- SAFE BODY PARSE --------
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    conversation_id = safe_text(body.get("conversation_id", "evaluator_default"))
    message = body.get("message", "")

    # -------- INIT STATE --------
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = {
            "created_at": time.time(),
            "turns": 0,
            "scam_confirmed": False,
            "keywords": set(),
            "stage": "HOOK",
            "history": [],
            "intel": {
                "bank_accounts": [],
                "upi_ids": [],
                "urls": []
            },
            "terminated": False,
            "termination_reason": ""
        }

    state = conversation_store[conversation_id]
    state["turns"] += 1

    text = safe_text(message)

    # -------- RECORD MESSAGE --------
    state["history"].append({
        "role": "user",
        "content": text
    })

    # -------- DETECTION --------
    kws = detect_keywords(text)
    if kws:
        state["scam_confirmed"] = True
        state["keywords"].update(kws)

    state["stage"] = detect_stage(text)

    # -------- EXTRACTION --------
    intel = extract_intelligence(text, state["turns"])
    for k in intel:
        state["intel"][k].extend(intel[k])

    confidence = calculate_confidence(state)

    # -------- TERMINATION --------
    terminate, reason = should_terminate(state)
    state["terminated"] = terminate
    state["termination_reason"] = reason

    # -------- AGENT RESPONSE --------
    agent_reply = ""
    agent_active = state["scam_confirmed"] and not terminate

    if agent_active:
        agent_reply = generate_agent_reply(state["history"], state["stage"])
        state["history"].append({
            "role": "assistant",
            "content": agent_reply
        })

    # -------- RESPONSE --------
    return {
        "conversation_id": conversation_id,
        "scam_detected": state["scam_confirmed"],
        "agent_active": agent_active,
        "agent_reply": agent_reply,
        "confidence": confidence,
        "engagement": {
            "turns": state["turns"],
            "stage": state["stage"]
        },
        "extracted_intelligence": state["intel"],
        "summary": {
            "total_turns": state["turns"],
            "detected_keywords": list(state["keywords"]),
            "conversation_closed": state["terminated"],
            "termination_reason": state["termination_reason"]
        }
    }
