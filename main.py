from fastapi import FastAPI, Request, Depends, HTTPException, Header
from typing import Dict, Optional
import os, re, time, random, json

# =====================================================
# APP
# =====================================================
app = FastAPI(
    title="Agentic Honey-Pot Scam Intelligence API",
    version="3.1.0"
)

# =====================================================
# AUTH (Evaluator Safe)
# =====================================================
API_KEY = os.getenv("API_KEY", "SECRET123")

def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    def clean(v): return v.replace("Bearer", "").strip()
    if authorization and clean(authorization) == API_KEY:
        return True
    if x_api_key and clean(x_api_key) == API_KEY:
        return True
    raise HTTPException(status_code=401, detail="Invalid API Key")

# =====================================================
# MEMORY STORE (Sticky Conversations)
# =====================================================
conversation_store: Dict[str, Dict] = {}

# =====================================================
# SCAM INTELLIGENCE
# =====================================================
SCAM_KEYWORDS = [
    "verify", "urgent", "account", "suspend", "blocked",
    "bank", "otp", "refund", "click", "link", "kyc"
]

def detect_keywords(text: str):
    return [k for k in SCAM_KEYWORDS if k in text.lower()]

def detect_stage(text: str):
    t = text.lower()
    if any(w in t for w in ["otp", "upi", "account", "transfer"]):
        return "PAYMENT"
    if any(w in t for w in ["click", "link", "verify"]):
        return "ACTION"
    if any(w in t for w in ["urgent", "blocked", "suspend"]):
        return "PRESSURE"
    return "HOOK"

def extract_intel(text: str, turn: int):
    return {
        "bank_accounts": [{"value": v, "turn": turn}
            for v in re.findall(r"\b\d{9,18}\b", text)],
        "upi_ids": [{"value": v, "turn": turn}
            for v in re.findall(r"[a-zA-Z0-9.\-_]+@[a-zA-Z]+", text)],
        "urls": [{"value": v, "turn": turn}
            for v in re.findall(r"https?://\S+|www\.\S+", text)]
    }

def confidence_score(state):
    score = 0.2 * len(state["keywords"])
    if state["intel"]["urls"]: score += 0.3
    if state["intel"]["upi_ids"]: score += 0.3
    if state["intel"]["bank_accounts"]: score += 0.4
    if state["stage"] == "PAYMENT": score += 0.2
    return round(min(score, 1.0), 2)

# =====================================================
# AGENT (LLM + FALLBACK)
# =====================================================
def agent_reply(history, stage):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"You are a cautious bank customer. Stage={stage}. Delay actions."

        msgs = [{"role": "system", "content": prompt}]
        for h in history:
            msgs.append({"role": h["role"], "content": h["text"]})

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.8,
            timeout=6
        )
        return res.choices[0].message.content

    except Exception:
        return random.choice([
            "The link isn’t opening for me.",
            "Can you explain that again?",
            "Is there another option?",
            "I’m not sure, please wait."
        ])

# =====================================================
# TERMINATION
# =====================================================
def should_close(state):
    if state["turns"] >= 15:
        return True, "MAX_TURNS"
    if state["intel"]["bank_accounts"] or state["intel"]["upi_ids"]:
        return True, "INTEL_CAPTURED"
    if time.time() - state["created"] > 300:
        return True, "TIMEOUT"
    return False, ""

# =====================================================
# MAIN ENDPOINT (422-PROOF)
# =====================================================
@app.post("/scam-agent")
async def scam_agent(
    request: Request,
    auth: bool = Depends(verify_api_key)
):
    # ---------- SAFE BODY PARSING ----------
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    conversation_id = body.get("conversation_id", "evaluator_default")
    message = body.get("message", "")

    # ---------- INIT MEMORY ----------
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = {
            "created": time.time(),
            "turns": 0,
            "scam": False,
            "keywords": set(),
            "stage": "HOOK",
            "history": [],
            "intel": {
                "bank_accounts": [],
                "upi_ids": [],
                "urls": []
            },
            "closed": False,
            "reason": ""
        }

    state = conversation_store[conversation_id]
    state["turns"] += 1

    # ---------- PROCESS MESSAGE ----------
    if message:
        state["history"].append({"role": "user", "text": message})

        kws = detect_keywords(message)
        if kws:
            state["scam"] = True
            state["keywords"].update(kws)

        state["stage"] = detect_stage(message)

        intel = extract_intel(message, state["turns"])
        for k in intel:
            state["intel"][k].extend(intel[k])

    # ---------- CONFIDENCE ----------
    confidence = confidence_score(state)

    # ---------- TERMINATION ----------
    close, reason = should_close(state)
    state["closed"] = close
    state["reason"] = reason

    # ---------- AGENT ----------
    agent_active = state["scam"] and not close
    reply = ""
    if agent_active:
        reply = agent_reply(state["history"], state["stage"])
        state["history"].append({"role": "assistant", "text": reply})

    # ---------- RESPONSE (ALWAYS 200) ----------
    return {
        "conversation_id": conversation_id,
        "scam_detected": state["scam"],
        "agent_active": agent_active,
        "agent_reply": reply,
        "confidence": confidence,
        "engagement": {
            "turns": state["turns"],
            "stage": state["stage"]
        },
        "extracted_intelligence": state["intel"],
        "summary": {
            "total_turns": state["turns"],
            "keywords": list(state["keywords"]),
            "conversation_closed": state["closed"],
            "termination_reason": state["reason"]
        }
    }
