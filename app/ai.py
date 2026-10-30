"""
All AI logic lives here.

PRIMARY path: Groq LLM does the real work (understands messy human text,
which rules cannot do reliably).
FALLBACK path: if Groq is unavailable (no key, timeout, bad JSON), a small
rule-based extractor still returns *something* so the app keeps working.

Every AI call is timed and its outcome (success / fallback / error) is logged.

AI features implemented:
  1. extract_recap()      -> summary + decisions + explicit action items (JSON)
  2. catch_commitments()  -> second pass that finds IMPLIED promises
  3. generate_brief()     -> a short catch-up brief for one person
  4. embed_text()         -> returns None (Groq has no embeddings endpoint)
"""

import json
import logging
import os
import re
import time

from dotenv import load_dotenv

from .logging_config import logger, log_event

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()

AI_ENABLED = False
_client = None
_selected_model: str | None = None

try:
    if GROQ_API_KEY:
        from groq import Groq

        _client = Groq(api_key=GROQ_API_KEY)

        if GROQ_MODEL:
            _selected_model = GROQ_MODEL
            logger.info("Groq AI configured with user-specified model: %s", _selected_model)
        else:
            models = _client.models.list()
            candidates = []
            for m in models.data:
                mid = m.id.lower()
                if any(skip in mid for skip in ("whisper", "tts", "guard", "vision", "embed")):
                    continue
                candidates.append(m.id)
            preferred = [c for c in candidates if "versatile" in c.lower() or ("llama" in c.lower() and "instruct" not in c.lower())]
            if not preferred:
                preferred = [c for c in candidates if "llama" in c.lower()]
            if not preferred:
                preferred = candidates
            if preferred:
                _selected_model = preferred[0]
            else:
                _selected_model = candidates[0] if candidates else None

            if _selected_model:
                logger.info("Groq AI auto-selected model: %s", _selected_model)
            else:
                logger.warning("No suitable Groq model found — running in fallback mode")

        if _selected_model:
            AI_ENABLED = True
    else:
        logger.warning("GROQ_API_KEY not set — running in rule-based fallback mode")
except Exception:
    logger.exception("Failed to configure Groq — running in rule-based fallback mode")


def _call_groq(prompt: str, purpose: str, *, json_mode: bool = False) -> str | None:
    """
    Make one Groq chat completion call, timed and logged.
    Returns the text, or None on any failure.
    """
    if not AI_ENABLED or not _client or not _selected_model:
        return None
    start = time.perf_counter()
    try:
        kwargs = {
            "model": _selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = _client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(
            logging.INFO,
            "AI call succeeded",
            type="ai.call",
            purpose=purpose,
            provider="groq",
            model=_selected_model,
            outcome="success",
            latency_ms=latency_ms,
        )
        return text
    except Exception:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log_event(
            logging.ERROR,
            "AI call failed",
            type="ai.call",
            purpose=purpose,
            provider="groq",
            model=_selected_model,
            outcome="error",
            latency_ms=latency_ms,
        )
        logger.exception("Groq call raised for purpose=%s", purpose)
        return None


def _extract_json(text: str):
    """
    LLMs sometimes wrap JSON in ```json ... ``` fences or add stray words.
    This pulls out the first {...} block and parses it. Returns None if it
    can't, so the caller can fall back.
    """
    if not text:
        return None
    text = re.sub(r"```(json)?", "", text).strip("` \n")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# 1. EXTRACTION: raw mess -> structured recap
# ---------------------------------------------------------------------------

def extract_recap(raw_text: str) -> dict:
    """
    Returns: {"summary": str, "decisions": [str], "action_items": [{description, owner, due_date}], "ai_used": bool}
    """
    prompt = f"""You are an assistant that turns messy meeting/handover notes into structured data.
Return ONLY valid JSON, no explanation, in exactly this shape:
{{
  "summary": "2-3 sentence plain-language summary",
  "decisions": ["each decision as a short sentence"],
  "action_items": [
    {{"description": "what needs doing", "owner": "person name or Unassigned", "due_date": "date/day if mentioned, else empty string"}}
  ]
}}

NOTES:
\"\"\"{raw_text}\"\"\"
"""
    text = _call_groq(prompt, purpose="extract_recap", json_mode=True)
    data = _extract_json(text)
    if data and isinstance(data, dict):
        return {
            "summary": str(data.get("summary", "")),
            "decisions": [str(d) for d in data.get("decisions", []) if d],
            "action_items": [
                {
                    "description": str(a.get("description", "")).strip(),
                    "owner": str(a.get("owner", "Unassigned")).strip() or "Unassigned",
                    "due_date": str(a.get("due_date", "")).strip(),
                }
                for a in data.get("action_items", [])
                if a.get("description")
            ],
            "ai_used": AI_ENABLED and text is not None,
        }

    log_event(logging.WARNING, "extract_recap using rule-based fallback", type="ai.fallback", purpose="extract_recap")
    return _rule_based_extract(raw_text)


def _rule_based_extract(raw_text: str) -> dict:
    """Very simple keyword extraction so the app still works without AI."""
    chunks = []
    for ln in raw_text.splitlines():
        chunks.extend(re.split(r"(?<=[.!?])\s+", ln))
    lines = [c.strip(" -*\t") for c in chunks if c.strip()]
    action_items = []
    decisions = []
    action_words = ("will ", "to do", "action", "follow up", "send", "prepare", "finish", "complete", "assign")
    decision_words = ("decided", "agreed", "we will go with", "chose", "approved")
    for ln in lines:
        low = ln.lower()
        if any(w in low for w in decision_words):
            decisions.append(ln)
        elif any(w in low for w in action_words):
            owner = "Unassigned"
            first_word = ln.split()[0] if ln.split() else ""
            if first_word[:1].isupper() and first_word.lower() not in ("we", "the", "a"):
                owner = first_word
            action_items.append({"description": ln, "owner": owner, "due_date": ""})
    summary = " ".join(lines[:2])[:400] if lines else ""
    return {
        "summary": summary,
        "decisions": decisions,
        "action_items": action_items,
        "ai_used": False,
    }


# ---------------------------------------------------------------------------
# 2. SPECIAL FEATURE: Commitment Catcher (second AI pass)
# ---------------------------------------------------------------------------

def catch_commitments(raw_text: str, already_found: list[str]) -> list[dict]:
    """
    Finds IMPLIED commitments that the normal extraction missed — casual
    promises people make and never log.
    """
    if not AI_ENABLED:
        return []

    found_text = "\n".join(f"- {d}" for d in already_found) or "(none)"
    prompt = f"""Read these notes. Some action items were already captured (listed below).
Find ADDITIONAL *implied* commitments or promises that were NOT already captured —
casual phrases like "I'll loop in X", "let me check and get back", "we should follow up on".
Do NOT repeat anything already captured. If there are none, return an empty list.

Return ONLY JSON: {{"suggestions": [{{"description": "...", "owner": "name or Unassigned"}}]}}

ALREADY CAPTURED:
{found_text}

NOTES:
\"\"\"{raw_text}\"\"\"
"""
    text = _call_groq(prompt, purpose="catch_commitments", json_mode=True)
    data = _extract_json(text)
    if data and isinstance(data, dict):
        return [
            {
                "description": str(s.get("description", "")).strip(),
                "owner": str(s.get("owner", "Unassigned")).strip() or "Unassigned",
                "due_date": "",
            }
            for s in data.get("suggestions", [])
            if s.get("description")
        ]
    return []


# ---------------------------------------------------------------------------
# 3. Catch-up brief for one person
# ---------------------------------------------------------------------------

def generate_brief(person: str, context_items: list[dict]) -> str:
    """
    Writes a short brief addressed to `person`, focused on what THEY need to
    know / act on. Falls back to a plain formatted list if AI is unavailable.
    """
    if not context_items:
        return f"There are currently no open items or recent context relevant to {person}."

    lines = "\n".join(
        f"- [{it['status']}] {it['description']} (owner: {it['owner']}, due: {it['due_date'] or 'n/a'})"
        for it in context_items
    )
    prompt = f"""Write a short, friendly catch-up brief (max 6 sentences) addressed directly to {person},
who missed a meeting or is taking over work. Lead with what THEY are responsible for.
Be specific and actionable. Here is the relevant context:

{lines}
"""
    text = _call_groq(prompt, purpose="generate_brief")
    if text:
        return text.strip()

    log_event(logging.WARNING, "generate_brief using fallback", type="ai.fallback", purpose="generate_brief")
    return f"Catch-up brief for {person} (AI unavailable, showing raw items):\n{lines}"


# ---------------------------------------------------------------------------
# 4. Embeddings — Groq has no embeddings endpoint
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float] | None:
    """Returns None — Groq does not provide an embeddings API."""
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measures how similar two vectors are: 1.0 = identical direction,
    0 = unrelated. Pure Python, no numpy needed for a small KB.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
