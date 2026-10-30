"""
The FastAPI application: all the HTTP routes (the API) plus the middleware that
assigns a correlation ID to every request and logs it.

This is the "entry point" — you run the app with:
    uvicorn app.main:app --reload

How a request flows (end to end):
  browser (static/index.html)  ->  HTTP request  ->  middleware assigns a
  correlation ID & logs start  ->  route function  ->  Pydantic validates the
  body  ->  crud.py talks to the DB / ai.py talks to Gemini  ->  response goes
  back  ->  middleware logs finish with status + latency.
"""

import io
import logging
import time

from fastapi import FastAPI, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import models, schemas, crud, ai
from .database import engine, get_db
from .logging_config import logger, log_event, correlation_id_ctx, new_correlation_id

# Create all database tables on startup (if they don't already exist).
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI-Powered Knowledge & Tasks Management Platform")


# ---------------------------------------------------------------------------
# Global exception handlers — every error returns {detail, correlation_id}
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    cid = correlation_id_ctx.get()
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "correlation_id": cid},
        headers={"X-Correlation-ID": cid},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    cid = correlation_id_ctx.get()
    parts = []
    for err in exc.errors():
        field = " → ".join(str(l) for l in err.get("loc", []) if l != "body")
        msg = err.get("msg", "invalid")
        parts.append(f"{field}: {msg}" if field else msg)
    readable = "; ".join(parts) or "Validation error"
    return JSONResponse(
        status_code=422,
        content={"detail": readable, "correlation_id": cid},
        headers={"X-Correlation-ID": cid},
    )


# ---------------------------------------------------------------------------
# Middleware: correlation ID + request logging (runs for EVERY request)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def correlation_and_logging(request: Request, call_next):
    cid = new_correlation_id()
    correlation_id_ctx.set(cid)
    start = time.perf_counter()
    log_event(logging.INFO, "request started", type="http.request_started",
              method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.exception("Unhandled error while processing request")
        log_event(logging.ERROR, "request failed", type="http.request_failed",
                  method=request.method, path=request.url.path, latency_ms=latency_ms)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error", "correlation_id": cid})
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    log_event(logging.INFO, "request finished", type="http.request_finished",
              method=request.method, path=request.url.path,
              status_code=response.status_code, latency_ms=latency_ms)
    # expose the correlation ID to the client for easy tracing
    response.headers["X-Correlation-ID"] = cid
    return response


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    result = {"status": "ok", "ai_enabled": ai.AI_ENABLED}
    if not ai.AI_ENABLED:
        if not ai.GROQ_API_KEY:
            result["ai_reason"] = "No GROQ_API_KEY configured"
        elif not ai._selected_model:
            result["ai_reason"] = "No suitable Groq model found"
        else:
            result["ai_reason"] = "AI initialization failed"
    return result


# ---------------------------------------------------------------------------
# Meetings
# ---------------------------------------------------------------------------
@app.post("/api/meetings", response_model=schemas.MeetingOut)
def create_meeting(payload: schemas.MeetingCreate,
                   carry_from: int | None = None,
                   db: Session = Depends(get_db)):
    """
    Create a meeting or handover. If carry_from=<previous meeting id> is passed,
    all still-open items from that meeting are carried forward into this one.
    """
    meeting = crud.create_meeting(db, payload.title, payload.kind,
                                  payload.participants, payload.agenda,
                                  payload.handover_from, payload.handover_to)
    if carry_from is not None:
        prev = crud.get_meeting(db, carry_from)
        if not prev:
            raise HTTPException(status_code=404, detail=f"carry_from meeting {carry_from} not found")
        crud.carry_forward_open_items(db, carry_from, meeting)
        db.refresh(meeting)
    return meeting


@app.get("/api/meetings", response_model=list[schemas.MeetingOut])
def list_meetings(db: Session = Depends(get_db)):
    return crud.list_meetings(db)


@app.get("/api/meetings/{meeting_id}", response_model=schemas.MeetingOut)
def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = crud.get_meeting(db, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Turn an uploaded file into plain text. Raises ValueError on failure."""
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_UPLOAD_EXTENSIONS)}")
    try:
        if ext in (".txt", ".md"):
            return content.decode("utf-8", errors="replace")
        elif ext == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        log_event(logging.ERROR, "file text extraction failed", type="upload.parse_error",
                  filename=filename, error=str(exc))
        raise ValueError(f"Could not parse '{filename}': {exc}")
    return ""


@app.post("/api/meetings/upload")
async def upload_meeting(
    title: str = Form(...),
    kind: str = Form("meeting"),
    participants: str = Form(""),
    agenda: str = Form(""),
    handover_from: str = Form(None),
    handover_to: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Create a meeting/handover from an uploaded transcript file in one call."""
    if kind not in ("meeting", "handover"):
        raise HTTPException(status_code=400, detail="kind must be 'meeting' or 'handover'")
    if not title.strip():
        raise HTTPException(status_code=400, detail="title is required")

    file_content = await file.read()
    log_event(logging.INFO, "file uploaded", type="upload.received",
              filename=file.filename, size_bytes=len(file_content),
              content_type=file.content_type)

    try:
        raw_text = extract_text_from_file(file.filename or "upload.txt", file_content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty or could not be read")

    parts = [p.strip() for p in participants.split(",") if p.strip()]
    meeting = crud.create_meeting(
        db, title.strip(), kind, parts, agenda,
        handover_from=handover_from or None,
        handover_to=handover_to or None,
    )

    recap = ai.extract_recap(raw_text)
    explicit_descriptions = [a["description"] for a in recap.get("action_items", [])]
    suggestions = ai.catch_commitments(raw_text, explicit_descriptions)

    if kind == "handover" and handover_to:
        assignee = handover_to.strip()
        for a in recap.get("action_items", []):
            if not a.get("owner") or a["owner"] == "Unassigned":
                a["owner"] = assignee
        for s in suggestions:
            if not s.get("owner") or s["owner"] == "Unassigned":
                s["owner"] = assignee

    meeting = crud.apply_recap_to_meeting(db, meeting, raw_text, recap, suggestions)
    meeting_data = schemas.MeetingOut.model_validate(meeting).model_dump()
    ai_used = recap.get("ai_used", False)
    meeting_data["ai_used"] = ai_used
    meeting_data["ai_message"] = "AI extraction succeeded" if ai_used else "AI unavailable — used basic keyword extraction"
    return meeting_data


@app.post("/api/meetings/{meeting_id}/extract")
def extract_into_meeting(meeting_id: int, payload: schemas.ExtractRequest,
                         db: Session = Depends(get_db)):
    """
    THE AI CORE: take raw messy text, extract a structured recap, run the
    Commitment Catcher, and save everything onto the meeting.
    """
    meeting = crud.get_meeting(db, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    recap = ai.extract_recap(payload.raw_text)
    explicit_descriptions = [a["description"] for a in recap.get("action_items", [])]
    suggestions = ai.catch_commitments(payload.raw_text, explicit_descriptions)

    meeting = crud.apply_recap_to_meeting(db, meeting, payload.raw_text, recap, suggestions)
    meeting_data = schemas.MeetingOut.model_validate(meeting).model_dump()
    ai_used = recap.get("ai_used", False)
    meeting_data["ai_used"] = ai_used
    meeting_data["ai_message"] = "AI extraction succeeded" if ai_used else "AI unavailable — used basic keyword extraction"
    return meeting_data


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------
@app.post("/api/meetings/{meeting_id}/items", response_model=schemas.ActionItemOut)
def add_item(meeting_id: int, payload: schemas.ActionItemCreate,
             db: Session = Depends(get_db)):
    if not crud.get_meeting(db, meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    return crud.create_action_item(db, meeting_id, payload)


@app.patch("/api/items/{item_id}", response_model=schemas.ActionItemOut)
def update_item(item_id: int, payload: schemas.ActionItemUpdate,
                db: Session = Depends(get_db)):
    item = crud.get_action_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")
    updated, error = crud.update_action_item(db, item, payload)
    if error:
        # invalid status transition -> 400 with a clear message
        raise HTTPException(status_code=400, detail=error)
    return updated


@app.get("/api/items/{item_id}/history", response_model=list[schemas.ActionItemOut])
def item_history(item_id: int, db: Session = Depends(get_db)):
    item = crud.get_action_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")
    return crud.get_item_history(db, item)


# ---------------------------------------------------------------------------
# Catch-up brief
# ---------------------------------------------------------------------------
@app.get("/api/brief/{person}")
def catch_up_brief(person: str, db: Session = Depends(get_db)):
    """Generate an AI catch-up brief for one person, based on their open items."""
    items = (
        db.query(models.ActionItem)
        .filter(models.ActionItem.owner.ilike(f"%{person}%"),
                models.ActionItem.status != "closed")
        .all()
    )
    context = [{"description": i.description, "owner": i.owner,
                "status": i.status, "due_date": i.due_date} for i in items]
    brief = ai.generate_brief(person, context)
    return {"person": person, "open_item_count": len(items), "brief": brief}


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
@app.post("/api/knowledge")
def add_knowledge(payload: schemas.KnowledgeDocCreate, db: Session = Depends(get_db)):
    doc = crud.create_knowledge_doc(db, payload.title, payload.content)
    return {"id": doc.id, "title": doc.title, "has_embedding": doc.embedding is not None}


@app.get("/api/knowledge")
def list_knowledge(db: Session = Depends(get_db)):
    docs = db.query(models.KnowledgeDoc).order_by(models.KnowledgeDoc.created_at.desc()).all()
    return [{"id": d.id, "title": d.title, "content": d.content} for d in docs]


@app.get("/api/knowledge/link")
def link_knowledge(q: str, db: Session = Depends(get_db)):
    """Given a query string q, return the most related knowledge docs."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="query 'q' cannot be empty")
    return {"query": q, "related": crud.link_related_docs(db, q)}


# ---------------------------------------------------------------------------
# Serve the single-page frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
