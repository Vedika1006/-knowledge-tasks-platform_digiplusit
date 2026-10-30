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

import logging
import time

from fastapi import FastAPI, Depends, HTTPException, Request
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
    return {"status": "ok", "ai_enabled": ai.AI_ENABLED}


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
                                  payload.participants, payload.agenda)
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


@app.post("/api/meetings/{meeting_id}/extract", response_model=schemas.MeetingOut)
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
    return meeting


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
