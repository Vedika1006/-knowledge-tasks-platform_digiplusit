"""
Database tables, defined as Python classes (SQLAlchemy ORM models).

Each class = one table. Each Column = one column. When the app starts we call
Base.metadata.create_all(), which creates these tables automatically if they
don't exist yet — so you never have to write CREATE TABLE SQL by hand.

Design decisions (be ready to explain these):

1. ONE "Meeting" table serves BOTH meetings and work-handovers, separated by a
   `kind` field. A handover is structurally the same thing: participants,
   notes, decisions, action items. One model = no duplicated logic.

2. Action items carry forward across meetings using `parent_item_id`. When a
   meeting ends with open items and the next meeting is created, each open item
   is COPIED to a new row whose parent points back to the original. Walking the
   parent chain gives the full history — nothing is ever overwritten, so we
   keep a real audit trail (this is the brief's "keeping their history").

3. Knowledge-base documents store an `embedding` (a vector of numbers that
   represents meaning) so we can link items by semantic similarity, not just
   keyword matching.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from .database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    # "meeting" or "handover"
    kind = Column(String(20), nullable=False, default="meeting")
    title = Column(String(200), nullable=False)
    # participants stored as a JSON list of names, e.g. ["Rahul", "Priya"]
    participants = Column(JSON, nullable=False, default=list)
    agenda = Column(Text, default="")
    # handover-specific: who is handing work over, and to whom
    handover_from = Column(String(120), nullable=True)
    handover_to = Column(String(120), nullable=True)
    # the original messy text the user pasted in
    raw_notes = Column(Text, default="")
    # AI-generated outputs
    summary = Column(Text, default="")
    # decisions stored as a JSON list of strings
    decisions = Column(JSON, default=list)
    created_at = Column(DateTime, default=utcnow)

    # one meeting has many action items
    action_items = relationship(
        "ActionItem", back_populates="meeting", cascade="all, delete-orphan"
    )


class ActionItem(Base):
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)

    description = Column(Text, nullable=False)
    owner = Column(String(120), default="Unassigned")
    due_date = Column(String(40), default="")  # kept as text to allow "Friday", "2026-11-03", etc.

    # lifecycle: open -> in_progress -> closed   (transitions are validated in crud.py)
    status = Column(String(20), nullable=False, default="open")
    # what actually happened, recorded when the item is resolved
    outcome = Column(Text, default="")

    # points to the previous version of this item from an earlier meeting (carry-forward history)
    parent_item_id = Column(Integer, ForeignKey("action_items.id"), nullable=True)

    # "explicit" = pulled from notes as a clear action item
    # "suggested" = caught by the Commitment Catcher (an implied promise, needs review)
    source = Column(String(20), default="explicit")

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    meeting = relationship("Meeting", back_populates="action_items")


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, default="")
    # the embedding vector, stored as a JSON list of floats
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)
