"""
Pydantic schemas = the shape of data coming IN (requests) and going OUT (responses).

Why this matters for the assessment:
FastAPI uses these classes to VALIDATE every incoming request automatically.
If a request is missing a required field, or has the wrong type, FastAPI rejects
it with a clear 422 error BEFORE our code ever runs. That satisfies the
"appropriate validation and error handling" requirement at the framework level,
and it is visible live on the auto-generated docs page at /docs.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------- Action items ----------

class ActionItemCreate(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    owner: str = Field(default="Unassigned", max_length=120)
    due_date: str = Field(default="", max_length=40)


class ActionItemUpdate(BaseModel):
    # all optional — you can update just the field you care about
    status: Optional[str] = None
    outcome: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[str] = None

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v):
        if v is not None and v not in {"open", "in_progress", "closed"}:
            raise ValueError("status must be one of: open, in_progress, closed")
        return v


class ActionItemOut(BaseModel):
    id: int
    meeting_id: int
    description: str
    owner: str
    due_date: str
    status: str
    outcome: str
    parent_item_id: Optional[int]
    source: str

    class Config:
        from_attributes = True  # lets Pydantic read directly from ORM objects


# ---------- Meetings ----------

class MeetingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    kind: str = Field(default="meeting")
    participants: list[str] = Field(default_factory=list)
    agenda: str = Field(default="", max_length=5000)

    @field_validator("kind")
    @classmethod
    def kind_must_be_valid(cls, v):
        if v not in {"meeting", "handover"}:
            raise ValueError("kind must be 'meeting' or 'handover'")
        return v


class MeetingOut(BaseModel):
    id: int
    kind: str
    title: str
    participants: list[str]
    agenda: str
    raw_notes: str
    summary: str
    decisions: list[str]
    action_items: list[ActionItemOut] = []

    class Config:
        from_attributes = True


# ---------- AI extraction ----------

class ExtractRequest(BaseModel):
    raw_text: str = Field(min_length=1, max_length=20000)


# ---------- Knowledge base ----------

class KnowledgeDocCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=20000)
