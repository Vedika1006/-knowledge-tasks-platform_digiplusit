# AI-Powered Knowledge & Tasks Management Platform

A small application that captures what was discussed and decided in meetings and
work-handovers, transfers that context to the right people, and follows up on
open action items until they are closed — using AI for the parts that a plain
database cannot do.

Built for the DigiPlus technical assessment.

---

## 1. What it does (mapped to the brief)

| Requirement | How it's met |
|---|---|
| Create & persist meetings / work transfers | One `meetings` table with a `kind` field (`meeting` or `handover`); agenda, participants, notes, decisions, action items all persisted |
| Manage lifecycle & track every action item | `action_items` with `owner`, `due_date`, `status` (open → in_progress → closed), `outcome`; status transitions are validated |
| Carry open items forward, keeping history | Open items are **copied** into the next meeting as new rows whose `parent_item_id` points back — the full chain is preserved, nothing is overwritten |
| AI: raw info → structured recap | `POST /api/meetings/{id}/extract` sends raw text to Gemini and gets back a strict-JSON `{summary, decisions, action_items}` |
| AI: catch-up brief for whoever missed it | `GET /api/brief/{person}` — a short brief focused on what *that person* must act on |
| Knowledge base + AI linking | `knowledge_docs` with embeddings; `GET /api/knowledge/link?q=` returns semantically related docs via cosine similarity |
| Record outcome of each item | `outcome` field, set when an item is resolved |
| Validation & error handling | Pydantic validates every request; invalid status transitions return 400; missing/invalid fields return 422; AI failures fall back gracefully |

---

## 2. Special feature — "Commitment Catcher"

After the normal extraction pulls out the **explicit** action items, a **second
AI pass** re-reads the same notes looking only for **implied commitments that
nobody logged** — phrases like *"I'll loop in finance"*, *"let me check and get
back"*, *"we should follow up on that"*. These surface as `suggested` items in a
review queue that the user can accept or ignore.

**Why it adds value:** the brief's own pain points are all about things
*silently dropping*. Explicit action items rarely get lost — it's the casual
verbal promises nobody wrote down that fall through the cracks. The first pass
captures what was *decided*; the second pass catches what was *promised*. This
attacks the root of the stated problem, not just the symptom.

---

## 3. Tech stack & why

| Layer | Choice | Reason |
|---|---|---|
| Backend | **Python + FastAPI** | Automatic request validation via Pydantic (satisfies the validation requirement at framework level); auto-generated live API docs at `/docs` |
| AI | **Groq** (auto-selected LLM, e.g. Llama 3) | Fast inference; auto-selects a current chat model on startup |
| Database | **SQLAlchemy ORM** → SQLite (dev) / Postgres (prod) | Same Python code runs on both; switch by changing one line in `.env` |
| Persistence (prod) | **Supabase** (hosted Postgres) | Free, durable — local files on free hosts get wiped on restart |
| Frontend | **Single HTML page + vanilla JS** | No build step; every line is explainable; one repo, one deploy |
| Hosting | **Render** (web service) + **Supabase** (DB) | Both free tiers |

---

## 4. AI configuration

- Provider: **Groq**. Get a free key at <https://console.groq.com/keys>.
- Set it in `.env` as `GROQ_API_KEY`.
- Optionally set `GROQ_MODEL` to pin a specific model. If not set, the app
  calls `client.models.list()` on startup and auto-selects a current
  general-purpose chat model (prefers Llama "versatile" variants; skips
  whisper/tts/guard/vision models). The chosen model is logged at startup.
- Uses Groq's `response_format: {"type": "json_object"}` for structured
  extraction when the model supports it.
- **Graceful failure:** if the key is missing, or a call times out, or the model
  returns invalid JSON, the app automatically falls back to a rule-based
  extractor and logs that it did so. The app never crashes because of AI. The
  rules are only a safety net — the primary path is always the LLM, so we are
  not "relying entirely on hard-coded rules".
- **Embeddings:** Groq does not provide an embeddings endpoint, so
  `embed_text()` returns `None`. The knowledge-base linking falls back to
  keyword overlap. Adding embeddings via a dedicated provider (e.g. OpenAI
  embeddings + pgvector) is a future step.

**Data flow:** browser → FastAPI route → Pydantic validation → `ai.py` builds a
prompt that forces strict-JSON output → Groq → JSON is parsed and validated →
saved to DB via `crud.py` → returned to the browser.

---

## 5. Logging

Structured JSON logging (`app/logging_config.py`), written to **`logs/app.log`**
and the console.

- **Log levels:** INFO / WARNING / ERROR.
- **Correlation ID:** a middleware assigns a unique ID to every HTTP request
  (stored in a `ContextVar`) and stamps it on every log line, so one user action
  can be traced end-to-end. It's also returned in the `X-Correlation-ID`
  response header.
- **Key events logged:** `meeting.created`, `item.created`, `item.status_changed`,
  `item.closed`, `item.reassigned`, `items.carried_forward`, `kb.created`.
- **AI calls:** each logged with `purpose`, `outcome` (success / error / fallback)
  and `latency_ms`.
- **Errors:** logged with full context and stack trace.
- **No secrets:** the API key is read from an environment variable and never logged.

To trace one request: find its correlation ID and grep for it:
```bash
grep "12e5ef44982c" logs/app.log
```

---

## 6. Run locally

```bash
# 1. create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. set up environment
cp .env.example .env             # Windows: copy .env.example .env
# open .env and paste your GROQ_API_KEY (leave DATABASE_URL as-is for SQLite)

# 4. run
uvicorn app.main:app --reload

# 5. open in browser
#    app:       http://127.0.0.1:8000
#    API docs:  http://127.0.0.1:8000/docs
```

The app works even **without** a Groq key — it runs in rule-based fallback
mode (you'll see a warning in the logs and an "AI off" badge in the UI).

---

## 7. Deploy (free)

**Database — Supabase**
1. Create a project at supabase.com → it gives you a Postgres database.
2. Project Settings → Database → copy the connection string.
3. Put it in `DATABASE_URL` (see `.env.example` for the format).

**App — Render**
1. Push this folder to a GitHub repo.
2. Render → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables `GROQ_API_KEY` and `DATABASE_URL` in the Render dashboard.

---

## 8. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status + whether AI is enabled |
| POST | `/api/meetings` (`?carry_from=<id>`) | create meeting/handover, optionally carrying open items forward |
| GET | `/api/meetings` | list all meetings with their items |
| GET | `/api/meetings/{id}` | one meeting |
| POST | `/api/meetings/{id}/extract` | AI: raw notes → recap + commitment catcher |
| POST | `/api/meetings/{id}/items` | add an action item manually |
| PATCH | `/api/items/{id}` | update status / owner / due date / outcome (validated) |
| GET | `/api/items/{id}/history` | carry-forward history chain |
| GET | `/api/brief/{person}` | AI catch-up brief for a person |
| POST | `/api/knowledge` | add a knowledge-base document |
| GET | `/api/knowledge` | list documents |
| GET | `/api/knowledge/link?q=` | semantically related documents |

---

## 9. Assumptions & known limitations

- **Auth is out of scope** — no login/users; the assessment is about knowledge &
  task flow, not identity. Owners are plain name strings.
- **KB similarity is brute-force cosine** in Python. Groq does not offer an
  embeddings endpoint, so `embed_text()` returns `None` and KB linking falls
  back to keyword overlap. The next step is adding a dedicated embeddings
  provider (e.g. OpenAI `text-embedding-3-small`) and `pgvector` in Postgres.
- **Due dates are stored as free text** (`"Friday"`, `"2026-11-03"`) because the
  AI extracts whatever the notes contain; normalising them to real dates would
  be the next improvement.
- **Carry-forward is triggered explicitly** by passing `carry_from` when creating
  the next meeting, rather than running automatically on a schedule.
- Single-user, single-process; no concurrency control beyond the DB.
