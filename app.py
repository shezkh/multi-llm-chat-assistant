import os
import uuid
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read environment variables at import
# time (database.py reads DATABASE_URL, chat_providers.py reads ACTIVE_MODEL).
load_dotenv()

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional
import secrets

# --- DB imports ---
from database import get_db
from models import ChatSession, ChatMessage

# --- Multi-model chat ---
from chat_providers import send_message, get_active_model, set_active_model, PROVIDERS

# --- Security Setup ---
security = HTTPBasic()

# Chat login (for regular users)
CHAT_USER = os.getenv("CHAT_USER", os.getenv("ADMIN_USER", "admin"))
CHAT_PASS = os.getenv("CHAT_PASS", os.getenv("ADMIN_PASS", "change-me"))

# Admin login (for /admin panel only)
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "change-me")


def get_chat_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticates regular chat users. Admin credentials also work."""
    is_chat = (
        secrets.compare_digest(credentials.username, CHAT_USER)
        and secrets.compare_digest(credentials.password, CHAT_PASS)
    )
    is_admin = (
        secrets.compare_digest(credentials.username, ADMIN_USER)
        and secrets.compare_digest(credentials.password, ADMIN_PASS)
    )
    if not (is_chat or is_admin):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def get_admin_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticates admin panel access, separate credentials."""
    is_user_correct = secrets.compare_digest(credentials.username, ADMIN_USER)
    is_pass_correct = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (is_user_correct and is_pass_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


IS_PROD = os.getenv("PRODUCTION", "false").lower() == "true"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allowed origins come from the environment so no deployment details live in code.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request models ---
class ChatMessagePayload(BaseModel):
    message: str


class UpdateSessionPayload(BaseModel):
    outcome: Optional[str] = None
    flagged: Optional[bool] = None
    notes: Optional[str] = None


class LanguagePayload(BaseModel):
    language: str


class ModelPayload(BaseModel):
    model: str


# --- helpers ---
def get_session_id(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            secure=IS_PROD,
        )
    return session_id


def detect_language(text: str) -> str:
    """Detect language using langdetect library, with character-based fallback."""
    try:
        from langdetect import detect
        lang = detect(text)
        # langdetect returns codes like 'en', 'tr', 'de', 'ar', 'ur', 'fr', etc.
        return lang
    except Exception:
        # Fallback: simple character-based detection
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        german_chars = set("äöüßÄÖÜ")
        if any(c in turkish_chars for c in text):
            return "tr"
        if any(c in german_chars for c in text):
            return "de"
        return "en"


def get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- DB helpers ---
def ensure_db_session(db, session_id: str, request: Request):
    """Create a DB session record if it doesn't exist yet."""
    if db is None:
        return
    existing = db.query(ChatSession).filter_by(id=session_id).first()
    if not existing:
        new_session = ChatSession(
            id=session_id,
            ip_address=get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
        )
        db.add(new_session)
        db.commit()


def load_history(db, session_id: str):
    """Load chat history from PostgreSQL in a common provider-agnostic format."""
    if db is None:
        return []
    try:
        messages = (
            db.query(ChatMessage)
            .filter_by(session_id=session_id)
            .order_by(ChatMessage.created_at)
            .all()
        )
        return [
            {"role": m.role, "parts": [{"text": m.text}]}
            for m in messages
        ]
    except Exception as e:
        print(f"DB Read Error: {e}")
        return []


def save_message(db, session_id: str, role: str, text: str, request: Request = None):
    """Persist a single message and update session metadata."""
    if db is None:
        return
    try:
        ensure_db_session(db, session_id, request)
        session_record = db.query(ChatSession).filter_by(id=session_id).first()
        if session_record:
            # Only auto-detect if no language was set by the picker
            if role == "user" and session_record.message_count == 0 and not session_record.language:
                session_record.language = detect_language(text)
            session_record.message_count += 1
            session_record.last_activity = datetime.now(timezone.utc)
        msg = ChatMessage(session_id=session_id, role=role, text=text)
        db.add(msg)
        db.commit()
    except Exception as e:
        print(f"DB Write Error: {e}")
        db.rollback()


def load_prompt(path="prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PROMPT = load_prompt()


# ===================== CHAT ENDPOINTS =====================
@app.post("/chat", dependencies=[Depends(get_chat_user)])
def chat_endpoint(payload: ChatMessagePayload, request: Request, response: Response):
    session_id = get_session_id(request, response)
    db = next(get_db())

    # Load full conversation history from PostgreSQL (common format)
    db_history = load_history(db, session_id)

    # Convert to simple format for the provider wrapper
    history = [
        {"role": m["role"], "text": m["parts"][0]["text"]}
        for m in db_history
    ]

    # Build system prompt, injecting the selected language if available
    system_prompt = PROMPT
    if db:
        session_record = db.query(ChatSession).filter_by(id=session_id).first()
        if session_record and session_record.language:
            lang_map = {
                "en": "English", "tr": "Turkish", "zh": "Chinese (Mandarin)",
                "hi": "Hindi", "es": "Spanish", "fr": "French", "ar": "Arabic",
                "bn": "Bengali", "pt": "Portuguese", "ru": "Russian", "ur": "Urdu",
            }
            lang_name = lang_map.get(session_record.language, session_record.language)
            system_prompt = PROMPT + f"\n\nIMPORTANT: The user has selected {lang_name} as their language. You MUST respond in {lang_name}."

    try:
        reply_text = send_message(history, system_prompt, payload.message, db)
        if reply_text is None:
            return {"reply": "I am unable to answer that specific question due to safety filters."}
    except Exception as e:
        print(f"All retries failed: {e}")
        return {"reply": "I am currently experiencing high traffic. Please try again in a moment."}

    # Persist both messages to PostgreSQL
    save_message(db, session_id, "user", payload.message, request)
    save_message(db, session_id, "model", reply_text, request)

    if db:
        db.close()
    return {"reply": reply_text}


@app.get("/history", dependencies=[Depends(get_chat_user)])
def get_chat_history(request: Request, response: Response):
    session_id = get_session_id(request, response)
    db = next(get_db())
    history = load_history(db, session_id)
    if db:
        db.close()
    return history


@app.post("/reset", dependencies=[Depends(get_chat_user)])
def reset_chat(request: Request, response: Response):
    # Give the user a new session cookie. The old conversation stays in DB for review.
    new_session_id = str(uuid.uuid4())
    response.set_cookie(
        key="session_id",
        value=new_session_id,
        httponly=True,
        samesite="lax",
        secure=IS_PROD,
    )
    return {"status": "History cleared"}


@app.post("/set-language", dependencies=[Depends(get_chat_user)])
def set_language(payload: LanguagePayload, request: Request, response: Response):
    """Save the user's chosen language to their session."""
    session_id = get_session_id(request, response)
    db = next(get_db())
    if db:
        try:
            ensure_db_session(db, session_id, request)
            session_record = db.query(ChatSession).filter_by(id=session_id).first()
            if session_record:
                session_record.language = payload.language
                db.commit()
        except Exception as e:
            print(f"Set language error: {e}")
            db.rollback()
        finally:
            db.close()
    return {"status": "language set", "language": payload.language}


# ===================== MODEL SWITCHING =====================
@app.get("/api/model", dependencies=[Depends(get_admin_user)])
def get_model():
    """Get the currently active model."""
    db = next(get_db())
    try:
        return {"model": get_active_model(db), "available": list(PROVIDERS.keys())}
    finally:
        if db:
            db.close()


@app.post("/api/model", dependencies=[Depends(get_admin_user)])
def switch_model(payload: ModelPayload):
    """Switch the active AI model."""
    db = next(get_db())
    try:
        set_active_model(payload.model, db)
        return {"status": "switched", "model": payload.model}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db:
            db.close()


# ===================== ADMIN API ENDPOINTS =====================
@app.get("/api/sessions", dependencies=[Depends(get_admin_user)])
def list_sessions(
    flagged: Optional[bool] = Query(None),
    language: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """List all chat sessions with optional filters."""
    db = next(get_db())
    if db is None:
        return {"sessions": [], "total": 0, "page": page}
    try:
        query = db.query(ChatSession)
        if flagged is not None:
            query = query.filter(ChatSession.flagged == flagged)
        if language:
            query = query.filter(ChatSession.language == language)
        if outcome:
            query = query.filter(ChatSession.outcome == outcome)
        total = query.count()
        sessions = (
            query.order_by(ChatSession.last_activity.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        return {
            "sessions": [
                {
                    "id": s.id,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "last_activity": s.last_activity.isoformat() if s.last_activity else None,
                    "language": s.language,
                    "ip_address": s.ip_address,
                    "country": s.country,
                    "message_count": s.message_count,
                    "outcome": s.outcome,
                    "flagged": s.flagged,
                    "notes": s.notes,
                }
                for s in sessions
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
        }
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/messages", dependencies=[Depends(get_admin_user)])
def get_session_messages(session_id: str):
    """Get all messages for a specific session."""
    db = next(get_db())
    if db is None:
        return {"messages": []}
    try:
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = (
            db.query(ChatMessage)
            .filter_by(session_id=session_id)
            .order_by(ChatMessage.created_at)
            .all()
        )
        return {
            "session": {
                "id": session.id,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "language": session.language,
                "ip_address": session.ip_address,
                "user_agent": session.user_agent,
                "message_count": session.message_count,
                "outcome": session.outcome,
                "flagged": session.flagged,
                "notes": session.notes,
            },
            "messages": [
                {
                    "role": m.role,
                    "text": m.text,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ],
        }
    finally:
        db.close()


@app.patch("/api/sessions/{session_id}", dependencies=[Depends(get_admin_user)])
def update_session(session_id: str, payload: UpdateSessionPayload):
    """Update session metadata (flag, outcome, notes)."""
    db = next(get_db())
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if payload.outcome is not None:
            session.outcome = payload.outcome
        if payload.flagged is not None:
            session.flagged = payload.flagged
        if payload.notes is not None:
            session.notes = payload.notes
        db.commit()
        return {"status": "updated"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ===================== PAGE ROUTES =====================
@app.get("/admin", dependencies=[Depends(get_admin_user)])
def read_admin():
    return FileResponse("static/admin.html")


@app.get("/", dependencies=[Depends(get_chat_user)])
def read_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000))
    )
