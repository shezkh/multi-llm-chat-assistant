# Multi-LLM Chat Platform

A production-grade conversational AI platform with a unified abstraction layer over **5 LLM providers and 20+ hot-swappable models**, PostgreSQL session persistence, multilingual support, and an admin review dashboard with human-in-the-loop moderation.

Built as a client project (persona and branding removed); the demo ships with a generic assistant persona in `prompt.txt` that you can replace with your own.

## Features

**Multi-provider LLM layer.** One common message format, translated per provider: OpenAI, Anthropic Claude, Google Gemini, OpenRouter, and Groq. Models are hot-swappable from the admin panel at runtime, and the active model persists in the database across restarts.

**Resilience by design.** Automatic retry with backoff on provider failures, graceful handling of safety-filter blocks (returned as a distinct signal, not an exception), and a user-facing fallback message when all retries are exhausted. The app also degrades gracefully when no database is configured: chat still works, logging is skipped.

**Session persistence.** Cookie-based anonymous sessions, full conversation history in PostgreSQL, and per-session metadata: language, IP (proxy-aware), user agent, message counts, and activity timestamps.

**Multilingual.** 11-language picker plus automatic language detection (langdetect with a character-heuristic fallback). The selected language is injected into the system prompt so the model consistently replies in the user's language.

**Admin review dashboard.** Session list with filters (language, outcome, flagged) and pagination, full conversation viewer, outcome tracking, flag-for-review, and admin notes. Protected by separate credentials from the chat login.

**Prompt-engineered persona.** A structured system prompt (`prompt.txt`) covering persona, conversational style rules, domain restrictions, output format, and language protocol.

## Architecture

```
Browser (static HTML/JS)
    |  Basic Auth
    v
FastAPI (app.py)
    |-- /chat, /history, /reset, /set-language     user endpoints
    |-- /api/sessions, /api/model, /admin          admin endpoints
    |
    |-- chat_providers.py    unified provider layer + retry/fallback
    |       |-- OpenAI / Claude / Gemini / OpenRouter / Groq
    |
    |-- database.py + models.py    SQLAlchemy -> PostgreSQL
            |-- sessions / messages / app_settings
```

## Quick start

```bash
git clone https://github.com/shezkh/multi-llm-chat-assistant.git
cd multi-llm-chat-assistant
pip install -r requirements.txt
cp .env.example .env        # add at least one provider API key
python app.py               # serves on http://localhost:10000
```

Log in with the credentials from your `.env` (`CHAT_USER` / `CHAT_PASS`). The admin dashboard lives at `/admin` with `ADMIN_USER` / `ADMIN_PASS`.

Without a `DATABASE_URL` the app runs in stateless mode: chat works, but history and the admin dashboard are disabled.

## Configuration

All configuration is environment-based; see `.env.example`. At minimum set one provider key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, or `GROQ_API_KEY`) and pick a matching `ACTIVE_MODEL`. For deployments, set `ALLOWED_ORIGINS` to your frontend origin(s), `PRODUCTION=true` for secure cookies, and change the default credentials.

## Docker

```bash
docker build -t multi-llm-chat .
docker run --env-file .env -p 10000:10000 multi-llm-chat
```

## Project structure

```
app.py                FastAPI app: auth, chat endpoints, admin API
chat_providers.py     provider abstraction, model registry, retry logic
database.py           engine/session setup, graceful no-DB fallback
models.py             ChatSession, ChatMessage, AppSettings
prompt.txt            system prompt (replace with your persona)
static/index.html     chat UI
static/admin.html     review dashboard
```

## License

MIT
