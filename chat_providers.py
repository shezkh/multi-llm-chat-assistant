"""
Unified chat interface for Gemini, OpenAI, Claude, OpenRouter, and Groq.
Each provider translates to/from a common message format.
Supports individual model selection from the admin panel.
"""
import os
import time

# ---- Common format ----
# Messages are stored in DB as: {"role": "user"|"model", "text": "..."}
# Each provider converts to/from its own format.


# ===================== GEMINI =====================
def _chat_gemini(model_id: str):
    """Factory: returns a chat function for any Gemini model."""
    def chat_fn(history: list, system_prompt: str, user_message: str) -> str:
        from google import genai
        from google.genai import types
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        client = genai.Client(api_key=api_key)
        # Convert history to Gemini format
        gemini_history = [
            {"role": m["role"], "parts": [{"text": m["text"]}]}
            for m in history
        ]
        chat = client.chats.create(
            model=model_id,
            history=gemini_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.9,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            ),
        )
        reply = chat.send_message(user_message)
        if not reply.candidates or not reply.candidates[0].content:
            return None  # blocked by safety filters
        return reply.candidates[0].content.parts[0].text
    return chat_fn


# ===================== OPENAI =====================
def chat_openai(history: list, system_prompt: str, user_message: str) -> str:
    """Send a message using OpenAI (GPT-4o)."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    # Build messages: system + history + new message
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = "assistant" if m["role"] == "model" else "user"
        messages.append({"role": role, "content": m["text"]})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.9,
    )
    return response.choices[0].message.content


# ===================== CLAUDE =====================
def chat_claude(history: list, system_prompt: str, user_message: str) -> str:
    """Send a message using Anthropic Claude."""
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    # Build messages: history + new message
    messages = []
    for m in history:
        role = "assistant" if m["role"] == "model" else "user"
        messages.append({"role": role, "content": m["text"]})
    messages.append({"role": "user", "content": user_message})
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        temperature=0.9,
    )
    return response.content[0].text


# ===================== OPENROUTER =====================
def _chat_openrouter(model_id: str):
    """Factory: returns a chat function for any OpenRouter model."""
    def chat_fn(history: list, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            role = "assistant" if m["role"] == "model" else "user"
            messages.append({"role": role, "content": m["text"]})
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.9,
        )
        return response.choices[0].message.content
    return chat_fn


# ===================== GROQ =====================
def _chat_groq(model_id: str):
    """Factory: returns a chat function for any Groq model."""
    def chat_fn(history: list, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            role = "assistant" if m["role"] == "model" else "user"
            messages.append({"role": role, "content": m["text"]})
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.9,
        )
        return response.choices[0].message.content
    return chat_fn


# ===================== DISPATCHER =====================
PROVIDERS = {
    # --- Paid (your own API keys) ---
    "openai":               chat_openai,
    "claude":               chat_claude,
    # --- Google Gemini (free via AI Studio) ---
    "gemini":               _chat_gemini("gemini-2.5-flash"),
    "gemini_35_flash":      _chat_gemini("gemini-3.5-flash"),
    "gemini_25_pro":        _chat_gemini("gemini-2.5-pro"),
    "gemini_20_flash":      _chat_gemini("gemini-2.0-flash"),
    "gemini_31_flash_lite": _chat_gemini("gemini-3.1-flash-lite"),
    # --- OpenRouter (free) ---
    "or_nemotron_ultra":    _chat_openrouter("nvidia/nemotron-3-ultra-550b-a55b:free"),
    "or_nemotron_super":    _chat_openrouter("nvidia/nemotron-3-super-120b-a12b:free"),
    "or_gpt_oss_120b":      _chat_openrouter("openai/gpt-oss-120b:free"),
    "or_kimi_k2.6":         _chat_openrouter("moonshotai/kimi-k2.6:free"),
    "or_hermes_405b":       _chat_openrouter("nousresearch/hermes-3-llama-3.1-405b:free"),
    "or_qwen3_next_80b":    _chat_openrouter("qwen/qwen3-next-80b-a3b-instruct:free"),
    "or_qwen3_coder":       _chat_openrouter("qwen/qwen3-coder:free"),
    "or_gemma4_31b":        _chat_openrouter("google/gemma-4-31b-it:free"),
    "or_llama3.3_70b":      _chat_openrouter("meta-llama/llama-3.3-70b-instruct:free"),
    "or_laguna_m1":         _chat_openrouter("poolside/laguna-m.1:free"),
    "or_north_mini_code":   _chat_openrouter("cohere/north-mini-code:free"),
    # --- Groq (free, ultra-fast) ---
    "groq_llama3.3_70b":    _chat_groq("llama-3.3-70b-versatile"),
    "groq_gpt_oss_120b":    _chat_groq("openai/gpt-oss-120b"),
    "groq_llama4_scout":    _chat_groq("meta-llama/llama-4-scout-17b-16e-instruct"),
    "groq_qwen3_32b":       _chat_groq("qwen/qwen3-32b"),
    "groq_qwen3.6_27b":     _chat_groq("qwen/qwen3.6-27b"),
    "groq_gpt_oss_20b":     _chat_groq("openai/gpt-oss-20b"),
}

# Default model, used if DB has no setting yet
_default_model = os.getenv("ACTIVE_MODEL", "gemini")


def get_active_model(db=None) -> str:
    """Get active model from DB, falling back to default."""
    if db:
        try:
            from models import AppSettings
            setting = db.query(AppSettings).filter_by(key="active_model").first()
            if setting and setting.value in PROVIDERS:
                return setting.value
        except Exception:
            pass
    return _default_model


def set_active_model(model: str, db=None):
    """Save active model to DB for persistence across restarts."""
    if model not in PROVIDERS:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(PROVIDERS.keys())}")
    if db:
        try:
            from models import AppSettings
            setting = db.query(AppSettings).filter_by(key="active_model").first()
            if setting:
                setting.value = model
            else:
                db.add(AppSettings(key="active_model", value=model))
            db.commit()
        except Exception as e:
            print(f"Failed to save model to DB: {e}")
            db.rollback()


def send_message(history: list, system_prompt: str, user_message: str, db=None) -> str:
    """
    Send a message using the currently active model.
    Returns the reply text, or None if blocked by safety filters.
    Retries up to 3 times on failure.
    """
    active = get_active_model(db)
    provider_fn = PROVIDERS[active]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            reply = provider_fn(history, system_prompt, user_message)
            if reply is None:
                return None  # safety filter block
            return reply
        except Exception as e:
            print(f"[{active}] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise  # re-raise after all retries exhausted
