"""Shared utilities for the StoryMorals workflows.

Holds the pieces used by both `summarization.py` and `moral_generation_plus.py`:
API-key loading (from the environment, never hard-coded), model dispatch
(OpenAI + Google Gemini) with retry/back-off, `models.txt` parsing, and small
JSON-extraction helpers.
"""

import json
import os
import re
import sys
import time
from getpass import getpass
from pathlib import Path

# ---------------------------------------------------------------------------
# API keys — read from the environment (or an optional .env), never hard-coded
# ---------------------------------------------------------------------------

def load_dotenv(path: Path = Path(".env")) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Existing environment variables win (we only set what is missing), so a real
    exported variable always overrides the file. Lines that are blank or start
    with `#` are ignored. No third-party dependency required.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()

# Which environment variables to consult for each provider (first match wins).
_KEY_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}
_key_cache: dict[str, str] = {}


def get_api_key(provider: str) -> str:
    """Return the API key for a provider without ever hard-coding it.

    Resolution order: cached value -> environment variable(s) -> (only if
    running interactively) a hidden `getpass` prompt. The key is kept in memory
    for the process lifetime and never written to disk.
    """
    if provider in _key_cache:
        return _key_cache[provider]

    for var in _KEY_ENV.get(provider, []):
        if os.environ.get(var):
            _key_cache[provider] = os.environ[var]
            return _key_cache[provider]

    # Not in the environment — fall back to an interactive prompt if we can.
    if sys.stdin.isatty():
        key = getpass(f"Enter your {provider} API key: ")
        if key:
            _key_cache[provider] = key
            return key

    envs = " or ".join(_KEY_ENV.get(provider, []))
    raise RuntimeError(
        f"No API key found for '{provider}'. Set {envs} in your environment "
        "or a .env file (see .env.example)."
    )


# ---------------------------------------------------------------------------
# models.txt — sectioned list of models per stage
# ---------------------------------------------------------------------------

def load_models(path: Path = Path("models.txt")) -> dict[str, list[str]]:
    """Parse an INI-style `models.txt` into {section: [model, ...]}.

    Format (blank lines and `#` comments ignored)::

        [moral_generation]
        gpt-5.4
        gemini-3.1-pro-preview

        [value_extraction]
        gpt-5.4
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — it lists the models per stage.")

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return sections


def models_for(section: str, path: Path = Path("models.txt")) -> list[str]:
    """Return the model list for one section of models.txt (error if empty)."""
    sections = load_models(path)
    models = sections.get(section)
    if not models:
        raise ValueError(
            f"Section [{section}] is missing or empty in {path}. "
            f"Found sections: {sorted(sections)}"
        )
    return models


# ---------------------------------------------------------------------------
# Model dispatch (OpenAI + Gemini) with retry/back-off
# ---------------------------------------------------------------------------

def provider_for_model(model: str) -> str:
    """Infer the provider from a model name prefix."""
    m = model.lower()
    if m.startswith(("gpt", "o1", "o3")):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    raise ValueError(f"Unsupported model: {model}")


_openai_client = None


def _openai():
    """Lazily construct and cache the OpenAI client (keeps import optional)."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=get_api_key("openai"))
    return _openai_client


def _call_openai(model, prompt, *, max_tokens, temperature, use_temperature,
                 json_mode, timeout, system):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(model=model, messages=messages,
                  max_completion_tokens=max_tokens, timeout=timeout)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if use_temperature:
        kwargs["temperature"] = temperature
    resp = _openai().chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _call_gemini(model, prompt, *, max_tokens, temperature, use_temperature,
                 json_mode, timeout, system):
    import requests
    gen: dict = {}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    if use_temperature:
        gen["temperature"] = temperature
    # NB: we deliberately do NOT set maxOutputTokens for Gemini. On reasoning
    # models (e.g. gemini-3.x) the cap counts thinking tokens too, so a modest
    # limit gets consumed by reasoning and truncates the actual JSON reply.
    # `max_tokens` therefore only applies to the OpenAI path.
    _ = max_tokens

    payload: dict = {"contents": [{"parts": [{"text": prompt}]}],
                     "generationConfig": gen}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": get_api_key("google"),
                 "content-type": "application/json"},
        json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini: no candidates ({json.dumps(data)[:300]})")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    # Join all real (non-thought) text parts; reasoning models may emit several.
    texts = [p["text"] for p in parts
             if isinstance(p, dict) and "text" in p and not p.get("thought")]
    if not texts:
        raise RuntimeError(
            "Gemini: empty reply "
            f"(finishReason={cand.get('finishReason')}; raw={json.dumps(data)[:300]})")
    return "".join(texts)


_TRANSIENT = ("429", "500", "502", "503", "529", "timeout", "overloaded")


def call_model(model, prompt, *, max_tokens, temperature=0, use_temperature=True,
               json_mode=False, system=None, timeout=120, max_retries=5,
               quiet=False):
    """Call a model by name, dispatching on provider, retrying transient errors."""
    provider = provider_for_model(model)
    for attempt in range(1, max_retries + 1):
        try:
            fn = _call_openai if provider == "openai" else _call_gemini
            return fn(model, prompt, max_tokens=max_tokens, temperature=temperature,
                      use_temperature=use_temperature, json_mode=json_mode,
                      timeout=timeout, system=system)
        except Exception as e:  # noqa: BLE001 — we classify then re-raise
            transient = any(s in str(e).lower() for s in _TRANSIENT)
            if attempt == max_retries or not transient:
                raise
            wait = 2 ** attempt
            if not quiet:
                print(f"      {model} error ({e}); retry in {wait}s "
                      f"({attempt}/{max_retries})")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json(txt):
    """Parse a JSON object from a model reply (tolerant of code fences / prose)."""
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t).rsplit("```", 1)[0].strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)  # fall back to the outermost {...}
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def extract_json_array(txt):
    """Strip code fences, keep the LAST [...] block, parse to a list of strings."""
    if not txt:
        return None
    t = re.sub(r"```(json)?", "", txt)
    arrays = re.findall(r"\[.*?\]", t, flags=re.S)
    if not arrays:
        return None
    try:
        parsed = json.loads(arrays[-1])
    except Exception:
        return None
    return [str(x) for x in parsed]


def read_text_file(path) -> str:
    """Read a text file as UTF-8, tolerant of bad bytes."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="utf-8", errors="replace")


def normalize_id(name: str) -> str:
    """Canonical book id = filename without a trailing .txt."""
    name = str(name).strip()
    return name[:-4] if name.lower().endswith(".txt") else name
