import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


def _env_if_set(*names: str) -> str | None:
    """First defined env value (may be empty string); None if none of the keys are set."""
    for name in names:
        if name in os.environ:
            return os.environ.get(name, "").strip()
    return None


def _secret_if_set(*names: str) -> str | None:
    """First defined Streamlit secret value, else None."""
    try:
        import streamlit as st  # Local import keeps non-UI scripts lightweight.
    except Exception:  # noqa: BLE001
        return None

    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001
        return None

    for name in names:
        try:
            if name in secrets:
                return str(secrets.get(name, "")).strip()
        except Exception:  # noqa: BLE001
            continue

    # Also support nested Streamlit secrets, e.g.:
    # [bird_vision] api_key = "..."
    nested_aliases = {
        "BIRD_VISION_API_KEY": ("bird_vision", "api_key"),
        "OPENROUTER_API_KEY": ("bird_vision", "api_key"),
        "DEEPSEEK_API_KEY": ("bird_vision", "api_key"),
        "BIRD_VISION_BASE_URL": ("bird_vision", "base_url"),
        "OPENROUTER_BASE_URL": ("bird_vision", "base_url"),
        "DEEPSEEK_BASE_URL": ("bird_vision", "base_url"),
        "BIRD_VISION_MODEL": ("bird_vision", "model"),
        "OPENROUTER_MODEL": ("bird_vision", "model"),
        "DEEPSEEK_MODEL": ("bird_vision", "model"),
        "BIRD_VISION_PROVIDER_IGNORE": ("bird_vision", "provider_ignore"),
        "OPENROUTER_PROVIDER_IGNORE": ("bird_vision", "provider_ignore"),
        "BIRD_VISION_HTTP_REFERER": ("bird_vision", "http_referer"),
        "OPENROUTER_HTTP_REFERER": ("bird_vision", "http_referer"),
        "BIRD_VISION_APP_TITLE": ("bird_vision", "app_title"),
        "OPENROUTER_TITLE": ("bird_vision", "app_title"),
        "BIRD_TRANSLATE_API_KEY": ("translation", "api_key"),
        "DEEPSEEK_TRANSLATE_API_KEY": ("translation", "api_key"),
        "BIRD_TRANSLATE_BASE_URL": ("translation", "base_url"),
        "DEEPSEEK_TRANSLATE_BASE_URL": ("translation", "base_url"),
        "BIRD_TRANSLATE_MODEL": ("translation", "model"),
        "DEEPSEEK_TRANSLATE_MODEL": ("translation", "model"),
        "EBIRD_API_TOKEN": ("ebird", "api_token"),
    }
    for name in names:
        alias = nested_aliases.get(name)
        if not alias:
            continue
        section, key = alias
        try:
            node = secrets.get(section, None)
            if node and key in node:
                return str(node.get(key, "")).strip()
        except Exception:  # noqa: BLE001
            continue
    return None


def _env_chain(default: str, *names: str) -> str:
    """First defined env value, else `default`."""
    for name in names:
        if name in os.environ:
            return os.environ.get(name, "").strip()
    return default


def _env_first_non_empty(*names: str) -> str:
    """Walk keys in order; use first present env with a non-empty value, else empty string."""
    for name in names:
        if name not in os.environ:
            continue
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return ""


@dataclass(frozen=True)
class AppSettings:
    api_key: str
    base_url: str
    model: str
    openrouter_provider_ignore: List[str]
    openrouter_http_referer: str
    openrouter_title: str
    timeout_seconds: int
    retry_count: int
    top_k: int
    max_upload_mb: int
    max_image_side: int
    db_path: str
    deepseek_translate_api_key: str
    deepseek_translate_base_url: str
    deepseek_translate_model: str
    ebird_api_token: str
    ebird_regions: List[str]
    ebird_recent_max_per_region: int
    moss_tts_nano_home: str
    moss_tts_cli: str
    moss_tts_backend: str
    moss_tts_voice: str
    moss_tts_prompt_speech: str
    moss_tts_timeout_seconds: int
    cantonese_tts_engine: str
    edge_tts_voice: str


def load_settings() -> AppSettings:
    load_dotenv(override=True)
    api_key = (
        _secret_if_set("BIRD_VISION_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY")
        or _env_first_non_empty(
            "BIRD_VISION_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
        )
    )
    base_url = (
        _env_chain(
            "",
            "BIRD_VISION_BASE_URL",
            "OPENROUTER_BASE_URL",
            "DEEPSEEK_BASE_URL",
        )
        or "https://openrouter.ai/api/v1"
    ).rstrip("/")
    model = _env_chain(
        "meta-llama/llama-3.2-11b-vision-instruct",
        "BIRD_VISION_MODEL",
        "OPENROUTER_MODEL",
        "DEEPSEEK_MODEL",
    )
    is_openrouter = "openrouter.ai" in base_url
    provider_ignore_raw = _env_if_set("BIRD_VISION_PROVIDER_IGNORE", "OPENROUTER_PROVIDER_IGNORE")
    if provider_ignore_raw is None:
        openrouter_provider_ignore = ["novita"] if is_openrouter else []
    else:
        openrouter_provider_ignore = [
            x.strip().lower() for x in provider_ignore_raw.split(",") if x.strip()
        ]
    openrouter_http_referer = (
        _env_chain(
            "https://github.com/local/bird_identifier",
            "BIRD_VISION_HTTP_REFERER",
            "OPENROUTER_HTTP_REFERER",
        )
        or "https://github.com/local/bird_identifier"
    )
    openrouter_title = _env_chain(
        "雀鳥辨識",
        "BIRD_VISION_APP_TITLE",
        "OPENROUTER_TITLE",
    ) or "雀鳥辨識"
    regions_raw = os.getenv("EBIRD_REGIONS", "HK,CN")
    ebird_regions = [x.strip().upper() for x in regions_raw.split(",") if x.strip()]
    return AppSettings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        openrouter_provider_ignore=openrouter_provider_ignore,
        openrouter_http_referer=openrouter_http_referer,
        openrouter_title=openrouter_title,
        timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        retry_count=int(os.getenv("REQUEST_RETRY_COUNT", "2")),
        top_k=int(os.getenv("TOP_K", "3")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "8")),
        max_image_side=int(os.getenv("MAX_IMAGE_SIDE", "1600")),
        db_path=os.getenv("HISTORY_DB_PATH", "bird_identifier.db"),
        deepseek_translate_api_key=(
            _secret_if_set("BIRD_TRANSLATE_API_KEY", "DEEPSEEK_TRANSLATE_API_KEY", "DEEPSEEK_API_KEY")
            or _env_first_non_empty(
            "BIRD_TRANSLATE_API_KEY",
            "DEEPSEEK_TRANSLATE_API_KEY",
            "DEEPSEEK_API_KEY",
            )
        ),
        deepseek_translate_base_url=(
            _env_chain("", "BIRD_TRANSLATE_BASE_URL", "DEEPSEEK_TRANSLATE_BASE_URL") or "https://api.deepseek.com"
        ).rstrip("/"),
        deepseek_translate_model=_env_chain(
            "deepseek-chat",
            "BIRD_TRANSLATE_MODEL",
            "DEEPSEEK_TRANSLATE_MODEL",
        ),
        ebird_api_token=(
            _secret_if_set("EBIRD_API_TOKEN")
            or os.getenv("EBIRD_API_TOKEN", "").strip()
        ),
        ebird_regions=ebird_regions or ["HK", "CN"],
        ebird_recent_max_per_region=int(os.getenv("EBIRD_RECENT_MAX", "10")),
        moss_tts_nano_home=os.getenv("MOSS_TTS_NANO_HOME", "").strip(),
        moss_tts_cli=os.getenv("MOSS_TTS_CLI", "").strip(),
        moss_tts_backend=(os.getenv("MOSS_TTS_BACKEND", "onnx").strip().lower() or "onnx"),
        moss_tts_voice=os.getenv("MOSS_TTS_VOICE", "Junhao").strip() or "Junhao",
        moss_tts_prompt_speech=os.getenv("MOSS_TTS_PROMPT_SPEECH", "").strip(),
        moss_tts_timeout_seconds=int(os.getenv("MOSS_TTS_TIMEOUT_SECONDS", "120")),
        cantonese_tts_engine=(os.getenv("CANTONESE_TTS_ENGINE", "edge").strip().lower() or "edge"),
        edge_tts_voice=os.getenv("EDGE_TTS_VOICE", "yue-HK-HiuGaaiNeural").strip() or "yue-HK-HiuGaaiNeural",
    )
