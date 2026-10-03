from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    state_dir: Path = Path("state")
    compact_threshold_tokens: int = 1000
    compact_keep_messages: int = 4
    model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            provider="openai", model_name="gpt-4o-mini", temperature=0.0
        )
    )
    judge_model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            provider="openai", model_name="gpt-4o-mini", temperature=0.0
        )
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables from .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Create `state/` if it does not exist
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"

    # Threshold settings
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Provider and model resolution
    raw_provider = os.getenv("LLM_PROVIDER", "openai")
    try:
        provider = normalize_provider(raw_provider)
    except ValueError:
        provider = "openai"

    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Resolve API keys and base URLs
    api_key_map = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama": None,
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    base_url_map = {
        "custom": os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1"),
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key_map.get(provider),
        base_url=base_url_map.get(provider),
    )

    judge_provider_raw = os.getenv("JUDGE_PROVIDER", raw_provider)
    try:
        judge_provider = normalize_provider(judge_provider_raw)
    except ValueError:
        judge_provider = provider
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)

    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key_map.get(judge_provider, model.api_key),
        base_url=base_url_map.get(judge_provider, model.base_url),
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model,
        judge_model=judge_model,
    )
