"""Configuration helpers for the Gemini token optimization notebook.

The defaults favor Google Cloud / Agent Platform authentication because the
notebook is intended to work after:

    gcloud auth application-default login

Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and GEMINI_MODEL in your shell
or override them from the notebook before calling load_config().
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Dict, Optional


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ModelPricing:
    """USD pricing per 1M tokens for planning-level cost comparisons."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0
    cache_storage_per_million_token_hour: float = 0.0


# Update these values if your billing contract, region, or API surface differs.
# Defaults are based on public Gemini Developer API paid-tier prices viewed
# 2026-06-23. Vertex/Agent Platform enterprise agreements may differ.
DEFAULT_PRICES: Dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(
        input_per_million=0.30,
        output_per_million=2.50,
        cached_input_per_million=0.075,
        cache_storage_per_million_token_hour=1.00,
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input_per_million=0.10,
        output_per_million=0.40,
        cached_input_per_million=0.025,
        cache_storage_per_million_token_hour=1.00,
    ),
    "gemini-3-flash-preview": ModelPricing(
        input_per_million=0.50,
        output_per_million=3.00,
        cached_input_per_million=0.05,
        cache_storage_per_million_token_hour=1.00,
    ),
}


@dataclass(frozen=True)
class ADKConfig:
    """Runtime config used by the notebook and optional ADK App examples."""

    project: str
    location: str
    model: str
    use_enterprise: bool
    pricing: ModelPricing
    context_cache_min_tokens: int = 2048
    context_cache_ttl_seconds: int = 600
    context_cache_intervals: int = 5
    compaction_token_threshold: int = 4000
    compaction_event_retention_size: int = 5
    compaction_interval: int = 3
    compaction_overlap_size: int = 1


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _gcloud_project() -> str:
    """Best-effort fallback to the active gcloud project."""

    try:
        completed = subprocess.run(
            ["gcloud", "config", "get-value", "project", "--quiet"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""

    value = completed.stdout.strip()
    if completed.returncode != 0 or not value or value == "(unset)":
        return ""
    return value


def get_active_gcloud_project() -> str:
    """Return the active project from `gcloud config get-value project`."""

    return _gcloud_project()


def load_config(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    model: Optional[str] = None,
    use_enterprise: Optional[bool] = None,
) -> ADKConfig:
    """Load notebook config from explicit values and environment variables."""

    resolved_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    default_pricing = DEFAULT_PRICES.get(
        resolved_model, DEFAULT_PRICES["gemini-2.5-flash"]
    )
    pricing = ModelPricing(
        input_per_million=_float_env(
            "GEMINI_INPUT_PRICE_PER_MILLION", default_pricing.input_per_million
        ),
        output_per_million=_float_env(
            "GEMINI_OUTPUT_PRICE_PER_MILLION", default_pricing.output_per_million
        ),
        cached_input_per_million=_float_env(
            "GEMINI_CACHED_INPUT_PRICE_PER_MILLION",
            default_pricing.cached_input_per_million,
        ),
        cache_storage_per_million_token_hour=_float_env(
            "GEMINI_CACHE_STORAGE_PRICE_PER_MILLION_TOKEN_HOUR",
            default_pricing.cache_storage_per_million_token_hour,
        ),
    )

    enterprise_default = _bool_env("GOOGLE_GENAI_USE_ENTERPRISE", True)
    enterprise_default = _bool_env("GOOGLE_GENAI_USE_VERTEXAI", enterprise_default)

    return ADKConfig(
        project=project or os.getenv("GOOGLE_CLOUD_PROJECT") or _gcloud_project(),
        location=location or os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        model=resolved_model,
        use_enterprise=enterprise_default
        if use_enterprise is None
        else use_enterprise,
        pricing=pricing,
        context_cache_min_tokens=_int_env("ADK_CONTEXT_CACHE_MIN_TOKENS", 2048),
        context_cache_ttl_seconds=_int_env("ADK_CONTEXT_CACHE_TTL_SECONDS", 600),
        context_cache_intervals=_int_env("ADK_CONTEXT_CACHE_INTERVALS", 5),
        compaction_token_threshold=_int_env("ADK_COMPACTION_TOKEN_THRESHOLD", 4000),
        compaction_event_retention_size=_int_env(
            "ADK_COMPACTION_EVENT_RETENTION_SIZE", 5
        ),
        compaction_interval=_int_env("ADK_COMPACTION_INTERVAL", 3),
        compaction_overlap_size=_int_env("ADK_COMPACTION_OVERLAP_SIZE", 1),
    )


def apply_environment(config: ADKConfig) -> None:
    """Apply config to environment variables used by ADK and google-genai."""

    if config.project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = config.project
    if config.location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = config.location
    os.environ["GEMINI_MODEL"] = config.model

    # Current google-genai docs use GOOGLE_GENAI_USE_ENTERPRISE. Some ADK docs
    # and older examples still use GOOGLE_GENAI_USE_VERTEXAI, so set both.
    flag = "true" if config.use_enterprise else "false"
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = flag
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE" if config.use_enterprise else "FALSE"


def create_genai_client(config: ADKConfig):
    """Create a Google GenAI SDK client.

    The current SDK accepts enterprise=True for Gemini Enterprise Agent Platform.
    Older SDKs used vertexai=True. This helper tries the current form first and
    falls back to the older keyword so the notebook is less brittle.
    """

    try:
        from google import genai
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Missing Google GenAI SDK in this Python/Jupyter kernel. "
            "Run `python -m pip install -r requirements.txt` from this project, "
            "or run the notebook dependency setup cell before creating the client."
        ) from exc

    if not config.use_enterprise:
        return genai.Client()

    kwargs = {}
    if config.project:
        kwargs["project"] = config.project
    if config.location:
        kwargs["location"] = config.location

    try:
        return genai.Client(enterprise=True, **kwargs)
    except TypeError:
        return genai.Client(vertexai=True, **kwargs)


def create_adk_agent(config: ADKConfig, *, name: str = "token_optimizer_agent"):
    """Create a minimal Gemini-backed ADK agent for experimentation."""

    try:
        from google.adk import Agent
    except ImportError:
        from google.adk.agents import LlmAgent as Agent

    return Agent(
        name=name,
        model=config.model,
        instruction=(
            "You answer questions using only the supplied context. "
            "Be concise, cite the relevant section names when present, and avoid "
            "restating irrelevant context."
        ),
    )


def create_adk_app(config: ADKConfig, root_agent=None):
    """Create an ADK App with context caching and compaction enabled."""

    if root_agent is None:
        root_agent = create_adk_agent(config)

    from google.adk.apps.app import App, EventsCompactionConfig
    from google.adk.agents.context_cache_config import ContextCacheConfig

    return App(
        name="token-optimization-app",
        root_agent=root_agent,
        context_cache_config=ContextCacheConfig(
            min_tokens=config.context_cache_min_tokens,
            ttl_seconds=config.context_cache_ttl_seconds,
            cache_intervals=config.context_cache_intervals,
        ),
        events_compaction_config=EventsCompactionConfig(
            token_threshold=config.compaction_token_threshold,
            event_retention_size=config.compaction_event_retention_size,
            compaction_interval=config.compaction_interval,
            overlap_size=config.compaction_overlap_size,
        ),
    )
