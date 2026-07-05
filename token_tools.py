"""Small utilities for Gemini token optimization comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config import ModelPricing


@dataclass(frozen=True)
class CountResult:
    tokens: int
    method: str
    error: str = ""


@dataclass(frozen=True)
class CostBreakdown:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    normal_input_tokens: int
    cache_storage_token_hours: int
    input_usd: float
    cached_input_usd: float
    output_usd: float
    cache_storage_usd: float
    total_usd: float

    def to_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def textify(contents: Any) -> str:
    """Best-effort conversion of notebook inputs into plain text."""

    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, Mapping):
        return "\n".join(f"{key}: {textify(value)}" for key, value in contents.items())
    if isinstance(contents, (list, tuple)):
        return "\n".join(textify(item) for item in contents)
    return str(contents)


def estimate_tokens(contents: Any) -> int:
    """Estimate Gemini tokens using the public rule of thumb: ~4 chars/token."""

    text = textify(contents).strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _extract_total_tokens(response: Any) -> int:
    for attr in ("total_tokens", "totalTokens"):
        value = getattr(response, attr, None)
        if value is not None:
            return int(value)
    if isinstance(response, Mapping):
        for key in ("total_tokens", "totalTokens"):
            value = response.get(key)
            if value is not None:
                return int(value)
    raise AttributeError("No total token field found on count_tokens response.")


def count_tokens(
    contents: Any,
    *,
    model: str,
    client: Any = None,
    allow_estimate_fallback: bool = True,
) -> CountResult:
    """Count Gemini input tokens, using the API when a client is provided."""

    if not textify(contents).strip():
        return CountResult(tokens=0, method="none")

    if client is None:
        if not allow_estimate_fallback:
            raise ValueError("A Gemini client is required for strict token counting.")
        return CountResult(tokens=estimate_tokens(contents), method="estimated")

    try:
        response = client.models.count_tokens(model=model, contents=contents)
        return CountResult(tokens=_extract_total_tokens(response), method="gemini")
    except Exception as exc:
        if not allow_estimate_fallback:
            raise
        return CountResult(
            tokens=estimate_tokens(contents),
            method="estimated_after_error",
            error=f"{type(exc).__name__}: {exc}",
        )


def usage_metadata_to_dict(usage_metadata: Any) -> Dict[str, int]:
    """Normalize google-genai usage metadata into snake_case integer fields."""

    if usage_metadata is None:
        return {}
    if hasattr(usage_metadata, "model_dump"):
        raw = usage_metadata.model_dump(exclude_none=True)
    elif isinstance(usage_metadata, Mapping):
        raw = dict(usage_metadata)
    else:
        raw = {
            name: getattr(usage_metadata, name)
            for name in dir(usage_metadata)
            if not name.startswith("_") and not callable(getattr(usage_metadata, name))
        }

    aliases = {
        "prompt_token_count": ("prompt_token_count", "promptTokenCount"),
        "candidates_token_count": ("candidates_token_count", "candidatesTokenCount"),
        "cached_content_token_count": (
            "cached_content_token_count",
            "cachedContentTokenCount",
        ),
        "thoughts_token_count": ("thoughts_token_count", "thoughtsTokenCount"),
        "total_token_count": ("total_token_count", "totalTokenCount"),
    }
    normalized: Dict[str, int] = {}
    for target, names in aliases.items():
        for name in names:
            if raw.get(name) is not None:
                normalized[target] = int(raw[name])
                break
    return normalized


def generate_and_measure(
    prompt: str,
    *,
    model: str,
    client: Any,
    max_output_tokens: int,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, int]]:
    """Run Gemini once and return response text plus usage metadata."""

    from google.genai import types

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if hasattr(types, "ThinkingConfig"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return extract_response_text(response), usage_metadata_to_dict(response.usage_metadata)


def extract_response_text(response: Any) -> str:
    """Extract text from a google-genai response across SDK/model variants."""

    text = getattr(response, "text", None)
    if text:
        return text

    pieces: List[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                pieces.append(part_text)
    return "\n".join(pieces)


def calculate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing,
    cached_input_tokens: int = 0,
    cache_storage_token_hours: int = 0,
) -> CostBreakdown:
    """Calculate USD cost from token counts and per-million-token prices."""

    cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    normal_input = max(input_tokens - cached, 0)
    input_usd = normal_input / 1_000_000 * pricing.input_per_million
    cached_input_usd = cached / 1_000_000 * pricing.cached_input_per_million
    output_usd = max(output_tokens, 0) / 1_000_000 * pricing.output_per_million
    cache_storage_usd = (
        max(cache_storage_token_hours, 0)
        / 1_000_000
        * pricing.cache_storage_per_million_token_hour
    )
    total_usd = input_usd + cached_input_usd + output_usd + cache_storage_usd
    return CostBreakdown(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        normal_input_tokens=normal_input,
        cache_storage_token_hours=cache_storage_token_hours,
        input_usd=input_usd,
        cached_input_usd=cached_input_usd,
        output_usd=output_usd,
        cache_storage_usd=cache_storage_usd,
        total_usd=total_usd,
    )


def pct_saving(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100


def compare_prompts(
    *,
    approach: str,
    before_prompt: str,
    after_prompt: str,
    model: str,
    after_model: Optional[str] = None,
    before_pricing: ModelPricing,
    after_pricing: Optional[ModelPricing] = None,
    client: Any = None,
    before_output_tokens: int,
    after_output_tokens: Optional[int] = None,
    before_cached_input_tokens: int = 0,
    after_cached_input_tokens: int = 0,
    before_cache_storage_token_hours: int = 0,
    after_cache_storage_token_hours: int = 0,
    allow_estimate_fallback: bool = True,
    note: str = "",
) -> Dict[str, Any]:
    """Create one before/after comparison row."""

    after_model = after_model or model
    after_pricing = after_pricing or before_pricing
    after_output_tokens = (
        before_output_tokens if after_output_tokens is None else after_output_tokens
    )
    before_count = count_tokens(
        before_prompt,
        model=model,
        client=client,
        allow_estimate_fallback=allow_estimate_fallback,
    )
    after_count = count_tokens(
        after_prompt,
        model=after_model,
        client=client,
        allow_estimate_fallback=allow_estimate_fallback,
    )

    before_cost = calculate_cost(
        input_tokens=before_count.tokens,
        output_tokens=before_output_tokens,
        pricing=before_pricing,
        cached_input_tokens=before_cached_input_tokens,
        cache_storage_token_hours=before_cache_storage_token_hours,
    )
    after_cost = calculate_cost(
        input_tokens=after_count.tokens,
        output_tokens=after_output_tokens,
        pricing=after_pricing,
        cached_input_tokens=after_cached_input_tokens,
        cache_storage_token_hours=after_cache_storage_token_hours,
    )

    row = {
        "approach": approach,
        "before_model": model,
        "after_model": after_model,
        "count_method": (
            before_count.method
            if before_count.method == after_count.method
            else f"{before_count.method}/{after_count.method}"
        ),
        "before_input_tokens": before_count.tokens,
        "after_input_tokens": after_count.tokens,
        "before_output_tokens": before_output_tokens,
        "after_output_tokens": after_output_tokens,
        "input_token_saving_pct": pct_saving(before_count.tokens, after_count.tokens),
        "output_token_saving_pct": pct_saving(
            before_output_tokens, after_output_tokens
        ),
        "cost_saving_pct": pct_saving(before_cost.total_usd, after_cost.total_usd),
        "before_cost_usd": before_cost.total_usd,
        "after_cost_usd": after_cost.total_usd,
        "before_cached_input_tokens": before_cached_input_tokens,
        "after_cached_input_tokens": after_cached_input_tokens,
        "before_cache_storage_token_hours": before_cache_storage_token_hours,
        "after_cache_storage_token_hours": after_cache_storage_token_hours,
        "note": note,
    }
    if before_count.error or after_count.error:
        row["count_warning"] = before_count.error or after_count.error
    return row


def compare_cached_reuse(
    *,
    approach: str,
    prompt: str,
    static_context: str,
    model: str,
    pricing: ModelPricing,
    client: Any = None,
    calls: int = 10,
    output_tokens_per_call: int = 220,
    cache_ttl_seconds: int = 600,
    allow_estimate_fallback: bool = True,
) -> Dict[str, Any]:
    """Compare repeated calls with and without Gemini context caching."""

    prompt_count = count_tokens(
        prompt,
        model=model,
        client=client,
        allow_estimate_fallback=allow_estimate_fallback,
    )
    static_count = count_tokens(
        static_context,
        model=model,
        client=client,
        allow_estimate_fallback=allow_estimate_fallback,
    )
    total_input = prompt_count.tokens * calls
    total_output = output_tokens_per_call * calls
    cached_reads = max(calls - 1, 0)
    cached_input = min(static_count.tokens, prompt_count.tokens) * cached_reads
    storage_token_hours = int(static_count.tokens * cache_ttl_seconds / 3600)

    before_cost = calculate_cost(
        input_tokens=total_input,
        output_tokens=total_output,
        pricing=pricing,
    )
    after_cost = calculate_cost(
        input_tokens=total_input,
        output_tokens=total_output,
        pricing=pricing,
        cached_input_tokens=cached_input,
        cache_storage_token_hours=storage_token_hours,
    )

    return {
        "approach": approach,
        "before_model": model,
        "after_model": model,
        "count_method": prompt_count.method,
        "before_input_tokens": total_input,
        "after_input_tokens": total_input,
        "before_output_tokens": total_output,
        "after_output_tokens": total_output,
        "input_token_saving_pct": 0.0,
        "output_token_saving_pct": 0.0,
        "cost_saving_pct": pct_saving(before_cost.total_usd, after_cost.total_usd),
        "before_cost_usd": before_cost.total_usd,
        "after_cost_usd": after_cost.total_usd,
        "before_cached_input_tokens": 0,
        "after_cached_input_tokens": cached_input,
        "before_cache_storage_token_hours": 0,
        "after_cache_storage_token_hours": storage_token_hours,
        "repeat_calls": calls,
        "note": (
            f"{calls} repeated calls; prompt tokens are unchanged, but "
            "the static prefix is billed as cached input after the first call."
        ),
    }


def normalize_words(text: str) -> List[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS and len(word) > 1
    ]


def split_sections(document: str) -> List[Tuple[str, str]]:
    """Split markdown-ish text into titled sections."""

    sections: List[Tuple[str, str]] = []
    current_title = "Context"
    current_lines: List[str] = []
    for line in document.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, body) for title, body in sections if body]


def select_relevant_sections(
    document: str,
    question: str,
    *,
    max_sections: int = 3,
) -> str:
    """Lazy-load only the sections that overlap with the question keywords."""

    query_terms = set(normalize_words(question))
    scored = []
    for title, body in split_sections(document):
        section_text = f"{title}\n{body}"
        section_terms = set(normalize_words(section_text))
        overlap = query_terms & section_terms
        title_overlap = query_terms & set(normalize_words(title))
        score = len(overlap) + (2 * len(title_overlap))
        scored.append((score, title, body))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [
        f"## {title}\n{body}"
        for score, title, body in scored[:max_sections]
        if score > 0
    ]
    if not selected:
        selected = [f"## {title}\n{body}" for _, title, body in scored[:1]]
    return "\n\n".join(selected)


def format_history(history: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(f"{role.upper()}: {message}" for role, message in history)


def compact_history(
    history: Sequence[Tuple[str, str]],
    *,
    keep_last_turns: int = 2,
    summary_sentences: int = 5,
) -> str:
    """A local, extractive approximation of context compaction."""

    if not history:
        return ""
    keep_items = keep_last_turns * 2
    older = history[:-keep_items] if len(history) > keep_items else []
    recent = history[-keep_items:] if keep_items else []
    summary_parts: List[str] = []
    for _, message in older:
        sentences = re.split(r"(?<=[.!?])\s+", message.strip())
        for sentence in sentences:
            if sentence and len(summary_parts) < summary_sentences:
                summary_parts.append(sentence)
    lines = []
    if summary_parts:
        lines.append("SUMMARY_OF_OLDER_TURNS: " + " ".join(summary_parts))
    if recent:
        lines.append("RECENT_RAW_TURNS:\n" + format_history(recent))
    return "\n\n".join(lines)


def local_nlp_answer(question: str, document: str) -> Optional[str]:
    """Answer a narrow policy question locally when the evidence is explicit."""

    question_l = question.lower()
    if "refund" not in question_l:
        return None

    evidence = select_relevant_sections(document, question, max_sections=2)
    refund_window = re.search(r"(\d+)[ -]day refund", evidence, re.IGNORECASE)
    purchase_age = re.search(r"(\d+)\s+days?", question_l)
    window_days = int(refund_window.group(1)) if refund_window else 30
    age_days = int(purchase_age.group(1)) if purchase_age else None

    if age_days is not None and age_days <= window_days:
        decision = (
            f"Eligible: the request is {age_days} days after purchase, within "
            f"the {window_days}-day refund window."
        )
    elif age_days is not None:
        decision = (
            f"Not automatically eligible: the request is {age_days} days after "
            f"purchase, beyond the {window_days}-day refund window."
        )
    else:
        decision = f"Check eligibility against the {window_days}-day refund window."

    return (
        f"{decision} Ask support to verify the order, confirm there was no "
        "abuse or chargeback, process the refund in Billing Admin, and send the "
        "approved refund template. Evidence used locally: Refund policy."
    )


def make_agent_prompt(
    *,
    question: str,
    context: str,
    history: str = "",
    output_contract: str = "Answer in 4-6 concise bullets.",
) -> str:
    """Build a stable prompt for before/after token comparisons."""

    return f"""SYSTEM
You are a support policy assistant. Use only the supplied context. If evidence is missing, say what is missing.

OUTPUT CONTRACT
{output_contract}

CONVERSATION HISTORY
{history or "No prior turns."}

CONTEXT
{context}

USER QUESTION
{question}
"""


def build_sample_dataset() -> Dict[str, Any]:
    """Return a small but realistic dataset for the notebook examples."""

    policy_document = """## Refund policy
Customers on monthly or annual self-serve plans have a 30-day refund window from the original purchase date. Support can approve one courtesy refund per account per calendar year. Refunds are blocked when there is evidence of account abuse, chargeback fraud, or more than 10,000 successful API calls after the purchase. Annual plan refunds should be prorated only after the 30-day window; inside the window, process a full refund. Before processing, verify the account email, workspace ID, payment processor ID, and invoice number. After processing, send the approved refund template and tag the ticket with billing_refund.

## Enterprise billing policy
Enterprise contracts are handled by Finance Ops and must not be refunded directly by support. Contract cancellation requests require the account executive, customer success manager, and legal owner. A support agent may acknowledge the request, collect the renewal date, and open an internal Finance Ops task.

## Data retention and privacy
Deleted workspaces enter a 14-day soft-delete period. During this period admins can restore projects, API keys, and notebooks. After 14 days, production data is queued for permanent deletion. Backups age out within 35 days. Privacy export requests must be routed to the privacy operations queue and require requester verification.

## Token optimization playbook
Prefer sending the smallest context that can answer the question. Use section retrieval for long policy documents, cache stable instructions and policy text across repeated calls, compact old conversation turns into summaries, and answer deterministic questions locally before calling Gemini. Keep output contracts narrow so the model does not spend tokens restating background material.

## Support escalation matrix
Billing tickets route to Billing Support first. Refund exceptions route to Billing Support plus Risk Review. Data deletion tickets route to Privacy Ops. Security incidents route to Security Response. Product bugs route to the product triage queue with logs, timestamps, workspace ID, and reproduction steps.

## Regional notes
Customers in Indonesia, Singapore, Australia, and Japan use the same self-serve refund window unless an enterprise contract says otherwise. Local taxes may be non-refundable depending on the payment processor. Support should not provide tax advice; instead, attach the processor receipt and direct the customer to their tax advisor.
"""

    long_background = """## Product overview
The platform helps teams build AI applications with hosted models, notebooks, datasets, and deployment tools. Teams can create workspaces, invite users, rotate API keys, and monitor usage. The billing system supports monthly and annual plans, invoice downloads, and tax receipt emails.

## Operational notes
When answering customers, support agents should be empathetic, precise, and avoid internal-only wording. Do not reveal risk scoring rules. Do not promise refund timelines; payment processors usually finish in 5-10 business days, but timing can vary by bank. Keep the answer brief unless the user explicitly asks for detailed policy text.

## Historical release notes
Version 1.8 added notebook sharing and improved usage dashboards. Version 1.9 added regional invoice templates. Version 2.0 changed workspace deletion from immediate deletion to a soft-delete flow. Version 2.1 added billing_refund tags and improved audit logging.
"""

    full_context = policy_document + "\n\n" + long_background
    history = [
        ("user", "I am preparing a support response for a billing ticket."),
        (
            "model",
            "I can help. Please provide the purchase date, plan type, and region.",
        ),
        ("user", "The customer is on an annual self-serve Pro plan."),
        (
            "model",
            "For self-serve billing, I will check the refund policy and regional notes.",
        ),
        ("user", "The user is in Jakarta and paid by card."),
        (
            "model",
            "Indonesia follows the self-serve refund policy unless an enterprise contract applies.",
        ),
        ("user", "They used only 200 API calls since buying."),
        (
            "model",
            "That usage is below the refund abuse threshold described in the policy.",
        ),
        ("user", "They bought the plan 12 days ago."),
        (
            "model",
            "A 12-day-old self-serve annual plan is inside the standard refund window.",
        ),
    ]
    question = (
        "A customer in Jakarta bought an annual Pro self-serve plan 12 days ago "
        "and wants a refund. What should support answer and what internal steps "
        "should be followed?"
    )
    return {
        "policy_document": policy_document,
        "long_background": long_background,
        "full_context": full_context,
        "history": history,
        "question": question,
    }
