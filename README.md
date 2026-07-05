# Gemini Token Optimization Notebook

This project is a small, notebook-first comparison of Gemini input/output token usage and cost before and after common optimization patterns, with Google ADK configuration helpers.

## Files

- `config.py`: Google ADK / Google GenAI SDK config, gcloud ADC environment setup, pricing defaults, and optional ADK App helpers for context caching and compaction.
- `token_tools.py`: token counting, cost calculation, lazy context selection, simple local NLP fallback, prompt builders, and sample data.
- `token_comparison.ipynb`: apple-to-apple before/after comparison notebook.
- `requirements.txt`: convenience dependencies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
jupyter lab token_comparison.ipynb
```

If Jupyter raises `ModuleNotFoundError: No module named 'google'`, run the first dependency setup cell in the notebook or install the requirements in the same kernel environment. Do not install the package named `google`; this project needs `google-genai` and `google-adk`.

The notebook reads the project from `gcloud config get-value project` by default. You can change it with:

```bash
gcloud config set project "your-project-id"
```

Inside the notebook, the important live settings are:

```python
USE_LIVE_GEMINI = True
RUN_LIVE_GENERATION = True
PROJECT_ID = get_active_gcloud_project()
LOCATION = "global"
MODEL = "gemini-2.5-flash"
```

The notebook is live-only: it calls Gemini `count_tokens` for input token counts and `generate_content` for real answers plus output-token usage metadata. It will raise an error instead of falling back to local token estimates when credentials, project, or model configuration are missing.

## Approaches Compared

The notebook compares the same support-policy question across:

1. Baseline full-context prompting.
2. Lazy loading / retrieval of only relevant sections.
3. Local NLP first, skipping Gemini when deterministic policy evidence is enough.
4. Gemini context caching for repeated stable context.
5. Context compaction of older conversation turns.
6. Output discipline with a tighter response contract.
7. Model routing to a cheaper model for simple questions.

## Notes

The default pricing values are planning defaults from public Gemini pricing checked on 2026-06-22. Your Google Cloud / Agent Platform pricing, region, committed-use discounts, or enterprise contract may differ, so update `DEFAULT_PRICES` in `config.py` or override with environment variables:

```bash
export GEMINI_INPUT_PRICE_PER_MILLION="1.50"
export GEMINI_OUTPUT_PRICE_PER_MILLION="9.00"
export GEMINI_CACHED_INPUT_PRICE_PER_MILLION="0.15"
export GEMINI_CACHE_STORAGE_PRICE_PER_MILLION_TOKEN_HOUR="1.00"
```

Useful official references:

- ADK Python quickstart: https://adk.dev/get-started/python/
- ADK Gemini model auth: https://adk.dev/agents/models/google-gemini/
- ADK context caching: https://adk.dev/context/caching/
- ADK context compaction: https://adk.dev/context/compaction/
- Gemini token counting: https://ai.google.dev/gemini-api/docs/tokens
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
