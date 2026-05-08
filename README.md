# Sepsis Atlas Hackathon Prototype

This project turns sepsis PDFs into a structured, source-grounded evidence table for mortality estimation use cases. It is optimized for a two-day hackathon demo: clear ingestion, retrieval, extraction, validation flags, and a Streamlit review UI.

## What It Does

1. Parses full-text PDFs from `articles/`.
2. Splits pages into passage chunks while preserving `source_file`, `source_page`, and `source_passage_id`.
3. Retrieves passages relevant to a natural-language clinical question.
4. Extracts an analysis-ready table with cohort, predictor, outcome, timing, method, effect size, performance, and source fields.
5. Marks missing values as `Not reported`.
6. Adds `support_status` and `support_notes` so reviewers can see whether values are lexically supported by the cited source anchor.

The pipeline uses OpenRouter for LLM extraction. Without a valid `OPENROUTER_API_KEY`, the default build command fails fast so missing credentials are visible.

## Setup

**Linux & Mac**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For Win**
``` bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Place the hackathon PDFs here:

```text
articles/
  paper_1.pdf
  paper_2.pdf
  ...
```

Add your OpenRouter key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_VISION_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=http://localhost:8501
OPENROUTER_APP_TITLE=Sepsis Atlas Hackathon
ATLAS_WORKERS=4
ATLAS_VISION_MODE=auto
ATLAS_VISION_ERROR_POLICY=warn
```

## Build The Evidence Table

OpenRouter extraction:

> Avoid using '--vision-mode all' - it consumes way too many tokens and could get you roasted in the general channel (don't ask how we found out).

**Linux & Mac**
```bash
python build_atlas.py \
  --articles ./articles \
  --backend llm \
  --workers 4 \
  --vision-mode off \
  --query "What predictors, biomarkers, severity scores, statistical methods, effect sizes, and model performance metrics are reported for mortality estimation in sepsis?"
```

**Win**
``` bash
python build_atlas.py `
  --articles ./articles `
  --backend llm `
  --workers 4 `
  --vision-mode off `
  --query "What predictors, biomarkers, severity scores, statistical methods, effect sizes, and model performance metrics are reported for mortality estimation in sepsis?"
```

Performance knobs:

- `--workers 4`: runs multiple per-study OpenRouter extractions concurrently. Increase for speed, lower it if OpenRouter rate limits requests.
- `--parse-workers 4`: parses multiple PDFs concurrently.
- `--vision-mode auto`: runs OpenRouter vision on scanned/image-heavy pages.
- `--vision-mode all`: renders every PDF page and extracts visual text from tables, figures, captions, and scanned pages. Use this for final judged runs when visual data matters.
- `--vision-error-policy warn`: keeps the build running if OpenRouter vision fails or credits run out. Use `error` for strict final runs.
- `--vision-dpi 144`: controls rendered page resolution. Higher values can improve OCR but increase latency and token/image cost.
- `--top-k-per-study 6`: fewer passages per paper usually reduces token use and latency.
- `--max-pages-per-pdf 12`: useful for quick demos when full-paper extraction is not needed.

Usage Pricing:

- `--vision-mode off` - approximately $0.04 per run
- `--vision-mode all` - approximately $1.50 per run

Optional regex fallback for offline debugging:

```bash
python build_atlas.py --articles ./articles --backend regex
```

Outputs:

- `sepsis_atlas_results.csv`: analysis-ready evidence table.
- `evidence_passages.jsonl`: retrieved passage audit trail.

## Run The Demo App

```bash
streamlit run app.py
```

The app accepts a natural-language clinical question, ranks the evidence rows by relevance, returns a structured table, and shows source snippets for rapid review.

## Current Demo Data

The repository includes a prebuilt `sepsis_atlas_results.csv` so the UI can be demonstrated even when PDFs are not present locally. Re-run `build_atlas.py` after adding PDFs to regenerate the table with page-level anchors and validation fields.

## Evidence Table Fields

- `study_name`
- `population`
- `sample_size`
- `predictor`
- `outcome`
- `timing`
- `method`
- `effect_size`
- `performance`
- `notes`
- `source_file`
- `source_page`
- `source_passage_id`
- `source_link`
- `source_anchor`
- `field_sources_json`
- `support_status`
- `support_notes`
- `query`
- `query_relevance_score`

## Limitations

This is a research-assistant prototype, not a clinical decision tool. The regex fallback is intentionally conservative and should be reviewed. LLM extraction improves flexibility, but values must still be verified against the source anchors before any downstream analysis.

