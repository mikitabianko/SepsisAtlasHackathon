from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_local_env() -> None:
    if load_dotenv is not None:
        load_dotenv()
        return

    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _usable_secret(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized not in {
        "",
        "replace_with_your_openrouter_api_key",
        "your_openrouter_api_key",
        "sk-or-v1-your-key-here",
    }


_load_local_env()


MISSING = "Not reported"

CANONICAL_COLUMNS = [
    "study_name",
    "population",
    "sample_size",
    "predictor",
    "outcome",
    "timing",
    "method",
    "effect_size",
    "performance",
    "notes",
    "source_file",
    "source_page",
    "source_passage_id",
    "source_link",
    "source_anchor",
    "field_sources_json",
    "support_status",
    "support_notes",
    "query",
    "query_relevance_score",
]

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
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patients",
    "relationship",
    "study",
    "the",
    "to",
    "using",
    "what",
    "with",
}
GENERIC_FIELD_WORDS = {
    "analysis",
    "criteria",
    "level",
    "levels",
    "model",
    "score",
    "scores",
    "test",
    "use",
    "value",
    "values",
    "variable",
    "variables",
}

QUERY_EXPANSIONS = {
    "mortality": {"mortality", "death", "died", "deceased", "fatality", "survival"},
    "death": {"mortality", "death", "died", "deceased", "fatality", "survival"},
    "lactate": {"lactate", "lactic"},
    "sofa": {"sofa", "sequential organ failure assessment"},
    "apache": {"apache", "acute physiology"},
    "auc": {"auc", "auroc", "area under", "roc"},
    "auroc": {"auc", "auroc", "area under", "roc"},
    "or": {"odds ratio", "or"},
    "hr": {"hazard ratio", "hr"},
    "risk": {"risk", "odds ratio", "hazard ratio", "regression"},
    "counterfactual": {"control", "untreated", "expected mortality", "risk", "prognostic"},
    "sepsis": {"sepsis", "septic", "infection", "shock"},
}

PREDICTOR_PATTERNS = [
    ("Lactate", r"\b(?:lactate|lactic acid)\b"),
    ("SOFA score", r"\bSOFA\b|sequential organ failure assessment"),
    ("qSOFA score", r"\bqSOFA\b"),
    ("APACHE score", r"\bAPACHE(?:\s+II|\s+III)?\b"),
    ("SIRS criteria", r"\bSIRS\b"),
    ("Lymphocyte count", r"\blymphocyte"),
    ("Neutrophil-to-lymphocyte ratio", r"\bNLR\b|neutrophil[- ]to[- ]lymphocyte"),
    ("IL-6", r"\bIL[- ]?6\b|interleukin[- ]?6"),
    ("Presepsin", r"\bpresepsin\b"),
    ("Procalcitonin", r"\bPCT\b|procalcitonin"),
    ("C-reactive protein", r"\bCRP\b|C-reactive protein"),
    ("Bilirubin", r"\bbilirubin\b"),
    ("Phosphate", r"\bphosphate|hypophosphat"),
    ("Hematocrit", r"\bhematocrit\b"),
    ("Mean corpuscular volume", r"\bMCV\b|mean corpuscular volume"),
    ("Age", r"\bage\b"),
    ("Charlson Comorbidity Index", r"\bCharlson\b|\bCCI\b"),
    ("Glasgow Coma Scale", r"\bGCS\b|Glasgow Coma"),
    ("Temperature", r"\btemperature\b"),
    ("Oxygenation index", r"\boxygenation index\b"),
    ("Albumin", r"\balbumin\b"),
    ("BUN", r"\bBUN\b|blood urea nitrogen"),
    ("Creatinine", r"\bcreatinine\b"),
    ("Platelet count", r"\bplatelet"),
    ("Vasopressor use", r"\bvasopressor"),
    ("Mechanical ventilation", r"\bmechanical ventilation\b|\bMV\b"),
    ("Renal replacement therapy", r"\brenal replacement therapy\b|\bRRT\b|\bCRRT\b"),
]

METHOD_PATTERNS = [
    ("ROC analysis", r"\bROC\b|receiver operating characteristic|AUROC|AUC"),
    ("Logistic regression", r"logistic regression"),
    ("Cox regression", r"\bCox\b|hazard ratio"),
    ("Multivariable regression", r"multivariable|multivariate"),
    ("LASSO regression", r"\bLASSO\b"),
    ("Kaplan-Meier analysis", r"Kaplan[- ]Meier"),
    ("Random forest", r"random forest"),
    ("Machine learning model", r"machine learning|XGBoost|gradient boosting|neural network"),
]

OUTCOME_PATTERNS = [
    r"\b(?:28|30|60|90|180|365)[- ]day (?:all[- ]cause )?(?:mortality|death)\b",
    r"\b(?:in[- ]hospital|hospital|ICU|one[- ]year|1[- ]year) (?:mortality|death)\b",
    r"\bmortality (?:within|at) (?:28|30|60|90|180|365) days\b",
    r"\bsurvival\b",
]

TIMING_PATTERNS = [
    r"\bwithin (?:the )?first \d+\s*(?:h|hr|hour|hours|day|days)\b",
    r"\bfirst \d+\s*(?:h|hr|hour|hours|day|days)\b",
    r"\b(?:on|at) (?:ICU |hospital |ED )?admission\b",
    r"\bat (?:the time of )?(?:sepsis )?diagnosis\b",
    r"\bbaseline\b",
    r"\bday [0-9]+\b",
]

EFFECT_REGEXES = [
    r"\b(?:OR|odds ratio)\s*(?:=|of|:)?\s*[0-9]+(?:\.[0-9]+)?(?:\s*\([^)]+\))?(?:\s*,?\s*p\s*[<=>]\s*0?\.[0-9]+)?",
    r"\b(?:HR|hazard ratio)\s*(?:=|of|:)?\s*[0-9]+(?:\.[0-9]+)?(?:\s*\([^)]+\))?(?:\s*,?\s*p\s*[<=>]\s*0?\.[0-9]+)?",
    r"\b(?:RR|relative risk|risk ratio)\s*(?:=|of|:)?\s*[0-9]+(?:\.[0-9]+)?(?:\s*\([^)]+\))?",
    r"\bcut[- ]?off(?: value)?\s*(?:=|of|:)?\s*[<>]?\s*[0-9]+(?:\.[0-9]+)?(?:\s*[x×]\s*10\^?[0-9]+)?",
]

PERFORMANCE_REGEXES = [
    r"\b(?:AUC|AUROC)\s*(?:=|of|:)?\s*0?\.[0-9]{2,3}(?:\s*\([^)]+\))?",
    r"\barea under (?:the )?(?:ROC |receiver operating characteristic )?curve\s*(?:=|of|:)?\s*0?\.[0-9]{2,3}(?:\s*\([^)]+\))?",
    r"\bsens(?:itivity)?\s*(?:=|:)?\s*[0-9]+(?:\.[0-9]+)?%?",
    r"\bspec(?:ificity)?\s*(?:=|:)?\s*[0-9]+(?:\.[0-9]+)?%?",
]


@dataclass
class Passage:
    passage_id: str
    source_file: str
    source_path: str
    page_number: int | str
    text: str
    score: float = 0.0


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_source_link(source_path: str | Path | None, source_file: str, source_page: Any) -> str:
    """Build a local PDF link with a page fragment for validation."""

    try:
        page_number = int(float(str(source_page)))
    except (TypeError, ValueError):
        page_number = None

    path: Path | None = None
    if source_path:
        candidate = Path(str(source_path))
        if candidate.exists():
            path = candidate.resolve()

    if path is None and source_file and source_file != MISSING:
        candidate = Path("articles") / source_file
        if candidate.exists():
            path = candidate.resolve()

    if path is None:
        return MISSING

    link = path.as_uri()
    if page_number:
        link = f"{link}#page={page_number}"
    return link


def sentence_split(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if part.strip()]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", text)]


def expand_query_terms(query: str) -> set[str]:
    terms = {token for token in tokenize(query) if token not in STOPWORDS}
    lower_query = query.lower()
    for key, expansion in QUERY_EXPANSIONS.items():
        if len(key) <= 2:
            matched = re.search(rf"\b{re.escape(key)}\b", lower_query) is not None
        else:
            matched = key in terms or key in lower_query
        if matched:
            terms.update(expansion)
    if not terms:
        terms.update({"sepsis", "mortality", "auc", "odds ratio", "lactate", "sofa"})
    return terms


def split_pages_into_passages(
    pages: Iterable[Any],
    *,
    max_chars: int = 2400,
    overlap_chars: int = 250,
) -> list[Passage]:
    passages: list[Passage] = []

    for page in pages:
        metadata = getattr(page, "metadata", {}) or {}
        text = getattr(page, "text", "") or ""
        source_file = str(metadata.get("file_name") or Path(str(metadata.get("file_path", "unknown"))).name)
        source_path = str(metadata.get("file_path") or "")
        page_number = metadata.get("page_number") or metadata.get("page_label") or MISSING

        raw_parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if not raw_parts:
            raw_parts = sentence_split(text)

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            chunk = normalize_text(" ".join(current))
            if chunk:
                chunks.append(chunk)
            if overlap_chars and len(chunk) > overlap_chars:
                current = [chunk[-overlap_chars:]]
                current_len = len(current[0])
            else:
                current = []
                current_len = 0

        for part in raw_parts:
            if len(part) > max_chars:
                for sentence in sentence_split(part):
                    if current and current_len + len(sentence) + 1 > max_chars:
                        flush()
                    current.append(sentence)
                    current_len += len(sentence) + 1
                continue

            if current and current_len + len(part) + 1 > max_chars:
                flush()
            current.append(part)
            current_len += len(part) + 1

        if current:
            flush()

        for index, chunk in enumerate(chunks, start=1):
            passages.append(
                Passage(
                    passage_id=f"{Path(source_file).stem}:p{page_number}:c{index}",
                    source_file=source_file,
                    source_path=source_path,
                    page_number=page_number,
                    text=chunk,
                )
            )

    return passages


def score_text_for_query(text: str, query: str) -> float:
    lower_text = text.lower()
    terms = expand_query_terms(query)
    score = 0.0

    for term in terms:
        term_lower = term.lower()
        if " " in term_lower:
            if term_lower in lower_text:
                score += 4.0
        else:
            matches = re.findall(rf"\b{re.escape(term_lower)}\b", lower_text)
            score += min(len(matches), 5) * 1.25

    clinical_bonuses = {
        "mortality": 2.0,
        "death": 1.5,
        "auc": 2.0,
        "auroc": 2.0,
        "odds ratio": 1.5,
        "hazard ratio": 1.5,
        "regression": 1.0,
        "sepsis": 1.0,
        "septic": 1.0,
    }
    for phrase, bonus in clinical_bonuses.items():
        if phrase in lower_text:
            score += bonus

    return round(score, 3)


def retrieve_passages(
    passages: Iterable[Passage],
    query: str,
    *,
    top_k: int = 8,
    per_file_limit: int | None = None,
) -> list[Passage]:
    scored = [
        Passage(
            passage_id=passage.passage_id,
            source_file=passage.source_file,
            source_path=passage.source_path,
            page_number=passage.page_number,
            text=passage.text,
            score=score_text_for_query(passage.text, query),
        )
        for passage in passages
    ]
    scored.sort(key=lambda item: item.score, reverse=True)

    selected: list[Passage] = []
    per_file_counts: dict[str, int] = {}
    for passage in scored:
        if passage.score <= 0:
            continue
        if per_file_limit is not None:
            count = per_file_counts.get(passage.source_file, 0)
            if count >= per_file_limit:
                continue
            per_file_counts[passage.source_file] = count + 1
        selected.append(passage)
        if len(selected) >= top_k:
            break
    return selected


def _unique_join(values: Iterable[str], *, limit: int = 6) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        value = normalize_text(value)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
        if len(output) >= limit:
            break
    return "; ".join(output) if output else MISSING


def _first_regex(patterns: Iterable[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(0))
    return MISSING


def _all_regex(patterns: Iterable[str], text: str, *, limit: int = 6) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        hits.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return [normalize_text(hit) for hit in hits[:limit]]


def _extract_population(text: str) -> str:
    candidates = []
    for sentence in sentence_split(text):
        lower = sentence.lower()
        if any(word in lower for word in ["patients", "cohort", "included", "enrolled", "admitted"]):
            candidates.append(sentence[:260])
    return _unique_join(candidates, limit=1)


def _extract_sample_size(text: str) -> str:
    patterns = [
        r"\bN\s*=\s*[\d,]+",
        r"\bn\s*=\s*[\d,]+",
        r"\btotal of [\d,]+ (?:eligible )?(?:patients|subjects|encounters)\b",
        r"\b[\d,]+ (?:eligible )?(?:patients|subjects|encounters) (?:were )?(?:included|enrolled|admitted)\b",
        r"\b[\d,]+ (?:patients|subjects|encounters)\b",
    ]
    hits = _all_regex(patterns, text, limit=4)

    death_hits = _all_regex(
        [
            r"\b(?:died|deaths?|non-survivors?)\s*(?:n\s*)?=?\s*[\d,]+",
            r"\b[\d,]+\s*(?:patients )?(?:died|deaths?|non-survivors?)\b",
            r"\bmortality (?:rate )?(?:was|of)?\s*[\d.]+%",
        ],
        text,
        limit=2,
    )
    return _unique_join(hits + death_hits, limit=5)


def _extract_predictors(text: str, query: str) -> str:
    found = [
        name
        for name, pattern in PREDICTOR_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]

    query_terms = expand_query_terms(query)
    query_matched = [
        name
        for name in found
        if any(term.lower() in name.lower() or name.lower() in term.lower() for term in query_terms)
    ]
    if query_matched:
        return _unique_join(query_matched + found, limit=6)
    return _unique_join(found, limit=6)


def _extract_outcome(text: str) -> str:
    hits = _all_regex(OUTCOME_PATTERNS, text, limit=3)
    if hits:
        return _unique_join(hits, limit=3)
    for sentence in sentence_split(text):
        if re.search(r"\bmortality|death|survival\b", sentence, flags=re.IGNORECASE):
            return normalize_text(sentence[:220])
    return MISSING


def _extract_method(text: str) -> str:
    methods = [
        name
        for name, pattern in METHOD_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    return _unique_join(methods, limit=5)


def _extract_timing(text: str) -> str:
    hits = _all_regex(TIMING_PATTERNS, text, limit=3)
    return _unique_join(hits, limit=3)


def _extract_effect_size(text: str) -> str:
    hits = _all_regex(EFFECT_REGEXES, text, limit=6)
    return _unique_join(hits, limit=6)


def _extract_performance(text: str) -> str:
    hits = _all_regex(PERFORMANCE_REGEXES, text, limit=8)
    return _unique_join(hits, limit=8)


def _best_anchor(text: str, row: dict[str, str]) -> str:
    sentences = sentence_split(text)
    important_terms = expand_query_terms(" ".join([row.get("predictor", ""), row.get("outcome", "")]))
    important_terms.update({"mortality", "death", "auc", "auroc", "odds ratio", "hazard ratio", "regression"})

    best_sentence = ""
    best_score = -1.0
    for idx, sentence in enumerate(sentences):
        window = " ".join(sentences[max(0, idx - 1) : min(len(sentences), idx + 2)])
        score = score_text_for_query(window, " ".join(sorted(important_terms)))
        if any(regex_hit.lower() in window.lower() for regex_hit in [row.get("effect_size", ""), row.get("performance", "")] if regex_hit != MISSING):
            score += 5
        if score > best_score:
            best_score = score
            best_sentence = window

    if not best_sentence and sentences:
        best_sentence = sentences[0]
    return normalize_text(best_sentence[:700]) or MISSING


def _source_for_anchor(anchor: str, passages: list[Passage]) -> Passage:
    anchor_lower = anchor.lower()
    for passage in passages:
        if anchor != MISSING and anchor_lower in passage.text.lower():
            return passage
    return passages[0]


def _anchor_for_value(text: str, value: str, fallback_anchor: str) -> str:
    value = normalize_text(value)
    if not value or value == MISSING:
        return MISSING

    sentences = sentence_split(text)
    if not sentences:
        return fallback_anchor

    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    words = [word for word in tokenize(value) if word not in STOPWORDS and len(word) > 2]

    best_anchor = fallback_anchor
    best_score = -1
    for idx, sentence in enumerate(sentences):
        window = " ".join(sentences[max(0, idx - 1) : min(len(sentences), idx + 2)])
        lower_window = window.lower()
        score = 0
        score += sum(3 for number in numbers if number in window)
        score += sum(1 for word in set(words[:10]) if word.lower() in lower_window)
        if value.lower() in lower_window:
            score += 8
        if score > best_score:
            best_score = score
            best_anchor = window

    return normalize_text(best_anchor[:700]) or fallback_anchor


def _field_sources(text: str, fallback_anchor: str, row: dict[str, str]) -> str:
    field_sources = {}
    for field in [
        "population",
        "sample_size",
        "predictor",
        "outcome",
        "timing",
        "method",
        "effect_size",
        "performance",
    ]:
        value = row.get(field, MISSING)
        if value and value != MISSING:
            field_sources[field] = _anchor_for_value(text, str(value), fallback_anchor)
    return json.dumps(field_sources, ensure_ascii=False)


def validate_support(row: dict[str, Any]) -> tuple[str, str]:
    anchor = normalize_text(str(row.get("source_anchor") or ""))
    if not anchor or anchor == MISSING:
        return "unsupported", "No source anchor was extracted."

    checked = 0
    supported = 0
    weak_fields: list[str] = []
    try:
        field_sources = json.loads(str(row.get("field_sources_json") or "{}"))
    except json.JSONDecodeError:
        field_sources = {}

    for field in ["sample_size", "predictor", "outcome", "timing", "method", "effect_size", "performance"]:
        value = normalize_text(str(row.get(field) or ""))
        if not value or value == MISSING:
            continue

        checked += 1
        field_anchor = normalize_text(str(field_sources.get(field) or anchor))
        lower_anchor = field_anchor.lower()
        numbers = re.findall(r"\d+(?:\.\d+)?", value)
        words = [
            word
            for word in tokenize(value)
            if word not in STOPWORDS and word not in GENERIC_FIELD_WORDS and len(word) > 2
        ]

        numeric_match = bool(numbers) and any(number in field_anchor for number in numbers)
        word_matches = sum(1 for word in set(words[:8]) if word.lower() in lower_anchor)
        word_match = word_matches >= min(2, max(1, len(set(words[:8]))))

        if numeric_match or word_match:
            supported += 1
        else:
            weak_fields.append(field)

    if checked == 0:
        return "needs review", "Only missing values were extracted."
    if supported == checked:
        return "supported", "All reported fields have lexical support in the source anchor."
    if supported:
        return "partial", f"Review fields: {', '.join(weak_fields)}."
    return "needs review", "Reported fields were not found lexically in the source anchor."


def _normalize_row(row: dict[str, Any], passages: list[Passage], query: str) -> dict[str, Any]:
    normalized = {column: row.get(column, MISSING) for column in CANONICAL_COLUMNS}
    normalized["study_name"] = normalize_text(str(normalized.get("study_name") or Path(passages[0].source_file).stem))
    normalized["query"] = query
    combined_text = " ".join(passage.text for passage in passages)

    anchor = normalize_text(str(normalized.get("source_anchor") or ""))
    if not anchor or anchor == MISSING:
        anchor = _best_anchor(combined_text, normalized)
    normalized["source_anchor"] = anchor

    source_passage = _source_for_anchor(anchor, passages)
    normalized["source_file"] = normalized.get("source_file") if normalized.get("source_file") != MISSING else source_passage.source_file
    normalized["source_page"] = normalized.get("source_page") if normalized.get("source_page") != MISSING else source_passage.page_number
    normalized["source_passage_id"] = (
        normalized.get("source_passage_id")
        if normalized.get("source_passage_id") != MISSING
        else source_passage.passage_id
    )
    if normalized.get("source_link") in ("", MISSING, None):
        normalized["source_link"] = build_source_link(
            source_passage.source_path,
            str(normalized.get("source_file") or source_passage.source_file),
            normalized.get("source_page") or source_passage.page_number,
        )

    if normalized.get("field_sources_json") in ("", MISSING, None):
        normalized["field_sources_json"] = _field_sources(combined_text, anchor, normalized)

    status, notes = validate_support(normalized)
    normalized["support_status"] = status
    normalized["support_notes"] = notes
    normalized["query_relevance_score"] = round(max((passage.score for passage in passages), default=0), 3)

    for column in CANONICAL_COLUMNS:
        if normalized.get(column) in (None, ""):
            normalized[column] = MISSING
    return normalized


def extract_rows_regex(study_name: str, passages: list[Passage], query: str) -> list[dict[str, Any]]:
    combined_text = "\n\n".join(passage.text for passage in passages)
    if not combined_text:
        return []

    row = {
        "study_name": study_name or Path(passages[0].source_file).stem,
        "population": _extract_population(combined_text),
        "sample_size": _extract_sample_size(combined_text),
        "predictor": _extract_predictors(combined_text, query),
        "outcome": _extract_outcome(combined_text),
        "timing": _extract_timing(combined_text),
        "method": _extract_method(combined_text),
        "effect_size": _extract_effect_size(combined_text),
        "performance": _extract_performance(combined_text),
        "notes": "Regex fallback extraction; review source anchor before clinical use.",
    }
    row["source_anchor"] = _best_anchor(combined_text, row)
    return [_normalize_row(row, passages, query)]


def _api_key_for_backend() -> tuple[str | None, str | None, dict[str, str]]:
    if _usable_secret(os.environ.get("OPENROUTER_API_KEY")):
        return (
            os.environ["OPENROUTER_API_KEY"],
            os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            {
                "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Sepsis Atlas Hackathon"),
            },
        )
    if _usable_secret(os.environ.get("OPENAI_API_KEY")):
        return os.environ["OPENAI_API_KEY"], os.environ.get("OPENAI_BASE_URL"), {}
    return None, None, {}


def _extract_rows_llm(
    study_name: str,
    passages: list[Passage],
    query: str,
    *,
    model: str,
) -> list[dict[str, Any]]:
    api_key, base_url, default_headers = _api_key_for_backend()
    if not api_key:
        raise RuntimeError("No OPENROUTER_API_KEY is configured. Add it to `.env` or your shell environment.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The optional `openai` package is required for LLM extraction.") from exc

    context_blocks = []
    for passage in passages:
        context_blocks.append(
            f"[{passage.passage_id} | file={passage.source_file} | page={passage.page_number}]\n{passage.text}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are extracting analysis-ready data for a Sepsis Atlas hackathon prototype.

Natural language query:
{query}

Study:
{study_name}

Use only the passages below. Extract rows for mortality prediction / counterfactual mortality estimation evidence.
Return JSON only with this shape:
{{
  "rows": [
    {{
      "study_name": "...",
      "population": "...",
      "sample_size": "...",
      "predictor": "...",
      "outcome": "...",
      "timing": "...",
      "method": "...",
      "effect_size": "...",
      "performance": "...",
      "notes": "...",
      "source_file": "...",
      "source_page": "...",
      "source_passage_id": "...",
      "source_anchor": "exact short quote copied from one source passage"
    }}
  ]
}}

Rules:
- Use "Not reported" for any field not explicitly supported.
- Every reported number must appear in source_anchor or in the same cited passage.
- source_anchor must be copied verbatim from the passages and should be under 700 characters.
- Prefer one row per distinct predictor/outcome/effect association. If only a risk model is reported, extract the model as one row.
- Do not infer treatment effects. This table supports mortality estimation; it does not compute counterfactual mortality.

Passages:
{context}
""".strip()

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You produce source-grounded clinical evidence extraction JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    rows = parsed.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [_normalize_row(row, passages, query) for row in rows if isinstance(row, dict)]


def extract_rows_for_study(
    study_name: str,
    passages: list[Passage],
    query: str,
    *,
    backend: str = "llm",
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    if not passages:
        return []

    backend = backend.lower()
    if backend not in {"auto", "llm", "regex"}:
        raise ValueError("backend must be one of: auto, llm, regex")

    if backend in {"auto", "llm"}:
        try:
            rows = _extract_rows_llm(study_name, passages, query, model=model)
            if rows:
                return rows
        except Exception as exc:
            if backend == "llm":
                raise
            print(f"LLM extraction skipped for {study_name}: {exc}")

    return extract_rows_regex(study_name, passages, query)


def row_relevance(row: dict[str, Any], query: str) -> float:
    text = " ".join(str(row.get(column, "")) for column in CANONICAL_COLUMNS)
    return score_text_for_query(text, query)
