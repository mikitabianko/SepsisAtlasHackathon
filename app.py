from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pandas as pd
import streamlit as st

from extract import CANONICAL_COLUMNS, MISSING, build_source_link, row_relevance, validate_support


DATA_PATH = Path("sepsis_atlas_results.csv")
STATIC_PDF_DIR = Path("static/source_pdfs")
STATIC_PDF_URL_PREFIX = "/app/static/source_pdfs"
DEFAULT_QUERY = "What is the relationship between lactate level and 28-day mortality in septic shock?"
VISIBLE_COLUMNS = [
    "study_name",
    "population",
    "sample_size",
    "predictor",
    "outcome",
    "timing",
    "method",
    "effect_size",
    "performance",
    "source_link",
    "source_file",
    "source_page",
    "support_status",
]
SOURCE_FIELDS = [
    "population",
    "sample_size",
    "predictor",
    "outcome",
    "timing",
    "method",
    "effect_size",
    "performance",
]


st.set_page_config(page_title="Sepsis Atlas", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        background: #ffffff;
    }
    .source-box {
        border-left: 4px solid #0f766e;
        padding: 0.65rem 0.8rem;
        background: #f8fafc;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_number(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def pdf_path_from_row(row: dict[str, object]) -> Path | None:
    source_link = str(row.get("source_link") or "")
    if source_link.startswith("file://"):
        parsed = urlparse(source_link)
        candidate = Path(unquote(parsed.path))
        if candidate.exists():
            return candidate

    source_file = str(row.get("source_file") or "")
    if source_file and source_file != MISSING:
        candidate = Path(source_file)
        if candidate.exists():
            return candidate.resolve()

        candidate = Path("articles") / source_file
        if candidate.exists():
            return candidate.resolve()

    study_name = str(row.get("study_name") or "")
    if study_name and study_name != MISSING:
        candidates = [Path("articles") / study_name]
        if not study_name.lower().endswith(".pdf"):
            candidates.append(Path("articles") / f"{study_name}.pdf")
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

    return None


def static_pdf_name(pdf_path: Path) -> str:
    resolved = pdf_path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in resolved.stem)
    stem = stem[:90] or "source"
    suffix = resolved.suffix if resolved.suffix.lower() == ".pdf" else ".pdf"
    return f"{stem}-{digest}{suffix}"


def app_pdf_link(row: dict[str, object]) -> str:
    pdf_path = pdf_path_from_row(row)
    if pdf_path is None:
        source_link = str(row.get("source_link") or "")
        if source_link and source_link != MISSING and not source_link.startswith("file://"):
            return source_link
        return MISSING

    STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)
    static_path = STATIC_PDF_DIR / static_pdf_name(pdf_path)
    if (
        not static_path.exists()
        or static_path.stat().st_size != pdf_path.stat().st_size
        or static_path.stat().st_mtime < pdf_path.stat().st_mtime
    ):
        shutil.copy2(pdf_path, static_path)

    link = f"{STATIC_PDF_URL_PREFIX}/{quote(static_path.name)}"
    page = page_number(row.get("source_page"))
    if page is not None:
        link = f"{link}#page={page}"
    return link


@st.cache_data
def load_evidence_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = MISSING

    if "source_file" not in df.columns or df["source_file"].eq(MISSING).all():
        df["source_file"] = df["study_name"]
    if "source_page" not in df.columns or df["source_page"].eq(MISSING).all():
        df["source_page"] = "Not indexed"
    if "source_passage_id" not in df.columns or df["source_passage_id"].eq(MISSING).all():
        df["source_passage_id"] = "Legacy CSV"
    if "source_link" not in df.columns:
        df["source_link"] = MISSING

    df = df.fillna(MISSING)
    rows = []
    for row in df.to_dict(orient="records"):
        row["source_link"] = app_pdf_link(row)
        if row.get("source_link") in ("", MISSING, None):
            row["source_link"] = build_source_link(
                None,
                str(row.get("source_file") or row.get("study_name") or ""),
                row.get("source_page"),
            )
        anchor = str(row.get("source_anchor", "")).strip()
        if row.get("field_sources_json") in ("", MISSING, None) and anchor:
            row["field_sources_json"] = json.dumps(
                {
                    field: anchor
                    for field in SOURCE_FIELDS
                    if str(row.get(field, MISSING)).strip() not in ("", MISSING)
                },
                ensure_ascii=False,
            )
        if row.get("support_status") in ("", MISSING, "legacy anchor", None):
            status, notes = validate_support(row)
            row["support_status"] = status
            row["support_notes"] = notes
        rows.append(row)

    return pd.DataFrame(rows)


def score_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    scored = df.copy()
    scored["_relevance"] = scored.apply(lambda row: row_relevance(row.to_dict(), query), axis=1)
    if query.strip():
        filtered = scored[scored["_relevance"] > 0].copy()
        if filtered.empty:
            filtered = scored.copy()
    else:
        filtered = scored.copy()
    return filtered.sort_values(["_relevance", "support_status"], ascending=[False, True])


def option_values(df: pd.DataFrame, column: str) -> list[str]:
    values = sorted({str(value) for value in df[column].dropna() if str(value).strip() and str(value) != MISSING})
    return values


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Evidence Set")
        st.metric("Rows", len(df))
        st.metric("Studies", df["study_name"].nunique())
        st.metric("Supported", int(df["support_status"].astype(str).str.contains("supported", case=False).sum()))

        statuses = option_values(df, "support_status")
        selected_statuses = st.multiselect("Support status", statuses, default=statuses)

        outcomes = option_values(df, "outcome")
        selected_outcomes = st.multiselect("Outcome", outcomes)

        predictors = option_values(df, "predictor")
        selected_predictors = st.multiselect("Predictor", predictors)

    filtered = df
    if selected_statuses:
        filtered = filtered[filtered["support_status"].isin(selected_statuses)]
    if selected_outcomes:
        filtered = filtered[filtered["outcome"].isin(selected_outcomes)]
    if selected_predictors:
        filtered = filtered[filtered["predictor"].isin(selected_predictors)]
    return filtered


def render_source_review(row: pd.Series) -> None:
    title = f"{row['study_name']} | {row.get('source_file', MISSING)} | page {row.get('source_page', MISSING)}"
    with st.expander(title):
        source_link = str(row.get("source_link", MISSING))
        if source_link and source_link != MISSING:
            st.link_button("Open PDF at source page", source_link)
        else:
            st.caption("PDF link unavailable. Rebuild the atlas from local PDFs or place the source PDF in `articles/`.")
        source_anchor = html.escape(str(row.get("source_anchor", MISSING)))
        st.markdown(f"<div class='source-box'>{source_anchor}</div>", unsafe_allow_html=True)
        support_notes = row.get("support_notes", MISSING)
        if support_notes != MISSING:
            st.caption(f"Support check: {support_notes}")

        field_sources = row.get("field_sources_json", MISSING)
        if isinstance(field_sources, str) and field_sources not in ("", MISSING):
            try:
                parsed = json.loads(field_sources)
            except json.JSONDecodeError:
                parsed = {}
            if parsed:
                grouped_sources: dict[str, list[str]] = {}
                for field, source in parsed.items():
                    grouped_sources.setdefault(str(source), []).append(str(field))
                source_rows = [
                    {"fields": ", ".join(fields), "source text": source}
                    for source, fields in grouped_sources.items()
                ]
                st.dataframe(pd.DataFrame(source_rows), hide_index=True, use_container_width=True)


def unique_source_review_rows(df: pd.DataFrame, *, limit: int = 12) -> pd.DataFrame:
    if df.empty:
        return df

    keyed = df.copy()
    keyed["_source_review_key"] = keyed.apply(
        lambda row: "|".join(
            [
                str(row.get("source_passage_id", "")),
                str(row.get("source_file", "")),
                str(row.get("source_page", "")),
                str(row.get("source_anchor", ""))[:300],
            ]
        ),
        axis=1,
    )
    return keyed.drop_duplicates("_source_review_key").head(limit).drop(columns=["_source_review_key"])


if not DATA_PATH.exists():
    st.title("Sepsis Atlas")
    st.error("No evidence table found.")
    st.code(
        "python build_atlas.py --articles ./articles --backend auto "
        "--query \"What predictors are reported for 28-day mortality in sepsis?\""
    )
    st.stop()


df = load_evidence_table(str(DATA_PATH))

st.title("Sepsis Atlas")
query = st.text_input("Clinical question", value=DEFAULT_QUERY)

ranked = score_rows(df, query)
ranked = apply_sidebar_filters(ranked)

top_metric_cols = st.columns(4)
top_metric_cols[0].metric("Returned rows", len(ranked))
top_metric_cols[1].metric("Studies", ranked["study_name"].nunique())
top_metric_cols[2].metric("Not reported cells", int((ranked[VISIBLE_COLUMNS] == MISSING).sum().sum()))
mean_relevance = ranked["_relevance"].mean()
if pd.isna(mean_relevance):
    mean_relevance = 0
top_metric_cols[3].metric("Mean relevance", round(float(mean_relevance), 2))

table = ranked[VISIBLE_COLUMNS + ["source_anchor", "_relevance"]].copy()
table.rename(columns={"_relevance": "query_relevance"}, inplace=True)

st.dataframe(
    table[VISIBLE_COLUMNS + ["query_relevance"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "study_name": "Study",
        "population": "Population",
        "sample_size": "Sample size",
        "predictor": "Predictor",
        "outcome": "Outcome",
        "timing": "Timing",
        "method": "Method",
        "effect_size": "Effect size",
        "performance": "Performance",
        "source_link": st.column_config.LinkColumn("Open PDF", display_text="Open"),
        "source_file": "Source file",
        "source_page": "Page",
        "support_status": "Support",
        "query_relevance": st.column_config.NumberColumn("Relevance", format="%.2f"),
    },
)

csv = ranked.drop(columns=["_relevance"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "Download current table",
    data=csv,
    file_name="sepsis_atlas_query_results.csv",
    mime="text/csv",
)

st.subheader("Source Review")
source_review_rows = unique_source_review_rows(ranked, limit=12)
st.caption(f"Showing {len(source_review_rows)} unique source anchors from {len(ranked)} evidence row(s).")
for _, row in source_review_rows.iterrows():
    render_source_review(row)
