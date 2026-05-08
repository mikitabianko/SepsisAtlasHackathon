from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from extract import (
    CANONICAL_COLUMNS,
    Passage,
    build_source_link,
    extract_rows_for_study,
    retrieve_passages,
    split_pages_into_passages,
)
from parse import load_medical_pdfs


DEFAULT_QUERY = (
    "What predictors, biomarkers, severity scores, statistical methods, effect sizes, "
    "and model performance metrics are reported for mortality estimation in sepsis?"
)
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def _group_passages_by_file(passages: list[Passage]) -> dict[str, list[Passage]]:
    grouped: dict[str, list[Passage]] = defaultdict(list)
    for passage in passages:
        grouped[passage.source_file].append(passage)
    return dict(grouped)


def _write_passages_jsonl(passages_by_file: dict[str, list[Passage]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for source_file, passages in passages_by_file.items():
            for passage in passages:
                handle.write(
                    json.dumps(
                        {
                            "source_file": source_file,
                            "source_page": passage.page_number,
                            "source_passage_id": passage.passage_id,
                            "source_link": build_source_link(
                                passage.source_path,
                                passage.source_file,
                                passage.page_number,
                            ),
                            "score": passage.score,
                            "text": passage.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _extract_study(
    source_file: str,
    retrieved: list[Passage],
    query: str,
    backend: str,
    model: str,
) -> list[dict[str, Any]]:
    return extract_rows_for_study(
        Path(source_file).stem,
        retrieved,
        query,
        backend=backend,
        model=model,
    )


def build_sepsis_atlas(
    *,
    articles_dir: str | Path = "./articles",
    output_csv: str | Path = "sepsis_atlas_results.csv",
    evidence_jsonl: str | Path = "evidence_passages.jsonl",
    query: str = DEFAULT_QUERY,
    backend: str = "llm",
    model: str = DEFAULT_MODEL,
    top_k_per_study: int = 8,
    min_score: float = 1.0,
    max_pages_per_pdf: int | None = None,
    workers: int = 4,
    parse_workers: int | None = None,
    vision_mode: str = "auto",
    vision_model: str | None = None,
    vision_dpi: int = 144,
    min_text_chars_for_vision: int = 500,
    vision_error_policy: str = "warn",
) -> pd.DataFrame:
    import pandas as pd

    print("Loading PDFs...")
    pages = load_medical_pdfs(
        articles_dir,
        max_pages_per_pdf=max_pages_per_pdf,
        workers=parse_workers,
        vision_mode=vision_mode,
        vision_model=vision_model,
        vision_dpi=vision_dpi,
        min_text_chars_for_vision=min_text_chars_for_vision,
        vision_error_policy=vision_error_policy,
    )
    if not pages:
        raise FileNotFoundError(
            f"No PDFs were found in {Path(articles_dir).resolve()}. "
            "Create an `articles/` folder and place the hackathon PDFs there."
        )

    print("Segmenting pages into traceable passages...")
    passages = split_pages_into_passages(pages)
    passages_by_file = _group_passages_by_file(passages)
    print(f"Prepared {len(passages)} passages from {len(passages_by_file)} studies.")

    all_rows: list[dict[str, Any]] = []
    selected_passages: dict[str, list[Passage]] = {}

    for source_file, study_passages in passages_by_file.items():
        retrieved = retrieve_passages(
            study_passages,
            query,
            top_k=top_k_per_study,
            per_file_limit=top_k_per_study,
        )
        if not retrieved:
            continue
        if max(passage.score for passage in retrieved) < min_score:
            continue

        selected_passages[source_file] = retrieved

    extraction_jobs = sorted(selected_passages.items(), key=lambda item: item[0].lower())
    worker_count = max(1, min(workers, len(extraction_jobs) or 1))
    if extraction_jobs:
        print(f"Extracting {len(extraction_jobs)} studies with {worker_count} worker(s)...")

    if worker_count == 1:
        for source_file, retrieved in extraction_jobs:
            print(f"Extracting {source_file} from {len(retrieved)} retrieved passages...")
            all_rows.extend(_extract_study(source_file, retrieved, query, backend, model))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_extract_study, source_file, retrieved, query, backend, model): (
                    source_file,
                    retrieved,
                )
                for source_file, retrieved in extraction_jobs
            }
            completed = 0
            for future in as_completed(futures):
                source_file, retrieved = futures[future]
                completed += 1
                try:
                    rows = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Extraction failed for {source_file}: {exc}") from exc
                all_rows.extend(rows)
                print(
                    f"Extracted {source_file} ({completed}/{len(extraction_jobs)}; "
                    f"{len(retrieved)} passages, {len(rows)} row(s))."
                )

    if not all_rows:
        raise RuntimeError(
            "No evidence rows were extracted. Try a broader query, lower --min-score, "
            "or inspect the parsed PDF text."
        )

    df = pd.DataFrame(all_rows)
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = "Not reported"
    df = df[CANONICAL_COLUMNS]
    df.sort_values(["support_status", "study_name", "predictor"], inplace=True)

    output_csv = Path(output_csv)
    df.to_csv(output_csv, index=False)
    print(f"Saved structured evidence table: {output_csv.resolve()}")

    evidence_jsonl = Path(evidence_jsonl)
    _write_passages_jsonl(selected_passages, evidence_jsonl)
    print(f"Saved retrieved source passages: {evidence_jsonl.resolve()}")

    print("\nExtraction summary")
    print(
        df[
            ["study_name", "predictor", "outcome", "effect_size", "performance", "support_status"]
        ].to_string(index=False)
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a source-grounded Sepsis Atlas evidence table.")
    parser.add_argument("--articles", default="./articles", help="Folder containing PDF articles.")
    parser.add_argument("--output", default="sepsis_atlas_results.csv", help="Output CSV path.")
    parser.add_argument("--evidence-jsonl", default="evidence_passages.jsonl", help="Retrieved passage audit file.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Clinical question to retrieve/extract against.")
    parser.add_argument(
        "--backend",
        choices=["auto", "llm", "regex"],
        default="llm",
        help="llm uses OpenRouter by default; auto falls back to regex only if no LLM call succeeds.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name for --backend llm/auto.")
    parser.add_argument("--top-k-per-study", type=int, default=8, help="Retrieved passages per PDF.")
    parser.add_argument("--min-score", type=float, default=1.0, help="Minimum retrieval score for a study.")
    parser.add_argument("--max-pages-per-pdf", type=int, default=None, help="Optional cap for fast demos.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ATLAS_WORKERS", "4")),
        help="Concurrent study extraction workers. Lower this if OpenRouter rate limits requests.",
    )
    parser.add_argument(
        "--parse-workers",
        type=int,
        default=None,
        help="Concurrent PDF parsing workers. Defaults to min(4, CPU count, number of PDFs).",
    )
    parser.add_argument(
        "--vision-mode",
        choices=["off", "auto", "all"],
        default=os.environ.get("ATLAS_VISION_MODE", "auto"),
        help="Use OpenRouter vision on page images: off, auto for scanned/image-heavy pages, or all pages.",
    )
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("OPENROUTER_VISION_MODEL"),
        help="OpenRouter vision model. Defaults to OPENROUTER_VISION_MODEL or OPENROUTER_MODEL.",
    )
    parser.add_argument("--vision-dpi", type=int, default=144, help="DPI used when rendering PDF pages for vision.")
    parser.add_argument(
        "--min-text-chars-for-vision",
        type=int,
        default=500,
        help="In auto mode, run vision on pages with less extracted text than this threshold.",
    )
    parser.add_argument(
        "--vision-error-policy",
        choices=["warn", "error"],
        default=os.environ.get("ATLAS_VISION_ERROR_POLICY", "warn"),
        help="warn continues without vision if OpenRouter vision fails; error aborts the build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_sepsis_atlas(
        articles_dir=args.articles,
        output_csv=args.output,
        evidence_jsonl=args.evidence_jsonl,
        query=args.query,
        backend=args.backend,
        model=args.model,
        top_k_per_study=args.top_k_per_study,
        min_score=args.min_score,
        max_pages_per_pdf=args.max_pages_per_pdf,
        workers=args.workers,
        parse_workers=args.parse_workers,
        vision_mode=args.vision_mode,
        vision_model=args.vision_model,
        vision_dpi=args.vision_dpi,
        min_text_chars_for_vision=args.min_text_chars_for_vision,
        vision_error_policy=args.vision_error_policy,
    )


if __name__ == "__main__":
    main()
