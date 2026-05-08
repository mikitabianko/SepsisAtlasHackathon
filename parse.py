from __future__ import annotations

import base64
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageDocument:
    """Small page object used by the atlas pipeline."""

    text: str
    metadata: dict[str, Any]


def _clean_page_text(raw_text: str) -> str:
    """Normalize PyMuPDF output while keeping paragraph boundaries useful."""

    raw_text = raw_text.replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraph = " ".join(buffer)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                paragraphs.append(paragraph)
            buffer.clear()

    for line in lines:
        if not line:
            flush()
            continue

        if buffer and buffer[-1].endswith("-") and line:
            buffer[-1] = buffer[-1][:-1] + line
        else:
            buffer.append(line)

    flush()
    return "\n\n".join(paragraphs)


def iter_pdf_paths(data_dir: str | Path = "./articles") -> list[Path]:
    """Return PDFs in a stable order."""

    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.pdf"), key=lambda path: path.name.lower())


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

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


def _openrouter_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Sepsis Atlas Hackathon"),
    }


def _should_use_vision(
    *,
    vision_mode: str,
    extracted_text: str,
    embedded_image_count: int,
    min_text_chars: int,
) -> bool:
    if vision_mode == "off":
        return False
    if vision_mode == "all":
        return True
    if vision_mode != "auto":
        raise ValueError("vision_mode must be one of: off, auto, all")
    return len(extracted_text.strip()) < min_text_chars or embedded_image_count > 0


def _render_page_data_url(page: Any, *, dpi: int) -> str:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_visual_page_text(
    *,
    page_image_url: str,
    file_name: str,
    page_number: int,
    model: str,
) -> str:
    _load_local_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not _usable_secret(api_key):
        raise RuntimeError("OPENROUTER_API_KEY is required for PDF vision extraction.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The `openai` package is required for OpenRouter vision extraction.") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers=_openrouter_headers(),
    )
    prompt = f"""
You are extracting visual evidence from a clinical sepsis paper PDF page.

Source: {file_name}, page {page_number}

Read the page image and transcribe analysis-relevant visual content that may not appear in selectable PDF text.
Focus on:
- tables, figure panels, captions, flow diagrams, legends, footnotes
- sepsis mortality predictors, biomarkers, severity scores, outcomes, sample sizes, timing
- effect sizes such as OR, HR, RR, beta coefficients, cutoffs
- performance metrics such as AUC/AUROC, sensitivity, specificity, CI, p-values

Rules:
- Preserve exact numbers, units, confidence intervals, p-values, and labels.
- Mark the output as visual/OCR-derived.
- Do not infer values that are not visible.
- If there is no readable or relevant visual content, return: Not reported.
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": page_image_url}},
                ],
            }
        ],
    )
    return _clean_page_text(response.choices[0].message.content or "")


def _is_insufficient_credits_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "402" in message and "insufficient credits" in message


def load_medical_pdfs(
    data_dir: str | Path = "./articles",
    *,
    max_pages_per_pdf: int | None = None,
    workers: int | None = None,
    vision_mode: str = "auto",
    vision_model: str | None = None,
    vision_dpi: int = 144,
    min_text_chars_for_vision: int = 500,
    vision_error_policy: str = "warn",
) -> list[PageDocument]:
    """Parse PDFs into page-level documents with file and page metadata.

    The hackathon scoring cares about traceability, so the parser keeps every
    page separate. Downstream extraction can then cite a file, page, and exact
    snippet for each table row.
    """

    pdf_paths = iter_pdf_paths(data_dir)
    if not pdf_paths:
        print(f"No PDF files found in {Path(data_dir).resolve()}.")
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF parsing. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    vision_model = vision_model or os.environ.get("OPENROUTER_VISION_MODEL") or os.environ.get(
        "OPENROUTER_MODEL", "openai/gpt-4o-mini"
    )
    if vision_error_policy not in {"warn", "error"}:
        raise ValueError("vision_error_policy must be one of: warn, error")

    vision_lock = threading.Lock()
    vision_disabled_reason: str | None = None

    def load_one_pdf(pdf_path: Path) -> list[PageDocument]:
        nonlocal vision_disabled_reason
        pdf_documents: list[PageDocument] = []
        with fitz.open(pdf_path) as pdf:
            pdf_meta = pdf.metadata or {}
            total_pages = len(pdf)
            page_count = min(total_pages, max_pages_per_pdf or total_pages)

            for page_index in range(page_count):
                page = pdf.load_page(page_index)
                text = _clean_page_text(page.get_text("text"))
                embedded_image_count = len(page.get_images(full=True))
                source_types = ["pdf_text"] if text else []
                vision_text = ""

                if _should_use_vision(
                    vision_mode=vision_mode,
                    extracted_text=text,
                    embedded_image_count=embedded_image_count,
                    min_text_chars=min_text_chars_for_vision,
                ):
                    page_number = page_index + 1
                    with vision_lock:
                        disabled_reason = vision_disabled_reason

                    if disabled_reason:
                        source_types.append("vision_skipped")
                    else:
                        try:
                            print(f"Vision extraction: {pdf_path.name} page {page_number}")
                            page_image_url = _render_page_data_url(page, dpi=vision_dpi)
                            vision_text = _extract_visual_page_text(
                                page_image_url=page_image_url,
                                file_name=pdf_path.name,
                                page_number=page_number,
                                model=vision_model,
                            )
                            if vision_text and vision_text != "Not reported":
                                source_types.append("openrouter_vision")
                        except Exception as exc:
                            if vision_error_policy == "error":
                                raise RuntimeError(
                                    f"Vision extraction failed for {pdf_path.name} page {page_number}: {exc}"
                                ) from exc

                            if _is_insufficient_credits_error(exc):
                                with vision_lock:
                                    vision_disabled_reason = "OpenRouter returned 402 insufficient credits."
                                print(
                                    "Vision extraction disabled for the rest of this run: "
                                    "OpenRouter returned 402 insufficient credits. "
                                    "Continuing with selectable PDF text only."
                                )
                            else:
                                print(
                                    f"Vision extraction skipped for {pdf_path.name} page {page_number}: {exc}. "
                                    "Continuing with selectable PDF text only."
                                )
                            source_types.append("vision_error")

                if vision_text and vision_text != "Not reported":
                    text = (
                        f"{text}\n\n[OpenRouter vision/OCR extraction from page image]\n{vision_text}"
                        if text
                        else f"[OpenRouter vision/OCR extraction from page image]\n{vision_text}"
                    )

                if not text:
                    continue

                pdf_documents.append(
                    PageDocument(
                        text=text,
                        metadata={
                            "file_name": pdf_path.name,
                            "file_path": str(pdf_path.resolve()),
                            "page_number": page_index + 1,
                            "page_label": str(page_index + 1),
                            "total_pages": total_pages,
                            "title": pdf_meta.get("title") or pdf_path.stem,
                            "embedded_image_count": embedded_image_count,
                            "source_types": ", ".join(source_types) if source_types else "unknown",
                            "vision_mode": vision_mode,
                            "vision_model": vision_model if "openrouter_vision" in source_types else "",
                        },
                    )
                )
        return pdf_documents

    documents: list[PageDocument] = []
    worker_count = workers or min(4, max(1, os.cpu_count() or 1), len(pdf_paths))
    if worker_count <= 1 or len(pdf_paths) <= 1:
        for pdf_path in pdf_paths:
            documents.extend(load_one_pdf(pdf_path))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(load_one_pdf, pdf_path): pdf_path for pdf_path in pdf_paths}
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    documents.extend(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse {pdf_path.name}: {exc}") from exc

    documents.sort(key=lambda doc: (str(doc.metadata.get("file_name", "")), int(doc.metadata.get("page_number", 0))))
    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDFs.")
    return documents


if __name__ == "__main__":
    docs = load_medical_pdfs("./articles")
    if docs:
        first = docs[0]
        print(first.metadata)
        print(first.text[:800])
