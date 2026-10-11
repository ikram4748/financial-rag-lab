"""Download FinanceBench (open-source sample): 150 questions and the 84 filings they are about.

    python scripts/setup_data.py

FinanceBench (Islam et al., 2023, Patronus AI) is licensed CC BY-NC 4.0. It is downloaded here, not
redistributed in this repository.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
QUESTIONS_URL = "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl"
PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{}.pdf"


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    for i in range(tries):
        try:
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if dest.suffix == ".pdf" and not tmp.read_bytes()[:5] == b"%PDF-":
                raise ValueError("not a PDF")
            tmp.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  retry {dest.name}: {e}")
            time.sleep(3)


def main() -> None:
    (RAW / "pdfs").mkdir(parents=True, exist_ok=True)
    qpath = RAW / "financebench.jsonl"
    if not qpath.exists():
        fetch(QUESTIONS_URL, qpath)
    docs = sorted({json.loads(line)["doc_name"] for line in qpath.read_text().splitlines() if line.strip()})
    for i, d in enumerate(docs, 1):
        dest = RAW / "pdfs" / f"{d}.pdf"
        if not dest.exists():
            fetch(PDF_URL.format(d), dest)
        print(f"\r{i}/{len(docs)} {d:40s}", end="", flush=True)
    print(f"\n{len(docs)} filings in {RAW / 'pdfs'}")


if __name__ == "__main__":
    sys.exit(main())
