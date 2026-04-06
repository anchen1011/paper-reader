#!/usr/bin/env python3
"""Download a paper if needed, then ask Codex to explain the local document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

from src.paper_reader.ai_summary import DEFAULT_MODEL, explain_document

DEFAULT_URL = "https://arxiv.org/pdf/2501.12948"
DEFAULT_DOCUMENT_PATH = Path("docs/papers/2501.12948.pdf")
DEFAULT_OUTPUT_PATH = Path("docs/papers/2501.12948.explained.zh.md")


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    with urlopen(url) as response:
        destination.write_bytes(response.read())
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a paper if needed and ask Codex to explain the local PDF / DOC / DOCX document."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Remote file URL to download before summarizing")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DOCUMENT_PATH,
        help=f"Local document path (default: {DEFAULT_DOCUMENT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output markdown path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and summarize the local document directly",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the file even if it already exists locally",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document_path = args.input

    if not args.skip_download:
        print("Downloading file...", file=sys.stderr)
        document_path = download_file(args.url, document_path, force=args.force_download)

    print(f"Summarizing {document_path}...", file=sys.stderr)
    summary = explain_document(document_path, model=args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary + "\n", encoding="utf-8")

    print(f"Document saved to: {document_path}")
    print(f"Explanation saved to: {args.output}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
