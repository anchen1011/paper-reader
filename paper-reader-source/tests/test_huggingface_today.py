from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reader_source.huggingface import fetch_daily_snapshot, filter_papers_by_upvotes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Hugging Face Daily Papers and print papers whose upvotes are above the threshold."
    )
    parser.add_argument(
        "--date",
        help="Optional Daily Papers date in YYYY-MM-DD format. Defaults to the date exposed by https://huggingface.co/papers.",
    )
    parser.add_argument(
        "--min-upvotes",
        type=int,
        default=5,
        help="Only keep papers with upvotes strictly greater than this value. Default: 5.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the filtered result as JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshot = fetch_daily_snapshot(date_string=args.date)
    filtered = filter_papers_by_upvotes(snapshot, min_upvotes=args.min_upvotes)

    if args.json:
        payload = {
            "source_url": snapshot.source_url,
            "date_string": snapshot.date_string,
            "threshold": args.min_upvotes,
            "paper_count": len(filtered),
            "papers": [paper.to_dict() for paper in filtered],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Resolved Hugging Face Daily Papers date: {snapshot.date_string}")
    print(f"Source URL: {snapshot.source_url}")
    print(f"Papers with upvotes > {args.min_upvotes}: {len(filtered)}")

    if not filtered:
        print("No papers matched the current filter.")
        return 0

    for index, paper in enumerate(filtered, start=1):
        authors = ", ".join(paper.authors[:5])
        if len(paper.authors) > 5:
            authors = f"{authors}, ..."
        print()
        print(f"{index}. {paper.title}")
        print(f"   upvotes={paper.upvotes} comments={paper.comment_count} published_at={paper.published_at}")
        print(f"   authors={authors or 'N/A'}")
        print(f"   url={paper.url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
