"""Utilities for collecting paper metadata from external sources."""

from .huggingface import DailyPapersSnapshot, PaperRecord, fetch_daily_snapshot, filter_papers_by_upvotes

__all__ = [
    "DailyPapersSnapshot",
    "PaperRecord",
    "fetch_daily_snapshot",
    "filter_papers_by_upvotes",
]
