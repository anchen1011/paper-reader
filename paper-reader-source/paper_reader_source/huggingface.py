from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HUGGING_FACE_PAPERS_URL = "https://huggingface.co/papers"
DAILY_PAPERS_PROPS_RE = re.compile(r'data-target="DailyPapers"\s+data-props="([^"]+)"')
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = "paper-reader-source/0.1 (+https://huggingface.co/papers)"


@dataclass(slots=True)
class PaperRecord:
    paper_id: str
    title: str
    url: str
    upvotes: int
    published_at: str | None
    authors: list[str]
    summary: str
    comment_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DailyPapersSnapshot:
    source_url: str
    date_string: str
    papers: list[PaperRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "date_string": self.date_string,
            "papers": [paper.to_dict() for paper in self.papers],
        }


class HuggingFaceParseError(RuntimeError):
    """Raised when the Daily Papers page cannot be parsed."""


def fetch_daily_snapshot(date_string: str | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> DailyPapersSnapshot:
    source_url = HUGGING_FACE_PAPERS_URL if date_string is None else f"{HUGGING_FACE_PAPERS_URL}/date/{date_string}"
    html_text = _fetch_html(source_url, timeout=timeout)
    payload = _extract_daily_papers_payload(html_text)

    papers = [_normalize_paper(item) for item in payload.get("dailyPapers", [])]
    papers.sort(key=lambda paper: (-paper.upvotes, paper.paper_id))

    return DailyPapersSnapshot(
        source_url=source_url,
        date_string=payload["dateString"],
        papers=papers,
    )


def filter_papers_by_upvotes(snapshot: DailyPapersSnapshot, min_upvotes: int, *, inclusive: bool = False) -> list[PaperRecord]:
    if inclusive:
        return [paper for paper in snapshot.papers if paper.upvotes >= min_upvotes]
    return [paper for paper in snapshot.papers if paper.upvotes > min_upvotes]


def _fetch_html(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset)
    except HTTPError as exc:
        raise RuntimeError(f"Hugging Face returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach Hugging Face for {url}: {exc.reason}") from exc


def _extract_daily_papers_payload(html_text: str) -> dict[str, Any]:
    match = DAILY_PAPERS_PROPS_RE.search(html_text)
    if match is None:
        raise HuggingFaceParseError("Could not find the DailyPapers payload in the page HTML.")

    raw_payload = html.unescape(match.group(1))
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HuggingFaceParseError("Failed to decode the DailyPapers payload JSON.") from exc

    if "dateString" not in payload or "dailyPapers" not in payload:
        raise HuggingFaceParseError("DailyPapers payload is missing required fields.")

    return payload


def _normalize_paper(item: dict[str, Any]) -> PaperRecord:
    paper = item.get("paper", {})
    paper_id = str(paper.get("id") or "")
    title = str(item.get("title") or paper.get("title") or "")
    authors = [author.get("name", "") for author in paper.get("authors", []) if author.get("name")]

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        url=f"{HUGGING_FACE_PAPERS_URL}/{paper_id}",
        upvotes=int(paper.get("upvotes") or 0),
        published_at=paper.get("publishedAt") or item.get("publishedAt"),
        authors=authors,
        summary=str(item.get("summary") or paper.get("summary") or ""),
        comment_count=int(item.get("numComments") or 0),
    )
