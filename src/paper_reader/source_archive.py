from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

MANIFEST_FILE_NAME = "manifest.json"
RUN_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(slots=True)
class SourcePaperRecord:
    paper_id: str
    title: str
    url: str
    upvotes: int
    published_at: str | None
    authors: list[str]
    summary: str
    comment_count: int
    pdf_url: str | None
    pdf_rel_path: str | None
    pdf_file_name: str | None
    pdf_exists: bool


@dataclass(slots=True)
class SourceDayRecord:
    run_date: str
    year: str
    month: str
    day: str
    manifest_path: Path
    day_dir: Path
    source: str
    source_url: str
    snapshot_date: str | None
    saved_at_beijing: str | None
    paper_count: int
    filter_operator: str
    filter_value: int
    papers: list[SourcePaperRecord]


class SourceArchiveError(RuntimeError):
    """Raised when source archive data is invalid."""



def load_source_days(root: Path) -> list[SourceDayRecord]:
    root = root.resolve()
    records: list[SourceDayRecord] = []
    if not root.exists():
        return records

    for manifest_path in sorted(root.glob("*/*/*/manifest.json"), reverse=True):
        try:
            records.append(load_source_day_from_manifest(manifest_path))
        except SourceArchiveError:
            continue
    return sorted(records, key=lambda item: item.run_date, reverse=True)



def load_source_day(root: Path, run_date: str) -> SourceDayRecord | None:
    if not RUN_DATE_RE.match(run_date):
        return None
    year, month, day = run_date.split("-")
    manifest_path = root.resolve() / year / month / day / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return None
    try:
        return load_source_day_from_manifest(manifest_path)
    except SourceArchiveError:
        return None



def load_source_day_from_manifest(manifest_path: Path) -> SourceDayRecord:
    manifest_path = manifest_path.resolve()
    day_dir = manifest_path.parent
    payload = _load_json(manifest_path)
    run_date = str(payload.get("run_date_beijing") or "")
    if not RUN_DATE_RE.match(run_date):
        raise SourceArchiveError(f"Invalid run date in {manifest_path}")

    papers: list[SourcePaperRecord] = []
    for item in payload.get("papers", []):
        if not isinstance(item, dict):
            continue
        pdf_rel_path = item.get("pdf_rel_path")
        pdf_path = day_dir / pdf_rel_path if isinstance(pdf_rel_path, str) and pdf_rel_path else None
        papers.append(
            SourcePaperRecord(
                paper_id=str(item.get("paper_id") or ""),
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                upvotes=int(item.get("upvotes") or 0),
                published_at=item.get("published_at"),
                authors=[str(author) for author in item.get("authors", []) if author],
                summary=str(item.get("summary") or ""),
                comment_count=int(item.get("comment_count") or 0),
                pdf_url=str(item.get("pdf_url") or "") or None,
                pdf_rel_path=pdf_rel_path if isinstance(pdf_rel_path, str) else None,
                pdf_file_name=str(item.get("pdf_file_name") or "") or None,
                pdf_exists=bool(pdf_path and pdf_path.exists() and pdf_path.is_file()),
            )
        )

    year, month, day = run_date.split("-")
    filter_payload = payload.get("filter") if isinstance(payload.get("filter"), dict) else {}
    return SourceDayRecord(
        run_date=run_date,
        year=year,
        month=month,
        day=day,
        manifest_path=manifest_path,
        day_dir=day_dir,
        source=str(payload.get("source") or "huggingface_daily_papers"),
        source_url=str(payload.get("source_url") or ""),
        snapshot_date=(str(payload.get("snapshot_date")) if payload.get("snapshot_date") else None),
        saved_at_beijing=(str(payload.get("saved_at_beijing")) if payload.get("saved_at_beijing") else None),
        paper_count=int(payload.get("paper_count") or len(papers)),
        filter_operator=str(filter_payload.get("operator") or ">="),
        filter_value=int(filter_payload.get("value") or 0),
        papers=papers,
    )



def day_paper_map(day_record: SourceDayRecord) -> dict[str, SourcePaperRecord]:
    return {paper.paper_id: paper for paper in day_record.papers if paper.paper_id}



def local_pdf_path_for(day_record: SourceDayRecord, paper: SourcePaperRecord) -> Path | None:
    if not paper.pdf_rel_path:
        return None
    path = (day_record.day_dir / paper.pdf_rel_path).resolve()
    if day_record.day_dir not in path.parents and path != day_record.day_dir:
        return None
    return path



def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceArchiveError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceArchiveError(f"Manifest payload must be an object: {path}")
    return payload
