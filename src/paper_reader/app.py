from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from flask import Flask, abort, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for
from markupsafe import Markup
from werkzeug.utils import secure_filename

from . import bib_import as bib_import_utils
from .ai_summary import DEFAULT_MODEL, DEFAULT_USER_PROMPT, run_prompt_on_document
from .document_utils import ALLOWED_EXTENSIONS, extract_document_metadata
from .markdown_render import render_markdown
from .offline_package import build_offline_manifest, manifest_json as build_manifest_json, offline_prompt_arcname, offline_source_arcname
from .prompt_manager import DEFAULT_PROMPT_SLUG, PromptDefinition, PromptStore, parse_checkbox
from .settings import SettingsStore
from .task_queue import PaperJobQueue

CACHE_FILE_NAME = ".paper_reader_index.json"
DONE_INDEX_FILE_NAME = ".paper_reader_done_index.json"
SUMMARY_DIR_NAME = ".paper-reader-ai"
DONE_DIR_NAME = "DONE"
BIB_IMPORT_DIR_NAME = ".paper-reader-bib-imports"
DEFAULT_BATCH_PANEL_PAGE_SIZE = 50
DEFAULT_LOGIN_USERNAME = "admin"
DEFAULT_LOGIN_PASSWORD = "paperpaperreaderreader12678"
MAX_LOGIN_FAILURES = 3
LOGIN_LOCK_SECONDS = 5 * 60
ARXIV_NEW_STYLE_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
ARXIV_LEGACY_STYLE_RE = re.compile(r"^[A-Za-z0-9.\-]+/[0-9]{7}(?:v\d+)?$", re.IGNORECASE)
ARXIV_INPUT_SPLIT_RE = re.compile(r"[\r\n,，;；]+")


@dataclass
class PaperRecord:
    rel_path: str
    file_name: str
    folder: str
    extension: str
    title: str
    display_title: str
    preview_text: str
    extracted_date: str | None
    date_precision: str | None
    date_source: str | None
    sort_date: str | None
    file_size: int
    modified_at: str
    preview_kind: str
    prompt_result_count: int
    prompt_result_slugs: list[str]
    is_done: bool


@dataclass
class ScanResult:
    papers: list[PaperRecord]
    folders: list[str]


@dataclass
class LoginAttemptState:
    failed_count: int = 0
    locked_until: float = 0.0


@dataclass
class BibImportJobRecord:
    id: str
    source_name: str
    target_folder: str
    status: str
    progress: int
    message: str
    error: str | None
    total_entries: int
    processed_entries: int
    imported_count: int
    duplicate_count: int
    unmatched_count: int
    current_label: str
    source_bib_rel_path: str | None
    unmatched_bib_rel_path: str | None
    unmatched_list_rel_path: str | None
    imported_rel_paths: list[str] = field(default_factory=list)
    duplicate_rel_paths: list[str] = field(default_factory=list)
    auto_prompt_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None


class LoginGuard:
    def __init__(self) -> None:
        self._attempts: dict[str, LoginAttemptState] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def key_for_request(self, req: Any) -> str:
        forwarded = (req.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return forwarded or req.remote_addr or "local"

    def status_for(self, key: str) -> dict[str, Any]:
        with self._lock:
            state = self._attempts.get(key)
            if state is None:
                return {"failed_count": 0, "locked": False, "remaining_seconds": 0}

            now = self._now()
            if state.locked_until and state.locked_until > now:
                remaining = int(state.locked_until - now)
                return {"failed_count": state.failed_count, "locked": True, "remaining_seconds": max(1, remaining)}

            if state.locked_until and state.locked_until <= now:
                state.locked_until = 0.0
                state.failed_count = 0
                self._attempts.pop(key, None)
            return {"failed_count": 0, "locked": False, "remaining_seconds": 0}

    def register_failure(self, key: str) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            state = self._attempts.get(key)
            if state is None:
                state = LoginAttemptState()
                self._attempts[key] = state

            if state.locked_until and state.locked_until > now:
                remaining = int(state.locked_until - now)
                return {"locked": True, "remaining_seconds": max(1, remaining), "failed_count": state.failed_count}

            if state.locked_until and state.locked_until <= now:
                state.failed_count = 0
                state.locked_until = 0.0

            state.failed_count += 1
            if state.failed_count >= MAX_LOGIN_FAILURES:
                state.locked_until = now + LOGIN_LOCK_SECONDS
                return {"locked": True, "remaining_seconds": LOGIN_LOCK_SECONDS, "failed_count": state.failed_count}

            return {"locked": False, "remaining_seconds": 0, "failed_count": state.failed_count}

    def register_success(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


class PaperLibrary:
    def __init__(self, root: Path, prompt_store: PromptStore):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.root / CACHE_FILE_NAME
        self.done_index_path = self.root / DONE_INDEX_FILE_NAME
        self.summary_root = self.root / SUMMARY_DIR_NAME
        self.summary_root.mkdir(parents=True, exist_ok=True)
        self.prompt_store = prompt_store
        self._hash_cache: dict[str, tuple[float, int, str]] = {}
        self._scan_lock = threading.RLock()
        self._scan_cache: dict[tuple[bool, bool], ScanResult] = {}

    def invalidate_scan_cache(self) -> None:
        with self._scan_lock:
            self._scan_cache.clear()

    def scan(self, *, force: bool = False, lightweight: bool = False, include_done: bool = False) -> ScanResult:
        with self._scan_lock:
            cache_key = (lightweight, include_done)
            if cache_key in self._scan_cache and not force:
                return self._scan_cache[cache_key]

            active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
            if force or not self.cache_path.exists():
                papers = self.rebuild_active_index(lightweight=lightweight)
            else:
                papers = self.load_active_index(active_prompt_slugs)

            folders = {""}
            for paper in papers:
                folders.add(paper.folder)

            if include_done:
                papers.extend(self.load_done_index(active_prompt_slugs))
                for paper in papers:
                    folders.add(paper.folder)

            result = ScanResult(papers=papers, folders=sorted(folders))
            self._scan_cache[cache_key] = result
            return result

    def iter_documents(self, *, include_done: bool = False) -> list[Path]:
        documents: list[Path] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name == CACHE_FILE_NAME:
                continue
            if path.name == self.prompt_store.store_path.name:
                continue
            if SUMMARY_DIR_NAME in path.parts:
                continue
            if not include_done and DONE_DIR_NAME in path.parts:
                continue
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            documents.append(path)
        return documents

    def iter_done_documents(self) -> list[Path]:
        done_root = self.root / DONE_DIR_NAME
        if not done_root.exists():
            return []

        documents: list[Path] = []
        for path in sorted(done_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            documents.append(path)
        return documents

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _record_from_index_item(self, item: dict[str, Any], active_prompt_slugs: list[str]) -> PaperRecord | None:
        try:
            record = PaperRecord(**item)
        except TypeError:
            return None

        visible_slugs = set(active_prompt_slugs)
        stored_slugs = [slug for slug in record.prompt_result_slugs if isinstance(slug, str)]
        active_result_slugs = [slug for slug in stored_slugs if slug in visible_slugs]
        return PaperRecord(
            rel_path=record.rel_path,
            file_name=record.file_name,
            folder=record.folder,
            extension=record.extension,
            title=record.title,
            display_title=record.display_title,
            preview_text=record.preview_text,
            extracted_date=record.extracted_date,
            date_precision=record.date_precision,
            date_source=record.date_source,
            sort_date=record.sort_date,
            file_size=record.file_size,
            modified_at=record.modified_at,
            preview_kind=record.preview_kind,
            prompt_result_count=len(active_result_slugs),
            prompt_result_slugs=stored_slugs,
            is_done=record.is_done,
        )

    def _load_active_index_payload(self) -> dict[str, Any]:
        payload = self._load_cache()
        if "records" in payload and isinstance(payload.get("records"), list):
            return payload

        # Backward compatibility for the old rel_path -> {signature, record} cache format.
        records: list[dict[str, Any]] = []
        for item in payload.values():
            if isinstance(item, dict) and isinstance(item.get("record"), dict):
                records.append(item["record"])
        if not records:
            return {}
        return {"records": records}

    def _write_active_index_payload(self, payload: dict[str, Any]) -> None:
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            errors="backslashreplace",
        )

    def load_active_index(self, active_prompt_slugs: list[str]) -> list[PaperRecord]:
        payload = self._load_active_index_payload()
        records: list[PaperRecord] = []
        for item in payload.get("records", []):
            if not isinstance(item, dict):
                continue
            record = self._record_from_index_item(item, active_prompt_slugs)
            if record is None or record.is_done:
                continue
            records.append(record)
        return records

    def rebuild_active_index(self, *, lightweight: bool = False) -> list[PaperRecord]:
        active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
        records = [
            self._build_record(path, active_prompt_slugs, lightweight=lightweight)
            for path in self.iter_documents(include_done=False)
        ]
        self._write_active_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(records),
                "records": [asdict(record) for record in records],
            }
        )
        self.invalidate_scan_cache()
        return records

    def _update_active_index_entry(self, rel_path: str, *, lightweight: bool = False) -> None:
        payload = self._load_active_index_payload()
        if not payload.get("records") and not self.cache_path.exists():
            active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
            payload = {
                "records": [
                    asdict(self._build_record(path, active_prompt_slugs, lightweight=lightweight))
                    for path in self.iter_documents(include_done=False)
                ]
            }
        items = [item for item in payload.get("records", []) if isinstance(item, dict) and item.get("rel_path") != rel_path]
        if rel_path and not self.is_done_rel_path(rel_path):
            try:
                absolute = self.resolve_relative_path(rel_path)
            except ValueError:
                absolute = None
            if absolute is not None and absolute.exists() and absolute.is_file():
                active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
                items.append(asdict(self._build_record(absolute, active_prompt_slugs, lightweight=lightweight)))
        self._write_active_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(items),
                "records": items,
            }
        )
        self.invalidate_scan_cache()

    def _load_done_index_payload(self) -> dict[str, Any]:
        if not self.done_index_path.exists():
            return {}
        try:
            payload = json.loads(self.done_index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_done_index_payload(self, payload: dict[str, Any]) -> None:
        self.done_index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            errors="backslashreplace",
        )

    def load_done_index(self, active_prompt_slugs: list[str]) -> list[PaperRecord]:
        payload = self._load_done_index_payload()
        records: list[PaperRecord] = []
        for item in payload.get("records", []):
            if not isinstance(item, dict):
                continue
            record = self._record_from_index_item(item, active_prompt_slugs)
            if record is None:
                continue
            if not self.is_done_rel_path(record.rel_path):
                continue
            records.append(record)
        return records

    def rebuild_done_index(self, *, lightweight: bool = True) -> list[PaperRecord]:
        active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
        records = [
            self._build_record(path, active_prompt_slugs, lightweight=lightweight)
            for path in self.iter_done_documents()
        ]
        self._write_done_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(records),
                "records": [asdict(record) for record in records],
            }
        )
        self.invalidate_scan_cache()
        return records

    def _update_done_index_entry(self, rel_path: str, *, lightweight: bool = False) -> None:
        payload = self._load_done_index_payload()
        if not payload.get("records") and not self.done_index_path.exists():
            active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
            payload = {
                "records": [
                    asdict(self._build_record(path, active_prompt_slugs, lightweight=lightweight))
                    for path in self.iter_done_documents()
                ]
            }
        items = [item for item in payload.get("records", []) if isinstance(item, dict) and item.get("rel_path") != rel_path]
        if self.is_done_rel_path(rel_path):
            try:
                absolute = self.resolve_relative_path(rel_path)
            except ValueError:
                absolute = None
            if absolute is not None and absolute.exists() and absolute.is_file():
                active_prompt_slugs = [prompt.slug for prompt in self.prompt_store.active_prompts()]
                items.append(asdict(self._build_record(absolute, active_prompt_slugs, lightweight=lightweight)))
        self._write_done_index_payload(
            {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "record_count": len(items),
                "records": items,
            }
        )
        self.invalidate_scan_cache()

    def _build_record(self, path: Path, active_prompt_slugs: list[str], *, lightweight: bool = False) -> PaperRecord:
        rel_path = path.relative_to(self.root).as_posix()
        folder = path.relative_to(self.root).parent.as_posix()
        if folder == ".":
            folder = ""

        if lightweight:
            meta = {"title": path.stem, "preview_text": "", "full_text": ""}
        else:
            try:
                meta = extract_document_metadata(path)
            except Exception:
                meta = {"title": path.stem, "preview_text": "", "full_text": ""}

        title = self._safe_text(meta.get("title") or path.stem)
        preview_text = self._safe_text(meta.get("preview_text") or "")
        date_info = self._extract_date(title, preview_text, path.name)
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        all_prompt_result_slugs = self.list_existing_prompt_slugs(rel_path)
        visible_prompt_result_slugs = [slug for slug in all_prompt_result_slugs if slug in set(active_prompt_slugs)]

        return PaperRecord(
            rel_path=self._safe_text(rel_path),
            file_name=self._safe_text(path.name),
            folder=self._safe_text(folder),
            extension=path.suffix.lower(),
            title=title,
            display_title=title if title else path.stem,
            preview_text=preview_text,
            extracted_date=date_info["display_date"],
            date_precision=date_info["precision"],
            date_source=date_info["source"],
            sort_date=date_info["sort_date"],
            file_size=path.stat().st_size,
            modified_at=modified.isoformat(timespec="seconds"),
            preview_kind=self.preview_kind(path),
            prompt_result_count=len(visible_prompt_result_slugs),
            prompt_result_slugs=all_prompt_result_slugs,
            is_done=self.is_done_rel_path(rel_path),
        )

    def _safe_text(self, value: str) -> str:
        if not value:
            return ""
        return value.encode("utf-8", "backslashreplace").decode("utf-8")

    def build_record_for_rel_path(self, rel_path: str, active_prompt_slugs: list[str]) -> PaperRecord:
        return self._build_record(self.resolve_relative_path(rel_path), active_prompt_slugs)

    def _prompt_state(self, rel_path: str, active_prompt_slugs: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        result_dir = self.prompt_result_dir_for(rel_path)
        if result_dir.exists():
            for path in sorted(result_dir.glob("*.md")):
                if path.stem not in active_prompt_slugs:
                    continue
                stat = path.stat()
                entries.append({"slug": path.stem, "mtime": stat.st_mtime, "size": stat.st_size})

        legacy_path = self.legacy_summary_path_for(rel_path)
        if DEFAULT_PROMPT_SLUG in active_prompt_slugs and legacy_path.exists():
            stat = legacy_path.stat()
            entries.append({"slug": DEFAULT_PROMPT_SLUG, "mtime": stat.st_mtime, "size": stat.st_size, "legacy": True})
        return entries

    def preview_kind(self, path: Path) -> str:
        return {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "doc",
        }.get(path.suffix.lower(), "download")

    def _extract_date(self, title: str, preview_text: str, file_name: str) -> dict[str, str | None]:
        candidate_text = "\n".join([title, preview_text[:4000], file_name])

        patterns = [
            (r"Submitted on\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %b %Y", "submitted_on"),
            (r"Submitted on\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %B %Y", "submitted_on"),
            (r"\[v\d+\]\s+\w{3},\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %b %Y", "arxiv_version"),
            (r"\[v\d+\]\s+\w{3},\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", "%d %B %Y", "arxiv_version"),
            (r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", "%B %d, %Y", "text_date"),
        ]
        for pattern, fmt, source in patterns:
            match = re.search(pattern, candidate_text, re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1)
            try:
                parsed = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            return {
                "display_date": parsed.strftime("%Y-%m-%d"),
                "precision": "day",
                "source": source,
                "sort_date": parsed.date().isoformat(),
            }

        modern_id = re.search(r"(?<!\d)(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?(?!\d)", candidate_text)
        if modern_id:
            year = int(modern_id.group(1))
            year += 2000 if year < 90 else 1900
            month = int(modern_id.group(2))
            try:
                parsed = datetime(year, month, 1)
            except ValueError:
                return {"display_date": None, "precision": None, "source": None, "sort_date": None}
            return {
                "display_date": parsed.strftime("%Y-%m"),
                "precision": "month",
                "source": "arxiv_id",
                "sort_date": parsed.date().isoformat(),
            }

        legacy_id = re.search(r"[a-z\-]+\/(\d{2})(\d{2})\d{3,4}", candidate_text, re.IGNORECASE)
        if legacy_id:
            year = int(legacy_id.group(1))
            year += 2000 if year < 90 else 1900
            month = int(legacy_id.group(2))
            try:
                parsed = datetime(year, month, 1)
            except ValueError:
                return {"display_date": None, "precision": None, "source": None, "sort_date": None}
            return {
                "display_date": parsed.strftime("%Y-%m"),
                "precision": "month",
                "source": "legacy_arxiv_id",
                "sort_date": parsed.date().isoformat(),
            }

        return {"display_date": None, "precision": None, "source": None, "sort_date": None}

    def resolve_relative_path(self, rel_path: str) -> Path:
        rel_path = unquote(rel_path).split("#", 1)[0].strip("/")
        candidate = (self.root / rel_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError("Path escapes library root")
        return candidate

    def make_unique_destination(self, folder: str, original_name: str) -> Path:
        folder_path = self.resolve_relative_path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = secure_filename(original_name)
        if not filename:
            raise ValueError("Invalid file name")
        candidate = folder_path / filename
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = folder_path / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def is_done_rel_path(self, rel_path: str) -> bool:
        parts = Path(rel_path).parts
        return bool(parts) and parts[0] == DONE_DIR_NAME

    def done_destination_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        base_folder = self.root / DONE_DIR_NAME / rel.parent
        base_folder.mkdir(parents=True, exist_ok=True)
        candidate = base_folder / rel.name
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = base_folder / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def restore_destination_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        if not self.is_done_rel_path(rel_path):
            raise ValueError("Paper is not in DONE folder")
        original_rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
        folder_path = self.root / original_rel.parent
        folder_path.mkdir(parents=True, exist_ok=True)
        candidate = folder_path / original_rel.name
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = folder_path / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def find_duplicate_by_hash(self, file_size: int, file_hash: str) -> str | None:
        for path in self.iter_documents():
            try:
                if path.stat().st_size != file_size:
                    continue
            except FileNotFoundError:
                continue
            if self.hash_for_path(path) == file_hash:
                return path.relative_to(self.root).as_posix()
        return None

    def hash_for_path(self, path: Path) -> str:
        rel_path = path.relative_to(self.root).as_posix()
        stat = path.stat()
        cached = self._hash_cache.get(rel_path)
        signature = (stat.st_mtime, stat.st_size)
        if cached and cached[:2] == signature:
            return cached[2]
        digest = sha256_for_path(path)
        self._hash_cache[rel_path] = (stat.st_mtime, stat.st_size, digest)
        return digest

    def _move_prompt_results(self, old_rel_path: str, new_rel_path: str) -> None:
        old_result_dir = self.prompt_result_dir_for(old_rel_path)
        new_result_dir = self.prompt_result_dir_for(new_rel_path)
        if old_result_dir.exists():
            new_result_dir.parent.mkdir(parents=True, exist_ok=True)
            if new_result_dir.exists():
                shutil.rmtree(new_result_dir)
            shutil.move(str(old_result_dir), str(new_result_dir))

        old_legacy = self.legacy_summary_path_for(old_rel_path)
        new_legacy = self.legacy_summary_path_for(new_rel_path)
        if old_legacy.exists():
            new_legacy.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_legacy), str(new_legacy))

    def create_folder(self, rel_folder: str) -> Path:
        rel_folder = rel_folder.strip().strip("/")
        if not rel_folder:
            raise ValueError("Folder name is required")
        destination = self.resolve_relative_path(rel_folder)
        destination.mkdir(parents=True, exist_ok=True)
        self.invalidate_scan_cache()
        return destination

    def rename_file(self, rel_path: str, new_name: str) -> str:
        source = self.resolve_relative_path(rel_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(rel_path)
        filename = secure_filename(new_name)
        if not filename:
            raise ValueError("Invalid target file name")
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Only PDF / DOC / DOCX files are allowed")
        destination = source.with_name(filename)
        if destination.exists() and destination != source:
            raise ValueError("Target file already exists")

        old_rel_path = source.relative_to(self.root).as_posix()
        source.rename(destination)
        new_rel_path = destination.relative_to(self.root).as_posix()
        self._move_prompt_results(old_rel_path, new_rel_path)
        self._hash_cache.pop(old_rel_path, None)
        if self.is_done_rel_path(old_rel_path) or self.is_done_rel_path(new_rel_path):
            self._update_done_index_entry(old_rel_path)
            self._update_done_index_entry(new_rel_path)
        else:
            self._update_active_index_entry(old_rel_path)
            self._update_active_index_entry(new_rel_path)

        return new_rel_path

    def delete_file(self, rel_path: str) -> None:
        target = self.resolve_relative_path(rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(rel_path)
        target.unlink()
        self._hash_cache.pop(rel_path, None)

        result_dir = self.prompt_result_dir_for(rel_path)
        if result_dir.exists():
            shutil.rmtree(result_dir)

        legacy_path = self.legacy_summary_path_for(rel_path)
        if legacy_path.exists():
            legacy_path.unlink()
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)

    def toggle_done(self, rel_path: str) -> str:
        source = self.resolve_relative_path(rel_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(rel_path)

        if self.is_done_rel_path(rel_path):
            destination = self.restore_destination_for(rel_path)
        else:
            destination = self.done_destination_for(rel_path)

        old_rel_path = source.relative_to(self.root).as_posix()
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        new_rel_path = destination.relative_to(self.root).as_posix()
        self._move_prompt_results(old_rel_path, new_rel_path)
        cached = self._hash_cache.pop(old_rel_path, None)
        if cached:
            self._hash_cache[new_rel_path] = cached
        self._update_active_index_entry(old_rel_path)
        self._update_active_index_entry(new_rel_path)
        self._update_done_index_entry(old_rel_path)
        self._update_done_index_entry(new_rel_path)
        return new_rel_path

    def legacy_summary_path_for(self, rel_path: str) -> Path:
        rel = Path(rel_path)
        return self.summary_root / rel.parent / f"{rel.stem}.explained.zh.md"

    def prompt_result_dir_for(self, rel_path: str) -> Path:
        return self.summary_root / Path(rel_path)

    def prompt_result_path_for(self, rel_path: str, prompt_slug: str) -> Path:
        return self.prompt_result_dir_for(rel_path) / f"{prompt_slug}.md"

    def list_existing_prompt_slugs(self, rel_path: str, visible_slugs: set[str] | None = None) -> list[str]:
        found: set[str] = set()
        result_dir = self.prompt_result_dir_for(rel_path)
        if result_dir.exists():
            for path in result_dir.glob("*.md"):
                if visible_slugs is None or path.stem in visible_slugs:
                    found.add(path.stem)

        legacy_path = self.legacy_summary_path_for(rel_path)
        if legacy_path.exists() and (visible_slugs is None or DEFAULT_PROMPT_SLUG in visible_slugs):
            found.add(DEFAULT_PROMPT_SLUG)
        return sorted(found)

    def _existing_prompt_result_path(self, rel_path: str, prompt_slug: str) -> Path | None:
        prompt_path = self.prompt_result_path_for(rel_path, prompt_slug)
        if prompt_path.exists():
            return prompt_path
        legacy_path = self.legacy_summary_path_for(rel_path)
        if prompt_slug == DEFAULT_PROMPT_SLUG and legacy_path.exists():
            return legacy_path
        return None

    def prompt_result_info(self, rel_path: str, prompt_slug: str) -> dict[str, Any]:
        path = self._existing_prompt_result_path(rel_path, prompt_slug)
        if path is None or not path.exists():
            return {"exists": False, "updated_at": None, "result_rel_path": None}
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        return {
            "exists": True,
            "updated_at": updated_at,
            "result_rel_path": path.relative_to(self.root).as_posix(),
        }

    def read_prompt_result(self, rel_path: str, prompt_slug: str) -> str | None:
        path = self._existing_prompt_result_path(rel_path, prompt_slug)
        if path is None or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def existing_prompt_result_path(self, rel_path: str, prompt_slug: str) -> Path | None:
        path = self._existing_prompt_result_path(rel_path, prompt_slug)
        if path is None or not path.exists():
            return None
        return path

    def _format_prompt_result(self, rel_path: str, prompt: PromptDefinition, content: str) -> str:
        generated_at = datetime.now().isoformat(timespec="seconds")
        return (
            f"# {prompt.name}\n\n"
            f"- Source file: `{rel_path}`\n"
            f"- Prompt slug: `{prompt.slug}`\n"
            f"- Model: `{prompt.model or DEFAULT_MODEL}`\n"
            f"- Generated at: `{generated_at}`\n\n"
            "---\n\n"
            f"{content.rstrip()}\n"
        )

    def write_prompt_result(self, rel_path: str, prompt: PromptDefinition, content: str) -> Path:
        result_path = self.prompt_result_path_for(rel_path, prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(self._format_prompt_result(rel_path, prompt, content), encoding="utf-8")

        legacy_path = self.legacy_summary_path_for(rel_path)
        if legacy_path.exists() and legacy_path != result_path:
            legacy_path.unlink()
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)
        return result_path

    def generate_prompt_result(
        self,
        rel_path: str,
        prompt: PromptDefinition,
        *,
        force: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        process_callback: Callable[[Any], None] | None = None,
    ) -> tuple[Path, bool]:
        document_path = self.resolve_relative_path(rel_path)
        if not document_path.exists() or not document_path.is_file():
            raise FileNotFoundError(rel_path)

        existing = self._existing_prompt_result_path(rel_path, prompt.slug)
        if existing is not None and existing.exists() and not force:
            return existing, False

        content = run_prompt_on_document(
            document_path,
            user_prompt=prompt.user_prompt,
            model=prompt.model or DEFAULT_MODEL,
            progress_callback=progress_callback,
            should_abort=should_abort,
            process_callback=process_callback,
        )
        return self.write_prompt_result(rel_path, prompt, content), True

    def run_prompt_batch(
        self,
        rel_paths: list[str],
        prompt_slugs: list[str],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        prompt_map = {prompt.slug: prompt for prompt in self.prompt_store.list_prompts()}
        prompts = [prompt_map[slug] for slug in prompt_slugs if slug in prompt_map]
        unique_rel_paths = list(dict.fromkeys(path for path in rel_paths if path))
        result = {"generated": 0, "skipped": 0, "failed": 0, "errors": []}

        for rel_path in unique_rel_paths:
            for prompt in prompts:
                try:
                    _, generated = self.generate_prompt_result(rel_path, prompt, force=force)
                    if generated:
                        result["generated"] += 1
                    else:
                        result["skipped"] += 1
                except Exception as exc:
                    result["failed"] += 1
                    result["errors"].append(f"{Path(rel_path).name} / {prompt.name}: {exc}")
        return result


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_for_filestorage(file_storage: Any) -> tuple[int, str]:
    stream = file_storage.stream
    stream.seek(0)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
    stream.seek(0)
    return total, digest.hexdigest()


def normalize_arxiv_id(value: str) -> str:
    raw = unquote(str(value or "").strip())
    if not raw:
        raise ValueError("请输入 arXiv ID。")

    if raw.lower().startswith("arxiv:"):
        raw = raw.split(":", 1)[1].strip()

    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        if not parsed.netloc.lower().endswith("arxiv.org"):
            raise ValueError("请输入 arXiv ID 或 arxiv.org 链接。")
        path = parsed.path.strip("/")
        if path.startswith("abs/"):
            raw = path[4:]
        elif path.startswith("pdf/"):
            raw = path[4:]
        else:
            raise ValueError("请输入 arXiv ID 或 arxiv.org 链接。")

    raw = raw.strip().strip("/")
    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]

    if ARXIV_NEW_STYLE_RE.fullmatch(raw) or ARXIV_LEGACY_STYLE_RE.fullmatch(raw):
        return raw

    raise ValueError("arXiv ID 格式不正确。示例：2501.12948、2501.12948v1 或 cs/0112017")


def parse_arxiv_input(value: str) -> list[str]:
    tokens = [token.strip() for token in ARXIV_INPUT_SPLIT_RE.split(str(value or ""))]
    return [token for token in tokens if token]


def arxiv_pdf_filename(arxiv_id: str) -> str:
    safe_name = secure_filename(arxiv_id.replace("/", "--")) or "arxiv-paper"
    return safe_name if safe_name.lower().endswith(".pdf") else f"{safe_name}.pdf"


def download_arxiv_pdf(arxiv_id: str, *, timeout: int = 45) -> bytes:
    request = Request(
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        headers={"User-Agent": "paper-reader/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"找不到 arXiv 论文：{arxiv_id}") from exc
        raise RuntimeError(f"arXiv 下载失败（HTTP {exc.code}）。") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 arXiv：{exc.reason}") from exc

    if not payload:
        raise RuntimeError("arXiv 返回了空文件。")
    if not payload[:1024].lstrip().startswith(b"%PDF-"):
        raise RuntimeError("arXiv 返回的内容不是 PDF 文件。")
    return payload



def build_groups(papers: list[PaperRecord]) -> list[dict[str, Any]]:
    groups: dict[str, list[PaperRecord]] = {}
    for paper in papers:
        label = paper.extracted_date[:7] if paper.extracted_date else "未提取日期"
        groups.setdefault(label, []).append(paper)
    ordered: list[dict[str, Any]] = []
    for label in sorted((item for item in groups if item != "未提取日期"), reverse=True):
        ordered.append({"label": label, "papers": groups[label], "count": len(groups[label])})
    if "未提取日期" in groups:
        ordered.append({"label": "未提取日期", "papers": groups["未提取日期"], "count": len(groups["未提取日期"])})
    return ordered


def build_sidebar_groups(papers: list[PaperRecord], selected_rel_path: str | None = None) -> list[dict[str, Any]]:
    tree: dict[str, dict[str, Any]] = {}
    for paper in papers:
        year_key = paper.sort_date[:4] if paper.sort_date else "unknown"
        month_key = paper.sort_date[:7] if paper.sort_date else "unknown"
        year_label = year_key if year_key != "unknown" else "未提取日期"
        month_label = month_key if month_key != "unknown" else "未分类"

        year_group = tree.setdefault(
            year_key,
            {"key": year_key, "label": year_label, "count": 0, "months": {}},
        )
        month_group = year_group["months"].setdefault(
            month_key,
            {"key": month_key, "label": month_label, "count": 0, "papers": [], "is_open": False},
        )
        year_group["count"] += 1
        month_group["count"] += 1
        month_group["papers"].append(paper)
        if selected_rel_path and paper.rel_path == selected_rel_path:
            month_group["is_open"] = True

    groups: list[dict[str, Any]] = []
    for year_key in sorted((key for key in tree if key != "unknown"), reverse=True):
        year_group = tree[year_key]
        months = [
            year_group["months"][month_key]
            for month_key in sorted((key for key in year_group["months"] if key != "unknown"), reverse=True)
        ]
        if "unknown" in year_group["months"]:
            months.append(year_group["months"]["unknown"])
        groups.append(
            {
                "key": year_group["key"],
                "label": year_group["label"],
                "count": year_group["count"],
                "months": months,
                "is_open": any(month["is_open"] for month in months) or not groups,
            }
        )

    if "unknown" in tree:
        year_group = tree["unknown"]
        months = list(year_group["months"].values())
        groups.append(
            {
                "key": year_group["key"],
                "label": year_group["label"],
                "count": year_group["count"],
                "months": months,
                "is_open": any(month["is_open"] for month in months),
            }
        )
    return groups



def filter_and_sort_papers(
    papers: list[PaperRecord],
    folder: str,
    query: str,
    sort_by: str,
    *,
    show_done: bool = False,
) -> list[PaperRecord]:
    query_text = query.strip().lower()
    folder = folder.strip().strip("/")
    filtered: list[PaperRecord] = []
    for paper in papers:
        if paper.is_done and not show_done:
            continue
        if folder and not (paper.folder == folder or paper.folder.startswith(folder + "/")):
            continue
        haystack = f"{paper.file_name} {paper.display_title}".lower()
        if query_text and query_text not in haystack:
            continue
        filtered.append(paper)

    if sort_by == "title":
        filtered.sort(key=lambda paper: (paper.display_title.lower(), paper.file_name.lower()))
    elif sort_by == "date_asc":
        filtered.sort(key=lambda paper: (paper.sort_date or "9999-99-99", paper.display_title.lower()))
    else:
        filtered.sort(key=lambda paper: (paper.sort_date or "0000-00-00", paper.display_title.lower()), reverse=True)
    return filtered


def parse_page(value: str | None, default: int = 1) -> int:
    try:
        page = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(1, page)


def paginate_items(items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    total = len(items)
    if total == 0:
        return {
            "items": [],
            "page": 1,
            "page_size": page_size,
            "total": 0,
            "total_pages": 1,
            "has_prev": False,
            "has_next": False,
            "prev_page": 1,
            "next_page": 1,
            "start_index": 0,
            "end_index": 0,
        }

    total_pages = max(1, (total + page_size - 1) // page_size)
    safe_page = min(max(1, page), total_pages)
    start = (safe_page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": safe_page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_prev": safe_page > 1,
        "has_next": safe_page < total_pages,
        "prev_page": safe_page - 1 if safe_page > 1 else 1,
        "next_page": safe_page + 1 if safe_page < total_pages else total_pages,
        "start_index": start + 1,
        "end_index": min(end, total),
    }


def load_env_file_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def resolve_login_credentials(base_dir: Path) -> tuple[str, str]:
    env_file_path = Path(os.environ.get("PAPER_READER_ENV_FILE", str(base_dir / ".env")))
    env_values = load_env_file_values(env_file_path)
    username = (
        env_values.get("PAPER_READER_LOGIN_USERNAME")
        or os.environ.get("PAPER_READER_LOGIN_USERNAME")
        or DEFAULT_LOGIN_USERNAME
    )
    password = (
        env_values.get("PAPER_READER_LOGIN_PASSWORD")
        or os.environ.get("PAPER_READER_LOGIN_PASSWORD")
        or DEFAULT_LOGIN_PASSWORD
    )
    return username, password



def redirect_to_index(
    current_folder: str,
    query: str,
    sort_by: str,
    selected_paper: str | None = None,
    tab: str | None = None,
    *,
    show_done: bool = False,
    batch_show_done: bool = False,
) -> Any:
    params = {"folder": current_folder, "q": query, "sort": sort_by}
    if selected_paper:
        params["paper"] = selected_paper
    if tab:
        params["tab"] = tab
    if show_done:
        params["show_done"] = "1"
    if batch_show_done:
        params["batch_show_done"] = "1"
    return redirect(url_for("index", **params))



def create_app(library_root: Path | None = None) -> Flask:
    base_dir = Path(__file__).resolve().parents[2]
    root = library_root or Path(base_dir / "docs" / "papers")
    login_username, login_password = resolve_login_credentials(base_dir)
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.config["SECRET_KEY"] = "paper-reader-dev-secret"
    app.config["LIBRARY_ROOT"] = root.resolve()
    app.config["LOGIN_USERNAME"] = login_username
    app.config["LOGIN_PASSWORD"] = login_password
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.settings_store = SettingsStore(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]
    app.prompt_store = PromptStore(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]
    app.library = PaperLibrary(app.config["LIBRARY_ROOT"], app.prompt_store)  # type: ignore[attr-defined]
    app.login_guard = LoginGuard()  # type: ignore[attr-defined]
    app.job_queue = PaperJobQueue(  # type: ignore[attr-defined]
        app.library,
        app.prompt_store,
        max_concurrency=app.settings_store.max_concurrency(),  # type: ignore[attr-defined]
    )

    @app.context_processor
    def inject_helpers() -> dict[str, Any]:
        return {
            "format_bytes": format_bytes,
            "allowed_extensions": ", ".join(sorted(ALLOWED_EXTENSIONS)),
        }

    @app.before_request
    def require_login() -> Any:
        endpoint = request.endpoint or ""
        allowed = {"login", "health"}
        if endpoint in allowed or endpoint.startswith("static"):
            return None
        if session.get("authenticated"):
            return None
        next_url = request.full_path if request.query_string else request.path
        return redirect(url_for("login", next=next_url.rstrip("?")))

    @app.get("/health")
    def health() -> Any:
        return {"ok": True}

    @app.route("/login", methods=["GET", "POST"])
    def login() -> Any:
        next_url = request.values.get("next", "") or url_for("index")
        client_key = app.login_guard.key_for_request(request)  # type: ignore[attr-defined]
        status = app.login_guard.status_for(client_key)  # type: ignore[attr-defined]

        if request.method == "POST":
            if status["locked"]:
                flash(f"密码连续输错过多，请等待 {status['remaining_seconds']} 秒后再试。", "error")
            else:
                username = request.form.get("username", "").strip()
                password = request.form.get("password", "")
                if username == app.config["LOGIN_USERNAME"] and password == app.config["LOGIN_PASSWORD"]:
                    app.login_guard.register_success(client_key)  # type: ignore[attr-defined]
                    session["authenticated"] = True
                    session["username"] = app.config["LOGIN_USERNAME"]
                    return redirect(next_url or url_for("index"))

                failure = app.login_guard.register_failure(client_key)  # type: ignore[attr-defined]
                if failure["locked"]:
                    flash(f"密码连续输错 3 次，已锁定 5 分钟。请等待 {failure['remaining_seconds']} 秒后再试。", "error")
                else:
                    remaining_attempts = MAX_LOGIN_FAILURES - failure["failed_count"]
                    flash(f"用户名或密码错误。还可再试 {remaining_attempts} 次。", "error")
                status = app.login_guard.status_for(client_key)  # type: ignore[attr-defined]

        return render_template(
            "login.html",
            next_url=next_url,
            locked=status["locked"],
            remaining_seconds=status["remaining_seconds"],
        )

    def serialize_paper_for_view(
        paper: PaperRecord,
        *,
        current_folder: str,
        query: str,
        sort_by: str,
        active_prompt_count: int,
        show_done: bool,
    ) -> dict[str, Any]:
        year_key = paper.sort_date[:4] if paper.sort_date else "unknown"
        month_key = paper.sort_date[:7] if paper.sort_date else "unknown"
        paper_url_params: dict[str, Any] = {
            "folder": current_folder,
            "q": query,
            "sort": sort_by,
            "paper": paper.rel_path,
            "tab": "source",
        }
        if show_done:
            paper_url_params["show_done"] = "1"
        return {
            "rel_path": paper.rel_path,
            "file_name": paper.file_name,
            "folder": paper.folder,
            "extension": paper.extension,
            "display_title": paper.display_title,
            "extracted_date": paper.extracted_date,
            "sort_date": paper.sort_date,
            "prompt_result_count": paper.prompt_result_count,
            "active_prompt_count": active_prompt_count,
            "is_done": paper.is_done,
            "year_key": year_key,
            "month_key": month_key,
            "year_label": year_key if year_key != "unknown" else "未提取日期",
            "month_label": month_key if month_key != "unknown" else "未分类",
            "paper_url": url_for(
                "index",
                **paper_url_params,
            ),
        }

    def build_saved_file_result(
        rel_path: str,
        *,
        current_folder: str,
        query: str,
        sort_by: str,
        submit_auto_prompts: bool,
        show_done: bool,
        success_label: str,
        submission_source: str,
        include_paper_view: bool = True,
    ) -> dict[str, Any]:
        if app.library.is_done_rel_path(rel_path):  # type: ignore[attr-defined]
            app.library._update_done_index_entry(rel_path)  # type: ignore[attr-defined]
        else:
            app.library._update_active_index_entry(rel_path)  # type: ignore[attr-defined]
        active_prompts = app.prompt_store.active_prompts()  # type: ignore[attr-defined]
        paper = app.library.build_record_for_rel_path(rel_path, [prompt.slug for prompt in active_prompts])  # type: ignore[attr-defined]
        visible_in_current_view = None
        paper_payload = None
        if include_paper_view:
            visible_in_current_view = bool(
                filter_and_sort_papers([paper], folder=current_folder, query=query, sort_by=sort_by, show_done=show_done)
            )
            paper_payload = serialize_paper_for_view(
                paper,
                current_folder=current_folder,
                query=query,
                sort_by=sort_by,
                active_prompt_count=len(active_prompts),
                show_done=show_done,
            )

        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        if submit_auto_prompts:
            auto_prompts = [prompt for prompt in active_prompts if prompt.auto_run]
            if auto_prompts:
                submission = app.job_queue.submit(  # type: ignore[attr-defined]
                    [rel_path],
                    [prompt.slug for prompt in auto_prompts],
                    force=False,
                    source=submission_source,
                )

        message = f"{success_label}：{Path(rel_path).name}"
        if submission["queued"]:
            message += f"；已提交 {submission['queued']} 个后台 Prompt 任务"
        elif submission["existing"]:
            message += "；相关 Prompt 任务已在队列中"
        elif submission["skipped"]:
            message += "；Prompt 结果已存在，未重复提交"

        return {
            "status": "saved",
            "message": message,
            "saved_rel_path": rel_path,
            "paper": paper_payload,
            "visible_in_current_view": visible_in_current_view,
            "submission": submission,
        }

    def process_uploaded_file(
        file: Any,
        *,
        target_folder: str,
        current_folder: str,
        query: str,
        sort_by: str,
        submit_auto_prompts: bool,
        show_done: bool,
    ) -> dict[str, Any]:
        filename = getattr(file, "filename", "") or ""
        if not filename:
            return {"status": "error", "message": "没有选择文件。", "saved_rel_path": None}

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return {"status": "error", "message": f"不支持的文件类型：{filename}", "saved_rel_path": None}

        file_size, file_hash = sha256_for_filestorage(file)
        duplicate_rel_path = app.library.find_duplicate_by_hash(file_size, file_hash)  # type: ignore[attr-defined]
        if duplicate_rel_path:
            return {
                "status": "duplicate",
                "message": f"检测到重复文件，已跳过：{Path(duplicate_rel_path).name}",
                "saved_rel_path": None,
                "duplicate_rel_path": duplicate_rel_path,
            }

        destination = app.library.make_unique_destination(target_folder, filename)  # type: ignore[attr-defined]
        try:
            file.save(destination)
        except Exception:
            if destination.exists():
                destination.unlink()
            raise

        rel_path = destination.relative_to(app.config["LIBRARY_ROOT"]).as_posix()
        return build_saved_file_result(
            rel_path,
            current_folder=current_folder,
            query=query,
            sort_by=sort_by,
            submit_auto_prompts=submit_auto_prompts,
            show_done=show_done,
            success_label="上传成功",
            submission_source="upload",
        )

    def process_arxiv_download(
        arxiv_id: str,
        *,
        target_folder: str,
        current_folder: str,
        query: str,
        sort_by: str,
        submit_auto_prompts: bool,
        show_done: bool,
        include_paper_view: bool = True,
    ) -> dict[str, Any]:
        normalized_id = normalize_arxiv_id(arxiv_id)
        payload = download_arxiv_pdf(normalized_id)
        filename = arxiv_pdf_filename(normalized_id)
        file_size = len(payload)
        file_hash = hashlib.sha256(payload).hexdigest()
        duplicate_rel_path = app.library.find_duplicate_by_hash(file_size, file_hash)  # type: ignore[attr-defined]
        if duplicate_rel_path:
            return {
                "status": "duplicate",
                "message": f"检测到重复文件，已跳过：{Path(duplicate_rel_path).name}",
                "saved_rel_path": None,
                "duplicate_rel_path": duplicate_rel_path,
                "normalized_arxiv_id": normalized_id,
            }

        destination = app.library.make_unique_destination(target_folder, filename)  # type: ignore[attr-defined]
        try:
            destination.write_bytes(payload)
        except Exception:
            if destination.exists():
                destination.unlink()
            raise

        rel_path = destination.relative_to(app.config["LIBRARY_ROOT"]).as_posix()
        result = build_saved_file_result(
            rel_path,
            current_folder=current_folder,
            query=query,
            sort_by=sort_by,
            submit_auto_prompts=submit_auto_prompts,
            show_done=show_done,
            success_label="下载成功",
            submission_source="arxiv",
            include_paper_view=include_paper_view,
        )
        result["normalized_arxiv_id"] = normalized_id
        return result

    class BibImportManager:
        ACTIVE_STATUSES = {"queued", "running"}
        TERMINAL_STATUSES = {"completed", "failed"}

        def __init__(self, library_root: Path):
            self.library_root = library_root.resolve()
            self.job_root = self.library_root / BIB_IMPORT_DIR_NAME
            self.job_root.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self._jobs: dict[str, BibImportJobRecord] = {}

        def _timestamp(self) -> str:
            return datetime.utcnow().isoformat(timespec="seconds")

        def _job_dir(self, job_id: str) -> Path:
            return self.job_root / job_id

        def _status_path(self, job_id: str) -> Path:
            return self._job_dir(job_id) / "status.json"

        def _persist_locked(self, job: BibImportJobRecord) -> None:
            payload = asdict(job)
            path = self._status_path(job.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(path)

        def _coerce_job(self, payload: dict[str, Any]) -> BibImportJobRecord | None:
            try:
                imported_rel_paths = payload.get("imported_rel_paths") or []
                duplicate_rel_paths = payload.get("duplicate_rel_paths") or []
                auto_prompt_summary = payload.get("auto_prompt_summary") or {}
                return BibImportJobRecord(
                    id=str(payload.get("id") or ""),
                    source_name=str(payload.get("source_name") or ""),
                    target_folder=str(payload.get("target_folder") or ""),
                    status=str(payload.get("status") or "failed"),
                    progress=int(payload.get("progress") or 0),
                    message=str(payload.get("message") or ""),
                    error=(str(payload.get("error")) if payload.get("error") is not None else None),
                    total_entries=int(payload.get("total_entries") or 0),
                    processed_entries=int(payload.get("processed_entries") or 0),
                    imported_count=int(payload.get("imported_count") or 0),
                    duplicate_count=int(payload.get("duplicate_count") or 0),
                    unmatched_count=int(payload.get("unmatched_count") or 0),
                    current_label=str(payload.get("current_label") or ""),
                    source_bib_rel_path=(str(payload.get("source_bib_rel_path")) if payload.get("source_bib_rel_path") else None),
                    unmatched_bib_rel_path=(str(payload.get("unmatched_bib_rel_path")) if payload.get("unmatched_bib_rel_path") else None),
                    unmatched_list_rel_path=(str(payload.get("unmatched_list_rel_path")) if payload.get("unmatched_list_rel_path") else None),
                    imported_rel_paths=[str(item) for item in imported_rel_paths if item],
                    duplicate_rel_paths=[str(item) for item in duplicate_rel_paths if item],
                    auto_prompt_summary=auto_prompt_summary if isinstance(auto_prompt_summary, dict) else {},
                    created_at=str(payload.get("created_at") or ""),
                    updated_at=str(payload.get("updated_at") or payload.get("created_at") or ""),
                    started_at=(str(payload.get("started_at")) if payload.get("started_at") else None),
                    finished_at=(str(payload.get("finished_at")) if payload.get("finished_at") else None),
                )
            except Exception:
                return None

        def _load_job_from_disk(self, job_id: str) -> BibImportJobRecord | None:
            path = self._status_path(job_id)
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            job = self._coerce_job(payload)
            if job is None or not job.id:
                return None
            self._jobs[job.id] = job
            return job

        def _update_job(self, job_id: str, **changes: Any) -> BibImportJobRecord | None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return None
                for key, value in changes.items():
                    setattr(job, key, value)
                job.updated_at = self._timestamp()
                self._persist_locked(job)
                return job

        def start(self, file_storage: Any, *, target_folder: str) -> BibImportJobRecord:
            filename = secure_filename(getattr(file_storage, "filename", "") or "") or "library.bib"
            if Path(filename).suffix.lower() != ".bib":
                raise ValueError("只支持导入 .bib 文件。")

            payload = file_storage.read()
            try:
                file_storage.stream.seek(0)
            except Exception:
                pass
            if not payload:
                raise ValueError("上传的 .bib 文件为空。")

            with self._lock:
                active = next((job for job in self._jobs.values() if job.status in self.ACTIVE_STATUSES), None)
                if active is not None:
                    raise RuntimeError("当前已有一个 Bib 导入任务在运行，请等待其完成后再开始新的导入。")

                job_id = uuid.uuid4().hex[:12]
                now = self._timestamp()
                source_path = self._job_dir(job_id) / filename
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes(payload)
                job = BibImportJobRecord(
                    id=job_id,
                    source_name=filename,
                    target_folder=target_folder,
                    status="queued",
                    progress=0,
                    message="Bib 导入任务已创建，等待开始。",
                    error=None,
                    total_entries=0,
                    processed_entries=0,
                    imported_count=0,
                    duplicate_count=0,
                    unmatched_count=0,
                    current_label="",
                    source_bib_rel_path=source_path.relative_to(self.library_root).as_posix(),
                    unmatched_bib_rel_path=None,
                    unmatched_list_rel_path=None,
                    created_at=now,
                    updated_at=now,
                )
                self._jobs[job.id] = job
                self._persist_locked(job)

            worker = threading.Thread(
                target=self._run_job,
                args=(job.id, source_path),
                name=f"paper-reader-bib-import-{job.id}",
                daemon=True,
            )
            worker.start()
            return job

        def get_job(self, job_id: str) -> BibImportJobRecord | None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    return job
                return self._load_job_from_disk(job_id)

        def latest_job(self) -> BibImportJobRecord | None:
            with self._lock:
                if self._jobs:
                    return max(self._jobs.values(), key=lambda item: (item.updated_at, item.created_at))
                paths = sorted(self.job_root.glob("*/status.json"), key=lambda item: item.stat().st_mtime, reverse=True)
                for path in paths:
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    job = self._coerce_job(payload)
                    if job is None or not job.id:
                        continue
                    self._jobs[job.id] = job
                    return job
            return None

        def _run_job(self, job_id: str, source_path: Path) -> None:
            self._update_job(job_id, status="running", progress=1, started_at=self._timestamp(), message="正在解析 Bib 文件。")
            try:
                text = source_path.read_text(encoding="utf-8", errors="replace")
                entries = bib_import_utils.load_bib_entries(text)
                if not entries:
                    raise ValueError("Bib 文件中没有可导入的文献条目。")
            except Exception as exc:
                self._update_job(
                    job_id,
                    status="failed",
                    progress=100,
                    error=str(exc),
                    message="Bib 文件解析失败。",
                    finished_at=self._timestamp(),
                )
                return

            unmatched_entries: list[dict[str, Any]] = []
            unmatched_items: list[bib_import_utils.BibImportUnmatchedItem] = []
            imported_rel_paths: list[str] = []
            duplicate_rel_paths: list[str] = []
            total_entries = len(entries)
            self._update_job(job_id, total_entries=total_entries, message=f"共检测到 {total_entries} 篇文献，开始导入。")

            for index, entry in enumerate(entries, start=1):
                cite_key = str(entry.get("ID") or "")
                title = str(entry.get("title") or "").strip() or cite_key or f"Entry {index}"
                self._update_job(
                    job_id,
                    current_label=title,
                    message=f"正在处理 {index}/{total_entries}: {title}",
                )

                extracted_hint = bib_import_utils.extract_arxiv_id_from_entry(entry)
                try:
                    arxiv_id, match_source = bib_import_utils.find_arxiv_id_for_bib_entry(entry)
                except Exception as exc:
                    unmatched_entries.append(dict(entry))
                    unmatched_items.append(
                        bib_import_utils.BibImportUnmatchedItem(
                            cite_key=cite_key,
                            title=title,
                            reason=str(exc),
                            raw_arxiv_hint=extracted_hint,
                        )
                    )
                else:
                    if not arxiv_id:
                        unmatched_entries.append(dict(entry))
                        unmatched_items.append(
                            bib_import_utils.BibImportUnmatchedItem(
                                cite_key=cite_key,
                                title=title,
                                reason="未找到可信 arXiv 匹配。",
                                raw_arxiv_hint=extracted_hint,
                            )
                        )
                    else:
                        try:
                            result = process_arxiv_download(
                                arxiv_id,
                                target_folder=self._jobs[job_id].target_folder,
                                current_folder=self._jobs[job_id].target_folder,
                                query="",
                                sort_by="date_desc",
                                submit_auto_prompts=False,
                                show_done=False,
                                include_paper_view=False,
                            )
                        except Exception as exc:
                            unmatched_entries.append(dict(entry))
                            reason = str(exc)
                            if match_source == "search":
                                reason = f"搜索到 arXiv {arxiv_id}，但导入失败：{reason}"
                            unmatched_items.append(
                                bib_import_utils.BibImportUnmatchedItem(
                                    cite_key=cite_key,
                                    title=title,
                                    reason=reason,
                                    raw_arxiv_hint=arxiv_id,
                                )
                            )
                        else:
                            if result["status"] == "saved" and result.get("saved_rel_path"):
                                imported_rel_paths.append(str(result["saved_rel_path"]))
                            elif result.get("duplicate_rel_path"):
                                duplicate_rel_paths.append(str(result["duplicate_rel_path"]))
                            else:
                                unmatched_entries.append(dict(entry))
                                unmatched_items.append(
                                    bib_import_utils.BibImportUnmatchedItem(
                                        cite_key=cite_key,
                                        title=title,
                                        reason="导入结果未知，已标记为未导入。",
                                        raw_arxiv_hint=arxiv_id,
                                    )
                                )

                progress = min(99, max(1, round(index * 100 / max(total_entries, 1))))
                self._update_job(
                    job_id,
                    processed_entries=index,
                    imported_count=len(imported_rel_paths),
                    duplicate_count=len(duplicate_rel_paths),
                    unmatched_count=len(unmatched_items),
                    imported_rel_paths=list(imported_rel_paths),
                    duplicate_rel_paths=list(duplicate_rel_paths),
                    progress=progress,
                )

            unmatched_bib_rel_path: str | None = None
            unmatched_list_rel_path: str | None = None
            if unmatched_entries:
                job_dir = self._job_dir(job_id)
                unmatched_bib_path = job_dir / "unimported.bib"
                unmatched_list_path = job_dir / "unimported.txt"
                unmatched_bib_path.write_text(
                    bib_import_utils.dump_bib_entries(unmatched_entries),
                    encoding="utf-8",
                )
                unmatched_list_path.write_text(
                    bib_import_utils.dump_unmatched_list(unmatched_items),
                    encoding="utf-8",
                )
                unmatched_bib_rel_path = unmatched_bib_path.relative_to(self.library_root).as_posix()
                unmatched_list_rel_path = unmatched_list_path.relative_to(self.library_root).as_posix()

            auto_prompt_summary = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
            if imported_rel_paths:
                auto_prompts = app.prompt_store.auto_prompts()  # type: ignore[attr-defined]
                if auto_prompts:
                    auto_prompt_summary = app.job_queue.submit(  # type: ignore[attr-defined]
                        imported_rel_paths,
                        [prompt.slug for prompt in auto_prompts],
                        force=False,
                        source="bib-import",
                    )

            summary_message = (
                f"Bib 导入完成：新增 {len(imported_rel_paths)} 篇，重复 {len(duplicate_rel_paths)} 篇，未导入 {len(unmatched_items)} 篇。"
            )
            self._update_job(
                job_id,
                status="completed",
                progress=100,
                message=summary_message,
                error=None,
                current_label="",
                unmatched_bib_rel_path=unmatched_bib_rel_path,
                unmatched_list_rel_path=unmatched_list_rel_path,
                auto_prompt_summary=auto_prompt_summary,
                finished_at=self._timestamp(),
            )

    app.bib_import_manager = BibImportManager(app.config["LIBRARY_ROOT"])  # type: ignore[attr-defined]

    def flash_submission_summary(result: dict[str, Any], *, action_label: str) -> None:
        if result["queued"]:
            flash(f"{action_label}：已提交 {result['queued']} 个后台任务。", "success")
        if result["existing"]:
            flash(f"{result['existing']} 个任务已在队列中，未重复提交。", "success")
        if result["skipped"]:
            flash(f"{result['skipped']} 个结果已存在，未重复提交。", "success")
        if result["invalid"]:
            flash(f"{result['invalid']} 个任务因 Prompt 缺失而未提交。", "error")

    def build_batch_papers(
        *,
        folder: str,
        query: str,
        sort_by: str,
        show_done: bool,
        batch_show_done: bool,
        selected_rel_path: str = "",
    ) -> list[PaperRecord]:
        include_done = show_done or batch_show_done or selected_rel_path.startswith(f"{DONE_DIR_NAME}/")
        scan = app.library.scan(include_done=include_done)  # type: ignore[attr-defined]
        batch_papers = filter_and_sort_papers(
            scan.papers,
            folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=(show_done or batch_show_done),
        )
        if not batch_show_done:
            batch_papers = [paper for paper in batch_papers if not paper.is_done]
        return batch_papers

    def current_page_state() -> dict[str, Any]:
        folder = request.args.get("folder", "")
        query = request.args.get("q", "")
        sort_by = request.args.get("sort", "date_desc")
        show_done = request.args.get("show_done", "").strip().lower() in {"1", "true", "yes", "on"}
        batch_show_done = request.args.get("batch_show_done", "").strip().lower() in {"1", "true", "yes", "on"}
        batch_page = parse_page(request.args.get("batch_page"))
        selected_rel_path = request.args.get("paper", "").strip("/")
        selected_tab = request.args.get("tab", "source")

        include_done = show_done or batch_show_done or selected_rel_path.startswith(f"{DONE_DIR_NAME}/")
        scan = app.library.scan(include_done=include_done)  # type: ignore[attr-defined]
        all_prompts = app.prompt_store.list_prompts()  # type: ignore[attr-defined]
        active_prompts = [prompt for prompt in all_prompts if prompt.enabled]

        papers = filter_and_sort_papers(scan.papers, folder=folder, query=query, sort_by=sort_by, show_done=show_done)
        batch_papers = filter_and_sort_papers(
            scan.papers,
            folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=(show_done or batch_show_done),
        )
        if not batch_show_done:
            batch_papers = [paper for paper in batch_papers if not paper.is_done]
        batch_library_papers = filter_and_sort_papers(
            scan.papers,
            folder="",
            query="",
            sort_by=sort_by,
            show_done=(show_done or batch_show_done),
        )
        if not batch_show_done:
            batch_library_papers = [paper for paper in batch_library_papers if not paper.is_done]
        sidebar_groups = build_sidebar_groups(papers, selected_rel_path)

        selected_paper = next((item for item in papers if item.rel_path == selected_rel_path), None)
        if selected_paper is None and papers:
            selected_paper = papers[0]

        return {
            "scan": scan,
            "all_prompts": all_prompts,
            "active_prompts": active_prompts,
            "folder": folder,
            "query": query,
            "sort_by": sort_by,
            "show_done": show_done,
            "batch_show_done": batch_show_done,
            "batch_page": batch_page,
            "selected_rel_path": selected_rel_path,
            "selected_tab": selected_tab,
            "papers": papers,
            "batch_papers": batch_papers,
            "batch_library_total": len(batch_library_papers),
            "sidebar_groups": sidebar_groups,
            "selected_paper": selected_paper,
        }

    @app.get("/")
    def index() -> str:
        state = current_page_state()
        all_prompts = state["all_prompts"]
        active_prompts = state["active_prompts"]
        folder = state["folder"]
        query = state["query"]
        sort_by = state["sort_by"]
        show_done = state["show_done"]
        selected_tab = state["selected_tab"]
        papers = state["papers"]
        batch_papers = state["batch_papers"]
        sidebar_groups = state["sidebar_groups"]
        selected_paper = state["selected_paper"]

        preview_paragraphs: list[str] = []
        selected_prompt: PromptDefinition | None = None
        selected_prompt_content: str | None = None
        selected_prompt_html: Markup | None = None
        selected_prompt_info: dict[str, Any] | None = None
        selected_prompt_job: dict[str, Any] | None = None
        viewer_tabs: list[dict[str, Any]] = []

        if selected_paper:
            if selected_paper.preview_text:
                preview_paragraphs = [chunk.strip() for chunk in selected_paper.preview_text.split("\n\n") if chunk.strip()]
            for prompt in active_prompts:
                info = app.library.prompt_result_info(selected_paper.rel_path, prompt.slug)  # type: ignore[attr-defined]
                latest_job = app.job_queue.latest_job_for(selected_paper.rel_path, prompt.slug)  # type: ignore[attr-defined]
                viewer_tabs.append({
                    "slug": prompt.slug,
                    "name": prompt.name,
                    "model": prompt.model,
                    "exists": info["exists"],
                    "updated_at": info["updated_at"],
                    "job_status": latest_job.status if latest_job else None,
                    "job_progress": latest_job.progress if latest_job else None,
                })
            valid_tabs = {"source"} | {prompt.slug for prompt in active_prompts}
            if selected_tab not in valid_tabs:
                selected_tab = "source"
            if selected_tab != "source":
                selected_prompt = next((prompt for prompt in active_prompts if prompt.slug == selected_tab), None)
                if selected_prompt is not None:
                    selected_prompt_content = app.library.read_prompt_result(selected_paper.rel_path, selected_prompt.slug)  # type: ignore[attr-defined]
                    if selected_prompt_content:
                        selected_prompt_html = Markup(render_markdown(selected_prompt_content))
                    selected_prompt_info = app.library.prompt_result_info(selected_paper.rel_path, selected_prompt.slug)  # type: ignore[attr-defined]
                    latest_job = app.job_queue.latest_job_for(selected_paper.rel_path, selected_prompt.slug)  # type: ignore[attr-defined]
                    if latest_job is not None:
                        selected_prompt_job = asdict(latest_job)
        else:
            selected_tab = "source"

        latest_bib_import_job = app.bib_import_manager.latest_job()  # type: ignore[attr-defined]
        return render_template(
            "index.html",
            papers=papers,
            folders=state["scan"].folders,
            current_folder=folder,
            query=query,
            sort_by=sort_by,
            show_done=show_done,
            selected_paper=selected_paper,
            selected_tab=selected_tab,
            preview_paragraphs=preview_paragraphs,
            sidebar_groups=sidebar_groups,
            viewer_tabs=viewer_tabs,
            selected_prompt=selected_prompt,
            selected_prompt_content=selected_prompt_content,
            selected_prompt_html=selected_prompt_html,
            selected_prompt_info=selected_prompt_info,
            selected_prompt_job=selected_prompt_job,
            active_prompts=active_prompts,
            active_prompt_count=len(active_prompts),
            library_root=app.config["LIBRARY_ROOT"],
            initial_job_snapshot=app.job_queue.snapshot(),  # type: ignore[attr-defined]
            initial_bib_import_snapshot=(serialize_bib_import_job(latest_bib_import_job) if latest_bib_import_job else None),
        )

    @app.get("/tool-panels/<panel_name>")
    def tool_panel_route(panel_name: str) -> Any:
        state = current_page_state()
        context = {
            "current_folder": state["folder"],
            "query": state["query"],
            "sort_by": state["sort_by"],
            "show_done": state["show_done"],
            "batch_show_done": state["batch_show_done"],
            "batch_page": state["batch_page"],
            "selected_paper": state["selected_paper"],
            "selected_tab": state["selected_tab"],
            "active_prompt_count": len(state["active_prompts"]),
        }

        if panel_name == "prompt-manager":
            return render_template(
                "panels/prompt_manager.html",
                **context,
                all_prompts=state["all_prompts"],
                new_prompt_defaults={
                    "name": "",
                    "slug": "",
                    "model": DEFAULT_MODEL,
                    "user_prompt": DEFAULT_USER_PROMPT,
                    "enabled": True,
                    "auto_run": True,
                },
            )

        if panel_name == "offline-package":
            return render_template(
                "panels/offline_package.html",
                **context,
                filtered_papers=state["batch_papers"],
            )

        if panel_name == "batch-run":
            pagination = paginate_items(state["batch_papers"], state["batch_page"], DEFAULT_BATCH_PANEL_PAGE_SIZE)
            return render_template(
                "panels/batch_run.html",
                **context,
                filtered_papers=pagination["items"],
                batch_pagination=pagination,
                batch_library_total=state["batch_library_total"],
                all_prompts=state["all_prompts"],
            )

        abort(404)

    @app.post("/upload")
    def upload() -> Any:
        files = request.files.getlist("files")
        target_folder = request.form.get("target_folder", "").strip().strip("/")
        current_folder = request.form.get("folder", target_folder or "").strip().strip("/")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        if not files or not any(file.filename for file in files):
            flash("请选择至少一个文件。", "error")
            return redirect_to_index(current_folder or target_folder, query, sort_by, show_done=show_done)

        saved_rel_paths: list[str] = []
        saved = 0
        duplicate_count = 0
        error_count = 0
        for file in files:
            try:
                result = process_uploaded_file(
                    file,
                    target_folder=target_folder,
                    current_folder=current_folder or target_folder,
                    query=query,
                    sort_by=sort_by,
                    submit_auto_prompts=False,
                    show_done=show_done,
                )
            except ValueError as exc:
                flash(str(exc), "error")
                continue

            if result["status"] == "saved" and result["saved_rel_path"]:
                saved += 1
                saved_rel_paths.append(result["saved_rel_path"])
                flash(result["message"], "success")
            elif result["status"] == "duplicate":
                duplicate_count += 1
                flash(result["message"], "success")
            else:
                error_count += 1
                flash(result["message"], "error")

        if saved:
            flash(f"本次成功上传 {saved} 个文件。", "success")
            auto_prompts = app.prompt_store.auto_prompts()  # type: ignore[attr-defined]
            if auto_prompts:
                submission = app.job_queue.submit(  # type: ignore[attr-defined]
                    saved_rel_paths,
                    [prompt.slug for prompt in auto_prompts],
                    force=False,
                    source="upload",
                )
                flash_submission_summary(submission, action_label="自动 Prompt 处理已转为后台任务")
        if duplicate_count:
            flash(f"检测到 {duplicate_count} 个重复文件，已自动跳过。", "success")
        if error_count and not saved:
            flash(f"有 {error_count} 个文件上传失败。", "error")
        selected_rel_path = saved_rel_paths[0] if saved_rel_paths else None
        return redirect_to_index(current_folder or target_folder, query, sort_by, selected_rel_path, "source", show_done=show_done)

    @app.post("/upload-file")
    def upload_file() -> Any:
        file = request.files.get("file")
        target_folder = request.form.get("target_folder", "").strip().strip("/")
        current_folder = request.form.get("folder", target_folder or "").strip().strip("/")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))

        if file is None or not file.filename:
            return {"status": "error", "message": "没有选择文件。"}, 400

        try:
            result = process_uploaded_file(
                file,
                target_folder=target_folder,
                current_folder=current_folder or target_folder,
                query=query,
                sort_by=sort_by,
                submit_auto_prompts=True,
                show_done=show_done,
            )
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}, 400
        except Exception as exc:
            return {"status": "error", "message": f"上传失败：{exc}"}, 500

        status_code = 200 if result["status"] in {"saved", "duplicate"} else 400
        return result, status_code

    @app.post("/arxiv-download")
    def arxiv_download_route() -> Any:
        target_folder = request.form.get("target_folder", "").strip().strip("/")
        current_folder = request.form.get("folder", target_folder or "").strip().strip("/")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        raw_inputs = request.form.get("arxiv_ids", "") or request.form.get("arxiv_id", "")
        requested_ids = parse_arxiv_input(raw_inputs)

        if not requested_ids:
            flash("请输入至少一个 arXiv ID 或 arxiv.org 链接。", "error")
            return redirect_to_index(current_folder or target_folder, query, sort_by, show_done=show_done)

        saved_rel_paths: list[str] = []
        duplicate_rel_paths: list[str] = []
        errors: list[str] = []
        seen_ids: set[str] = set()
        skipped_input_duplicates = 0

        for requested_id in requested_ids:
            try:
                normalized_id = normalize_arxiv_id(requested_id)
            except ValueError as exc:
                errors.append(f"{requested_id}: {exc}")
                continue

            if normalized_id in seen_ids:
                skipped_input_duplicates += 1
                continue
            seen_ids.add(normalized_id)

            try:
                result = process_arxiv_download(
                    normalized_id,
                    target_folder=target_folder,
                    current_folder=current_folder or target_folder,
                    query=query,
                    sort_by=sort_by,
                    submit_auto_prompts=False,
                    show_done=show_done,
                )
            except (ValueError, RuntimeError) as exc:
                errors.append(f"{normalized_id}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{normalized_id}: arXiv 下载失败：{exc}")
                continue

            if result["status"] == "saved" and result.get("saved_rel_path"):
                saved_rel_paths.append(str(result["saved_rel_path"]))
            elif result.get("duplicate_rel_path"):
                duplicate_rel_paths.append(str(result["duplicate_rel_path"]))

        if saved_rel_paths:
            flash(f"arXiv 下载完成：新增 {len(saved_rel_paths)} 篇论文。", "success")
            auto_prompts = app.prompt_store.auto_prompts()  # type: ignore[attr-defined]
            if auto_prompts:
                submission = app.job_queue.submit(  # type: ignore[attr-defined]
                    saved_rel_paths,
                    [prompt.slug for prompt in auto_prompts],
                    force=False,
                    source="arxiv",
                )
                flash_submission_summary(submission, action_label="arXiv 自动 Prompt 处理已转为后台任务")
        if duplicate_rel_paths:
            flash(f"检测到 {len(duplicate_rel_paths)} 篇重复论文，已自动跳过。", "success")
        if skipped_input_duplicates:
            flash(f"输入中有 {skipped_input_duplicates} 个重复 arXiv ID，已忽略。", "success")
        if errors:
            for message in errors[:5]:
                flash(message, "error")
            if len(errors) > 5:
                flash(f"还有 {len(errors) - 5} 个 arXiv ID 下载失败，请检查输入格式或网络。", "error")

        selected_rel_path = saved_rel_paths[0] if saved_rel_paths else (duplicate_rel_paths[0] if duplicate_rel_paths else None)
        return redirect_to_index(
            current_folder or target_folder,
            query,
            sort_by,
            selected_rel_path,
            "source" if selected_rel_path else None,
            show_done=show_done,
        )

    def serialize_bib_import_job(job: BibImportJobRecord) -> dict[str, Any]:
        payload = asdict(job)
        payload["unmatched_bib_url"] = (
            url_for("serve_file", rel_path=job.unmatched_bib_rel_path) if job.unmatched_bib_rel_path else None
        )
        payload["unmatched_list_url"] = (
            url_for("serve_file", rel_path=job.unmatched_list_rel_path) if job.unmatched_list_rel_path else None
        )
        payload["source_bib_url"] = url_for("serve_file", rel_path=job.source_bib_rel_path) if job.source_bib_rel_path else None
        return payload

    @app.post("/bib-import/start")
    def bib_import_start_route() -> Any:
        file = request.files.get("bib_file")
        target_folder = request.form.get("target_folder", "").strip().strip("/")
        if file is None or not file.filename:
            return {"status": "error", "message": "请选择一个 .bib 文件。"}, 400
        try:
            job = app.bib_import_manager.start(file, target_folder=target_folder)  # type: ignore[attr-defined]
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}, 400
        except RuntimeError as exc:
            return {"status": "error", "message": str(exc)}, 409
        except Exception as exc:
            return {"status": "error", "message": f"Bib 导入启动失败：{exc}"}, 500
        return serialize_bib_import_job(job)

    @app.get("/bib-import/status/<job_id>")
    def bib_import_status_route(job_id: str) -> Any:
        job = app.bib_import_manager.get_job(job_id)  # type: ignore[attr-defined]
        if job is None:
            return {"status": "error", "message": "Bib 导入任务不存在。"}, 404
        return serialize_bib_import_job(job)

    @app.post("/folders")
    def create_folder_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        folder_name = request.form.get("new_folder", "").strip()
        parent_folder = request.form.get("parent_folder", "").strip().strip("/")
        target = "/".join(part for part in [parent_folder, folder_name] if part)
        try:
            app.library.create_folder(target)  # type: ignore[attr-defined]
            flash(f"已创建文件夹：{target}", "success")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/rename")
    def rename_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        tab = request.form.get("tab", "source")
        rel_path = request.form.get("rel_path", "").strip("/")
        new_name = request.form.get("new_name", "").strip()
        try:
            new_rel_path = app.library.rename_file(rel_path, new_name)  # type: ignore[attr-defined]
            flash("文件已重命名。", "success")
            return redirect_to_index(current_folder, query, sort_by, new_rel_path, tab, show_done=show_done)
        except (FileNotFoundError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/delete")
    def delete_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        try:
            app.library.delete_file(rel_path)  # type: ignore[attr-defined]
            flash("文件已删除。", "success")
        except FileNotFoundError:
            flash("文件不存在。", "error")
        return redirect_to_index(current_folder, query, sort_by, show_done=show_done)

    @app.post("/done-toggle")
    def done_toggle_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        tab = request.form.get("tab", "source")
        try:
            new_rel_path = app.library.toggle_done(rel_path)  # type: ignore[attr-defined]
            moved_to_done = app.library.is_done_rel_path(new_rel_path)  # type: ignore[attr-defined]
            flash("论文已标记为 DONE。" if moved_to_done else "论文已恢复到未完成列表。", "success")
            selected_rel_path = new_rel_path if show_done else None
            next_show_done = show_done
            next_folder = current_folder
            if next_folder and not (
                Path(new_rel_path).parent.as_posix() == next_folder
                or new_rel_path.startswith(next_folder + "/")
            ):
                next_folder = ""
            next_tab = tab if selected_rel_path else "source"
            return redirect_to_index(
                next_folder,
                query,
                sort_by,
                selected_rel_path,
                next_tab,
                show_done=next_show_done,
            )
        except (FileNotFoundError, ValueError) as exc:
            flash(str(exc), "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, tab, show_done=show_done)

    @app.post("/prompt-run")
    def prompt_run_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        prompt_slug = request.form.get("prompt_slug", "").strip()
        force = parse_checkbox(request.form.get("force"))
        prompt = app.prompt_store.get_prompt(prompt_slug)  # type: ignore[attr-defined]
        if prompt is None:
            flash("Prompt 不存在。", "error")
            return redirect_to_index(current_folder, query, sort_by, rel_path or None, "source", show_done=show_done)
        submission = app.job_queue.submit([rel_path], [prompt.slug], force=force, source="manual")  # type: ignore[attr-defined]
        flash_submission_summary(submission, action_label=f"《{prompt.name}》后台任务已提交")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, prompt_slug, show_done=show_done)

    @app.post("/prompt-save")
    def prompt_save_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        existing_slug = request.form.get("existing_slug", "").strip() or None
        try:
            prompt = app.prompt_store.save_prompt(  # type: ignore[attr-defined]
                existing_slug=existing_slug,
                name=request.form.get("name", ""),
                slug=request.form.get("slug", ""),
                user_prompt=request.form.get("user_prompt", ""),
                model=request.form.get("model", DEFAULT_MODEL),
                enabled=parse_checkbox(request.form.get("enabled")),
                auto_run=parse_checkbox(request.form.get("auto_run")),
            )
            flash(f"Prompt《{prompt.name}》已保存。", "success")
            app.library.invalidate_scan_cache()  # type: ignore[attr-defined]
            if tab != "source" and tab == prompt.slug and not prompt.enabled:
                tab = "source"
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/prompt-delete")
    def prompt_delete_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        prompt_slug = request.form.get("prompt_slug", "").strip()
        try:
            removed = app.prompt_store.delete_prompt(prompt_slug)  # type: ignore[attr-defined]
            flash(f"Prompt《{removed.name}》已删除。", "success")
            app.library.invalidate_scan_cache()  # type: ignore[attr-defined]
            if tab == prompt_slug:
                tab = "source"
        except FileNotFoundError:
            flash("Prompt 不存在。", "error")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/prompt-batch-run")
    def prompt_batch_run_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        batch_show_done = parse_checkbox(request.form.get("batch_show_done"))
        batch_page = parse_page(request.form.get("batch_page"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        rel_paths = request.form.getlist("rel_paths")
        prompt_slugs = request.form.getlist("prompt_slugs")
        select_all_filtered = parse_checkbox(request.form.get("select_all_filtered"))
        force = parse_checkbox(request.form.get("force"))

        if select_all_filtered:
            rel_paths = [
                paper.rel_path
                for paper in build_batch_papers(
                    folder=current_folder,
                    query=query,
                    sort_by=sort_by,
                    show_done=show_done,
                    batch_show_done=batch_show_done,
                    selected_rel_path=(selected_paper or ""),
                )
            ]

        if not rel_paths:
            flash("请至少选择一篇论文。", "error")
            return redirect(
                url_for(
                    "index",
                    folder=current_folder,
                    q=query,
                    sort=sort_by,
                    paper=selected_paper,
                    tab=tab,
                    show_done="1" if show_done else None,
                    batch_show_done="1" if batch_show_done else None,
                    batch_page=batch_page,
                )
            )
        if not prompt_slugs:
            flash("请至少选择一个 Prompt。", "error")
            return redirect(
                url_for(
                    "index",
                    folder=current_folder,
                    q=query,
                    sort=sort_by,
                    paper=selected_paper,
                    tab=tab,
                    show_done="1" if show_done else None,
                    batch_show_done="1" if batch_show_done else None,
                    batch_page=batch_page,
                )
            )

        submission = app.job_queue.submit(rel_paths, prompt_slugs, force=force, source="batch")  # type: ignore[attr-defined]
        flash_submission_summary(submission, action_label="批量后台任务已提交")
        return redirect(
            url_for(
                "index",
                folder=current_folder,
                q=query,
                sort=sort_by,
                paper=(selected_paper or (rel_paths[0] if rel_paths else None)),
                tab=tab,
                show_done="1" if show_done else None,
                batch_show_done="1" if batch_show_done else None,
                batch_page=batch_page,
            )
        )

    @app.post("/offline-package")
    def offline_package_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        rel_paths = list(dict.fromkeys(path.strip("/") for path in request.form.getlist("rel_paths") if path.strip("/")))

        if not rel_paths:
            flash("请至少选择一篇论文来生成离线阅读包。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        scan = app.library.scan()  # type: ignore[attr-defined]
        paper_map = {paper.rel_path: paper for paper in scan.papers}
        selected_records: list[PaperRecord] = []
        for rel_path in rel_paths:
            paper = paper_map.get(rel_path)
            if paper is None or paper.is_done:
                continue
            selected_records.append(paper)

        if not selected_records:
            flash("当前选择中没有可导出的未完成论文。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        manifest = build_offline_manifest(app.library, app.prompt_store, selected_records)  # type: ignore[attr-defined]
        manifest_json = build_manifest_json(manifest)
        html = render_template("offline_reader.html", manifest_json=manifest_json)

        static_root = Path(app.static_folder or "")
        with tempfile.NamedTemporaryFile(prefix="paper-reader-offline-", suffix=".zip", delete=False) as handle:
            zip_path = Path(handle.name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("index.html", html.encode("utf-8"))
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
            archive.writestr("assets/style.css", (static_root / "style.css").read_bytes())
            archive.writestr("assets/offline-reader.css", (static_root / "offline-reader.css").read_bytes())
            archive.writestr("assets/offline-reader.js", (static_root / "offline-reader.js").read_bytes())

            for paper in selected_records:
                source_path = app.library.resolve_relative_path(paper.rel_path)  # type: ignore[attr-defined]
                archive.write(source_path, offline_source_arcname(paper.rel_path))
                for prompt_slug in app.library.list_existing_prompt_slugs(paper.rel_path):  # type: ignore[attr-defined]
                    result_path = app.library.existing_prompt_result_path(paper.rel_path, prompt_slug)  # type: ignore[attr-defined]
                    if result_path is None:
                        continue
                    archive.write(result_path, offline_prompt_arcname(paper.rel_path, prompt_slug))

        package_name = f"paper-reader-offline-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.zip"
        response = send_file(zip_path, as_attachment=True, download_name=package_name, mimetype="application/zip")
        response.call_on_close(lambda: zip_path.unlink(missing_ok=True))
        return response

    @app.post("/ai-summary")
    def ai_summary_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        rel_path = request.form.get("rel_path", "").strip("/")
        prompt = app.prompt_store.get_prompt(DEFAULT_PROMPT_SLUG)  # type: ignore[attr-defined]
        if prompt is None:
            prompt = app.prompt_store.save_prompt(  # type: ignore[attr-defined]
                existing_slug=None,
                name="核心解读",
                slug=DEFAULT_PROMPT_SLUG,
                user_prompt=DEFAULT_USER_PROMPT,
                model=request.form.get("model", DEFAULT_MODEL),
                enabled=True,
                auto_run=True,
            )
        submission = app.job_queue.submit([rel_path], [prompt.slug], force=True, source="legacy-ai-summary")  # type: ignore[attr-defined]
        flash_submission_summary(submission, action_label="AI 摘要后台任务已提交")
        return redirect_to_index(current_folder, query, sort_by, rel_path or None, DEFAULT_PROMPT_SLUG, show_done=show_done)

    @app.get("/jobs/status")
    def jobs_status() -> Any:
        rel_path = request.args.get("paper", "").strip("/") or None
        return app.job_queue.snapshot(rel_path=rel_path)  # type: ignore[attr-defined]

    @app.post("/jobs/config")
    def jobs_config_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        requested = request.form.get("max_concurrency", "")
        try:
            value = app.settings_store.save_max_concurrency(int(requested))  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            flash("Max concurrency 必须是 1 到 32 之间的整数。", "error")
            return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

        app.job_queue.update_max_concurrency(value)  # type: ignore[attr-defined]
        flash(f"Max concurrency 已更新为 {value}。", "success")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.post("/jobs/stop-all")
    def jobs_stop_all_route() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        summary = app.job_queue.stop_all()  # type: ignore[attr-defined]
        interrupted = summary["queued"] + summary["running"]
        if interrupted:
            flash(f"已请求停止 {interrupted} 个后台任务（运行中 {summary['running']}，排队中 {summary['queued']}）。", "success")
        else:
            flash("当前没有可停止的后台任务。", "success")
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    @app.get("/files/<path:rel_path>")
    def serve_file(rel_path: str) -> Any:
        rel_path = rel_path.strip("/")
        try:
            absolute = app.library.resolve_relative_path(rel_path)  # type: ignore[attr-defined]
        except ValueError:
            abort(404)
        if not absolute.exists() or not absolute.is_file():
            abort(404)
        safe_rel_path = absolute.relative_to(app.config["LIBRARY_ROOT"]).as_posix()
        return send_from_directory(app.config["LIBRARY_ROOT"], safe_rel_path, as_attachment=False)

    @app.post("/reindex")
    def reindex() -> Any:
        current_folder = request.form.get("folder", "")
        query = request.form.get("q", "")
        sort_by = request.form.get("sort", "date_desc")
        show_done = parse_checkbox(request.form.get("show_done"))
        selected_paper = request.form.get("paper", "") or None
        tab = request.form.get("tab", "source")
        active_records = app.library.rebuild_active_index(lightweight=True)  # type: ignore[attr-defined]
        done_records = app.library.rebuild_done_index(lightweight=True)  # type: ignore[attr-defined]
        flash(
            f"已完成文件夹快速扫描：普通目录已更新（{len(active_records)} 篇），DONE 轻量索引已刷新（{len(done_records)} 篇），未触发 Prompt，也未执行重型解析。",
            "success",
        )
        return redirect_to_index(current_folder, query, sort_by, selected_paper, tab, show_done=show_done)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8022, debug=False)
