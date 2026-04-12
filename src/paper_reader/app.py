from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import string
import tempfile
import threading
import time
from urllib.parse import unquote
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from flask import Flask, abort, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for
from markupsafe import Markup
from werkzeug.utils import secure_filename

from .ai_summary import DEFAULT_MODEL, DEFAULT_USER_PROMPT, run_prompt_on_document
from .document_utils import ALLOWED_EXTENSIONS, extract_document_metadata
from .markdown_render import render_markdown
from .offline_package import build_offline_manifest, manifest_json as build_manifest_json, offline_prompt_arcname, offline_source_arcname
from .prompt_manager import DEFAULT_PROMPT_SLUG, PromptDefinition, PromptStore, parse_checkbox
from .settings import SettingsStore
from .source_archive import day_paper_map, load_source_day, load_source_days, local_pdf_path_for
from .task_queue import PaperJobQueue

CACHE_FILE_NAME = ".paper_reader_index.json"
DONE_INDEX_FILE_NAME = ".paper_reader_done_index.json"
SUMMARY_DIR_NAME = ".paper-reader-ai"
DONE_DIR_NAME = "DONE"
DEFAULT_BATCH_PANEL_PAGE_SIZE = 50
DEFAULT_LOGIN_USERNAME = "admin"
DEFAULT_LOGIN_PASSWORD = "paperpaperreaderreader12678"
MAX_LOGIN_FAILURES = 3
LOGIN_LOCK_SECONDS = 5 * 60


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

    def import_external_file(self, source_path: Path, target_folder: str, *, preferred_name: str | None = None) -> dict[str, Any]:
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(str(source_path))

        source_name = preferred_name or source_path.name
        suffix = Path(source_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {source_name}")

        file_size = source_path.stat().st_size
        file_hash = sha256_for_path(source_path)
        duplicate_rel_path = self.find_duplicate_by_hash(file_size, file_hash)
        if duplicate_rel_path:
            return {
                "status": "duplicate",
                "message": f"检测到重复文件，已跳过：{Path(duplicate_rel_path).name}",
                "saved_rel_path": None,
                "duplicate_rel_path": duplicate_rel_path,
            }

        destination = self.make_unique_destination(target_folder, source_name)
        shutil.copy2(source_path, destination)
        rel_path = destination.relative_to(self.root).as_posix()
        if self.is_done_rel_path(rel_path):
            self._update_done_index_entry(rel_path)
        else:
            self._update_active_index_entry(rel_path)
        return {
            "status": "saved",
            "message": f"导入成功：{destination.name}",
            "saved_rel_path": rel_path,
            "duplicate_rel_path": None,
        }

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


def safe_download_name(value: str, *, fallback: str) -> str:
    allowed = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned = "".join(char if char in allowed else "_" for char in value).strip().rstrip(".")
    return cleaned or fallback


def build_source_groups(days: list[Any]) -> list[dict[str, Any]]:
    tree: dict[str, dict[str, Any]] = {}
    for day in days:
        year_group = tree.setdefault(day.year, {"key": day.year, "label": day.year, "count": 0, "months": {}})
        month_group = year_group["months"].setdefault(
            day.month,
            {"key": day.month, "label": f"{day.year}-{day.month}", "count": 0, "days": []},
        )
        year_group["count"] += 1
        month_group["count"] += 1
        month_group["days"].append(day)

    groups: list[dict[str, Any]] = []
    for year_key in sorted(tree.keys(), reverse=True):
        year_group = tree[year_key]
        months = [year_group["months"][month_key] for month_key in sorted(year_group["months"].keys(), reverse=True)]
        groups.append(
            {
                "key": year_group["key"],
                "label": year_group["label"],
                "count": year_group["count"],
                "months": months,
            }
        )
    return groups



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



def create_app(library_root: Path | None = None, source_archive_root: Path | None = None) -> Flask:
    base_dir = Path(__file__).resolve().parents[2]
    root = library_root or Path(base_dir / "docs" / "papers")
    source_root = source_archive_root or Path(base_dir / "paper-reader-source" / "data" / "huggingface_daily")
    login_username, login_password = resolve_login_credentials(base_dir)
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).with_name("templates")),
        static_folder=str(Path(__file__).with_name("static")),
    )
    app.config["SECRET_KEY"] = "paper-reader-dev-secret"
    app.config["LIBRARY_ROOT"] = root.resolve()
    app.config["SOURCE_ARCHIVE_ROOT"] = source_root.resolve()
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
        if app.library.is_done_rel_path(rel_path):  # type: ignore[attr-defined]
            app.library._update_done_index_entry(rel_path)  # type: ignore[attr-defined]
        else:
            app.library._update_active_index_entry(rel_path)  # type: ignore[attr-defined]
        active_prompts = app.prompt_store.active_prompts()  # type: ignore[attr-defined]
        paper = app.library.build_record_for_rel_path(rel_path, [prompt.slug for prompt in active_prompts])  # type: ignore[attr-defined]
        visible_in_current_view = bool(
            filter_and_sort_papers([paper], folder=current_folder, query=query, sort_by=sort_by, show_done=show_done)
        )

        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        if submit_auto_prompts:
            auto_prompts = [prompt for prompt in active_prompts if prompt.auto_run]
            if auto_prompts:
                submission = app.job_queue.submit(  # type: ignore[attr-defined]
                    [rel_path],
                    [prompt.slug for prompt in auto_prompts],
                    force=False,
                    source="upload",
                )

        message = f"上传成功：{destination.name}"
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
            "paper": serialize_paper_for_view(
                paper,
                current_folder=current_folder,
                query=query,
                sort_by=sort_by,
                active_prompt_count=len(active_prompts),
                show_done=show_done,
            ),
            "visible_in_current_view": visible_in_current_view,
            "submission": submission,
        }

    def flash_submission_summary(result: dict[str, Any], *, action_label: str) -> None:
        if result["queued"]:
            flash(f"{action_label}：已提交 {result['queued']} 个后台任务。", "success")
        if result["existing"]:
            flash(f"{result['existing']} 个任务已在队列中，未重复提交。", "success")
        if result["skipped"]:
            flash(f"{result['skipped']} 个结果已存在，未重复提交。", "success")
        if result["invalid"]:
            flash(f"{result['invalid']} 个任务因 Prompt 缺失而未提交。", "error")

    def selected_source_papers(day_record: Any, selected_ids: list[str]) -> list[Any]:
        paper_map = day_paper_map(day_record)
        normalized_ids = [paper_id.strip() for paper_id in selected_ids if paper_id.strip()]
        if not normalized_ids:
            return list(day_record.papers)
        return [paper_map[paper_id] for paper_id in normalized_ids if paper_id in paper_map]

    def import_source_day_papers(day_record: Any, papers: list[Any]) -> dict[str, Any]:
        target_folder = f"Sources/HuggingFace/{day_record.year}/{day_record.month}/{day_record.day}"
        saved_rel_paths: list[str] = []
        duplicate_count = 0
        error_messages: list[str] = []

        for paper in papers:
            source_pdf_path = local_pdf_path_for(day_record, paper)
            if source_pdf_path is None or not source_pdf_path.exists():
                error_messages.append(f"{paper.paper_id or paper.title} 缺少可用 PDF。")
                continue
            preferred_name = paper.pdf_file_name or f"{paper.paper_id}.pdf"
            try:
                result = app.library.import_external_file(  # type: ignore[attr-defined]
                    source_pdf_path,
                    target_folder,
                    preferred_name=preferred_name,
                )
            except (FileNotFoundError, ValueError) as exc:
                error_messages.append(f"{paper.paper_id or paper.title} 导入失败：{exc}")
                continue

            if result["status"] == "saved" and result["saved_rel_path"]:
                saved_rel_paths.append(result["saved_rel_path"])
            elif result["status"] == "duplicate":
                duplicate_count += 1

        submission = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        if saved_rel_paths:
            auto_prompts = app.prompt_store.auto_prompts()  # type: ignore[attr-defined]
            if auto_prompts:
                submission = app.job_queue.submit(  # type: ignore[attr-defined]
                    saved_rel_paths,
                    [prompt.slug for prompt in auto_prompts],
                    force=False,
                    source="source-import",
                )

        return {
            "target_folder": target_folder,
            "saved_rel_paths": saved_rel_paths,
            "duplicate_count": duplicate_count,
            "error_messages": error_messages,
            "submission": submission,
        }

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

    @app.get("/sources")
    def sources_index() -> Any:
        source_root = Path(app.config["SOURCE_ARCHIVE_ROOT"])
        day_records = load_source_days(source_root)
        source_groups = build_source_groups(day_records)
        return render_template(
            "sources.html",
            source_root=source_root,
            source_groups=source_groups,
            source_day_count=len(day_records),
        )

    @app.get("/sources/open/<run_date>/<paper_id>")
    def source_pdf_route(run_date: str, paper_id: str) -> Any:
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            abort(404)
        paper = day_paper_map(day_record).get(paper_id)
        if paper is None:
            abort(404)
        source_pdf_path = local_pdf_path_for(day_record, paper)
        if source_pdf_path is None or not source_pdf_path.exists() or not source_pdf_path.is_file():
            abort(404)
        return send_file(source_pdf_path, as_attachment=False, download_name=paper.pdf_file_name or source_pdf_path.name)

    @app.post("/sources/download-zip")
    def source_download_zip_route() -> Any:
        run_date = request.form.get("run_date", "").strip()
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            flash("没有找到对应日期的 Source 数据。", "error")
            return redirect(url_for("sources_index"))

        selected_papers = selected_source_papers(day_record, request.form.getlist("paper_ids"))
        valid_papers = []
        for paper in selected_papers:
            source_pdf_path = local_pdf_path_for(day_record, paper)
            if source_pdf_path is None or not source_pdf_path.exists() or not source_pdf_path.is_file():
                continue
            valid_papers.append((paper, source_pdf_path))

        if not valid_papers:
            flash("当前选择中没有可打包的本地 PDF。", "error")
            return redirect(url_for("sources_index"))

        with tempfile.NamedTemporaryFile(prefix="paper-reader-source-", suffix=".zip", delete=False) as handle:
            zip_path = Path(handle.name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(day_record.manifest_path, "manifest.json")
            for index, (paper, source_pdf_path) in enumerate(valid_papers, start=1):
                base_name = paper.pdf_file_name or f"{paper.paper_id}.pdf"
                archive_name = safe_download_name(
                    f"{index:02d}-{paper.paper_id}-{paper.title}.pdf",
                    fallback=base_name,
                )
                archive.write(source_pdf_path, archive_name)

        package_name = f"paper-reader-source-{day_record.run_date}.zip"
        response = send_file(zip_path, as_attachment=True, download_name=package_name, mimetype="application/zip")
        response.call_on_close(lambda: zip_path.unlink(missing_ok=True))
        return response

    @app.post("/sources/import")
    def source_import_route() -> Any:
        run_date = request.form.get("run_date", "").strip()
        day_record = load_source_day(Path(app.config["SOURCE_ARCHIVE_ROOT"]), run_date)
        if day_record is None:
            flash("没有找到对应日期的 Source 数据。", "error")
            return redirect(url_for("sources_index"))

        selected = selected_source_papers(day_record, request.form.getlist("paper_ids"))
        if not selected:
            flash("请至少选择一篇论文。", "error")
            return redirect(url_for("sources_index"))

        summary = import_source_day_papers(day_record, selected)
        if summary["saved_rel_paths"]:
            flash(
                f"已导入 {len(summary['saved_rel_paths'])} 篇论文到 `{summary['target_folder']}`。",
                "success",
            )
            flash_submission_summary(summary["submission"], action_label="自动 Prompt 处理已转为后台任务")
        if summary["duplicate_count"]:
            flash(f"有 {summary['duplicate_count']} 篇论文已存在于阅读器中，已跳过。", "success")
        for message in summary["error_messages"]:
            flash(message, "error")
        return redirect(url_for("sources_index"))

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
