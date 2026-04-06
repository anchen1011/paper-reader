from __future__ import annotations

import json
import os
import queue
import signal
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .prompt_manager import PromptStore

JOB_STORE_NAME = ".paper-reader-jobs.json"
ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"completed", "failed", "skipped", "stopped"}
DEFAULT_MAX_CONCURRENCY = 12
MAX_WORKER_THREADS = 32
MAX_PERSISTED_TERMINAL_JOBS = 400


@dataclass
class JobRecord:
    id: str
    rel_path: str
    file_name: str
    prompt_slug: str
    prompt_name: str
    model: str
    force: bool
    source: str
    status: str
    progress: int
    message: str
    error: str | None
    result_rel_path: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None


class PaperJobQueue:
    def __init__(self, library: Any, prompt_store: PromptStore, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
        self.library = library
        self.prompt_store = prompt_store
        self.max_concurrency = max(1, int(max_concurrency))
        self.state_path = self.library.root / JOB_STORE_NAME
        self._lock = threading.Lock()
        self._slot_condition = threading.Condition()
        self._active_executions = 0
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, Any] = {}
        self._cancelled_jobs: set[str] = set()
        self._load_state()
        self._workers: list[threading.Thread] = []
        for index in range(MAX_WORKER_THREADS):
            worker = threading.Thread(
                target=self._run_loop,
                name=f"paper-reader-worker-{index + 1}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _load_state(self) -> None:
        payload: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}

        now = datetime.utcnow().isoformat(timespec="seconds")
        for item in payload.get("jobs", []):
            try:
                job = JobRecord(**item)
            except TypeError:
                continue
            if job.status in ACTIVE_STATUSES:
                job.status = "queued"
                job.progress = min(job.progress, 5)
                job.message = "服务重启后已重新排队。"
                job.updated_at = now
            self._jobs[job.id] = job

        self._persist_locked()
        for job in self._jobs.values():
            if job.status == "queued":
                self._queue.put(job.id)

    def _persist_locked(self) -> None:
        terminal_jobs = sorted(
            (job for job in self._jobs.values() if job.status in TERMINAL_STATUSES),
            key=lambda item: (item.updated_at, item.created_at),
            reverse=True,
        )
        keep_terminal_ids = {job.id for job in terminal_jobs[:MAX_PERSISTED_TERMINAL_JOBS]}
        self._jobs = {
            job_id: job
            for job_id, job in self._jobs.items()
            if job.status in ACTIVE_STATUSES or job_id in keep_terminal_ids
        }
        payload = {
            "jobs": [asdict(job) for job in sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)],
        }
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.state_path)

    def _timestamp(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def _create_job(
        self,
        *,
        rel_path: str,
        prompt_slug: str,
        prompt_name: str,
        model: str,
        force: bool,
        source: str,
        status: str,
        progress: int,
        message: str,
        error: str | None = None,
        result_rel_path: str | None = None,
    ) -> JobRecord:
        now = self._timestamp()
        return JobRecord(
            id=uuid.uuid4().hex[:12],
            rel_path=rel_path,
            file_name=Path(rel_path).name,
            prompt_slug=prompt_slug,
            prompt_name=prompt_name,
            model=model,
            force=force,
            source=source,
            status=status,
            progress=progress,
            message=message,
            error=error,
            result_rel_path=result_rel_path,
            created_at=now,
            updated_at=now,
            started_at=now if status == "running" else None,
            finished_at=now if status in TERMINAL_STATUSES else None,
        )

    def _find_active_duplicate(self, rel_path: str, prompt_slug: str, force: bool) -> JobRecord | None:
        for job in sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True):
            if job.rel_path == rel_path and job.prompt_slug == prompt_slug and job.status in ACTIVE_STATUSES and job.force == force:
                return job
        return None

    def submit(
        self,
        rel_paths: list[str],
        prompt_slugs: list[str],
        *,
        force: bool = False,
        source: str = "manual",
    ) -> dict[str, Any]:
        prompt_map = {prompt.slug: prompt for prompt in self.prompt_store.list_prompts()}
        result = {"queued": 0, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}
        unique_rel_paths = list(dict.fromkeys(path for path in rel_paths if path))
        unique_prompt_slugs = list(dict.fromkeys(slug for slug in prompt_slugs if slug))

        with self._lock:
            for rel_path in unique_rel_paths:
                for prompt_slug in unique_prompt_slugs:
                    prompt = prompt_map.get(prompt_slug)
                    if prompt is None:
                        result["invalid"] += 1
                        continue

                    duplicate = self._find_active_duplicate(rel_path, prompt_slug, force)
                    if duplicate is not None:
                        result["existing"] += 1
                        result["job_ids"].append(duplicate.id)
                        result["jobs"].append(asdict(duplicate))
                        continue

                    prompt_info = self.library.prompt_result_info(rel_path, prompt_slug)
                    if prompt_info["exists"] and not force:
                        job = self._create_job(
                            rel_path=rel_path,
                            prompt_slug=prompt_slug,
                            prompt_name=prompt.name,
                            model=prompt.model,
                            force=force,
                            source=source,
                            status="skipped",
                            progress=100,
                            message="结果已存在，未重复提交。",
                            result_rel_path=prompt_info["result_rel_path"],
                        )
                        self._jobs[job.id] = job
                        result["skipped"] += 1
                        result["job_ids"].append(job.id)
                        result["jobs"].append(asdict(job))
                        continue

                    job = self._create_job(
                        rel_path=rel_path,
                        prompt_slug=prompt_slug,
                        prompt_name=prompt.name,
                        model=prompt.model,
                        force=force,
                        source=source,
                        status="queued",
                        progress=0,
                        message="任务已提交，等待处理。",
                    )
                    self._jobs[job.id] = job
                    self._queue.put(job.id)
                    result["queued"] += 1
                    result["job_ids"].append(job.id)
                    result["jobs"].append(asdict(job))
            self._persist_locked()
        return result

    def _update_job(self, job_id: str, **changes: Any) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = self._timestamp()
            if job.status == "running" and job.started_at is None:
                job.started_at = job.updated_at
            if job.status in TERMINAL_STATUSES and job.finished_at is None:
                job.finished_at = job.updated_at
            self._persist_locked()
            return job

    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES or job.status == "stopped":
                return

        if not self._acquire_execution_slot(job_id):
            return

        try:
            job = self._update_job(job_id, status="running", progress=1, message="任务开始执行。")
            if job is None or self._is_cancelled(job_id):
                self._mark_stopped(job_id)
                return

            prompt = self.prompt_store.get_prompt(job.prompt_slug)
            if prompt is None:
                self._update_job(job_id, status="failed", progress=100, message="Prompt 不存在。", error="Prompt missing")
                return

            def report(progress: int, message: str) -> None:
                if self._is_cancelled(job_id):
                    raise InterruptedError("Job interrupted.")
                self._update_job(job_id, status="running", progress=max(1, min(progress, 99)), message=message)

            def process_callback(process: Any) -> None:
                with self._lock:
                    if process is None:
                        self._processes.pop(job_id, None)
                    else:
                        self._processes[job_id] = process

            result_path, generated = self.library.generate_prompt_result(
                job.rel_path,
                prompt,
                force=job.force,
                progress_callback=report,
                should_abort=lambda: self._is_cancelled(job_id),
                process_callback=process_callback,
            )
        except InterruptedError:
            self._mark_stopped(job_id)
            return
        except Exception as exc:
            if self._is_cancelled(job_id):
                self._mark_stopped(job_id)
                return
            self._update_job(
                job_id,
                status="failed",
                progress=100,
                message="任务执行失败。",
                error=str(exc),
            )
            return
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
            self._release_execution_slot()

        info = self.library.prompt_result_info(job.rel_path, job.prompt_slug)
        self._update_job(
            job_id,
            status="completed" if generated else "skipped",
            progress=100,
            message="任务完成。" if generated else "结果已存在，未重复生成。",
            error=None,
            result_rel_path=info["result_rel_path"] or result_path.relative_to(self.library.root).as_posix(),
        )

    def _acquire_execution_slot(self, job_id: str) -> bool:
        while True:
            with self._slot_condition:
                if self._is_cancelled(job_id):
                    return False
                if self._active_executions < self.max_concurrency:
                    self._active_executions += 1
                    return True
                self._slot_condition.wait(timeout=0.5)

    def _release_execution_slot(self) -> None:
        with self._slot_condition:
            self._active_executions = max(0, self._active_executions - 1)
            self._slot_condition.notify_all()

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled_jobs

    def _mark_stopped(self, job_id: str) -> None:
        self._update_job(
            job_id,
            status="stopped",
            progress=100,
            message="任务已停止。",
            error="Interrupted",
        )

    def list_jobs(self, *, limit: int = 80, rel_path: str | None = None) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())
        if rel_path:
            jobs = [job for job in jobs if job.rel_path == rel_path]
        jobs.sort(key=lambda item: (item.updated_at, item.created_at), reverse=True)
        return jobs[:limit]

    def update_max_concurrency(self, value: int) -> int:
        with self._slot_condition:
            self.max_concurrency = max(1, int(value))
            self._slot_condition.notify_all()
        return self.max_concurrency

    def stop_all(self) -> dict[str, int]:
        running_pids: list[int] = []
        summary = {"queued": 0, "running": 0}
        with self._lock:
            for job in self._jobs.values():
                if job.status == "queued":
                    self._cancelled_jobs.add(job.id)
                    summary["queued"] += 1
                    job.status = "stopped"
                    job.progress = 100
                    job.message = "任务已停止。"
                    job.error = "Interrupted"
                    job.finished_at = self._timestamp()
                    job.updated_at = job.finished_at
                elif job.status == "running":
                    self._cancelled_jobs.add(job.id)
                    summary["running"] += 1
                    job.message = "正在停止任务..."
                    job.updated_at = self._timestamp()
                    process = self._processes.get(job.id)
                    if process is not None and getattr(process, "pid", None):
                        running_pids.append(int(process.pid))
            self._persist_locked()

        with self._slot_condition:
            self._slot_condition.notify_all()

        for pid in running_pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
            except OSError:
                continue

        return summary

    def latest_job_for(self, rel_path: str, prompt_slug: str) -> JobRecord | None:
        jobs = self.list_jobs(limit=200, rel_path=rel_path)
        for job in jobs:
            if job.prompt_slug == prompt_slug:
                return job
        return None

    def snapshot(self, *, rel_path: str | None = None, limit: int = 32) -> dict[str, Any]:
        jobs = self.list_jobs(limit=limit, rel_path=rel_path)
        finished_durations = [duration for duration in (self._job_duration_seconds(job) for job in jobs) if duration is not None]
        with self._slot_condition:
            active_executions = self._active_executions
        return {
            "jobs": [asdict(job) for job in jobs],
            "active_count": sum(1 for job in jobs if job.status in ACTIVE_STATUSES),
            "queued_count": sum(1 for job in jobs if job.status == "queued"),
            "running_count": sum(1 for job in jobs if job.status == "running"),
            "average_duration_seconds": (
                round(sum(finished_durations) / len(finished_durations), 2) if finished_durations else None
            ),
            "max_concurrency": self.max_concurrency,
            "active_executions": active_executions,
        }

    def _job_duration_seconds(self, job: JobRecord) -> float | None:
        if not job.started_at or not job.finished_at:
            return None
        try:
            started_at = datetime.fromisoformat(job.started_at)
            finished_at = datetime.fromisoformat(job.finished_at)
        except ValueError:
            return None
        return max(0.0, (finished_at - started_at).total_seconds())
