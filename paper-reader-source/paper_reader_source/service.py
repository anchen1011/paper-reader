from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import time as time_module
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .huggingface import PaperRecord, filter_papers_by_upvotes, fetch_daily_snapshot

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_SCHEDULE_TIME = time(hour=18, minute=30)
DEFAULT_POLL_SECONDS = 30
DEFAULT_MIN_UPVOTES = 5
DEFAULT_DATA_DIR_NAME = "data/huggingface_daily"
STATE_FILE_NAME = "service_state.json"
MANIFEST_FILE_NAME = "manifest.json"
PDF_SUBDIR_NAME = "papers"
DEFAULT_TIMEOUT_SECONDS = 60
ARXIV_PDF_URLS = (
    "https://arxiv.org/pdf/{paper_id}.pdf",
    "https://arxiv.org/pdf/{paper_id}",
)
DOWNLOAD_USER_AGENT = "paper-reader-source/0.2 (+https://huggingface.co/papers)"


@dataclass(slots=True)
class ServiceState:
    last_successful_run_date_beijing: str | None = None
    last_scheduled_run_date_beijing: str | None = None
    last_saved_file: str | None = None
    last_snapshot_date: str | None = None
    updated_at_utc: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ServiceState":
        return cls(
            last_successful_run_date_beijing=payload.get("last_successful_run_date_beijing")
            or payload.get("last_run_date_beijing"),
            last_scheduled_run_date_beijing=payload.get("last_scheduled_run_date_beijing"),
            last_saved_file=payload.get("last_saved_file"),
            last_snapshot_date=payload.get("last_snapshot_date"),
            updated_at_utc=payload.get("updated_at_utc"),
        )


@dataclass(slots=True)
class DownloadResult:
    pdf_url: str | None
    pdf_rel_path: str | None
    pdf_file_name: str | None
    downloaded: bool
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Paper Reader Source Hugging Face Daily Papers service.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / DEFAULT_DATA_DIR_NAME),
        help="Directory where daily snapshots and PDFs are saved.",
    )
    parser.add_argument(
        "--min-upvotes",
        type=int,
        default=DEFAULT_MIN_UPVOTES,
        help="Keep papers with upvotes greater than or equal to this value. Default: 5.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="How often the scheduler wakes up to check whether it should run. Default: 30.",
    )
    parser.add_argument(
        "--run-on-start",
        action="store_true",
        help="Collect one snapshot immediately on startup before switching to the daily schedule.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after the first collection run.",
    )
    return parser



def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    log(
        "service_start",
        data_dir=str(data_dir),
        min_upvotes=args.min_upvotes,
        schedule_time=DEFAULT_SCHEDULE_TIME.strftime("%H:%M"),
        timezone=str(BEIJING_TZ),
    )

    state = load_state(data_dir)

    if args.run_on_start:
        state = run_collection(
            data_dir=data_dir,
            min_upvotes=args.min_upvotes,
            run_reason="startup",
            previous_state=state,
        )
        if args.once:
            return 0

    last_announced_next_run: str | None = None

    while True:
        now_beijing = datetime.now(BEIJING_TZ)
        today_str = now_beijing.date().isoformat()
        scheduled_for_today = combine_beijing(now_beijing.date(), DEFAULT_SCHEDULE_TIME)

        if now_beijing >= scheduled_for_today and state.last_scheduled_run_date_beijing != today_str:
            try:
                state = run_collection(
                    data_dir=data_dir,
                    min_upvotes=args.min_upvotes,
                    run_reason="scheduled",
                    previous_state=state,
                )
            except Exception as exc:  # pragma: no cover - operational safety path
                log("run_failed", error=str(exc))
                time_module.sleep(max(30, args.poll_seconds))
                continue

            if args.once:
                return 0

            last_announced_next_run = None
            continue

        next_run = next_run_after(now_beijing)
        next_run_key = next_run.isoformat()
        if next_run_key != last_announced_next_run:
            seconds_until = max(0, int((next_run - now_beijing).total_seconds()))
            log(
                "waiting",
                now_beijing=now_beijing.isoformat(),
                next_run_beijing=next_run.isoformat(),
                seconds_until_next_run=seconds_until,
                last_scheduled_run_date_beijing=state.last_scheduled_run_date_beijing,
            )
            last_announced_next_run = next_run_key

        sleep_seconds = min(max(5, args.poll_seconds), max(5, int((next_run - now_beijing).total_seconds())))
        time_module.sleep(sleep_seconds)



def run_collection(
    *,
    data_dir: Path,
    min_upvotes: int,
    run_reason: str,
    previous_state: ServiceState | None = None,
) -> ServiceState:
    now_beijing = datetime.now(BEIJING_TZ)
    now_utc = datetime.now(timezone.utc)
    snapshot = fetch_daily_snapshot()
    filtered = filter_papers_by_upvotes(snapshot, min_upvotes=min_upvotes, inclusive=True)

    day_dir = day_directory(data_dir, now_beijing.date())
    pdf_dir = day_dir / PDF_SUBDIR_NAME
    pdf_dir.mkdir(parents=True, exist_ok=True)

    manifest_papers: list[dict[str, Any]] = []
    downloaded_count = 0
    existing_count = 0
    failed_count = 0
    for paper in filtered:
        download_result = ensure_pdf_downloaded(pdf_dir, paper)
        if download_result.downloaded:
            if download_result.error == "already_exists":
                existing_count += 1
            else:
                downloaded_count += 1
        else:
            failed_count += 1

        paper_payload = paper.to_dict()
        paper_payload.update(
            {
                "pdf_url": download_result.pdf_url,
                "pdf_rel_path": download_result.pdf_rel_path,
                "pdf_file_name": download_result.pdf_file_name,
                "pdf_downloaded": download_result.downloaded,
            }
        )
        if download_result.error and download_result.error != "already_exists":
            paper_payload["pdf_error"] = download_result.error
        manifest_papers.append(paper_payload)

    manifest_path = day_dir / MANIFEST_FILE_NAME
    payload = {
        "run_reason": run_reason,
        "run_date_beijing": now_beijing.date().isoformat(),
        "saved_at_beijing": now_beijing.isoformat(),
        "saved_at_utc": now_utc.isoformat(),
        "schedule_timezone": str(BEIJING_TZ),
        "schedule_time_beijing": DEFAULT_SCHEDULE_TIME.strftime("%H:%M"),
        "source": "huggingface_daily_papers",
        "source_url": snapshot.source_url,
        "snapshot_date": snapshot.date_string,
        "filter": {
            "field": "upvotes",
            "operator": ">=",
            "value": min_upvotes,
        },
        "paper_count": len(manifest_papers),
        "download_summary": {
            "downloaded": downloaded_count,
            "existing": existing_count,
            "failed": failed_count,
        },
        "papers": manifest_papers,
    }
    write_json_atomic(manifest_path, payload)

    state = ServiceState(
        last_successful_run_date_beijing=now_beijing.date().isoformat(),
        last_scheduled_run_date_beijing=(
            now_beijing.date().isoformat()
            if run_reason == "scheduled"
            else (previous_state.last_scheduled_run_date_beijing if previous_state else None)
        ),
        last_saved_file=str(manifest_path),
        last_snapshot_date=snapshot.date_string,
        updated_at_utc=now_utc.isoformat(),
    )
    write_json_atomic(data_dir / STATE_FILE_NAME, asdict(state))

    log(
        "run_complete",
        run_reason=run_reason,
        run_date_beijing=state.last_successful_run_date_beijing,
        snapshot_date=snapshot.date_string,
        saved_file=str(manifest_path),
        paper_count=len(manifest_papers),
        pdf_downloaded=downloaded_count,
        pdf_existing=existing_count,
        pdf_failed=failed_count,
    )
    return state



def day_directory(data_dir: Path, run_date: date) -> Path:
    return data_dir / run_date.strftime("%Y") / run_date.strftime("%m") / run_date.strftime("%d")



def ensure_pdf_downloaded(pdf_dir: Path, paper: PaperRecord) -> DownloadResult:
    pdf_file_name = f"{paper.paper_id}.pdf"
    destination = pdf_dir / pdf_file_name
    if destination.exists() and destination.stat().st_size > 0:
        return DownloadResult(
            pdf_url=ARXIV_PDF_URLS[0].format(paper_id=paper.paper_id),
            pdf_rel_path=f"{PDF_SUBDIR_NAME}/{pdf_file_name}",
            pdf_file_name=pdf_file_name,
            downloaded=True,
            error="already_exists",
        )

    last_error: str | None = None
    for template in ARXIV_PDF_URLS:
        pdf_url = template.format(paper_id=paper.paper_id)
        try:
            download_pdf(pdf_url, destination)
            return DownloadResult(
                pdf_url=pdf_url,
                pdf_rel_path=f"{PDF_SUBDIR_NAME}/{pdf_file_name}",
                pdf_file_name=pdf_file_name,
                downloaded=True,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            destination.unlink(missing_ok=True)

    return DownloadResult(
        pdf_url=ARXIV_PDF_URLS[0].format(paper_id=paper.paper_id),
        pdf_rel_path=None,
        pdf_file_name=pdf_file_name,
        downloaded=False,
        error=last_error or "unknown download failure",
    )



def download_pdf(pdf_url: str, destination: Path) -> None:
    request = Request(pdf_url, headers={"User-Agent": DOWNLOAD_USER_AGENT})
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                raise RuntimeError(f"Unexpected content type for {pdf_url}: {content_type}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {pdf_url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {pdf_url}: {exc.reason}") from exc



def load_state(data_dir: Path) -> ServiceState:
    state_path = data_dir / STATE_FILE_NAME
    if not state_path.exists():
        return ServiceState()

    try:
        return ServiceState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        log("state_load_failed", path=str(state_path), error=str(exc))
        return ServiceState()



def next_run_after(now_beijing: datetime) -> datetime:
    today_target = combine_beijing(now_beijing.date(), DEFAULT_SCHEDULE_TIME)
    if now_beijing < today_target:
        return today_target
    return combine_beijing(now_beijing.date() + timedelta(days=1), DEFAULT_SCHEDULE_TIME)



def combine_beijing(run_date: date, run_time: time) -> datetime:
    return datetime.combine(run_date, run_time, tzinfo=BEIJING_TZ)



def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)



def log(event: str, **fields: Any) -> None:
    now = datetime.now(timezone.utc).isoformat()
    if fields:
        details = " ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in fields.items())
        print(f"[{now}] {event} {details}", flush=True)
        return
    print(f"[{now}] {event}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
