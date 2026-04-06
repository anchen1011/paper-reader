from __future__ import annotations

import json
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from .document_utils import UnsupportedDocumentError

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_USER_PROMPT = (
    "请直接阅读本地论文文件 `{document_path}`，不要先把论文分段再汇总。\n\n"
    "请用通俗、准确、结构化的中文解释这篇论文，重点回答：\n"
    "1. 这篇论文到底想解决什么问题\n"
    "2. 核心思想是什么，为什么这样做有效\n"
    "3. 具体实现方法是什么，按步骤讲清楚输入、关键模块、训练/推理流程\n"
    "4. 实验结果说明了什么\n"
    "5. 这篇论文的优点、局限和适用场景\n\n"
    "输出要求：\n"
    "- 标题\n"
    "- 一句话概括\n"
    "- 核心思想（通俗解释）\n"
    "- 具体实现方法（分步骤）\n"
    "- 实验结果怎么看\n"
    "- 优点与局限\n"
    "- 如果我要自己复现，最该先做什么\n"
)
ProgressCallback = Callable[[int, str], None]
AbortCallback = Callable[[], bool]
ProcessCallback = Callable[[Any], None]
SUPPORTED_SUMMARY_EXTENSIONS = {".pdf", ".doc", ".docx"}
MAX_CODEX_RETRIES = 5


class SafePromptValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_user_prompt(user_prompt: str, document_path: Path) -> str:
    values = SafePromptValues(
        document_name=document_path.name,
        document_path=str(document_path.resolve()),
        document_dir=str(document_path.resolve().parent),
        document_stem=document_path.stem,
        document_suffix=document_path.suffix.lower(),
    )
    rendered = user_prompt.format_map(values).strip()
    if "{document_path}" not in user_prompt and str(document_path.resolve()) not in rendered:
        rendered = (
            f"{rendered}\n\n"
            f"目标论文文件：`{document_path.resolve()}`\n"
            "请直接读取这个本地文件，不要让我再粘贴正文，也不要先自行分段总结。"
        )
    return rendered


def _progress_from_event(line_index: int, payload: dict[str, object] | None) -> tuple[int, str]:
    if payload:
        text = str(payload.get("msg") or payload.get("message") or payload.get("type") or "Codex 正在处理任务。")
    else:
        text = "Codex 正在处理任务。"
    progress = min(90, 15 + line_index * 4)
    return progress, text[:220]


def run_prompt_on_document(
    document_path: Path,
    *,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    progress_callback: ProgressCallback | None = None,
    should_abort: AbortCallback | None = None,
    process_callback: ProcessCallback | None = None,
) -> str:
    document_path = document_path.resolve()
    if not document_path.exists() or not document_path.is_file():
        raise FileNotFoundError(document_path)
    if document_path.suffix.lower() not in SUPPORTED_SUMMARY_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported file type for Codex processing: {document_path.suffix}")
    if shutil.which("codex") is None:
        raise RuntimeError("`codex` command is not available in PATH.")

    final_user_prompt = render_user_prompt(user_prompt, document_path)
    if progress_callback:
        progress_callback(5, "正在启动 Codex YOLO 后台任务。")

    last_error: RuntimeError | None = None
    for attempt in range(1, MAX_CODEX_RETRIES + 1):
        if should_abort and should_abort():
            raise InterruptedError("Job interrupted before Codex launch.")
        with tempfile.TemporaryDirectory(prefix="paper-reader-codex-") as temp_dir:
            process = None
            try:
                output_path = Path(temp_dir) / "codex-last-message.md"
                command = [
                    "codex",
                    "exec",
                    "--json",
                    "--skip-git-repo-check",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--cd",
                    str(document_path.parent),
                    "--model",
                    model or DEFAULT_MODEL,
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
                if process_callback:
                    process_callback(process)

                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("Failed to start Codex process.")

                process.stdin.write(final_user_prompt)
                process.stdin.close()

                last_lines: list[str] = []
                for line_index, raw_line in enumerate(process.stdout, start=1):
                    if should_abort and should_abort():
                        process.terminate()
                        raise InterruptedError("Job interrupted during Codex execution.")
                    line = raw_line.strip()
                    if not line:
                        continue
                    last_lines.append(line)
                    last_lines = last_lines[-20:]
                    payload = None
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        payload = None
                    if progress_callback:
                        progress, message = _progress_from_event(line_index, payload if isinstance(payload, dict) else None)
                        progress_callback(progress, message)

                return_code = process.wait()
                if return_code == 0:
                    if not output_path.exists():
                        raise RuntimeError("Codex finished without writing the final response file.")
                    answer = output_path.read_text(encoding="utf-8").strip()
                    if not answer:
                        raise RuntimeError("Codex returned an empty final response.")
                    if progress_callback:
                        progress_callback(98, "Codex 已完成输出，正在写入结果。")
                    return answer

                detail = "\n".join(last_lines[-8:]).strip()
                last_error = RuntimeError(detail or f"Codex exited with code {return_code}.")
                if "429" not in str(last_error) or attempt >= MAX_CODEX_RETRIES:
                    raise last_error

                if progress_callback:
                    backoff_seconds = 2 ** (attempt - 1)
                    progress_callback(
                        min(95, 70 + attempt * 4),
                        f"遇到 429，正在第 {attempt} 次重试前等待 {backoff_seconds} 秒。",
                    )
                time.sleep(2 ** (attempt - 1))
            finally:
                if process_callback:
                    process_callback(None)

    raise last_error or RuntimeError("Codex execution failed.")



def explain_document(document_path: Path, *, model: str = DEFAULT_MODEL) -> str:
    return run_prompt_on_document(document_path, user_prompt=DEFAULT_USER_PROMPT, model=model)
