from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from src.paper_reader.app import create_app
from src.paper_reader.markdown_render import render_markdown


DOCX_CONTENT_TYPES = """<?xml version='1.0' encoding='UTF-8'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
  <Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
  <Default Extension='xml' ContentType='application/xml'/>
  <Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>
  <Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>
</Types>
"""

DOCX_RELS = """<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>
  <Relationship Id='rId2' Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties' Target='docProps/core.xml'/>
</Relationships>
"""

DOCX_DOC = """<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
  <w:body>
    <w:p><w:r><w:t>{title}</w:t></w:r></w:p>
    <w:p><w:r><w:t>{body}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""

DOCX_CORE = """<?xml version='1.0' encoding='UTF-8'?>
<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
 xmlns:dc='http://purl.org/dc/elements/1.1/'>
  <dc:title>{title}</dc:title>
</cp:coreProperties>
"""


class PaperReaderAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.library = Path(self.tempdir.name)
        self.app = create_app(self.library)
        self.app.testing = True
        self.client = self.app.test_client()
        with self.client.session_transaction() as session:
            session["authenticated"] = True
            session["username"] = "admin"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_pdf(self, path: Path, title: str) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_metadata({"/Title": title})
        with path.open("wb") as handle:
            writer.write(handle)

    def make_docx(self, path: Path, title: str, body: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", DOCX_CONTENT_TYPES)
            archive.writestr("_rels/.rels", DOCX_RELS)
            archive.writestr("word/document.xml", DOCX_DOC.format(title=title, body=body))
            archive.writestr("docProps/core.xml", DOCX_CORE.format(title=title))

    def create_prompt(self, slug: str, name: str) -> None:
        self.app.prompt_store.save_prompt(
            existing_slug=None,
            name=name,
            slug=slug,
            user_prompt="请直接阅读 `{document_path}`，总结这篇论文。",
            model="gpt-5.4",
            enabled=True,
            auto_run=False,
        )

    def test_login_required_for_index(self) -> None:
        client = self.app.test_client()
        response = client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_login_allows_access_with_correct_credentials(self) -> None:
        client = self.app.test_client()
        response = client.post(
            "/login",
            data={
                "username": "admin",
                "password": "paperpaperreaderreader12678",
                "next": "/",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")

    def test_login_locks_for_five_minutes_after_three_failures(self) -> None:
        client = self.app.test_client()
        guard = self.app.login_guard
        original_now = guard._now
        timeline = {"value": 1000.0}
        guard._now = lambda: timeline["value"]
        self.addCleanup(setattr, guard, "_now", original_now)

        for _ in range(3):
            response = client.post(
                "/login",
                data={"username": "admin", "password": "wrong", "next": "/"},
                follow_redirects=True,
            )

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("已锁定 5 分钟", html)

        blocked = client.post(
            "/login",
            data={"username": "admin", "password": "paperpaperreaderreader12678", "next": "/"},
            follow_redirects=True,
        )
        self.assertIn("当前已锁定", blocked.get_data(as_text=True))

        timeline["value"] += 301
        success = client.post(
            "/login",
            data={"username": "admin", "password": "paperpaperreaderreader12678", "next": "/"},
            follow_redirects=False,
        )
        self.assertEqual(success.status_code, 302)
        self.assertEqual(success.headers["Location"], "/")

    def test_index_lists_existing_pdf_docx_and_default_prompt(self) -> None:
        self.make_pdf(self.library / "2501.12948.pdf", "DeepSeek-R1")
        self.make_docx(self.library / "notes.docx", "RL Notes", "submitted on 25 Jan 2025")

        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("DeepSeek-R1", html)
        self.assertIn("RL Notes", html)
        self.assertIn("2025-01", html)
        self.assertIn("核心解读", html)

    def test_upload_saves_supported_file_and_triggers_auto_prompts(self) -> None:
        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        upload_bytes.seek(0)

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/upload",
                data={
                    "target_folder": "arxiv/2025",
                    "files": (upload_bytes, "paper.pdf"),
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.library / "arxiv" / "2025" / "paper.pdf").exists())
        mocked.assert_called_once_with(["arxiv/2025/paper.pdf"], ["core-zh"], force=False, source="upload")

    def test_upload_file_endpoint_returns_json_for_single_success(self) -> None:
        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        upload_bytes.seek(0)

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/upload-file",
                data={
                    "target_folder": "incoming",
                    "folder": "incoming",
                    "q": "",
                    "sort": "date_desc",
                    "file": (upload_bytes, "single.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "saved")
        self.assertEqual(response.json["saved_rel_path"], "incoming/single.pdf")
        self.assertTrue(response.json["visible_in_current_view"])
        self.assertEqual(response.json["paper"]["file_name"], "single.pdf")
        mocked.assert_called_once_with(["incoming/single.pdf"], ["core-zh"], force=False, source="upload")

    def test_upload_file_endpoint_skips_duplicate_content(self) -> None:
        original = io.BytesIO(b"same-content")
        duplicate = io.BytesIO(b"same-content")
        (self.library / "existing.pdf").write_bytes(original.getvalue())

        with patch.object(self.app.job_queue, "submit") as mocked:
            response = self.client.post(
                "/upload-file",
                data={
                    "target_folder": "",
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "file": (duplicate, "renamed.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "duplicate")
        self.assertEqual(response.json["duplicate_rel_path"], "existing.pdf")
        self.assertFalse((self.library / "renamed.pdf").exists())
        mocked.assert_not_called()

    def test_upload_route_keeps_successful_files_when_some_fail(self) -> None:
        upload_bytes = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(upload_bytes)
        upload_bytes.seek(0)

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/upload",
                data={
                    "target_folder": "mixed",
                    "folder": "mixed",
                    "q": "",
                    "sort": "date_desc",
                    "files": [
                        (upload_bytes, "ok.pdf"),
                        (io.BytesIO(b"bad"), "bad.txt"),
                    ],
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.library / "mixed" / "ok.pdf").exists())
        self.assertFalse((self.library / "mixed" / "bad.txt").exists())
        mocked.assert_called_once_with(["mixed/ok.pdf"], ["core-zh"], force=False, source="upload")

    def test_prompt_save_route_creates_custom_prompt(self) -> None:
        response = self.client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "方法拆解",
                "slug": "method-breakdown",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
                "user_prompt": "请直接阅读 `{document_path}`，从实现角度解释这篇论文。",
            },
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("方法拆解", html)
        self.assertIsNotNone(self.app.prompt_store.get_prompt("method-breakdown"))

    def test_prompt_save_route_auto_generates_slug_for_chinese_name(self) -> None:
        first = self.client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "实验摘要",
                "slug": "",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
                "user_prompt": "请直接阅读 `{document_path}`，总结实验部分。",
            },
            follow_redirects=True,
        )
        second = self.client.post(
            "/prompt-save",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "name": "复现建议",
                "slug": "",
                "model": "gpt-5.4",
                "enabled": "on",
                "auto_run": "on",
                "user_prompt": "请直接阅读 `{document_path}`，给出复现建议。",
            },
            follow_redirects=True,
        )

        prompts = self.app.prompt_store.list_prompts()
        slugs = [prompt.slug for prompt in prompts]

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("prompt", slugs)
        self.assertIn("prompt-2", slugs)
        self.assertEqual(len(slugs), len(set(slugs)))

    def test_prompt_store_can_manage_multiple_prompts(self) -> None:
        self.app.prompt_store.save_prompt(
            existing_slug=None,
            name="方法拆解",
            slug="method-breakdown",
            user_prompt="请直接阅读 `{document_path}`，解释方法。",
            model="gpt-5.4",
            enabled=True,
            auto_run=True,
        )
        self.app.prompt_store.save_prompt(
            existing_slug=None,
            name="实验摘要",
            slug="experiment-summary",
            user_prompt="请直接阅读 `{document_path}`，解释实验。",
            model="gpt-5.4",
            enabled=False,
            auto_run=False,
        )

        prompts = self.app.prompt_store.list_prompts()
        names = [prompt.name for prompt in prompts]

        self.assertIn("核心解读", names)
        self.assertIn("方法拆解", names)
        self.assertIn("实验摘要", names)
        self.assertEqual(len(prompts), 3)

    def test_prompt_missing_tab_shows_empty_state(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        self.create_prompt("experiment-read", "实验摘要")

        response = self.client.get("/?paper=paper.pdf&tab=experiment-read")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("这个 Prompt 还没有应用到当前论文上", html)
        self.assertIn("实验摘要", html)

    def test_sidebar_groups_by_year_month_and_done_toggle(self) -> None:
        self.make_pdf(self.library / "2025-paper.pdf", "2025 Paper")
        self.make_docx(self.library / "2024-notes.docx", "2024 Notes", "submitted on 11 Dec 2024")

        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("paper-year-group", html)
        self.assertIn("2025", html)
        self.assertIn("2024-12", html)
        self.assertIn("显示 DONE 论文", html)

    def test_done_toggle_moves_file_and_hides_it_by_default(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Done Me")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("# Cached\n", encoding="utf-8")

        response = self.client.post(
            "/done-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "",
                "tab": "source",
                "rel_path": "paper.pdf",
            },
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertFalse((self.library / "paper.pdf").exists())
        self.assertTrue((self.library / "DONE" / "paper.pdf").exists())
        self.assertTrue(self.app.library.prompt_result_path_for("DONE/paper.pdf", prompt.slug).exists())
        self.assertNotIn("Done Me", html)

        show_done_response = self.client.get("/?show_done=1")
        self.assertIn("Done Me", show_done_response.get_data(as_text=True))

    def test_done_toggle_can_restore_paper(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Restore Me")
        done_path = self.app.library.toggle_done("paper.pdf")

        response = self.client.post(
            "/done-toggle",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "show_done": "1",
                "tab": "source",
                "rel_path": done_path,
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.library / "paper.pdf").exists())
        self.assertFalse((self.library / done_path).exists())

    def test_batch_section_hides_done_papers_even_when_show_done_enabled(self) -> None:
        self.make_pdf(self.library / "todo.pdf", "Todo Paper")
        self.make_pdf(self.library / "done.pdf", "Done Paper")
        self.app.library.toggle_done("done.pdf")

        response = self.client.get("/tool-panels/batch-run?show_done=1")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Todo Paper", html)
        self.assertNotIn("Done Paper", html)

    def test_batch_section_shows_select_all_controls(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Prompt Target")

        response = self.client.get("/tool-panels/batch-run")
        html = response.get_data(as_text=True)

        self.assertIn('data-check-action="all" data-check-group="prompt"', html)
        self.assertIn('data-check-action="all" data-check-group="paper"', html)
        self.assertIn('data-check-action="none" data-check-group="prompt"', html)
        self.assertIn('data-check-action="none" data-check-group="paper"', html)

    def test_prompt_tab_renders_markdown_html(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            "# 标题\n\n## 小节\n\n- 列表项\n\n包含 **加粗** 和 `code`。\n\n<script>alert('x')</script>\n",
            encoding="utf-8",
        )

        response = self.client.get("/?paper=paper.pdf&tab=core-zh")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('<article class="markdown-viewer markdown-render">', html)
        self.assertIn("<h1>标题</h1>", html)
        self.assertIn("<h2>小节</h2>", html)
        self.assertIn("<strong>加粗</strong>", html)
        self.assertIn("<code>code</code>", html)
        self.assertIn("&lt;script&gt;alert", html)
        self.assertNotIn("<script>alert", html)

    def test_generate_prompt_result_skips_existing_markdown(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        existing_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("# Existing\n", encoding="utf-8")

        with patch("src.paper_reader.app.run_prompt_on_document", side_effect=AssertionError("should not run")):
            path, generated = self.app.library.generate_prompt_result("paper.pdf", prompt, force=False)

        self.assertFalse(generated)
        self.assertEqual(path, existing_path)

    def test_rename_and_delete_file_move_prompt_results(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        prompt = self.app.prompt_store.get_prompt("core-zh")
        assert prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("# Cached\n", encoding="utf-8")

        rename_response = self.client.post(
            "/rename",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "tab": prompt.slug,
                "rel_path": "paper.pdf",
                "new_name": "paper-renamed.pdf",
            },
            follow_redirects=True,
        )
        self.assertEqual(rename_response.status_code, 200)
        moved_result = self.app.library.prompt_result_path_for("paper-renamed.pdf", prompt.slug)
        self.assertTrue((self.library / "paper-renamed.pdf").exists())
        self.assertTrue(moved_result.exists())
        self.assertFalse(result_path.exists())

        delete_response = self.client.post(
            "/delete",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "rel_path": "paper-renamed.pdf",
            },
            follow_redirects=True,
        )
        self.assertEqual(delete_response.status_code, 200)
        self.assertFalse((self.library / "paper-renamed.pdf").exists())
        self.assertFalse(moved_result.exists())

    def test_prompt_batch_route_runs_selected_prompts(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        self.create_prompt("method-breakdown", "方法拆解")

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 2, "existing": 0, "skipped": 1, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/prompt-batch-run",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "paper.pdf",
                    "tab": "source",
                    "rel_paths": ["paper.pdf"],
                    "prompt_slugs": ["core-zh", "method-breakdown"],
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(["paper.pdf"], ["core-zh", "method-breakdown"], force=False, source="batch")

    def test_prompt_run_route_submits_background_job(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")

        with patch.object(self.app.job_queue, "submit", return_value={"queued": 1, "existing": 0, "skipped": 0, "invalid": 0, "job_ids": [], "jobs": []}) as mocked:
            response = self.client.post(
                "/prompt-run",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "rel_path": "paper.pdf",
                    "prompt_slug": "core-zh",
                },
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(["paper.pdf"], ["core-zh"], force=False, source="manual")

    def test_serve_file_tolerates_zoom_fragment_encoded_in_path(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Zoom Safe")

        response = self.client.get("/files/paper.pdf%23zoom=page-width")
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")

    def test_offline_package_route_builds_complete_zip_bundle(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Offline PDF")
        self.make_docx(self.library / "notes.docx", "Offline Notes", "docx body for offline preview")
        self.create_prompt("method-breakdown", "方法拆解")

        core_prompt = self.app.prompt_store.get_prompt("core-zh")
        assert core_prompt is not None
        result_path = self.app.library.prompt_result_path_for("paper.pdf", core_prompt.slug)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("# 离线摘要\n\n- 要点一\n", encoding="utf-8")

        response = self.client.post(
            "/offline-package",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "paper.pdf",
                "tab": "source",
                "rel_paths": ["paper.pdf", "notes.docx"],
            },
        )
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")

        archive = zipfile.ZipFile(io.BytesIO(response.data))
        names = set(archive.namelist())

        self.assertIn("index.html", names)
        self.assertIn("manifest.json", names)
        self.assertIn("assets/style.css", names)
        self.assertIn("assets/offline-reader.css", names)
        self.assertIn("assets/offline-reader.js", names)
        self.assertIn("papers/paper.pdf", names)
        self.assertIn("papers/notes.docx", names)
        self.assertIn("prompt-results/paper.pdf/core-zh.md", names)

        index_html = archive.read("index.html").decode("utf-8")
        manifest_payload = archive.read("manifest.json").decode("utf-8")

        self.assertIn("论文离线阅读包", index_html)
        self.assertIn("offline-manifest", index_html)
        self.assertIn("Offline PDF", manifest_payload)
        self.assertIn("Offline Notes", manifest_payload)

    def test_offline_package_route_ignores_done_papers(self) -> None:
        self.make_pdf(self.library / "todo.pdf", "Todo Export")
        self.make_pdf(self.library / "done.pdf", "Done Export")
        done_rel_path = self.app.library.toggle_done("done.pdf")

        response = self.client.post(
            "/offline-package",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "todo.pdf",
                "tab": "source",
                "rel_paths": ["todo.pdf", done_rel_path],
            },
        )
        self.addCleanup(response.close)

        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(response.data))
        names = set(archive.namelist())

        self.assertIn("papers/todo.pdf", names)
        self.assertNotIn(f"papers/{done_rel_path}", names)

    def test_jobs_status_endpoint_returns_snapshot(self) -> None:
        self.make_pdf(self.library / "paper.pdf", "Test Title")
        with patch.object(self.app.job_queue, "snapshot", return_value={"jobs": [{"id": "job-1", "status": "queued"}], "active_count": 1, "queued_count": 1, "running_count": 0, "average_duration_seconds": 12.5, "max_concurrency": 32, "active_executions": 0}) as mocked:
            response = self.client.get("/jobs/status?paper=paper.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["active_count"], 1)
        self.assertEqual(response.json["max_concurrency"], 32)
        mocked.assert_called_once_with(rel_path="paper.pdf")

    def test_job_queue_defaults_to_12_workers(self) -> None:
        self.assertEqual(self.app.job_queue.max_concurrency, 12)
        self.assertEqual(len(self.app.job_queue._workers), 32)

    def test_jobs_config_route_updates_max_concurrency(self) -> None:
        response = self.client.post(
            "/jobs/config",
            data={
                "folder": "",
                "q": "",
                "sort": "date_desc",
                "paper": "",
                "tab": "source",
                "max_concurrency": "7",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.app.job_queue.max_concurrency, 7)
        self.assertEqual(self.app.settings_store.max_concurrency(), 7)

    def test_jobs_stop_all_route_interrupts_running_and_queued_jobs(self) -> None:
        self.make_pdf(self.library / "paper-a.pdf", "Paper A")
        self.make_pdf(self.library / "paper-b.pdf", "Paper B")
        self.app.job_queue.update_max_concurrency(1)

        started = threading.Event()
        release = threading.Event()

        def slow_generate(rel_path, prompt, *, force=False, progress_callback=None, should_abort=None, process_callback=None):
            started.set()
            while not release.is_set():
                if should_abort and should_abort():
                    raise InterruptedError("Job interrupted.")
                time.sleep(0.02)
            return self.app.library.prompt_result_path_for(rel_path, prompt.slug), True

        with patch.object(self.app.library, "generate_prompt_result", side_effect=slow_generate):
            self.app.job_queue.submit(["paper-a.pdf", "paper-b.pdf"], ["core-zh"], force=True, source="batch")
            self.assertTrue(started.wait(timeout=2))

            for _ in range(100):
                snapshot = self.app.job_queue.snapshot()
                if snapshot["running_count"] == 1 and snapshot["queued_count"] >= 1:
                    break
                time.sleep(0.02)

            response = self.client.post(
                "/jobs/stop-all",
                data={
                    "folder": "",
                    "q": "",
                    "sort": "date_desc",
                    "paper": "",
                    "tab": "source",
                },
                follow_redirects=True,
            )
            release.set()

        self.assertEqual(response.status_code, 200)
        for _ in range(100):
            snapshot = self.app.job_queue.snapshot()
            if snapshot["active_count"] == 0:
                break
            time.sleep(0.02)
        statuses = {job.status for job in self.app.job_queue.list_jobs(limit=10)}
        self.assertIn("stopped", statuses)

    def test_render_markdown_supports_rule_and_blockquote(self) -> None:
        rendered = render_markdown("# 标题\n\n> 引用内容\n\n---\n\n1. 第一项\n2. 第二项")

        self.assertIn("<h1>标题</h1>", rendered)
        self.assertIn("<blockquote><p>引用内容</p></blockquote>", rendered)
        self.assertIn("<hr>", rendered)
        self.assertIn("<ol><li>第一项</li><li>第二项</li></ol>", rendered)


if __name__ == "__main__":
    unittest.main()
