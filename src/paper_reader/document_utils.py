from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from olefile import OleFileIO
from pypdf import PdfReader

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}
NAMESPACE = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


class UnsupportedDocumentError(RuntimeError):
    pass


def guess_title_from_lines(lines: list[str], fallback: str) -> str:
    blacklist_prefixes = (
        "arxiv:",
        "submitted on",
        "abstract",
        "keywords",
        "authors",
    )
    cleaned: list[str] = []
    for line in lines[:12]:
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in blacklist_prefixes):
            continue
        if len(line) < 8:
            continue
        cleaned.append(line)
    if not cleaned:
        return fallback

    title_parts: list[str] = []
    for line in cleaned[:3]:
        candidate = " ".join(title_parts + [line]).strip()
        if len(candidate) > 180:
            break
        title_parts.append(line)
        if line.endswith("?") or line.endswith(":"):
            break
    return " ".join(title_parts).strip() or fallback


def guess_title_from_text(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return guess_title_from_lines(lines, fallback)


def extract_pdf_metadata(path: Path, *, include_full_text: bool = False, preview_page_limit: int = 2) -> dict[str, str]:
    title = path.stem
    preview_text = ""
    full_text = ""
    reader = PdfReader(str(path))
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or path.stem

    snippets: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        if not include_full_text and page_number > preview_page_limit:
            break
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        if page_number <= 2:
            snippets.append(page_text)
        if include_full_text:
            full_text += ("" if not full_text else "\n\n") + f"[Page {page_number}]\n{page_text}"

    preview_text = "\n\n".join(snippets)
    if title == path.stem and preview_text:
        title = guess_title_from_text(preview_text, path.stem)
    return {"title": title, "preview_text": preview_text, "full_text": full_text}


def extract_docx_metadata(path: Path, *, include_full_text: bool = False, preview_paragraph_limit: int = 30) -> dict[str, str]:
    title = path.stem
    paragraphs: list[str] = []

    with zipfile.ZipFile(path) as archive:
        if "docProps/core.xml" in archive.namelist():
            core_xml = archive.read("docProps/core.xml")
            root = ET.fromstring(core_xml)
            title_node = root.find("dc:title", NAMESPACE)
            if title_node is not None and title_node.text:
                title = title_node.text.strip() or path.stem

        if "word/document.xml" in archive.namelist():
            document_xml = archive.read("word/document.xml")
            root = ET.fromstring(document_xml)
            for paragraph in root.findall(".//w:p", NAMESPACE):
                runs = [node.text or "" for node in paragraph.findall(".//w:t", NAMESPACE)]
                text = "".join(runs).strip()
                if text:
                    paragraphs.append(text)
                    if not include_full_text and len(paragraphs) >= preview_paragraph_limit:
                        break

    preview_text = "\n\n".join(paragraphs[:30])
    full_text = "\n\n".join(paragraphs) if include_full_text else ""
    if title == path.stem and paragraphs:
        title = guess_title_from_lines(paragraphs, path.stem)
    return {"title": title, "preview_text": preview_text, "full_text": full_text}


def extract_doc_metadata(path: Path) -> dict[str, str]:
    title = path.stem
    try:
        with OleFileIO(str(path)) as ole:
            metadata = ole.get_metadata()
            doc_title = getattr(metadata, "title", "")
            if doc_title:
                title = str(doc_title).strip() or path.stem
    except Exception:
        pass
    return {"title": title, "preview_text": "", "full_text": ""}


def extract_document_metadata(path: Path, *, include_full_text: bool = False) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_metadata(path, include_full_text=include_full_text)
    if suffix == ".docx":
        return extract_docx_metadata(path, include_full_text=include_full_text)
    if suffix == ".doc":
        return extract_doc_metadata(path)
    raise UnsupportedDocumentError(f"Unsupported file type: {suffix}")


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_metadata(path, include_full_text=True)["full_text"]
    if suffix == ".docx":
        return extract_docx_metadata(path, include_full_text=True)["full_text"]
    if suffix == ".doc":
        raise UnsupportedDocumentError("Legacy .doc files cannot be text-extracted in this environment.")
    raise UnsupportedDocumentError(f"Unsupported file type: {suffix}")
