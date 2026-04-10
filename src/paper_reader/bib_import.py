from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter

ARXIV_ABS_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#{}]+)", re.IGNORECASE)
ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.([^\s{}]+)", re.IGNORECASE)
ARXIV_NOTE_RE = re.compile(r"arXiv:([^\s\[\]{}]+)", re.IGNORECASE)
LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
BIB_SPLIT_AUTHOR_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class BibImportUnmatchedItem:
    cite_key: str
    title: str
    reason: str
    raw_arxiv_hint: str | None = None


@dataclass
class ArxivCandidate:
    arxiv_id: str
    title: str
    authors: list[str]
    published_year: str | None


def load_bib_entries(text: str) -> list[dict[str, Any]]:
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    database = bibtexparser.loads(text, parser=parser)
    return [dict(entry) for entry in database.entries]


def dump_bib_entries(entries: list[dict[str, Any]]) -> str:
    database = BibDatabase()
    database.entries = [dict(entry) for entry in entries]
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None
    return writer.write(database)


def dump_unmatched_list(items: list[BibImportUnmatchedItem]) -> str:
    lines = ["# Unimported papers", ""]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. [{item.cite_key or 'unknown-key'}] {item.title or '(untitled)'}")
        lines.append(f"   reason: {item.reason}")
        if item.raw_arxiv_hint:
            lines.append(f"   arxiv_hint: {item.raw_arxiv_hint}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def extract_arxiv_id_from_entry(entry: dict[str, Any]) -> str | None:
    fields = [entry.get("url"), entry.get("doi"), entry.get("note"), entry.get("eprint")]
    for value in fields:
        text = str(value or "").strip()
        if not text:
            continue
        for pattern in (ARXIV_ABS_RE, ARXIV_DOI_RE, ARXIV_NOTE_RE):
            match = pattern.search(text)
            if match:
                candidate = match.group(1).strip().removesuffix(".pdf").strip("/")
                if candidate:
                    return candidate
        if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", text, re.IGNORECASE):
            return text
        if re.fullmatch(r"[A-Za-z0-9.\-]+/\d{7}(?:v\d+)?", text, re.IGNORECASE):
            return text
    return None


def normalize_title(value: str) -> str:
    text = str(value or "")
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ").replace("$", " ")
    text = text.replace("\n", " ").replace("\t", " ")
    text = NON_ALNUM_RE.sub(" ", text.lower())
    return " ".join(text.split())


def parse_author_surnames(value: str) -> list[str]:
    surnames: list[str] = []
    for raw_author in BIB_SPLIT_AUTHOR_RE.split(str(value or "")):
        author = raw_author.strip()
        if not author:
            continue
        if "," in author:
            surname = author.split(",", 1)[0].strip().lower()
        else:
            surname = author.split()[-1].strip().lower()
        if surname:
            surnames.append(surname)
    return surnames


def fetch_arxiv_candidates(title: str, *, timeout: int = 20, max_results: int = 5) -> list[ArxivCandidate]:
    query = urlencode(
        {
            "search_query": f'ti:"{title}"',
            "start": 0,
            "max_results": max_results,
        }
    )
    request = Request(
        f"https://export.arxiv.org/api/query?{query}",
        headers={"User-Agent": "paper-reader/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    candidates: list[ArxivCandidate] = []
    for node in root.findall("atom:entry", ATOM_NS):
        entry_id = (node.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        title_text = (node.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        if not entry_id or not title_text:
            continue
        arxiv_id = entry_id.rstrip("/").rsplit("/", 1)[-1]
        author_names = [
            (author_node.text or "").strip()
            for author_node in node.findall("atom:author/atom:name", ATOM_NS)
            if (author_node.text or "").strip()
        ]
        published = (node.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
        published_year = published[:4] if len(published) >= 4 else None
        candidates.append(
            ArxivCandidate(
                arxiv_id=arxiv_id,
                title=title_text,
                authors=author_names,
                published_year=published_year,
            )
        )
    return candidates


def find_arxiv_id_for_bib_entry(entry: dict[str, Any], *, timeout: int = 20) -> tuple[str | None, str]:
    direct_id = extract_arxiv_id_from_entry(entry)
    if direct_id:
        return direct_id, "direct"

    title = str(entry.get("title") or "").strip()
    if not title:
        return None, "缺少 title，无法检索 arXiv。"

    target_title = normalize_title(title)
    if not target_title:
        return None, "标题为空，无法检索 arXiv。"

    entry_authors = set(parse_author_surnames(str(entry.get("author") or "")))
    entry_year = str(entry.get("year") or "").strip()

    candidates = fetch_arxiv_candidates(title, timeout=timeout)
    best_candidate: ArxivCandidate | None = None
    best_score = 0.0

    for candidate in candidates:
        candidate_title = normalize_title(candidate.title)
        if not candidate_title:
            continue

        title_score = SequenceMatcher(None, target_title, candidate_title).ratio()
        title_exact = target_title == candidate_title
        title_contains = target_title in candidate_title or candidate_title in target_title
        candidate_authors = set(parse_author_surnames(" and ".join(candidate.authors)))
        author_overlap = len(entry_authors & candidate_authors)
        year_matches = bool(entry_year and candidate.published_year and entry_year == candidate.published_year)

        accepted = False
        if title_exact or title_contains:
            accepted = True
        elif title_score >= 0.97:
            accepted = True
        elif title_score >= 0.92 and (author_overlap > 0 or year_matches):
            accepted = True
        elif title_score >= 0.88 and author_overlap >= 2:
            accepted = True

        if accepted and title_score >= best_score:
            best_score = title_score
            best_candidate = candidate

    if best_candidate is not None:
        return best_candidate.arxiv_id, "search"
    return None, "未找到可信 arXiv 匹配。"
