from __future__ import annotations

import html
import re


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")
_BULLET_RE = re.compile(r"^[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\d+\.\s+(.*)$")
_RULE_RE = re.compile(r"^([-*_])(?:\s*\1){2,}\s*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def render_markdown(text: str) -> str:
    parts: list[str] = []
    paragraph: list[str] = []
    quote_lines: list[str] = []
    list_mode: str | None = None
    list_items: list[str] = []
    code_lines: list[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        joined = " ".join(line.strip() for line in paragraph if line.strip())
        if joined:
            parts.append(f"<p>{render_inline(joined)}</p>")
        paragraph = []

    def flush_quote() -> None:
        nonlocal quote_lines
        if not quote_lines:
            return
        content = " ".join(line.strip() for line in quote_lines if line.strip())
        if content:
            parts.append(f"<blockquote><p>{render_inline(content)}</p></blockquote>")
        quote_lines = []

    def flush_list() -> None:
        nonlocal list_mode, list_items
        if list_mode and list_items:
            tag = "ol" if list_mode == "ol" else "ul"
            items_html = "".join(f"<li>{render_inline(item)}</li>" for item in list_items)
            parts.append(f"<{tag}>{items_html}</{tag}>")
        list_mode = None
        list_items = []

    def flush_code_block() -> None:
        nonlocal code_lines, in_code_block
        code_html = html.escape("\n".join(code_lines))
        parts.append(f"<pre><code>{code_html}</code></pre>")
        code_lines = []
        in_code_block = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_quote()
            flush_list()
            if in_code_block:
                flush_code_block()
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_quote()
            flush_list()
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            flush_quote()
            flush_list()
            level = len(heading_match.group(1))
            parts.append(f"<h{level}>{render_inline(heading_match.group(2).strip())}</h{level}>")
            continue

        if _RULE_RE.match(stripped):
            flush_paragraph()
            flush_quote()
            flush_list()
            parts.append("<hr>")
            continue

        quote_match = _BLOCKQUOTE_RE.match(stripped)
        if quote_match:
            flush_paragraph()
            flush_list()
            quote_lines.append(quote_match.group(1))
            continue
        flush_quote()

        bullet_match = _BULLET_RE.match(stripped)
        ordered_match = _ORDERED_RE.match(stripped)
        if bullet_match or ordered_match:
            flush_paragraph()
            mode = "ol" if ordered_match else "ul"
            item_text = (ordered_match or bullet_match).group(1).strip()
            if list_mode != mode:
                flush_list()
                list_mode = mode
            list_items.append(item_text)
            continue

        if list_mode:
            flush_list()
        paragraph.append(stripped)

    flush_paragraph()
    flush_quote()
    flush_list()
    if in_code_block:
        flush_code_block()

    return "\n".join(parts)


def render_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = _INLINE_CODE_RE.sub(_render_inline_code, escaped)
    escaped = _LINK_RE.sub(_render_link, escaped)
    escaped = _BOLD_RE.sub(lambda match: f"<strong>{match.group(1)}</strong>", escaped)
    escaped = _ITALIC_RE.sub(lambda match: f"<em>{match.group(1)}</em>", escaped)
    return escaped


def _render_inline_code(match: re.Match[str]) -> str:
    return f"<code>{html.escape(match.group(1))}</code>"


def _render_link(match: re.Match[str]) -> str:
    label = match.group(1)
    href = html.escape(match.group(2), quote=True)
    return f'<a href="{href}" target="_blank" rel="noreferrer">{label}</a>'
