from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .markdown_render import render_markdown


def offline_source_arcname(rel_path: str) -> str:
    return Path("papers", rel_path).as_posix()


def offline_prompt_arcname(rel_path: str, prompt_slug: str) -> str:
    return Path("prompt-results", rel_path, f"{prompt_slug}.md").as_posix()


def build_offline_manifest(library: Any, prompt_store: Any, papers: list[Any]) -> dict[str, Any]:
    prompt_map = {prompt.slug: prompt for prompt in prompt_store.list_prompts()}
    active_prompts = prompt_store.active_prompts()
    prompt_order: list[str] = [prompt.slug for prompt in active_prompts]

    for paper in papers:
        for slug in library.list_existing_prompt_slugs(paper.rel_path):
            if slug not in prompt_order:
                prompt_order.append(slug)

    prompts_payload = []
    for slug in prompt_order:
        prompt = prompt_map.get(slug)
        prompts_payload.append(
            {
                "slug": slug,
                "name": prompt.name if prompt is not None else slug,
                "enabled": prompt.enabled if prompt is not None else False,
            }
        )

    paper_payload = []
    for paper in papers:
        preview_paragraphs = [chunk.strip() for chunk in paper.preview_text.split("\n\n") if chunk.strip()]
        prompt_results = {}
        for prompt_info in prompts_payload:
            slug = prompt_info["slug"]
            result_info = library.prompt_result_info(paper.rel_path, slug)
            if result_info["exists"]:
                content = library.read_prompt_result(paper.rel_path, slug) or ""
                prompt_results[slug] = {
                    "exists": True,
                    "updated_at": result_info["updated_at"],
                    "result_rel_path": offline_prompt_arcname(paper.rel_path, slug),
                    "html": render_markdown(content),
                }
            else:
                prompt_results[slug] = {
                    "exists": False,
                    "updated_at": None,
                    "result_rel_path": None,
                    "html": "",
                }

        paper_payload.append(
            {
                "rel_path": paper.rel_path,
                "file_name": paper.file_name,
                "folder": paper.folder,
                "extension": paper.extension,
                "display_title": paper.display_title,
                "extracted_date": paper.extracted_date,
                "sort_date": paper.sort_date,
                "file_size": paper.file_size,
                "modified_at": paper.modified_at,
                "preview_kind": paper.preview_kind,
                "preview_paragraphs": preview_paragraphs,
                "source_rel_path": offline_source_arcname(paper.rel_path),
                "prompt_result_count": paper.prompt_result_count,
                "prompt_results": prompt_results,
            }
        )

    manifest = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "paper_count": len(paper_payload),
        "prompts": prompts_payload,
        "papers": paper_payload,
    }
    return manifest


def manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
