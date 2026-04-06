from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .ai_summary import DEFAULT_MODEL, DEFAULT_USER_PROMPT

PROMPT_STORE_NAME = ".paper-reader-prompts.json"
DEFAULT_PROMPT_SLUG = "core-zh"


@dataclass
class PromptDefinition:
    slug: str
    name: str
    user_prompt: str
    model: str
    enabled: bool
    auto_run: bool
    created_at: str
    updated_at: str


class PromptStore:
    def __init__(self, library_root: Path):
        self.library_root = library_root.resolve()
        self.store_path = self.library_root / PROMPT_STORE_NAME

    def list_prompts(self) -> list[PromptDefinition]:
        payload = self._load_payload()
        prompts: list[PromptDefinition] = []
        for item in payload.get("prompts", []):
            prompt = self._coerce_prompt(item)
            if prompt is not None:
                prompts.append(prompt)
        if prompts:
            return sorted(
                prompts,
                key=lambda prompt: (
                    prompt.slug != DEFAULT_PROMPT_SLUG,
                    not prompt.enabled,
                    prompt.name.lower(),
                    prompt.created_at,
                ),
            )

        default_prompt = self.default_prompt()
        self._write_payload({"prompts": [asdict(default_prompt)]})
        return [default_prompt]

    def default_prompt(self) -> PromptDefinition:
        now = datetime.utcnow().isoformat(timespec="seconds")
        return PromptDefinition(
            slug=DEFAULT_PROMPT_SLUG,
            name="核心解读",
            user_prompt=DEFAULT_USER_PROMPT,
            model=DEFAULT_MODEL,
            enabled=True,
            auto_run=True,
            created_at=now,
            updated_at=now,
        )

    def get_prompt(self, slug: str) -> PromptDefinition | None:
        for prompt in self.list_prompts():
            if prompt.slug == slug:
                return prompt
        return None

    def active_prompts(self) -> list[PromptDefinition]:
        return [prompt for prompt in self.list_prompts() if prompt.enabled]

    def auto_prompts(self) -> list[PromptDefinition]:
        return [prompt for prompt in self.active_prompts() if prompt.auto_run]

    def save_prompt(
        self,
        *,
        existing_slug: str | None,
        name: str,
        slug: str,
        user_prompt: str,
        model: str,
        enabled: bool,
        auto_run: bool,
    ) -> PromptDefinition:
        name = name.strip()
        requested_slug = slug.strip()
        user_prompt = user_prompt.strip()
        model = model.strip() or DEFAULT_MODEL
        if not name:
            raise ValueError("Prompt 名称不能为空。")
        if not user_prompt:
            raise ValueError("Prompt 内容不能为空。")

        payload = self._load_payload()
        prompts = payload.get("prompts", [])
        if not prompts:
            prompts = [asdict(self.default_prompt())]
        used_slugs = {str(item.get("slug", "")).strip() for item in prompts if item.get("slug")}
        if existing_slug:
            used_slugs.discard(existing_slug)
            resolved_slug = existing_slug.strip()
        else:
            resolved_slug = self._choose_slug(name=name, requested_slug=requested_slug, used_slugs=used_slugs)

        now = datetime.utcnow().isoformat(timespec="seconds")
        updated_prompt: PromptDefinition | None = None
        seen = False
        next_prompts: list[dict[str, Any]] = []

        for item in prompts:
            item_slug = item.get("slug")
            if existing_slug and item_slug == existing_slug:
                created_at = item.get("created_at") or now
                updated_prompt = PromptDefinition(
                    slug=existing_slug,
                    name=name,
                    user_prompt=user_prompt,
                    model=model,
                    enabled=enabled,
                    auto_run=auto_run,
                    created_at=created_at,
                    updated_at=now,
                )
                next_prompts.append(asdict(updated_prompt))
                seen = True
                continue
            if item_slug == resolved_slug:
                raise ValueError("Prompt 标识已存在，请换一个。")
            next_prompts.append(self._normalize_payload_item(item))

        if not seen:
            updated_prompt = PromptDefinition(
                slug=resolved_slug,
                name=name,
                user_prompt=user_prompt,
                model=model,
                enabled=enabled,
                auto_run=auto_run,
                created_at=now,
                updated_at=now,
            )
            next_prompts.append(asdict(updated_prompt))

        self._write_payload({"prompts": next_prompts})
        return updated_prompt

    def delete_prompt(self, slug: str) -> PromptDefinition:
        payload = self._load_payload()
        prompts = payload.get("prompts", [])
        next_prompts: list[dict[str, Any]] = []
        removed: PromptDefinition | None = None
        for item in prompts:
            prompt = self._coerce_prompt(item)
            if prompt is None:
                continue
            if prompt.slug == slug:
                removed = prompt
                continue
            next_prompts.append(asdict(prompt))
        if removed is None:
            raise FileNotFoundError(slug)
        self._write_payload({"prompts": next_prompts})
        return removed

    def _coerce_prompt(self, item: Any) -> PromptDefinition | None:
        if not isinstance(item, dict):
            return None
        try:
            slug = str(item.get("slug", "")).strip()
            name = str(item.get("name", "")).strip()
            user_prompt = str(item.get("user_prompt") or item.get("prompt") or DEFAULT_USER_PROMPT).strip()
            model = str(item.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
            enabled = bool(item.get("enabled", True))
            auto_run = bool(item.get("auto_run", True))
            created_at = str(item.get("created_at") or datetime.utcnow().isoformat(timespec="seconds"))
            updated_at = str(item.get("updated_at") or created_at)
        except Exception:
            return None
        if not slug or not name or not user_prompt:
            return None
        return PromptDefinition(
            slug=slug,
            name=name,
            user_prompt=user_prompt,
            model=model,
            enabled=enabled,
            auto_run=auto_run,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _normalize_payload_item(self, item: Any) -> dict[str, Any]:
        prompt = self._coerce_prompt(item)
        if prompt is None:
            return asdict(self.default_prompt())
        return asdict(prompt)

    def _load_payload(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {}
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _choose_slug(self, *, name: str, requested_slug: str, used_slugs: set[str]) -> str:
        normalized_requested = slugify(requested_slug)
        if normalized_requested:
            if normalized_requested in used_slugs:
                raise ValueError("Prompt 标识已存在，请换一个。")
            return normalized_requested

        normalized_name = slugify(name)
        base = normalized_name or "prompt"
        slug = base
        counter = 2
        while slug in used_slugs:
            slug = f"{base}-{counter}"
            counter += 1
        return slug


_slug_pattern = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    normalized = _slug_pattern.sub("-", lowered).strip("-")
    return normalized[:80]


def parse_checkbox(value: str | None) -> bool:
    return value == "on"
