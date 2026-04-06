from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SETTINGS_FILE_NAME = ".paper-reader-settings.json"
DEFAULT_MAX_CONCURRENCY = 12
MAX_CONCURRENCY_LIMIT = 32


class SettingsStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / SETTINGS_FILE_NAME

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def max_concurrency(self) -> int:
        payload = self.load()
        return self._clamp(payload.get("max_concurrency", DEFAULT_MAX_CONCURRENCY))

    def save_max_concurrency(self, value: int) -> int:
        payload = self.load()
        payload["max_concurrency"] = self._clamp(value)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return int(payload["max_concurrency"])

    def _clamp(self, value: Any) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = DEFAULT_MAX_CONCURRENCY
        return max(1, min(MAX_CONCURRENCY_LIMIT, numeric))
