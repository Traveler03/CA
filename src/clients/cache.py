from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.hash import stable_hash


@dataclass
class JsonRequestCache:
    root: Path
    enabled: bool = True

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, namespace: str, payload: Any) -> str:
        return f"{namespace}-{stable_hash(payload, 32)}"

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.path_for(key).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        except OSError:
            return
