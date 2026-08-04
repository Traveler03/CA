from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


def mask_secret(value: str | None, prefix: int = 3, suffix: int = 4) -> str:
    """Return a stable non-reversible display form for a secret."""
    if not value:
        return ""
    if len(value) <= prefix + suffix:
        return "****"
    return f"{value[:prefix]}****{value[-suffix:]}"


def host_from_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc or parsed.path.split("/")[0]


@dataclass(frozen=True)
class SecretValue:
    value: str
    source: str

    @property
    def masked(self) -> str:
        return mask_secret(self.value)

