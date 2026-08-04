from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from src.ca_mem.schemas import MemoryNode


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_memory_key(subject: str, concept: str, description: str) -> str:
    return f"Subject: {subject}\nConcept: {concept}\nDescription: {description}"


def key_for_node(node: MemoryNode) -> str:
    return build_memory_key(node.subject, node.concept, node.description)


@dataclass
class HashingTextEmbedder:
    model_name: str = "deterministic-hashing-embedding"
    dim: int = 256
    prefix: str = "ca_mem_build:"
    backend: str = "hash"

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        for token in TOKEN_RE.findall((self.prefix + text).lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=float)
        return np.vstack([self.embed_one(text) for text in texts])
