"""Semantic vector cache service with feature hashing, TTL expiration, and O(1) LRU eviction."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.exceptions import ValidationError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class CacheEntry:
    """Stored semantic cache item with embedding, expiry timestamp, and metadata."""

    key: tuple[str, str]
    namespace: str
    query: str
    response: str
    embedding: np.ndarray
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0


class SemanticCacheService:
    """In-memory semantic vector cache supporting L2 feature hashing, TTL, and LRU eviction."""

    def __init__(
        self,
        threshold: float = 0.88,
        dim: int = 64,
        max_entries: int = 1024,
        ttl_seconds: float | None = 3600.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when specified")

        self.threshold = threshold
        self.dim = dim
        self.max_entries = max_entries
        self.default_ttl_seconds = ttl_seconds
        self._entries: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _bucket_index(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") % self.dim

    def _embed(self, text: str) -> np.ndarray:
        """Compute deterministic L2-normalized word + character-trigram hashed embedding."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)

        if tokens:
            for token, freq in Counter(tokens).items():
                weight = 1.0 + math.log(freq)
                vec[self._bucket_index(f"w:{token}")] += weight
                if len(token) >= 3:
                    padded = f"^{token}$"
                    trigram_w = 0.35 * weight
                    for i in range(len(padded) - 2):
                        vec[self._bucket_index(f"c:{padded[i : i + 3]}")] += trigram_w
        else:
            cleaned = text.lower().strip()
            for i in range(max(0, len(cleaned) - 2)):
                vec[self._bucket_index(f"raw:{cleaned[i : i + 3]}")] += 1.0

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def _purge_expired(self, now: float | None = None) -> int:
        current = time.monotonic() if now is None else now
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= current
        ]
        for key in expired_keys:
            del self._entries[key]
        self._expirations += len(expired_keys)
        return len(expired_keys)

    @staticmethod
    def _normalize_key(query: str, namespace: str) -> tuple[str, str]:
        return (namespace.strip() or "default", " ".join(query.strip().split()).lower())

    def lookup(
        self,
        query: str,
        namespace: str = "default",
        threshold: float | None = None,
    ) -> tuple[bool, str | None, float]:
        """Search cache within namespace for semantic hit above threshold; updates LRU order."""
        if not query or not query.strip():
            raise ValidationError("Query must not be empty")

        self._purge_expired()
        ns = namespace.strip() or "default"
        eff_threshold = self.threshold if threshold is None else threshold

        # Fast-path exact normalized key lookup
        exact_key = self._normalize_key(query, ns)
        exact_entry = self._entries.get(exact_key)
        if exact_entry is not None:
            self._entries.move_to_end(exact_key, last=True)
            exact_entry.hit_count += 1
            self._hits += 1
            return True, exact_entry.response, 1.0

        ns_entries = [entry for entry in self._entries.values() if entry.namespace == ns]
        if not ns_entries:
            self._misses += 1
            return False, None, 0.0

        q_emb = self._embed(query)
        best_sim = 0.0
        best_entry: CacheEntry | None = None
        for entry in ns_entries:
            sim = float(np.dot(q_emb, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= eff_threshold:
            self._entries.move_to_end(best_entry.key, last=True)
            best_entry.hit_count += 1
            self._hits += 1
            return True, best_entry.response, round(min(1.0, best_sim), 4)

        self._misses += 1
        return False, None, round(max(0.0, best_sim), 4)

    def store(
        self,
        query: str,
        response: str,
        namespace: str = "default",
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or refresh an entry in the LRU cache with optional per-item TTL."""
        if not query or not query.strip():
            raise ValidationError("Query must not be empty")

        now = time.monotonic()
        self._purge_expired(now)

        key = self._normalize_key(query, namespace)
        ns = key[0]
        eff_ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = (now + eff_ttl) if (eff_ttl is not None and eff_ttl > 0) else None

        if key in self._entries:
            entry = self._entries[key]
            entry.query = query
            entry.response = response
            entry.embedding = self._embed(query)
            entry.expires_at = expires_at
            if metadata is not None:
                entry.metadata = dict(metadata)
            self._entries.move_to_end(key, last=True)
            return

        while len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

        self._entries[key] = CacheEntry(
            key=key,
            namespace=ns,
            query=query,
            response=response,
            embedding=self._embed(query),
            expires_at=expires_at,
            metadata=dict(metadata) if metadata else {},
        )

    def batch_store(
        self,
        items: list[dict[str, Any]],
        namespace: str = "default",
    ) -> int:
        """Pipeline-style batch cache insertion returning count of stored items."""
        for item in items:
            self.store(
                query=item["query"],
                response=item["response"],
                namespace=item.get("namespace", namespace),
                ttl_seconds=item.get("ttl_seconds"),
                metadata=item.get("metadata"),
            )
        return len(items)

    def batch_lookup(
        self,
        queries: list[str],
        namespace: str = "default",
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Pipeline-style batch cache lookup returning structured hit statuses."""
        results = []
        for q in queries:
            hit, resp, sim = self.lookup(q, namespace=namespace, threshold=threshold)
            results.append(
                {
                    "query": q,
                    "hit": hit,
                    "response": resp,
                    "similarity_score": sim,
                }
            )
        return results

    def invalidate(self, query: str | None = None, namespace: str | None = None) -> int:
        """Remove a specific cached query or purge an entire namespace/cache; returns deleted count."""
        self._purge_expired()
        if query is not None:
            ns = (namespace or "default").strip() or "default"
            key = self._normalize_key(query, ns)
            if key in self._entries:
                del self._entries[key]
                return 1
            return 0

        if namespace is not None:
            ns = namespace.strip() or "default"
            targets = [k for k, v in self._entries.items() if v.namespace == ns]
            for k in targets:
                del self._entries[k]
            return len(targets)

        count = len(self._entries)
        self._entries.clear()
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache efficiency, eviction, and expiration counters."""
        self._purge_expired()
        total = self._hits + self._misses
        hit_rate = round(self._hits / total, 4) if total > 0 else 0.0
        return {
            "active_entries": len(self._entries),
            "capacity": self.max_entries,
            "threshold": self.threshold,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "expirations": self._expirations,
        }
