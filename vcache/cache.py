"""Backward-compatible SemanticCache facade delegating to SemanticCacheService."""

from __future__ import annotations

from typing import Any

from src.services.service import SemanticCacheService


class SemanticCache:
    """High-level semantic cache wrapper supporting LRU, TTL, and feature hashing."""

    def __init__(
        self,
        threshold: float = 0.88,
        dim: int = 64,
        max_entries: int = 1024,
        ttl_seconds: float | None = 3600.0,
    ) -> None:
        self._svc = SemanticCacheService(
            threshold=threshold,
            dim=dim,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )

    @property
    def entries(self) -> dict[Any, Any]:
        return self._svc._entries

    def get(self, query: str, namespace: str = "default") -> tuple[str | None, float]:
        """Lookup query in cache and return (response_or_none, similarity_score)."""
        hit, resp, sim = self._svc.lookup(query, namespace=namespace)
        return (resp if hit else None), sim

    def set(
        self,
        query: str,
        response: str,
        namespace: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        """Store response for query in semantic cache."""
        self._svc.store(query, response, namespace=namespace, ttl_seconds=ttl_seconds)

    def invalidate(self, query: str | None = None, namespace: str | None = None) -> int:
        """Invalidate entry or namespace from cache."""
        return self._svc.invalidate(query=query, namespace=namespace)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return self._svc.stats()
