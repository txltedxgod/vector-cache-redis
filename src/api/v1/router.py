"""Versioned REST endpoints for semantic vector cache operations."""

from typing import Any

from fastapi import APIRouter, Query

from src.schemas.models import (
    BatchLookupRequest,
    BatchStoreRequest,
    CacheLookupRequest,
    CacheLookupResponse,
    CacheStoreRequest,
)
from src.services.service import SemanticCacheService

router = APIRouter(prefix="/api/v1/cache", tags=["Vector Cache"])
cache = SemanticCacheService(threshold=0.80)


@router.post("/lookup", response_model=CacheLookupResponse)
def lookup(payload: CacheLookupRequest) -> CacheLookupResponse:
    """Lookup semantically similar cached response within namespace."""
    hit, resp, sim = cache.lookup(
        payload.query,
        namespace=payload.namespace or "default",
        threshold=payload.threshold,
    )
    return CacheLookupResponse(hit=hit, response=resp, similarity_score=sim)


@router.post("/store")
def store(payload: CacheStoreRequest) -> dict[str, str]:
    """Store a query-response embedding pair in semantic cache."""
    cache.store(
        payload.query,
        payload.response,
        namespace=payload.namespace or "default",
        ttl_seconds=payload.ttl_seconds,
        metadata=payload.metadata,
    )
    return {"status": "stored"}


@router.post("/pipeline/store")
def pipeline_store(payload: BatchStoreRequest) -> dict[str, Any]:
    """Store multiple entries in a single pipelined request."""
    stored_count = cache.batch_store(
        [
            {**item.model_dump(), "namespace": item.namespace or payload.namespace or "default"}
            for item in payload.items
        ],
        namespace=payload.namespace or "default",
    )
    return {"status": "stored", "count": stored_count}


@router.post("/pipeline/lookup")
def pipeline_lookup(payload: BatchLookupRequest) -> dict[str, Any]:
    """Resolve multiple semantic queries in a single pipelined request."""
    results = cache.batch_lookup(
        payload.queries,
        namespace=payload.namespace or "default",
        threshold=payload.threshold,
    )
    return {"results": results}


@router.delete("")
def invalidate_cache(
    query: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
) -> dict[str, Any]:
    """Invalidate matching entry, namespace, or entire cache."""
    removed = cache.invalidate(query=query, namespace=namespace)
    return {"status": "invalidated", "removed": removed}


@router.get("/stats")
def cache_stats() -> dict[str, Any]:
    """Return semantic cache hit rate, evictions, expirations, and occupancy."""
    return cache.stats()
