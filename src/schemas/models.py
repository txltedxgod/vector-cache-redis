"""Pydantic v2 schemas for semantic cache lookup, store, batch pipelining, and invalidation."""

from typing import Any

from pydantic import BaseModel, Field


class CacheLookupRequest(BaseModel):
    """Single semantic cache lookup request."""

    query: str = Field(..., min_length=2, max_length=2000)
    namespace: str | None = Field(default=None, min_length=1, max_length=64)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class CacheLookupResponse(BaseModel):
    """Semantic cache lookup result with hit status and similarity score."""

    hit: bool
    response: str | None = None
    similarity_score: float


class CacheStoreRequest(BaseModel):
    """Store query-response pair with optional namespace, TTL, and metadata."""

    query: str = Field(..., min_length=2, max_length=2000)
    response: str = Field(..., min_length=1, max_length=50000)
    namespace: str | None = Field(default=None, min_length=1, max_length=64)
    ttl_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] | None = Field(default=None)


class BatchStoreRequest(BaseModel):
    """Pipeline batch store payload."""

    items: list[CacheStoreRequest] = Field(..., min_length=1, max_length=100)
    namespace: str | None = Field(default=None, min_length=1, max_length=64)


class BatchLookupRequest(BaseModel):
    """Pipeline batch lookup payload."""

    queries: list[str] = Field(..., min_length=1, max_length=100)
    namespace: str | None = Field(default=None, min_length=1, max_length=64)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
