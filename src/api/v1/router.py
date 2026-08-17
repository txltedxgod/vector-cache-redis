from fastapi import APIRouter
from src.schemas.models import CacheLookupRequest, CacheLookupResponse, CacheStoreRequest
from src.services.service import SemanticCacheService

router = APIRouter(prefix="/api/v1/cache", tags=["Vector Cache"])
cache = SemanticCacheService(threshold=0.88)

@router.post("/lookup", response_model=CacheLookupResponse)
def lookup(payload: CacheLookupRequest):
    hit, resp, sim = cache.lookup(payload.query)
    return CacheLookupResponse(hit=hit, response=resp, similarity_score=sim)

@router.post("/store")
def store(payload: CacheStoreRequest):
    cache.store(payload.query, payload.response)
    return {"status": "stored"}
