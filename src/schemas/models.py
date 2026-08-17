from pydantic import BaseModel, Field
from typing import Optional

class CacheLookupRequest(BaseModel):
    query: str = Field(..., min_length=2)

class CacheLookupResponse(BaseModel):
    hit: bool
    response: Optional[str]
    similarity_score: float

class CacheStoreRequest(BaseModel):
    query: str
    response: str
