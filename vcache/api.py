from fastapi import FastAPI
from pydantic import BaseModel
from vcache.cache import SemanticCache

app = FastAPI(title="Vector Semantic Cache", version="0.1.0")
cache = SemanticCache(threshold=0.90)

class QueryReq(BaseModel):
    query: str

class StoreReq(BaseModel):
    query: str
    response: str

@app.post("/api/v1/lookup")
def lookup(req: QueryReq):
    resp, sim = cache.get(req.query)
    return {"hit": resp is not None, "response": resp, "similarity": sim}

@app.post("/api/v1/store")
def store(req: StoreReq):
    cache.set(req.query, req.response)
    return {"status": "cached"}
