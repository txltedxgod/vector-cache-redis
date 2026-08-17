def test_cache_roundtrip(client):
    client.post("/api/v1/cache/store", json={"query": "How to tune PostgreSQL?", "response": "Use proper indexing and connection pooling."})
    resp = client.post("/api/v1/cache/lookup", json={"query": "How to tune PostgreSQL?"})
    assert resp.status_code == 200
    assert resp.json()["hit"] is True
