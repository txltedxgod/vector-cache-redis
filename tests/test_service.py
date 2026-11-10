def test_cache_roundtrip(client) -> None:
    store_resp = client.post(
        "/api/v1/cache/store",
        json={
            "query": "How to tune PostgreSQL?",
            "response": "Use proper indexing and connection pooling.",
        },
    )
    assert store_resp.status_code == 200

    resp = client.post("/api/v1/cache/lookup", json={"query": "How to tune PostgreSQL?"})
    assert resp.status_code == 200
    assert resp.json()["hit"] is True
    assert resp.json()["similarity_score"] >= 0.99


def test_pipeline_invalidation_and_stats_endpoints(client) -> None:
    client.delete("/api/v1/cache", params={"namespace": "bench"})

    pipe_store = client.post(
        "/api/v1/cache/pipeline/store",
        json={
            "namespace": "bench",
            "items": [
                {"query": "Redis persistence AOF vs RDB", "response": "Combine RDB snapshots with AOF."},
                {"query": "Kafka consumer group rebalancing", "response": "Cooperative sticky assignor."},
            ],
        },
    )
    assert pipe_store.status_code == 200
    assert pipe_store.json()["count"] == 2

    pipe_lookup = client.post(
        "/api/v1/cache/pipeline/lookup",
        json={
            "namespace": "bench",
            "threshold": 0.55,
            "queries": ["Redis AOF RDB persistence", "unrelated astrophysics question"],
        },
    )
    assert pipe_lookup.status_code == 200
    results = pipe_lookup.json()["results"]
    assert results[0]["hit"] is True
    assert results[1]["hit"] is False

    stats = client.get("/api/v1/cache/stats")
    assert stats.status_code == 200
    assert stats.json()["active_entries"] >= 2

    deleted = client.delete("/api/v1/cache", params={"namespace": "bench"})
    assert deleted.status_code == 200
    assert deleted.json()["removed"] == 2
