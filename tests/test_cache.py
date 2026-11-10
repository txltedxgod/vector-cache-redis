import time

from vcache.cache import SemanticCache


def test_cache_hit_and_miss() -> None:
    cache = SemanticCache(threshold=0.85)
    cache.set("What is Python?", "Python is a high-level programming language.")

    resp, sim = cache.get("What is Python?")
    assert resp is not None
    assert sim >= 0.99

    resp2, _sim2 = cache.get("Completely different topic about cooking")
    assert resp2 is None


def test_paraphrase_semantic_hit_and_namespace_isolation() -> None:
    cache = SemanticCache(threshold=0.60, dim=128)
    cache.set(
        "How to tune PostgreSQL connection pooling?",
        "Use PgBouncer in transaction pooling mode.",
        namespace="gpt-4o",
    )

    hit_resp, hit_sim = cache.get("PostgreSQL connection pooling tuning", namespace="gpt-4o")
    assert hit_resp == "Use PgBouncer in transaction pooling mode."
    assert hit_sim >= 0.60

    # Other model namespace remains isolated
    miss_resp, miss_sim = cache.get("PostgreSQL connection pooling tuning", namespace="claude")
    assert miss_resp is None
    assert miss_sim == 0.0


def test_lru_eviction_and_ttl_expiration() -> None:
    cache = SemanticCache(threshold=0.85, max_entries=2, ttl_seconds=60.0)
    cache.set("alpha query", "A")
    cache.set("beta query", "B")

    # Touch alpha so beta becomes least recently used
    assert cache.get("alpha query")[0] == "A"

    cache.set("gamma query", "C")
    assert cache.get("beta query")[0] is None
    assert cache.get("alpha query")[0] == "A"
    assert cache.get("gamma query")[0] == "C"
    assert cache.stats()["evictions"] == 1

    # Explicit short TTL expiration
    cache.set("ephemeral", "temp", ttl_seconds=0.05)
    time.sleep(0.08)
    assert cache.get("ephemeral")[0] is None
    assert cache.stats()["expirations"] >= 1
