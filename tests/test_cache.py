from vcache.cache import SemanticCache

def test_cache_hit_and_miss():
    c = SemanticCache(threshold=0.85)
    c.set("What is Python?", "Python is a high-level programming language.")

    resp, sim = c.get("What is Python?")
    assert resp is not None
    assert sim >= 0.99

    resp2, sim2 = c.get("Completely different topic about cooking")
    assert resp2 is None
