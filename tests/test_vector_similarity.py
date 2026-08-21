import math

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def test_identical_vectors():
    vec = [0.1, 0.5, 0.9, -0.2]
    assert math.isclose(cosine_similarity(vec, vec), 1.0, rel_tol=1e-5)

def test_orthogonal_vectors():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert math.isclose(cosine_similarity(v1, v2), 0.0, abs_tol=1e-5)
