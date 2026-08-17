import numpy as np
import hashlib
from typing import Optional, Dict, Any, Tuple

class SemanticCache:
    def __init__(self, threshold: float = 0.88, dim: int = 64):
        self.threshold = threshold
        self.dim = dim
        self.entries: Dict[str, Dict[str, Any]] = {}

    def _embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.lower().strip().encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim)
        return v / np.linalg.norm(v)

    def get(self, query: str) -> Tuple[Optional[str], float]:
        q_emb = self._embed(query)
        best_sim = 0.0
        best_response = None

        for k, item in self.entries.items():
            sim = float(np.dot(q_emb, item["embedding"]))
            if sim > best_sim:
                best_sim = sim
                if sim >= self.threshold:
                    best_response = item["response"]

        return (best_response, round(best_sim, 4))

    def set(self, query: str, response: str):
        emb = self._embed(query)
        self.entries[query] = {"embedding": emb, "response": response}
