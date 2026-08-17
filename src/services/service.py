import numpy as np
import hashlib
from typing import Dict, Any, Optional, Tuple

class SemanticCacheService:
    def __init__(self, threshold: float = 0.88, dim: int = 64):
        self.threshold = threshold
        self.dim = dim
        self._entries: Dict[str, Dict[str, Any]] = {}

    def _embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.lower().strip().encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def lookup(self, query: str) -> Tuple[bool, Optional[str], float]:
        if not self._entries:
            return False, None, 0.0
        q_emb = self._embed(query)
        best_sim = 0.0
        best_resp = None
        for k, v in self._entries.items():
            sim = float(np.dot(q_emb, v["embedding"]))
            if sim > best_sim:
                best_sim = sim
                if sim >= self.threshold:
                    best_resp = v["response"]
        return (best_resp is not None, best_resp, round(best_sim, 4))

    def store(self, query: str, response: str):
        emb = self._embed(query)
        self._entries[query] = {"embedding": emb, "response": response}
