# vector-cache-redis

[![CI](https://github.com/txltedxgod/vector-cache-redis/actions/workflows/ci.yml/badge.svg)](https://github.com/txltedxgod/vector-cache-redis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

High-throughput semantic vector cache in Python 3.11+ for LLM API cost reduction with deterministic feature-hashed embeddings, `OrderedDict` `O(1)` LRU eviction, monotonic TTL expiration, namespace isolation, and pipeline batch endpoints.

## Features
- **Semantic Feature Hashing**: L2-normalized word unigrams + character trigrams compute meaningful cosine similarity for paraphrased queries (zero external model overhead).
- **LRU + TTL State Machine**: `O(1)` access reordering, capacity-bounded eviction, automatic expiration sweeps, and hit-rate / eviction telemetry.
- **Namespace Isolation & Pipelining**: Isolate caches by tenant or LLM model (`namespace`) and execute batch `pipeline/store` & `pipeline/lookup`.
- **REST API**: `/health`, `POST /api/v1/cache/store`, `POST /api/v1/cache/lookup`, `POST /api/v1/cache/pipeline/{store,lookup}`, `DELETE /api/v1/cache`, `GET /api/v1/cache/stats`.

## Quick Start
```bash
ruff check .
pytest -v tests/
```
