# vector-cache-redis

High-throughput semantic vector cache for LLM API cost reduction with cosine similarity deduplication.

## Architecture & Design

Built with modern Python 3.11 asynchronous patterns, strict Pydantic v2 schemas, and standard 12-factor application conventions.

### Directory Layout

```
├── src/
│   ├── api/v1/         # Versioned REST controllers
│   ├── core/           # Settings, structured logging & domain exceptions
│   ├── schemas/        # Request / Response validation schemas
│   ├── services/       # Core business logic and storage state machines
│   └── main.py         # Application lifespan & middleware integration
├── tests/
│   ├── conftest.py     # Shared fixtures and mock clients
│   └── test_service.py # Automated pytest test cases
├── Dockerfile          # Multi-stage production container
├── Makefile            # Standard developer commands (lint, test, run)
└── pyproject.toml      # Tooling configuration (Ruff, mypy, pytest)
```

## Quick Start

```bash
make install
make test
make run
```

## Production Container

```bash
docker compose up -d --build
```

## License
MIT License
