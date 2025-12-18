# toolhub

Local documentation index for AI coding agents.

## Prerequisites

Tool Hub requires **PostgreSQL** (with pgvector extension) and **MinIO** (S3-compatible storage).

### macOS (Homebrew) - Native Setup

```bash
# Install dependencies
brew install postgresql@17 pgvector minio/stable/minio minio/stable/mc go-task

# Start services
brew services start postgresql@17
minio server ~/minio-data &

# Run idempotent setup (creates database, schema, bucket, config)
task setup
```

**Individual setup commands:**
```bash
task setup:check   # Verify prerequisites are installed and running
task setup:db      # Create database and apply schema
task setup:s3      # Create MinIO bucket
task setup:config  # Create ~/.toolhub/config.toml
```

### Docker Compose (Alternative)

```bash
docker compose -f docker-compose.test.yml up -d
```

This starts:
- PostgreSQL with pgvector on port 5433
- MinIO on port 9010 (API) and 9011 (console)

Note: Test containers use different ports/credentials. Update `~/.toolhub/config.toml` accordingly.

## Installation

```bash
# Install globally (available from anywhere)
uv tool install --editable .

# Or run directly without installing
uv run toolhub --help
```

## Usage

```bash
# Add a documentation source
toolhub add fastapi https://github.com/fastapi/fastapi
toolhub add bittensor https://github.com/opentensor/bittensor
toolhub add mylib https://docs.example.com --max-pages 100

# Re-index existing source
toolhub add fastapi https://github.com/fastapi/fastapi --replace

# Search indexed documentation
toolhub search "how to create routes"
toolhub search "authentication" --tool fastapi
toolhub search "validators" --format json

# Manage indexed tools
toolhub list
toolhub info fastapi
toolhub status
toolhub update          # Update stale sources
toolhub remove fastapi
```

## Daemon (Fast Queries)

The first search auto-starts a background daemon that keeps the embedding model warm. Subsequent searches are near-instant (~100ms vs ~5s).

```bash
# First search starts daemon automatically
toolhub search "authentication"   # ~5s (starts daemon + loads model)

# Subsequent searches are fast
toolhub search "how to create routes"   # ~100ms
```

### Daemon Management

```bash
toolhub daemon status   # Check if daemon is running
toolhub daemon start    # Start daemon manually
toolhub daemon stop     # Stop daemon
```

### Bypass Daemon

```bash
# Run directly without daemon (useful for debugging)
toolhub search "query" --no-daemon
```

### HTTP API

The daemon exposes an HTTP API on `localhost:9742`:

```bash
# Health check (no version prefix)
curl http://localhost:9742/health

# Search via API
curl -X POST http://localhost:9742/api/v1/tools/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how to authenticate", "limit": 5}'

# List tools
curl http://localhost:9742/api/v1/tools

# Add a tool
curl -X POST http://localhost:9742/api/v1/tools/add \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/fastapi/fastapi", "name": "fastapi"}'
```

## Entities & Evidence

Tool Hub includes a structured knowledge system for managing entities with evidence-backed claims.

### Entity Types

```bash
# List registered entity types
toolhub entity type list

# Show entity type schema
toolhub entity type show competitor

# Register a new entity type with JSON Schema
toolhub entity type register competitor '{"type": "object", "properties": {"name": {"type": "string"}, "features": {"type": "array", "items": {"type": "string"}}}}' --description "Competitor for analysis"
```

### Entities

```bash
# Create an entity
toolhub entity create competitor "Acme Corp" --profile '{"name": "Acme Corp", "features": ["api", "dashboard"]}'

# List entities
toolhub entity list --type competitor

# Show entity details
toolhub entity show <entity-id> --citations
```

### Evidence & Reports

```bash
# Add evidence linking claims to source chunks
toolhub evidence add <entity-id> "features" --chunk-id <chunk-id> --quote "Acme offers API and dashboard"

# Generate reports
toolhub report list
toolhub report generate competitor-feature-matrix
toolhub report generate fundraising-timeline --collection research
```

**Note:** Reports expect specific entity types. Run `task setup:entities` to register the default schemas (competitor, wisdom).

## Configuration

Settings are stored in `~/.toolhub/config.toml`:

```toml
[daemon]
host = "127.0.0.1"
port = 9742

[embedding]
model = "all-MiniLM-L6-v2"
chunk_size_tokens = 500

[search]
default_limit = 5

[crawling]
github_token = ""  # Optional: for higher GitHub API rate limits

[updates]
enabled = false
interval_hours = 24
max_age_hours = 168

[postgres]
url = "postgresql://localhost:5432/toolhub"
pool_size = 5       # Not yet implemented
pool_timeout = 30   # Not yet implemented

[s3]
endpoint_url = "http://localhost:9000"
bucket = "toolhub"
access_key = "minioadmin"   # Change in production!
secret_key = "minioadmin"   # Change in production!
region = "us-east-1"
use_ssl = false
```

## Development

### Task Commands

All commands use [Task](https://taskfile.dev/) (`brew install go-task`):

```bash
# Setup
task setup          # Full setup (check + db + s3 + config + entities)
task setup:check    # Verify prerequisites
task setup:db       # Create database and schema
task setup:s3       # Create MinIO bucket
task setup:config   # Create config file
task setup:entities # Register entity types (competitor, wisdom)

# Development
task dev:up         # Start services via honcho
task dev:down       # Stop services

# Database
task db:migrate     # Apply pending migrations
task db:status      # Show migration status
task db:reset       # Drop and recreate (DESTRUCTIVE)

# Testing
task test           # Run full test suite
task test:integration  # Integration tests only
task test:docker:up    # Start test containers
task test:docker:down  # Stop test containers

# Code quality
task lint           # Run ruff linter
task format         # Run ruff formatter

# API
task api:health     # Check daemon health
```

### Running Tests

```bash
# Using native setup
task setup
uv run pytest tests/ -v

# Using Docker (isolated)
task test:docker:up
uv run pytest tests/ -v
task test:docker:down
```

## Uninstall

```bash
uv tool uninstall toolhub
```
