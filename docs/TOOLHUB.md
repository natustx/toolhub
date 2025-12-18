# Toolhub Documentation

Toolhub is a local documentation index for AI coding agents. It crawls, chunks, embeds, and indexes documentation from various sources, then provides semantic search via CLI and HTTP API.

## User Interface

### CLI Commands

All commands are available via `toolhub <command>`.

#### Adding Documentation

```bash
# Add from GitHub (auto-detects docs, README, etc.)
toolhub add fastapi https://github.com/fastapi/fastapi

# Add multiple sources to same tool
toolhub add bittensor https://github.com/opentensor/bittensor
toolhub add bittensor https://github.com/opentensor/btcli

# Replace all sources for a tool
toolhub add fastapi https://github.com/fastapi/fastapi --replace

# Add from llms.txt
toolhub add anthropic https://docs.anthropic.com/llms.txt

# Add from website (crawls linked pages)
toolhub add example https://docs.example.com --max-pages 500
```

#### Searching

```bash
# Basic search (searches all indexed tools)
toolhub search "how to create routes"

# Search specific tool
toolhub search "authentication" --tool fastapi

# JSON output for programmatic use
toolhub search "error handling" --format json

# More results
toolhub search "validation" --limit 10

# Bypass daemon (useful for debugging)
toolhub search "query" --no-daemon

# Show timing breakdown
toolhub search "query" --profile
```

#### Managing Tools

```bash
# List all indexed tools
toolhub list

# Detailed info about a tool
toolhub info fastapi

# Remove a tool completely
toolhub remove fastapi

# Remove specific source from a tool
toolhub remove fastapi --source https://github.com/fastapi/fastapi

# Update stale documentation
toolhub update           # All stale tools
toolhub update fastapi   # Specific tool
toolhub update --force   # Force update all
```

#### Status and Daemon

```bash
# Overall status
toolhub status

# Daemon management
toolhub daemon status    # Check if running
toolhub daemon start     # Start manually
toolhub daemon stop      # Stop daemon
```

### HTTP API

The daemon exposes a REST API on `localhost:9742` (configurable).

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with tool/chunk counts |
| POST | `/tools/query` | Semantic search |
| GET | `/tools` | List all tools |
| GET | `/tools/{id}` | Get tool details |
| POST | `/tools/add` | Add and index a source |
| DELETE | `/tools/{id}` | Remove a tool |
| POST | `/stop` | Gracefully stop daemon |

#### Example: Search via API

```bash
curl -X POST http://localhost:9742/tools/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how to authenticate", "tool_ids": ["fastapi"], "limit": 5}'
```

Response:
```json
{
  "query": "how to authenticate",
  "tools_searched": ["fastapi"],
  "result_count": 5,
  "results": [
    {
      "tool_id": "fastapi",
      "content": "...",
      "source_file": "docs/security.md",
      "heading": "OAuth2",
      "heading_path": "Security > OAuth2",
      "is_code": false,
      "similarity": 0.87
    }
  ],
  "timings": {"embed_query": 0.012, "vector_search": 0.003}
}
```

### Configuration

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

[updates]
enabled = true
max_age_hours = 168   # 7 days

[crawling]
github_token = ""     # Optional, for private repos
```

## Architecture

### High-Level Data Flow

```
┌─────────────┐     ┌──────────┐     ┌─────────┐     ┌────────────┐
│   Source    │────▶│ Crawler  │────▶│ Indexer │────▶│   Store    │
│ (URL/repo)  │     │          │     │         │     │ (LanceDB)  │
└─────────────┘     └──────────┘     └─────────┘     └────────────┘
                         │                │                 │
                         ▼                ▼                 ▼
                    Cache Dir        Embeddings       Vector Index
                   (~/.toolhub/     (sentence-       (per-tool DB)
                     cache/)        transformers)
```

### Component Overview

```
src/toolhub/
├── cli.py              # Typer CLI entry point
├── daemon.py           # FastAPI HTTP server
├── client.py           # HTTP client for daemon communication
├── config.py           # TOML configuration loading
├── paths.py            # Path management (~/.toolhub/...)
├── registry.py         # Tool metadata (sources, chunk counts)
├── scheduler.py        # Staleness detection and update scheduling
│
├── crawler/            # Documentation fetching
│   ├── base.py         # Abstract Crawler class
│   ├── github.py       # GitHub repo crawler (clones, extracts docs)
│   ├── llmstxt.py      # llms.txt file crawler
│   └── website.py      # Website crawler (follows links)
│
├── indexer/            # Document processing
│   ├── chunker.py      # Markdown/text chunking with heading context
│   ├── embedder.py     # Sentence-transformers embedding
│   └── openapi.py      # OpenAPI spec parsing
│
└── store/              # Vector storage and search
    ├── lance.py        # LanceDB wrapper (per-tool stores)
    ├── search.py       # Semantic search orchestration
    └── operations.py   # SQLite FTS for API operations
```

### Core Components

#### Crawlers (`crawler/`)

Each crawler implements the `Crawler` abstract base class:

```python
class Crawler(ABC):
    @property
    def source_type(self) -> str: ...
    def can_handle(self, url: str) -> bool: ...
    def crawl(self, url: str, cache_dir: Path) -> CrawlResult: ...
```

**GitHubCrawler**: Clones repos (shallow), extracts markdown files from `docs/`, `README.md`, etc. Supports GitHub token for private repos.

**LlmsTxtCrawler**: Fetches `llms.txt` files (a standard for LLM-readable documentation). Extracts linked resources.

**WebsiteCrawler**: Recursively crawls websites. Converts HTML to markdown. Configurable page limit. Respects same-domain restriction.

#### Indexer (`indexer/`)

**Chunker**: Splits documents into semantic chunks:
- Respects heading hierarchy (preserves `heading_path` context)
- Token-aware splitting (default 500 tokens/chunk)
- Preserves code blocks as atomic units
- Stores metadata: source file, headings, is_code flag

**Embedder**: Wraps sentence-transformers:
- Default model: `all-MiniLM-L6-v2` (fast, good quality)
- Batched embedding for efficiency
- Model stays warm in daemon for fast subsequent queries

**OpenAPI Parser**: Extracts API operations from OpenAPI/Swagger specs:
- Parses endpoints, methods, parameters, responses
- Stores in SQLite with FTS for hybrid search

#### Store (`store/`)

**VectorStore (LanceDB)**: Per-tool vector databases:
- Each tool gets its own LanceDB table
- Cosine similarity search
- Stores: content, embedding, source_file, heading, heading_path, is_code

**OperationsStore (SQLite)**: Full-text search for API operations:
- Complements vector search with keyword matching
- Fast FTS5 queries on method, path, summary, description

**Search**: Orchestrates hybrid search:
1. Embeds query using same model as indexing
2. Searches all (or filtered) tool vector stores
3. Optionally searches operations store
4. Merges and ranks by similarity

#### Daemon (`daemon.py`, `client.py`)

**Purpose**: Keeps embedding model warm in memory. First load takes ~5s, subsequent queries ~100ms.

**Lifecycle**:
1. First `toolhub search` auto-starts daemon in background
2. Daemon loads model, writes PID file
3. Subsequent searches use HTTP API
4. `toolhub daemon stop` or system shutdown cleans up

**Client**: HTTP client with connection pooling. Falls back to direct search if daemon unavailable.

### Storage Layout

```
~/.toolhub/
├── config.toml              # User configuration
├── registry.json            # Tool metadata (sources, stats)
├── daemon.pid               # Running daemon PID
├── cache/                   # Crawled content
│   └── {tool_id}/
│       └── {source_hash}/   # Per-source cache
│           └── *.md
├── stores/                  # Vector databases
│   └── {tool_id}/
│       └── vectors.lance/   # LanceDB files
└── operations.db            # SQLite FTS for API ops
```

### Search Flow (Detailed)

```
Query: "how to authenticate users"

1. CLI receives query
   └── toolhub search "how to authenticate users"

2. Check daemon
   ├── Running? → HTTP POST to daemon
   └── Not running? → Auto-start, then HTTP POST

3. Daemon processes request
   ├── Load config (model name, limits)
   ├── Embed query → 384-dim vector
   ├── For each tool store:
   │   └── LanceDB cosine search → top N results
   ├── Optional: FTS search on operations
   └── Merge, sort by similarity, limit

4. Return results
   ├── JSON: raw structured data
   └── Markdown: formatted with headings, snippets
```

### Design Decisions

**Per-tool stores**: Each tool gets isolated storage. Enables easy add/remove without re-indexing everything. Trade-off: can't do cross-tool similarity in single query (searches each store separately).

**Lazy daemon**: Model loading is expensive (~5s). Daemon amortizes this cost. Auto-start on first search provides good UX without manual daemon management.

**Hybrid search**: Vector search excels at semantic similarity. FTS excels at exact keyword matching (API paths, method names). Combining both improves recall for technical queries.

**Heading context**: Chunks store full heading path (`Security > OAuth2 > Scopes`). This context helps LLMs understand where the snippet fits in the documentation hierarchy.
