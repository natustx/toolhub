---
name: toolhub-search
description: Search local documentation index for library information
arguments:
  - name: query
    description: "Search query with optional tool prefix (e.g., 'beads:dependencies' or 'fastapi:authentication')"
    required: true
---

Search the toolhub documentation index for: $ARGUMENTS.query

## Query Format

The query supports `tool:query` syntax to search a specific tool:
- `beads:dependencies` → searches only the beads tool
- `fastapi:authentication` → searches only fastapi
- `how to create routes` → searches all indexed tools

## Instructions

1. Parse the query to check for `tool:query` format (split on first colon)
2. Run the appropriate search command:

If query contains a colon (e.g., `beads:dependencies`):
```bash
toolhub search "QUERY_PART" --tool TOOL_PART --limit 5 --format markdown
```

If no colon (searches all tools):
```bash
toolhub search "$ARGUMENTS.query" --limit 5 --format markdown
```

3. Summarize the results in 2-3 sentences, highlighting the most relevant information found.

If no results are found, suggest:
1. Check what tools are indexed with `toolhub list`
2. Add the relevant documentation with `toolhub add <name> <url>`
