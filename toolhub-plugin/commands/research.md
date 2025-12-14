---
name: research
description: Deep-dive documentation research for complex questions requiring synthesis
arguments:
  - name: query
    description: "Research query with optional tool prefix (e.g., 'bittensor:subnet setup process' or 'explain fastapi authentication flow')"
    required: true
---

Research the toolhub documentation for: $ARGUMENTS.query

This is a **deep-dive** query, not a quick lookup. Provide a comprehensive, synthesized answer.

## Query Format

The query supports `tool:query` syntax to research a specific tool:
- `bittensor:subnet setup process` → deep research on bittensor subnets
- `fastapi:authentication flow` → comprehensive auth explanation
- `how dependency injection works` → research across all indexed tools

## Instructions

1. Parse the query to check for `tool:query` format (split on first colon)

2. **Decompose into 2-3 focused searches** based on different aspects of the question:
   - For "subnet setup process": search setup, configuration, deployment
   - For "authentication flow": search auth basics, tokens, middleware

3. Run sequential searches with higher limits:

If query contains a colon (e.g., `bittensor:subnet setup`):
```bash
# Search 1: Core concept
toolhub search "ASPECT_1" --tool TOOL_PART --limit 10 --format markdown

# Search 2: Configuration/details
toolhub search "ASPECT_2" --tool TOOL_PART --limit 10 --format markdown

# Search 3: Implementation/advanced (if needed)
toolhub search "ASPECT_3" --tool TOOL_PART --limit 10 --format markdown
```

If no colon (searches all tools):
```bash
toolhub search "ASPECT_1" --limit 10 --format markdown
# ... repeat for other aspects
```

4. **Synthesize** the results into a comprehensive response:

```
## Overview
[2-3 sentence high-level summary]

## Step-by-Step Guide

### 1. [First Phase]
[Detailed explanation with context]

### 2. [Second Phase]
[Implementation details]

### 3. [Third Phase]
[Completion/advanced topics]

## Code Examples
[Relevant snippets from docs]

## Key Considerations
- [Important gotchas]
- [Best practices]

## Sources
- [tool]: [file paths referenced]
```

5. Target **500-1000 tokens** - this is meant to be thorough, not brief.

If results are sparse, acknowledge gaps and suggest adding more documentation with `toolhub add <url>`.
