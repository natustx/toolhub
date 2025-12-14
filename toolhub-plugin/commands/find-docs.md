---
name: find-docs
description: Find and add documentation sources for a library/tool to toolhub
arguments:
  - name: tool
    description: "Name of the tool/library to find docs for"
    required: true
  - name: hint
    description: "Optional context to disambiguate (e.g., 'python web framework', 'graphql client')"
    required: false
---

# Find Documentation Sources for: $ARGUMENTS.tool

## Instructions

Find documentation sources for **$ARGUMENTS.tool** and help the user add them to toolhub.

### Step 1: Search

Use WebSearch with this query:
```
"$ARGUMENTS.tool" $ARGUMENTS.hint documentation OR github OR docs
```

(If no hint provided, omit it from the query)

### Step 2: Probe llms.txt

Use WebFetch to check for llms.txt at common locations. Try these URLs:
- `https://$ARGUMENTS.tool.dev/llms.txt`
- `https://$ARGUMENTS.tool.io/llms.txt`
- `https://docs.$ARGUMENTS.tool.dev/llms.txt`
- `https://$ARGUMENTS.tool.com/llms.txt`

Also try any domain found in the search results (e.g., if you find `fastapi.tiangolo.com`, try `https://fastapi.tiangolo.com/llms.txt`).

A successful fetch (not 404) means llms.txt exists - this is a high-quality source.

### Step 3: Verify Sources

For promising URLs from the search:
1. WebFetch to verify accessible
2. Confirm it contains documentation (not login page, 404, or unrelated)
3. Note a 1-line description

### Step 4: Present Results

Show sources ranked by quality (llms.txt first, then GitHub, then docs sites):

```
Found documentation sources for $ARGUMENTS.tool:

[1] https://github.com/org/repo (GitHub - description)
[2] https://tool.dev/llms.txt (llms.txt - AI-optimized docs)
[3] https://docs.tool.dev (Documentation site)
```

URLs should be full and clickable (user can cmd+click to preview in browser).

### Step 5: User Selection

Use AskUserQuestion with multi-select to ask which sources to add.

If only 1-2 good sources found, can use simpler confirmation.

### Step 6: Add to Toolhub

For each selected source, use the tool name (from Step 1) as the first argument:

```bash
toolhub add <tool-name> <url>
```

**Important:** All sources for the same tool should use the same tool name. For example:
```bash
toolhub add bittensor https://github.com/opentensor/bittensor
toolhub add bittensor https://github.com/opentensor/btcli
toolhub add bittensor https://github.com/opentensor/subtensor
```

This groups all sources under one tool entry.

Report success/failure. The add command may take 30-60 seconds per source as it indexes the documentation.

### Step 7: Confirm

Summarize what was added:
```
Added to toolhub (tool: <tool-name>):
- source 1 (X chunks)
- source 2 (Y chunks)

You can now search with: toolhub search "your query" --tool <tool-name>
```

## Error Handling

- If no sources found: Suggest user provide more context or check spelling
- If toolhub add fails: Report error, continue with remaining sources
- If WebFetch fails: Skip that source, note it in results
