# Find Documentation Sources

Use when the user wants to find and add documentation for a library, tool, or framework to toolhub. Triggers on: "find docs for X", "add X to toolhub", "where can I get docs for X", "I need X documentation indexed", or when working with an unfamiliar library that isn't indexed.

## When to Use

- User asks to add a tool/library to toolhub
- User needs documentation for something not yet indexed
- User asks "where can I find docs for X"
- You're about to search toolhub but the tool isn't indexed

## Process

### Step 1: Extract Tool Name and Context

From the user's request, identify:
- **Tool name**: The library/framework/tool to find (e.g., "lucide", "bittensor")
- **Hint** (optional): Context to disambiguate (e.g., "decentralized AI protocol", "python web framework")

### Step 2: Search for Documentation Sources

Use WebSearch to find documentation:

```
Query: "{tool}" {hint} documentation OR github OR docs
```

Examples:
- `"lucide" icon library documentation OR github OR docs`
- `"bittensor" decentralized AI documentation OR github OR docs`

### Step 3: Probe for llms.txt

Use WebFetch to check common llms.txt locations. These are AI-optimized documentation files.

Try these URLs (replace {tool} with the tool name):
- `https://{tool}.dev/llms.txt`
- `https://{tool}.io/llms.txt`
- `https://docs.{tool}.dev/llms.txt`
- `https://{tool}.com/llms.txt`

Also check any domain found in search results:
- If search found `lucide.dev`, try `https://lucide.dev/llms.txt`

For each probe, a successful response (not 404/error) means llms.txt exists.

### Step 4: Verify and Describe Sources

For each candidate URL from search results:
1. Use WebFetch to verify it's accessible
2. Check it contains documentation (not a login page, 404, or unrelated content)
3. Extract a brief 1-line description of what the source contains

### Step 5: Present Sources to User

Show a numbered list with clickable URLs, ranked by quality:
1. llms.txt sources first (best for AI)
2. GitHub repos second (reliable)
3. Documentation sites third

Format:
```
Found documentation sources for {tool}:

[1] https://github.com/org/repo (GitHub - brief description)
[2] https://tool.dev/llms.txt (llms.txt - AI-optimized docs)
[3] https://tool.dev/docs (Documentation site)

Which sources do you want to add to toolhub?
```

### Step 6: Let User Select

Use AskUserQuestion with multi-select to let user choose which sources to add.

If only 1-2 sources found, can ask simpler yes/no confirmation.

### Step 7: Add Selected Sources

For each selected source, use the tool name as the first argument:

```bash
toolhub add <tool-name> <url>
```

**Important:** All sources for the same tool must use the same tool name. For example:
```bash
toolhub add bittensor https://github.com/opentensor/bittensor
toolhub add bittensor https://github.com/opentensor/btcli
```

This groups all sources under one tool entry in toolhub.

Report success/failure for each.

## Example Interaction

**User:** "Add bittensor to toolhub, it's a decentralized AI protocol"

**Assistant:**
1. WebSearch: `"bittensor" decentralized AI protocol documentation OR github OR docs`
2. WebFetch probe: `https://bittensor.dev/llms.txt`, `https://docs.bittensor.com/llms.txt`
3. Verify top results from search
4. Present:
   ```
   Found documentation sources for bittensor:

   [1] https://github.com/opentensor/bittensor (GitHub - decentralized AI network)
   [2] https://docs.bittensor.com (Official documentation)

   Which sources do you want to add to toolhub?
   ```
5. User selects [1, 2]
6. Run `toolhub add bittensor https://github.com/opentensor/bittensor`
7. Run `toolhub add bittensor https://docs.bittensor.com`
8. Confirm: "Added 2 sources for bittensor to toolhub"

## Important Notes

- **Clickable URLs**: Always show full URLs so user can cmd+click to preview
- **Multiple sources OK**: User may want both GitHub (code examples) and llms.txt (structured docs)
- **Verify before showing**: Don't show 404s or login pages
- **Be efficient**: Run WebFetch probes in parallel when possible
- **Handle failures gracefully**: If `toolhub add` fails, report error and continue with others
