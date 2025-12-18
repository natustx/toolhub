# Using Toolhub for Documentation Lookups

Use when working with unfamiliar libraries, APIs, or tools to search indexed documentation. Triggers on mentions of library usage questions, API lookups, "how do I use X", implementation questions, or when you need documentation for a dependency.

## Choose the Right Agent

Toolhub has two agents for different needs:

### toolhub-librarian (Quick Lookups)
Use for simple, focused questions:
- "What's the syntax for X?"
- "Show me an example of Y"
- "How do I call this function?"
- Quick API reference checks

Returns: 200-300 token concise summary

### toolhub-researcher (Deep Dives)
Use for complex, open-ended questions:
- "Walk me through setting up X"
- "Explain how Y works end-to-end"
- "Describe the full process for Z"
- "Help me understand the architecture of X"

Returns: 500-1000 token comprehensive guide with steps

## Agent Selection Decision Tree

```
User asks documentation question?
├── Is it a quick lookup? (syntax, example, specific fact)
│   └── Yes → toolhub-librarian
├── Is it a deep explanation? (walkthrough, process, how it works)
│   └── Yes → toolhub-researcher
└── Unclear?
    └── Default to librarian; switch to researcher if answer seems incomplete
```

**Signal words for researcher:**
- "walk me through", "step by step"
- "explain how", "help me understand"
- "full process", "end to end"
- "comprehensive", "detailed guide"

## When to Use Toolhub

Use toolhub agents when:

- You need documentation for a library, framework, or tool
- The user asks "how do I use X" for an indexed tool
- You're implementing code that uses an unfamiliar API
- You need to verify correct usage of a library function
- You want to find examples or patterns for a specific library

## How It Works

Toolhub is a local documentation index. It stores embeddings of documentation from GitHub repos and searches them semantically.

### Query Syntax

Use `tool:query` syntax to search a specific tool:
- `beads:dependencies` → searches only the beads tool
- `fastapi:authentication` → searches only fastapi
- `pandas:merge dataframes` → searches only pandas

### Check What's Indexed

Before searching, check what tools are available:

```bash
toolhub list
```

This shows all indexed tools with their chunk counts.

### Search Documentation

Use the `toolhub-librarian` agent for searches. It will:
1. Query the vector store
2. Return 200-300 token summaries of relevant documentation
3. Include source file and heading information

Example: "Search toolhub for beads:how to configure dependencies"

### Add New Documentation

If a tool isn't indexed, add it:

```bash
toolhub add <name> <url>
# Example:
toolhub add fastapi https://github.com/fastapi/fastapi
```

## IMPORTANT: Tool Selection

**When the user asks a documentation question without specifying which tool:**

1. First, run `toolhub list` to see available tools
2. Use AskUserQuestion to ask which tool they want to search:

```
Which tool's documentation should I search?

Options based on `toolhub list` output:
- beads (issue tracking)
- fastapi (web framework)
- [other indexed tools...]
```

3. Only proceed with the search after the user selects a tool

**When the tool IS clear** (user mentions it by name, or context makes it obvious):
- Proceed directly with `tool:query` syntax
- Example: "How do beads dependencies work?" → search `beads:dependencies`

## Decision Tree

```
User asks documentation question?
├── Tool explicitly mentioned? (e.g., "in beads", "for fastapi")
│   └── Yes → Search with tool:query syntax
├── Tool obvious from context? (e.g., working on beads-related code)
│   └── Yes → Search with tool:query syntax
└── Tool unclear?
    └── Ask user which tool to search (use AskUserQuestion)
```

## Important Notes

1. **Toolhub is local** - it only searches what you've indexed
2. **Semantic search** - queries match meaning, not just keywords
3. **Respects context** - results include file paths and headings
4. **Fast** - uses daemon with warm model for ~85ms queries

## Anti-Patterns

- Don't search all tools when the user clearly means a specific one
- Don't guess which tool - ask if unclear
- Don't search toolhub for general coding questions unrelated to indexed tools
- Don't expect toolhub to have tools you haven't indexed
