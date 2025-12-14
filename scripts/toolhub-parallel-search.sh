#!/bin/bash
# toolhub-parallel-search.sh - Run multiple toolhub searches in parallel
# Usage: toolhub-parallel-search.sh <tool> <query1> <query2> [query3] ...
#        toolhub-parallel-search.sh bittensor "subnet setup" "subnet config"

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: toolhub-parallel-search.sh <tool> <query1> <query2> [query3] ..."
    echo "Example: toolhub-parallel-search.sh bittensor 'subnet setup' 'subnet config'"
    exit 1
fi

TOOL="$1"
shift
QUERIES=("$@")

# Create temp directory, auto-cleanup on exit
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Launch all searches in parallel
i=0
for query in "${QUERIES[@]}"; do
    toolhub search "$query" --tool "$TOOL" --limit 10 --format markdown > "$TMPDIR/$i.txt" 2>&1 &
    ((i++))
done

# Wait for all to complete
wait

# Combine results in order
for j in $(seq 0 $((i-1))); do
    if [[ -f "$TMPDIR/$j.txt" ]]; then
        cat "$TMPDIR/$j.txt"
        echo ""
    fi
done
