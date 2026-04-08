#!/bin/bash
# Multi-turn conversation test: verify the model produces correct
# answers across multiple turns. Uses greedy decoding for determinism.
set -e

LLMC="${1:-./llmc}"
MODEL="${2:-mistral-7b-instruct-v0.3-Q4_K_M.gguf}"

pass=0
fail=0

check() {
    local desc="$1"
    local input="$2"
    local expect="$3"
    
    output=$(echo -e "$input" | "$LLMC" --greedy "$MODEL" 2>/dev/null)
    
    if echo "$output" | grep -qi "$expect"; then
        echo "  ok: $desc"
        pass=$((pass + 1))
    else
        echo "  FAIL: $desc"
        echo "    expected to contain: $expect"
        echo "    got: $(echo "$output" | head -c 200)"
        fail=$((fail + 1))
    fi
}

echo "multiturn tests ($MODEL):"

# Single turn
check "single turn: capital of France" \
    "What is the capital of France?" \
    "Paris"

# Multi-turn: follow-up question
check "multi-turn: capital then follow-up" \
    "What is the capital of France?\nWhat about Germany?" \
    "Berlin"

# Multi-turn: context retention
check "multi-turn: remembers first answer" \
    "What is the capital of France?\nWhat about Germany?" \
    "Paris"

# Verify both turns produce output (not empty)
output=$(echo -e "Hello\nHow are you?" | "$LLMC" --greedy "$MODEL" 2>/dev/null)
lines=$(echo "$output" | grep -c '.' || true)
if [ "$lines" -ge 2 ]; then
    echo "  ok: multi-turn produces multiple responses"
    pass=$((pass + 1))
else
    echo "  FAIL: multi-turn should produce at least 2 responses"
    echo "    got $lines lines: $(echo "$output" | head -c 200)"
    fail=$((fail + 1))
fi

# EOS: model should stop reasonably (not blabber for 200 tokens)
output=$(echo "What is 2+2?" | "$LLMC" --greedy "$MODEL" 2>/dev/null)
len=${#output}
if [ "$len" -lt 500 ]; then
    echo "  ok: concise response ($len chars)"
    pass=$((pass + 1))
else
    echo "  FAIL: response too long ($len chars), EOS not triggered"
    echo "    first 200 chars: ${output:0:200}"
    fail=$((fail + 1))
fi

echo ""
if [ "$fail" -gt 0 ]; then
    echo "multiturn: $fail/$((pass+fail)) FAILED"
    exit 1
else
    echo "multiturn: all $pass tests passed"
fi
