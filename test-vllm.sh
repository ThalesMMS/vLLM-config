#!/bin/bash
#
# Test script for vLLM Server
# Auto-detects running model or allows specifying
#
# Usage:
#   ./test-vllm.sh              # Auto-detect model
#   ./test-vllm.sh llama        # Test Llama 3.2 3B
#   ./test-vllm.sh phi4         # Test Phi-4-mini
#

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Available models
declare -A MODELS
MODELS[llama]="meta-llama/Llama-3.2-3B-Instruct"
MODELS[phi4]="microsoft/Phi-4-mini-instruct"

echo -e "${CYAN}=== Testing vLLM server ===${NC}"
echo ""

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}Error: Server is not running on localhost:8000${NC}"
    echo "Start with: /home/user/start-vllm.sh"
    exit 1
fi

# Detect model or use argument
if [ -n "$1" ]; then
    case "$1" in
        llama|llama32) MODEL="${MODELS[llama]}" ;;
        phi4|phi) MODEL="${MODELS[phi4]}" ;;
        *) echo "Usage: $0 [llama|phi4]"; exit 1 ;;
    esac
else
    # Auto-detect by querying the server
    echo "Detecting model..."
    AVAILABLE=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')" 2>/dev/null)
    if [ -n "$AVAILABLE" ]; then
        MODEL="$AVAILABLE"
        echo "Detected model: $MODEL"
    else
        echo -e "${RED}Could not detect model${NC}"
        exit 1
    fi
fi

echo ""
echo "Testing model: $MODEL"
echo ""

# Send test request WITH system message
echo "Testing with system message..."
response=$(curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [
            {\"role\": \"system\", \"content\": \"You are a helpful assistant. Always respond in a single line.\"},
            {\"role\": \"user\", \"content\": \"What is 2+2? Answer with just the number.\"}
        ],
        \"max_tokens\": 10
    }")

# Check response
if echo "$response" | python3 -m json.tool > /dev/null 2>&1; then
    echo "Server response:"
    echo "$response" | python3 -m json.tool
    echo ""

    # Extract response content or tool output
    extracted=$(echo "$response" | python3 - <<'PY' 2>/dev/null
import json
import sys

data = json.load(sys.stdin)
message = data.get("choices", [{}])[0].get("message", {})
content = message.get("content") or ""
if content:
    print(f"CONTENT:{content}")
    raise SystemExit(0)
tool_calls = message.get("tool_calls") or []
if isinstance(tool_calls, list) and tool_calls:
    first_call = tool_calls[0]
    if isinstance(first_call, dict):
        function = first_call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            print(f"TOOL:{arguments}")
            raise SystemExit(0)
        if arguments is not None:
            print("TOOL:" + json.dumps(arguments, ensure_ascii=True))
            raise SystemExit(0)
print("EMPTY:")
PY
)

    kind="${extracted%%:*}"
    value="${extracted#*:}"
    if [ "$kind" = "CONTENT" ]; then
        echo -e "${GREEN}✅ Server is working!${NC}"
        echo "Model response: $value"
    elif [ "$kind" = "TOOL" ]; then
        echo -e "${GREEN}✅ Server is working (tool output)!${NC}"
        echo "Model tool response: $value"
    else
        echo -e "${RED}⚠️ Server responded but no content${NC}"
    fi
else
    echo -e "${RED}❌ Error: Server returned an invalid response${NC}"
    echo "Response: $response"
    exit 1
fi
