#!/bin/bash
#
# Start script for vLLM Server
# Supports multiple models: GPT-OSS-20B and Phi-4-mini
#
# Usage:
#   ./start-vllm.sh              # Interactive menu
#   ./start-vllm.sh gptoss       # Direct GPT-OSS-20B
#   ./start-vllm.sh phi4         # Direct Phi-4-mini
#

set -e

# Check if running as root; otherwise re-exec with sudo
if [ "$EUID" -ne 0 ]; then
    echo "Elevating privileges to root..."
    exec sudo "$0" "$@"
fi

# Output colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Model configuration
declare -A MODELS
MODELS[gptoss]="openai/gpt-oss-20b"
MODELS[phi4]="microsoft/Phi-4-mini-instruct"

declare -A MODEL_CONFIGS
# Format: "max_model_len:max_num_seqs:gpu_mem_util:enforce_eager"
MODEL_CONFIGS[gptoss]="512:2:0.95:true"      # Higher context; more VRAM pressure
MODEL_CONFIGS[phi4]="16384:16:0.90:false"     # 3.8B model (~3GB VRAM, 128K context)

# Chat templates for system message support (GPT-OSS uses the built-in template)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A CHAT_TEMPLATES
CHAT_TEMPLATES[gptoss]=""
CHAT_TEMPLATES[phi4]=""       # Uses built-in chat template

declare -A MODEL_NAMES
MODEL_NAMES[gptoss]="GPT-OSS-20B (OpenAI)"
MODEL_NAMES[phi4]="Phi-4-mini-instruct (3.8B)"

show_menu() {
    echo -e "${CYAN}=== vLLM Server - Model Selection ===${NC}"
    echo ""
    echo "Available models:"
    echo ""
    echo -e "  ${GREEN}1)${NC} GPT-OSS-20B (OpenAI)"
    echo "     - 21B params (3.6B active), MXFP4"
    echo "     - VRAM: ~14.6GB | Context: 512 tokens"
    echo ""
    echo -e "  ${GREEN}2)${NC} Phi-4-mini-instruct (3.8B)"
    echo "     - 3.8B params, excellent reasoning"
    echo "     - VRAM: ~3GB | Context: 128K tokens"
    echo ""
    echo -e "  ${YELLOW}0)${NC} Exit"
    echo ""
}

select_model() {
    while true; do
        show_menu
        read -p "Choose a model [1/2/0]: " choice
        case $choice in
            1) MODEL_KEY="gptoss"; break ;;
            2) MODEL_KEY="phi4"; break ;;
            0) echo "Exiting."; exit 0 ;;
            *) echo -e "${YELLOW}Invalid option. Try again.${NC}"; echo "" ;;
        esac
    done
}

stop_existing_vllm() {
    # Check processes using the GPU
    local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')

    # Check "vllm serve" process (server, not this script)
    local serve_pids=$(pgrep -f "vllm serve" 2>/dev/null || true)

    if [ -n "$gpu_pids" ] || [ -n "$serve_pids" ]; then
        echo -e "${YELLOW}vLLM instance detected. Shutting down...${NC}"

        # Kill "vllm serve" specifically
        if [ -n "$serve_pids" ]; then
            echo "$serve_pids" | xargs kill -9 2>/dev/null || true
        fi
        sleep 2

        # Kill processes still using the GPU
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
        for pid in $gpu_pids; do
            echo "Killing GPU process: $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2

        echo "Previous instance shut down."
    fi

    # Check if memory was freed
    local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$free_mem" ] && [ "$free_mem" -lt 10000 ]; then
        echo -e "${YELLOW}Warning: GPU still low on free memory (${free_mem} MiB). Waiting...${NC}"
        sleep 3
    fi
}

start_server() {
    local model_key=$1
    local model_id="${MODELS[$model_key]}"
    local model_name="${MODEL_NAMES[$model_key]}"
    local config="${MODEL_CONFIGS[$model_key]}"
    local chat_template="${CHAT_TEMPLATES[$model_key]}"

    # Parse config
    IFS=':' read -r max_model_len max_num_seqs gpu_mem_util enforce_eager <<< "$config"

    echo ""
    echo -e "${GREEN}=== Starting $model_name ===${NC}"
    echo ""

    # Stop any existing vLLM instance
    stop_existing_vllm

    # Stop display manager to free VRAM
    echo "Stopping display manager..."
    sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

    # Check available GPU memory
    echo "Free GPU memory:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader
    echo ""

    # Activate virtual environment
    echo "Activating virtual environment..."
    source ~/vllm-env/bin/activate

    # Required environment variable
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Build command
    # --served-model-name lets you use "default" instead of the full model name
    # --chat-template adds support for OpenAI-style system messages
    cmd="vllm serve $model_id --port 8000 --served-model-name default --max-model-len $max_model_len --max-num-seqs $max_num_seqs --gpu-memory-utilization $gpu_mem_util"

    # Add chat template if specified
    if [ -n "$chat_template" ]; then
        if [ -f "$chat_template" ]; then
            cmd="$cmd --chat-template $chat_template"
            echo -e "${CYAN}Chat template: $chat_template${NC}"
        else
            echo -e "${YELLOW}Warning: Chat template not found: $chat_template${NC}"
            echo -e "${YELLOW}System messages may not work correctly.${NC}"
        fi
    fi

    if [ "$enforce_eager" = "true" ]; then
        cmd="$cmd --enforce-eager"
    fi

    echo ""
    echo "Command:"
    echo "$cmd"
    echo ""
    echo -e "${GREEN}Starting server on port 8000...${NC}"
    echo ""

    # Run
    eval $cmd
}

# Main
if [ -n "$1" ]; then
    # Argument passed directly
    case "$1" in
        gptoss|gpt|1) MODEL_KEY="gptoss" ;;
        phi4|phi|2) MODEL_KEY="phi4" ;;
        *) echo "Usage: $0 [gptoss|phi4]"; exit 1 ;;
    esac
else
    # Interactive menu
    select_model
fi

start_server "$MODEL_KEY"
