# Running LLMs on vLLM - Complete Tutorial

This tutorial documents the complete setup process for running LLMs locally using vLLM on an NVIDIA RTX 5080 (16GB VRAM).

## Reproducing on a New System (Step by Step)

### Mandatory Prerequisites

Before running the installation script, the new system **must have**:

1. **NVIDIA driver installed and working**
   ```bash
   # Check if the driver is installed
   nvidia-smi

   # Should show the GPU and driver version. Example:
   # +-----------------------------------------------------------------------------+
   # | NVIDIA-SMI 570.xx.xx    Driver Version: 570.xx.xx    CUDA Version: 12.8    |
   # | GPU Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   # | NVIDIA GeForce RTX 5080  Off |   00000000:01:00.0  On |                  N/A |
   # +-----------------------------------------------------------------------------+

   # If not installed on Pop!_OS:
   sudo apt install system76-driver-nvidia
   # Or on Ubuntu:
   sudo apt install nvidia-driver-570  # or the latest version
   ```

2. **Internet connection** (for downloading ~50GB of models)

3. **Disk space**: ~60GB free in `~/.cache/huggingface/`

4. **Recommended RAM**: 32GB+ (to load the model before sending to GPU)

### Steps for a New Installation

```bash
# 1. Copy the 4 files to the new system
scp setup-vllm.sh start-vllm.sh test-vllm.sh vllm-tutorial.md user@new-pc:~

# 2. On the new system, verify prerequisites
nvidia-smi                    # Should work
df -h ~/.cache                # Check space

# 3. Run the installation script (installs CUDA, vLLM, configures the system)
sudo bash ~/setup-vllm.sh

# 4. Copy the improved scripts (multi-model menu)
cp ~/start-vllm.sh ~/start-vllm.sh
cp ~/test-vllm.sh ~/test-vllm.sh
chmod +x ~/start-vllm.sh ~/test-vllm.sh

# 5. First run (downloads the model, ~50GB)
~/start-vllm.sh

# 6. In another terminal, test
~/test-vllm.sh
```

### Required Files

| File | Required | Purpose |
|------|----------|---------|
| `setup-vllm.sh` | ✅ Yes | Full installation (CUDA, vLLM, systemd) |
| `start-vllm.sh` | ✅ Yes | Start server (menu with 2 models) |
| `test-vllm.sh` | Optional | Test whether the server responds |
| `vllm-tutorial.md` | Optional | Documentation/reference |

---

## Available Models

| Model | Params | VRAM | Context | Use |
|-------|--------|------|---------|-----|
| **Llama 3.2 3B Instruct** | 3B | ~3GB | 128K tokens | Tool calling, chat |
| **Phi-4-mini-instruct** | 3.8B | ~3GB | 128K tokens | Reasoning, fast |

## System Requirements

- **GPU**: NVIDIA RTX 5080 (16GB VRAM) or similar with 16GB+ VRAM
- **OS**: Pop!_OS / Ubuntu 24.04
- **CUDA**: 12.8+ (required for Blackwell architecture GPUs)
- **RAM**: 32GB+ recommended
- **Storage**: ~60GB for model files (both models)

## Prerequisites Installed

### 1. Disable System Sleep (for SSH access)

```bash
# Mask sleep targets to prevent system from sleeping
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Create logind config to ignore lid close and idle
sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/nosleep.conf > /dev/null << 'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
EOF

sudo systemctl restart systemd-logind
```

### 2. Install Python and pip

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
```

### 3. Install CUDA 12.8 (Required for Blackwell GPUs)

```bash
# Add NVIDIA CUDA repository
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12.8 toolkit
sudo apt install -y cuda-toolkit-12-8

# Verify installation
/usr/local/cuda-12.8/bin/nvcc --version
```

## vLLM Installation

### 1. Create Virtual Environment

```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip
```

### 2. Install vLLM

```bash
pip install vllm
```

This installs vLLM v0.13.0 along with all dependencies including:
- PyTorch 2.9.0
- CUDA libraries (bundled)
- Transformers, tokenizers, etc.

## Running Models

### Important: Free GPU Memory First

For optimal performance, stop the display manager to free GPU memory:

```bash
# Stop display manager to free GPU memory
sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

# Verify GPU memory is free
nvidia-smi --query-gpu=memory.free --format=csv,noheader
# Should show ~15.5GB+ free
```

### Start the vLLM Server

**Quick method** (interactive menu):

```bash
/home/user/start-vllm.sh
```

The script shows a menu to choose the model:
```
=== vLLM Server - Model Selection ===

Available models:

  1) Llama 3.2 3B Instruct (Meta)
     - 3B params, tool calling support
     - VRAM: ~3GB | Context: 128K tokens

  2) Phi-4-mini-instruct (3.8B)
     - 3.8B params, excellent reasoning
     - VRAM: ~3GB | Context: 128K tokens

  0) Exit

Choose a model [1/2/0]:
```

**Direct method** (no menu):

```bash
# Llama 3.2 3B
/home/user/start-vllm.sh llama

# Phi-4-mini
/home/user/start-vllm.sh phi4
```

**Manual method** (individual commands):

```bash
# 1. Stop display manager to free VRAM
sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

# 2. Activate virtual environment
source /root/vllm-env/bin/activate

# 3. Required environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Start server (Llama 3.2 3B)
vllm serve unsloth/Llama-3.2-3B-Instruct \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90

# Or for Phi-4-mini:
vllm serve microsoft/Phi-4-mini-instruct \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--max-model-len` | 512 | Maximum context length (higher VRAM use) |
| `--max-num-seqs` | 2 | Maximum concurrent sequences |
| `--gpu-memory-utilization` | 0.95 | Use 95% of available VRAM |
| `--enforce-eager` | - | Disable CUDA graphs (saves memory) |

### Test the Server

**Test script** (auto-detects model):

```bash
/home/user/test-vllm.sh
```

**Manual test via cURL:**

```bash
# Using --served-model-name default (recommended):
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }' | python3 -m json.tool

# Or using full model name (Llama 3.2 3B):
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

### Example Response

```json
{
    "id": "chatcmpl-b4cc7de53622ebbe",
    "model": "default",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "4"
            },
            "finish_reason": "stop"
        }
    ]
}
```

## Optimization Strategies Used

These lightweight models (~3B parameters each) run comfortably on 16GB of VRAM with room to spare. Below are optimization techniques that can be applied for larger models.

---

### 1. System Sleep Prevention

**Problem:** Loss of SSH connection when the system enters sleep mode.

**Solution:** Disable all systemd sleep targets and configure logind.

```bash
# Mask sleep targets
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Configure logind to ignore lid close and idle
sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/nosleep.conf > /dev/null << 'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
EOF

sudo systemctl restart systemd-logind
```

**Impact:**
- System stays active 24/7
- SSH/VNC connections are not interrupted
- Essential for headless servers

---

### 2. CUDA 12.8 Installation (Blackwell Architecture)

**Problem:** RTX 50-series (Blackwell) GPUs require CUDA 12.8+. Earlier versions fail with `compute_120a not supported`.

**Solution:** Install CUDA 12.8 from the official NVIDIA repository.

```bash
# Add NVIDIA repository
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12.8
sudo apt install -y cuda-toolkit-12-8

# Configure PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

**Impact:**
- Enables support for RTX 5080/5090
- CUDA kernels compile correctly
- Marlin backend (MXFP4) works

---

### 3. MXFP4 Quantization (MicroScaling FP4)

**Problem:** An FP16 model requires ~40GB of VRAM, impossible on a 16GB GPU.

**Solution:** GPT-OSS-20B ships pre-quantized with MXFP4 by OpenAI.

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│  MXFP4 (MicroScaling FP4)                                   │
├─────────────────────────────────────────────────────────────┤
│  • Original weights: FP16 (16 bits per parameter)           │
│  • Quantized weights: FP4 (4 bits per parameter)            │
│  • Technique: Block scaling (groups of 32 elements)         │
│  • Each block has a shared scale factor                     │
│  • Precision maintained via microscaling                    │
├─────────────────────────────────────────────────────────────┤
│  Memory without quantization: ~40GB (FP16)                  │
│  Memory with MXFP4:         ~14.6GB                         │
│  Savings:                   63% reduction                   │
└─────────────────────────────────────────────────────────────┘
```

**Inference Backend:**
- vLLM uses the **Marlin** kernel to decode MXFP4
- Marlin is optimized for NVIDIA GPUs with Tensor Cores
- Efficient inference without significant quality loss

---

### 4. Mixture of Experts (MoE)

**Problem:** Dense models with 20B+ parameters are computationally expensive.

**Solution:** MoE architecture that activates only a subset of parameters.

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│  Mixture of Experts (MoE)                                   │
├─────────────────────────────────────────────────────────────┤
│  Total parameters:     21B                                  │
│  Active parameters:    3.6B (per token)                     │
│  Number of experts:    Multiple specialists                 │
│  Active experts:       Top-K selected by router             │
├─────────────────────────────────────────────────────────────┤
│  Token → Router → Selects Top-K Experts → Output            │
│                                                             │
│  [Expert 1] [Expert 2] [Expert 3] ... [Expert N]            │
│      ↓          ↓                                           │
│   Active     Active     (others inactive)                   │
└─────────────────────────────────────────────────────────────┘
```

**Impact:**
- Large model capacity (21B parameters of knowledge)
- Small-model compute cost (3.6B active)
- Faster inference than an equivalent dense model

---

### 5. Eager Mode (--enforce-eager)

**Problem:** CUDA graphs pre-allocate memory, consuming additional VRAM.

**Solution:** Disable CUDA graphs with `--enforce-eager`.

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│  CUDA Graphs vs Eager Mode                                  │
├─────────────────────────────────────────────────────────────┤
│  CUDA Graphs (default):                                     │
│  • Captures a sequence of GPU operations                    │
│  • Pre-allocates memory for fast replay                     │
│  • Overhead: ~500MB+ additional VRAM                        │
│  • Benefit: 10-20% faster                                   │
├─────────────────────────────────────────────────────────────┤
│  Eager Mode (--enforce-eager):                              │
│  • Executes operations one by one                           │
│  • Allocates memory dynamically                             │
│  • Savings: ~500MB VRAM                                     │
│  • Trade-off: 10-20% slower                                 │
└─────────────────────────────────────────────────────────────┘
```

**When to use:**
- GPUs with limited VRAM (<=16GB)
- Models that run near the memory limit
- Priority: stability over speed

---

### 6. PyTorch Memory Management (PYTORCH_CUDA_ALLOC_CONF)

**Problem:** GPU memory fragmentation causes OOM even when VRAM appears "available".

**Solution:** Configure the PyTorch allocator for expandable segments.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│  GPU Memory Fragmentation                                   │
├─────────────────────────────────────────────────────────────┤
│  Without expandable_segments:                               │
│  [Used][Free][Used][Free][Used][Free]                       │
│          ↑              ↑              ↑                    │
│     Small fragments cannot be combined                      │
│     Large allocation FAILS despite total space              │
├─────────────────────────────────────────────────────────────┤
│  With expandable_segments:True:                             │
│  [Used][    Expandable Segment    ][Used]                   │
│          ↑                                                  │
│     PyTorch expands segments dynamically                    │
│     Large allocations WORK                                  │
└─────────────────────────────────────────────────────────────┘
```

**Impact:**
- Critical when operating >90% VRAM
- Prevents OOM errors due to fragmentation
- Enables maximum GPU utilization

---

### 7. Chunked Prefill (Automatic in vLLM)

**Problem:** Processing long prompts at once causes a memory spike.

**Solution:** vLLM automatically splits the prefill into chunks.

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│  Chunked Prefill                                            │
├─────────────────────────────────────────────────────────────┤
│  Prompt: "Explain the theory of relativity in detail..."    │
│                                                             │
│  Without chunking:                                          │
│  [████████████████████████████] → High memory spike         │
│                                                             │
│  With chunking (vLLM automatic):                            │
│  [████]→[████]→[████]→[████]→[████]                         │
│    ↓       ↓       ↓       ↓       ↓                        │
│  Processed in chunks, stable memory                         │
└─────────────────────────────────────────────────────────────┘
```

**Impact:**
- Reduces memory spikes during prefill
- Enables more efficient batching
- Enabled automatically by vLLM V1

---

### 8. Context Control (--max-model-len)

**Problem:** KV cache grows linearly with context length.

**Solution:** Limit maximum context to fit the available VRAM.

```bash
--max-model-len 512  # Limited to fit VRAM
```

**KV cache memory calculation:**
```
┌─────────────────────────────────────────────────────────────┐
│  KV Cache Memory                                            │
├─────────────────────────────────────────────────────────────┤
│  Formula: 2 × num_layers × hidden_size × context_len × dtype│
│                                                             │
│  For GPT-OSS-20B:                                            │
│  • Each additional token: ~0.5MB VRAM                       │
│  • 512-token context: ~256MB of KV cache                    │
│  • 128K-token context: ~64GB (impossible on 16GB GPU)       │
├─────────────────────────────────────────────────────────────┤
│  Model supports: 128K tokens                                │
│  We limit to: 512 tokens (to fit VRAM)                      │
│  Savings: ~63.7GB of KV cache not allocated                 │
└─────────────────────────────────────────────────────────────┘
```

---

### 9. GPU Memory Release (Display Manager)

**Problem:** Desktop environments consume VRAM even when idle.

**Solution:** Stop the display manager on systems accessed via SSH/VNC.

```bash
# Stop common display managers
sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

# Check freed memory
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

**Typical VRAM usage by component:**
```
┌─────────────────────────────────────────────────────────────┐
│  Component                     │ VRAM Used                 │
├────────────────────────────────┼────────────────────────────┤
│  GNOME Desktop (gdm3)          │ 400-600MB                  │
│  COSMIC Desktop (cosmic-comp)  │ 300-500MB                  │
│  X11/Wayland compositor        │ 200-400MB                  │
│  GUI apps (browser, etc.)      │ 100-500MB each             │
├────────────────────────────────┼────────────────────────────┤
│  Total possible savings:       │ 500-800MB                  │
└─────────────────────────────────────────────────────────────┘
```

**Impact:**
- Frees 500-800MB of critical VRAM
- Allows a larger model or larger context
- SSH/VNC continue working (do not use GPU)

---

### Strategy NOT Used: CPU Offloading

**Attempt:** Use `--cpu-offload-gb` to move layers to 128GB of RAM.

```bash
# We tried, but there are incompatibilities with vLLM V1 + MXFP4
vllm serve openai/gpt-oss-20b --cpu-offload-gb 10  # FAILED
```

**Why it does not work:**
```
┌─────────────────────────────────────────────────────────────┐
│  CPU Offload - Current Status                               │
├─────────────────────────────────────────────────────────────┤
│  ✗ vLLM V1 engine has bugs with MXFP4 + CPU offload          │
│  ✗ Marlin kernel does not support partial offload            │
│  ✗ Model is already at the VRAM limit                        │
│  ? Future vLLM versions may resolve this                     │
└─────────────────────────────────────────────────────────────┘
```

**Alternatives to use 128GB RAM:**
- **llama.cpp**: Convert to GGUF, allows full CPU offload
- **Slower**: CPU inference is 10-50x slower than GPU
- **Larger context**: Allows much larger contexts

---

### Optimization Summary

| # | Strategy | Savings/Impact | Status |
|---|----------|----------------|--------|
| 1 | Sleep prevention | SSH stability | ✅ Applied |
| 2 | CUDA 12.8 | Blackwell support | ✅ Applied |
| 3 | MXFP4 quantization | ~25GB (63%) | ✅ Native to model |
| 4 | Mixture of Experts | 17.4B inactive params | ✅ Native to model |
| 5 | Eager mode | ~500MB VRAM | ✅ Applied |
| 6 | Expandable Segments | Avoids fragmentation | ✅ Applied |
| 7 | Chunked Prefill | Reduced memory spike | ✅ vLLM automatic |
| 8 | Context control | ~63.7GB KV cache | ✅ Applied |
| 9 | Stop display manager | ~500-800MB VRAM | ✅ Applied |
| 10 | CPU Offload | N/A | ✗ Incompatible |

## Model Details

### Llama 3.2 3B Instruct Specifications

- **Parameters**: 3B (dense)
- **Quantization**: FP16 (native)
- **VRAM Usage**: ~3GB for weights + KV cache
- **Context**: 128K tokens (very large)
- **License**: Llama 3.2 Community License
- **Model Size on Disk**: ~6GB
- **HuggingFace ID**: `unsloth/Llama-3.2-3B-Instruct`

**Key Features:**
- Native tool/function calling support
- Excellent for agentic applications
- Multilingual support
- Fast inference (small model)

### Phi-4-mini-instruct Specifications

- **Parameters**: 3.8B (dense)
- **Quantization**: FP16 (native)
- **VRAM Usage**: ~3GB for weights + KV cache
- **Context**: 128K tokens (very large)
- **License**: MIT
- **Model Size on Disk**: ~8GB
- **HuggingFace ID**: `microsoft/Phi-4-mini-instruct`

**Key Features:**
- Excellent reasoning capabilities
- Very large context (128K tokens)
- Fast inference (small model)
- Supports multiple concurrent requests
- Great for coding and mathematics

### Model Comparison

| Aspect | Llama 3.2 3B | Phi-4-mini-instruct |
|--------|--------------|---------------------|
| **VRAM** | ~3GB | ~3GB |
| **Context** | 128K tokens | 128K tokens |
| **Speed** | Fast | Fast |
| **Reasoning** | Good | Excellent |
| **Tool calling** | Native | Supported |
| **Ideal use** | Agentic, chat | Coding, reasoning |

## Troubleshooting

### Out of Memory Errors

If you see OOM errors:

1. **Stop display manager**: `sudo systemctl stop gdm3`
2. **Reduce max-model-len**: Try `--max-model-len 128`
3. **Reduce max-num-seqs**: Try `--max-num-seqs 1`
4. **Lower GPU utilization**: Try `--gpu-memory-utilization 0.90`

### CUDA Architecture Not Supported

For newer GPUs (RTX 50-series), you need CUDA 12.8+:

```bash
# Check current CUDA version
nvcc --version

# If using older CUDA, install 12.8
sudo apt install cuda-toolkit-12-8
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### Model Not Found

The model downloads automatically from Hugging Face on first run. Ensure:
- Internet connection is available
- ~50GB free disk space in `~/.cache/huggingface/`

## API Usage Examples

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum computing briefly"}],
    max_tokens=200
)
print(response.choices[0].message.content)
```

### cURL with Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a haiku about coding"}],
    "stream": true
  }'
```

## Running as a Service

Create a systemd service for auto-start:

```bash
sudo tee /etc/systemd/system/vllm.service << 'EOF'
[Unit]
Description=vLLM LLM Server
After=network.target

[Service]
Type=simple
User=root
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
ExecStart=/root/vllm-env/bin/vllm serve unsloth/Llama-3.2-3B-Instruct --port 8000 --served-model-name default --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.90
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

## Reproducibility and Automation

This section provides scripts and configurations to replicate this setup on other machines.

### Option 1: Full Bash Script (Recommended)

**File available at:** `/home/user/setup-vllm.sh`

**To use on another machine:**
```bash
# 1. Copy to the new machine
scp /home/user/setup-vllm.sh user@new-machine:~

# 2. Run on the new machine
ssh user@new-machine
sudo bash setup-vllm.sh

# 3. Start the server (first run downloads ~50GB)
~/start-vllm.sh

# 4. Test
~/test-vllm.sh
```

**Script contents** `setup-vllm.sh`:

```bash
#!/bin/bash
#
# Setup script for vLLM + Llama 3.2 3B
# Tested on: Pop!_OS / Ubuntu 24.04 with NVIDIA RTX 5080
#
# Usage: sudo bash setup-vllm.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if running as root
[[ $EUID -ne 0 ]] && error "This script must be run as root (sudo)"

# Configuration
VLLM_USER="${VLLM_USER:-$SUDO_USER}"
VLLM_HOME="/home/$VLLM_USER"
VENV_PATH="$VLLM_HOME/vllm-env"
CUDA_VERSION="12-8"

log "Starting vLLM + Llama 3.2 3B setup for user: $VLLM_USER"

# ============================================================
# 1. Disable System Sleep (for SSH access)
# ============================================================
log "Configuring system to prevent sleep..."

systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || true

mkdir -p /etc/systemd/logind.conf.d
cat > /etc/systemd/logind.conf.d/nosleep.conf << 'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
EOF

systemctl restart systemd-logind

# ============================================================
# 2. Install System Dependencies
# ============================================================
log "Installing system dependencies..."

apt update
apt install -y python3-pip python3-venv wget curl

# ============================================================
# 3. Install CUDA 12.8
# ============================================================
log "Installing CUDA $CUDA_VERSION..."

if ! command -v /usr/local/cuda-12.8/bin/nvcc &> /dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update
    apt install -y cuda-toolkit-$CUDA_VERSION
    rm -f cuda-keyring_1.1-1_all.deb
else
    log "CUDA 12.8 already installed"
fi

# ============================================================
# 4. Create Python Virtual Environment
# ============================================================
log "Creating Python virtual environment..."

sudo -u $VLLM_USER python3 -m venv $VENV_PATH
sudo -u $VLLM_USER $VENV_PATH/bin/pip install --upgrade pip

# ============================================================
# 5. Install vLLM
# ============================================================
log "Installing vLLM (this may take a few minutes)..."

sudo -u $VLLM_USER $VENV_PATH/bin/pip install vllm

# ============================================================
# 6. Create systemd Service
# ============================================================
log "Creating systemd service..."

cat > /etc/systemd/system/vllm.service << EOF
[Unit]
Description=vLLM LLM Server
After=network.target

[Service]
Type=simple
User=$VLLM_USER
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="PATH=/usr/local/cuda-12.8/bin:\$PATH"
ExecStartPre=/bin/bash -c 'systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true'
ExecStart=$VENV_PATH/bin/vllm serve unsloth/Llama-3.2-3B-Instruct --port 8000 --served-model-name default --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.90
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

# ============================================================
# 7. Create Helper Scripts
# ============================================================
log "Creating helper scripts..."

# Start script
cat > $VLLM_HOME/start-vllm.sh << 'EOF'
#!/bin/bash
# Manual start script for vLLM (Llama 3.2 3B)

# Stop display manager to free GPU memory
sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

# Activate environment
source ~/vllm-env/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Start vLLM
vllm serve unsloth/Llama-3.2-3B-Instruct \
    --port 8000 \
    --served-model-name default \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90
EOF

# Test script
cat > $VLLM_HOME/test-vllm.sh << 'EOF'
#!/bin/bash
# Test if vLLM server is responding

echo "Testing vLLM server..."
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 50
    }' | python3 -m json.tool

echo ""
echo "If you see a response above, the server is working!"
EOF

chmod +x $VLLM_HOME/start-vllm.sh $VLLM_HOME/test-vllm.sh
chown $VLLM_USER:$VLLM_USER $VLLM_HOME/start-vllm.sh $VLLM_HOME/test-vllm.sh

# ============================================================
# 8. Summary
# ============================================================
log "Setup complete!"

echo ""
echo "============================================================"
echo "  vLLM + Llama 3.2 3B Installation Complete"
echo "============================================================"
echo ""
echo "To start the server manually:"
echo "  ~/start-vllm.sh"
echo ""
echo "To start as a service:"
echo "  sudo systemctl start vllm"
echo "  sudo systemctl enable vllm  # Auto-start on boot"
echo ""
echo "To test the server:"
echo "  ~/test-vllm.sh"
echo ""
echo "NOTE: First run will download ~50GB of model files."
echo "============================================================"
```

### Option 2: YAML Configuration File

Create `vllm-config.yaml` to document the configuration:

```yaml
# vLLM Llama 3.2 3B Configuration
# Use with: vllm serve --config vllm-config.yaml

# Model
model: unsloth/Llama-3.2-3B-Instruct
served_model_name: default

# Server
port: 8000
host: 0.0.0.0

# Memory Optimization (for 16GB VRAM)
max_model_len: 16384
max_num_seqs: 16
gpu_memory_utilization: 0.90

# Environment variables required:
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Note:** vLLM v0.13.0 has limited support for config files. Use mainly for documentation.

### Option 3: Python with Requirements

Create `requirements.txt`:

```text
# requirements.txt for vLLM + Llama 3.2 3B
# Install with: pip install -r requirements.txt

vllm>=0.13.0
openai>=1.0.0  # For Python client
```

Create `run_server.py`:

```python
#!/usr/bin/env python3
"""
vLLM Server Launcher for Llama 3.2 3B

Usage:
    python run_server.py
    python run_server.py --max-model-len 16384
"""

import os
import subprocess
import argparse

# Configuration defaults
CONFIG = {
    "model": "unsloth/Llama-3.2-3B-Instruct",
    "port": 8000,
    "max_model_len": 16384,
    "max_num_seqs": 16,
    "gpu_memory_utilization": 0.90,
}

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM with Llama 3.2 3B")
    parser.add_argument("--port", type=int, default=CONFIG["port"])
    parser.add_argument("--max-model-len", type=int, default=CONFIG["max_model_len"])
    parser.add_argument("--max-num-seqs", type=int, default=CONFIG["max_num_seqs"])
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=CONFIG["gpu_memory_utilization"])
    args = parser.parse_args()

    # Set required environment variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Build command
    cmd = [
        "vllm", "serve", CONFIG["model"],
        "--port", str(args.port),
        "--served-model-name", "default",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    print(f"Starting vLLM server...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
```

### Option 4: Docker (Alternative)

```dockerfile
# Dockerfile for vLLM + Llama 3.2 3B
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install vLLM
RUN python3 -m venv /opt/vllm-env
RUN /opt/vllm-env/bin/pip install --upgrade pip vllm

# Environment
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PATH="/opt/vllm-env/bin:$PATH"

# Expose port
EXPOSE 8000

# Default command
CMD ["vllm", "serve", "unsloth/Llama-3.2-3B-Instruct", \
     "--port", "8000", \
     "--served-model-name", "default", \
     "--max-model-len", "16384", \
     "--max-num-seqs", "16", \
     "--gpu-memory-utilization", "0.90"]
```

**Run with Docker:**
```bash
docker build -t vllm .
docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface vllm
```

### Reproducibility Checklist

Use this checklist when configuring a new machine:

```
[ ] 1. NVIDIA GPU with 16GB+ VRAM
[ ] 2. Ubuntu 24.04 / Pop!_OS installed
[ ] 3. NVIDIA driver installed (nvidia-smi works)
[ ] 4. CUDA 12.8 installed
[ ] 5. Python 3.10+ with venv
[ ] 6. vLLM v0.13.0+ installed
[ ] 7. Display manager stopped (to free VRAM)
[ ] 8. PYTORCH_CUDA_ALLOC_CONF configured
[ ] 9. ~50GB of disk space for the model
[ ] 10. Internet connectivity (model download)
```

### Files Created in This Installation

```
/home/user/
├── setup-vllm.sh          # ✅ Installation script (for new machines)
├── start-vllm.sh                 # ✅ Script to start the server
├── test-vllm.sh                  # ✅ Script to test the server
├── vllm-tutorial.md      # ✅ This tutorial/documentation
│
/root/
└── vllm-env/                     # ✅ Virtual environment with vLLM

/etc/systemd/logind.conf.d/
└── nosleep.conf                  # ✅ Anti-sleep configuration

# Created by setup-vllm.sh on new machines:
# ~/vllm-env/                     # Virtual environment
# ~/start-vllm.sh                 # Start script
# ~/test-vllm.sh                  # Test script
# /etc/systemd/system/vllm.service
```

## Summary

Successfully running LLMs locally on a 16GB GPU:

**Current Models:**
- **Llama 3.2 3B Instruct**: Tool calling, agentic applications, chat
- **Phi-4-mini-instruct**: Excellent reasoning, coding, mathematics

**Key Requirements:**
1. **CUDA 12.8+** for Blackwell architecture support (RTX 50-series)
2. **vLLM 0.13.0+** for optimal performance
3. **HuggingFace models** downloaded to cache

Both models are lightweight (~3GB VRAM each) and support 128K context, making them ideal for local AI development and experimentation.

---

*Tutorial generated on 2026-01-11*
*Hardware: NVIDIA RTX 5080 (16GB) | Software: vLLM 0.13.0, CUDA 12.8*
