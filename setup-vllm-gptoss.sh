#!/bin/bash
#
# Setup script for vLLM + GPT-OSS-20B
# Tested on: Pop!_OS / Ubuntu 24.04 with NVIDIA RTX 5080
#
# Usage: sudo bash setup-vllm-gptoss.sh
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
[[ -z "$VLLM_USER" ]] && error "Could not determine user. Set VLLM_USER environment variable."
VLLM_HOME="/home/$VLLM_USER"
VENV_PATH="$VLLM_HOME/vllm-env"
CUDA_VERSION="12-8"

log "Starting vLLM + GPT-OSS-20B setup for user: $VLLM_USER"

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
log "Checking CUDA installation..."

if ! command -v /usr/local/cuda-12.8/bin/nvcc &> /dev/null; then
    log "Installing CUDA $CUDA_VERSION..."
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

if [[ -d "$VENV_PATH" ]]; then
    warn "Virtual environment already exists at $VENV_PATH"
else
    sudo -u "$VLLM_USER" python3 -m venv "$VENV_PATH"
fi
sudo -u "$VLLM_USER" "$VENV_PATH/bin/pip" install --upgrade pip

# ============================================================
# 5. Install vLLM
# ============================================================
log "Installing vLLM (this may take a few minutes)..."

sudo -u "$VLLM_USER" "$VENV_PATH/bin/pip" install vllm

# ============================================================
# 6. Create systemd Service
# ============================================================
log "Creating systemd service..."

cat > /etc/systemd/system/vllm-gptoss.service << EOF
[Unit]
Description=vLLM GPT-OSS-20B Server
After=network.target

[Service]
Type=simple
User=$VLLM_USER
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin"
ExecStartPre=/bin/bash -c 'systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true'
ExecStart=$VENV_PATH/bin/vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 512 --max-num-seqs 2 --gpu-memory-utilization 0.95 --enforce-eager
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
cat > "$VLLM_HOME/start-vllm.sh" << 'EOF'
#!/bin/bash
# Manual start script for vLLM GPT-OSS-20B

# Stop display manager to free GPU memory
sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

# Activate environment
source ~/vllm-env/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Start vLLM
vllm serve openai/gpt-oss-20b \
    --port 8000 \
    --max-model-len 512 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager
EOF

# Test script
cat > "$VLLM_HOME/test-vllm.sh" << 'EOF'
#!/bin/bash
# Test if vLLM server is responding

echo "Testing vLLM server..."
response=$(curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 50
    }')

if echo "$response" | python3 -m json.tool 2>/dev/null; then
    echo ""
    echo "Server is working!"
else
    echo "Error: Server not responding or returned invalid JSON"
    echo "Response: $response"
    exit 1
fi
EOF

chmod +x "$VLLM_HOME/start-vllm.sh" "$VLLM_HOME/test-vllm.sh"
chown "$VLLM_USER:$VLLM_USER" "$VLLM_HOME/start-vllm.sh" "$VLLM_HOME/test-vllm.sh"

# ============================================================
# 8. Summary
# ============================================================
log "Setup complete!"

echo ""
echo "============================================================"
echo "  vLLM + GPT-OSS-20B Installation Complete"
echo "============================================================"
echo ""
echo "To start the server manually:"
echo "  ~/start-vllm.sh"
echo ""
echo "To start as a service:"
echo "  sudo systemctl start vllm-gptoss"
echo "  sudo systemctl enable vllm-gptoss  # Auto-start on boot"
echo ""
echo "To test the server:"
echo "  ~/test-vllm.sh"
echo ""
echo "NOTE: First run will download ~50GB of model files."
echo "============================================================"
