# vLLM Server Configuration

This directory contains configuration files for running vLLM, a high-performance serving framework for large language models. vLLM is optimized for fast inference with PagedAttention and supports a wide range of models.

## Available Models

- **Llama 3.2 3B Instruct** (unsloth/Llama-3.2-3B-Instruct)
  - 3B parameters
  - Native 128K context window
  - Tool calling support
  - VRAM usage: ~3GB

- **Phi-4-mini-instruct** (microsoft/Phi-4-mini-instruct)
  - 3.8B parameters
  - Native 128K context window
  - Excellent reasoning capabilities
  - VRAM usage: ~3GB

## Quick Start

### Installation

Run the setup script to install vLLM and all dependencies:

```bash
sudo bash setup-vllm.sh
```

This will:
- Configure system to prevent sleep (for remote access)
- Install Python and system dependencies
- Install CUDA 12.8 (if not already installed)
- Create a Python virtual environment at `~/vllm-env`
- Install vLLM with optimized kernels
- Create a systemd service for auto-start

### Starting the Server

#### Interactive Mode
```bash
./start-vllm.sh
```

Choose from the menu:
1. Llama 3.2 3B Instruct
2. Phi-4-mini-instruct

#### Direct Launch
```bash
./start-vllm.sh llama    # Launch Llama 3.2 3B
./start-vllm.sh phi4     # Launch Phi-4-mini
```

#### Using Command Line
```bash
source ~/vllm-env/bin/activate
vllm serve unsloth/Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92
```

### Testing the Server

```bash
./test-vllm.sh
```

This will run several tests:
1. Health check
2. Model info retrieval
3. Completion API test
4. Chat completion API test

### Using as a System Service

```bash
# Start the service
sudo systemctl start vllm

# Enable auto-start on boot
sudo systemctl enable vllm

# Check status
sudo systemctl status vllm

# View logs
sudo journalctl -u vllm -f
```

## Performance Features

### PagedAttention
vLLM uses PagedAttention for efficient memory management:
- Reduces memory fragmentation
- Enables higher batch sizes
- Improves throughput significantly

### Continuous Batching
- Processes requests as they arrive
- No waiting for batch completion
- Optimal for real-time applications

### Optimized Kernels
- Custom CUDA kernels for attention
- FlashAttention integration
- Quantization support (INT8/INT4)

## API Usage

vLLM provides an OpenAI-compatible API:

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.0
  }'
```

### Python Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't require an API key
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Advanced Configuration

### Custom Context Length

To use the full 128K context (requires more VRAM):

```bash
vllm serve unsloth/Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95
```

### Enable Metrics and Logging

Add these flags to your launch command:

```bash
--enable-metrics \
--log-requests \
--log-level info
```

### Multi-GPU Setup

For tensor parallelism across multiple GPUs:

```bash
vllm serve unsloth/Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

### Quantization

Enable INT8 quantization for reduced memory usage:

```bash
vllm serve unsloth/Llama-3.2-3B-Instruct \
  --quantization int8 \
  --gpu-memory-utilization 0.95
```

## Performance Tuning

### Memory Optimization

Adjust `gpu-memory-utilization` based on available VRAM:
- `0.80` - Conservative (leaves room for other processes)
- `0.92` - Balanced (default)
- `0.98` - Aggressive (maximum performance)

### Batch Size Tuning

Adjust `max-num-seqs` based on your workload:
- Lower values: Better latency per request
- Higher values: Better throughput for many concurrent requests

### Context Length Optimization

Current configuration uses 64K context (65536 tokens):
- Balances memory usage with most use cases
- Can be increased to 128K for document processing
- Reduce to 32K for lower memory usage

## Chat Templates

Custom chat templates are available in the `chat-templates/` directory:

- `mistral-instruct.jinja` - For Mistral-style models
- `openai-compat.jinja` - OpenAI-compatible format

Use with `--chat-template` flag:

```bash
vllm serve model-name \
  --chat-template ./chat-templates/openai-compat.jinja
```

## Troubleshooting

### Server Won't Start

1. Check if port 8000 is already in use:
   ```bash
   sudo netstat -tulpn | grep 8000
   ```

2. Check GPU availability:
   ```bash
   nvidia-smi
   ```

3. View detailed logs:
   ```bash
   sudo journalctl -u vllm -n 100
   ```

### Out of Memory Errors

Reduce memory usage:
- Lower `max-model-len` (e.g., 32768)
- Reduce `max-num-seqs` (e.g., 16)
- Lower `gpu-memory-utilization` (e.g., 0.85)
- Enable quantization: `--quantization int8`

### Slow Response Times

Increase performance:
- Stop display manager: `sudo systemctl stop gdm3`
- Increase `gpu-memory-utilization`
- Ensure CUDA is properly installed
- Check for GPU throttling: `nvidia-smi -q -d TEMPERATURE`

### Model Loading Issues

1. Clear cache:
   ```bash
   rm -rf ~/.cache/huggingface
   ```

2. Check model access:
   ```bash
   huggingface-cli whoami
   ```

3. Verify internet connection for model download

## Files in This Directory

- `setup-vllm.sh` - Installation script
- `start-vllm.sh` - Interactive server launcher
- `test-vllm.sh` - Server testing script
- `chat-templates/` - Custom chat template files
- `vllm-tutorial.md` - Comprehensive tutorial and guide
- `README.md` - This file

## Configuration Comparison

| Parameter | Current Setting | Description |
|-----------|----------------|-------------|
| **Context Length** | 65536 tokens | Maximum sequence length |
| **Max Concurrent** | 32 requests | Maximum simultaneous requests |
| **GPU Memory** | 92% utilization | GPU memory allocation |
| **Port** | 8000 | API server port |
| **Host** | 0.0.0.0 | Network interface binding |
| **Data Type** | FP16 | Half precision inference |

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## License

This configuration follows the same license as the underlying models:
- Llama 3.2: Llama 3.2 Community License
- Phi-4: MIT License
