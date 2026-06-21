# DGX Spark: Install DeepSeek V4 Flash with DwarfStar ds4

A step-by-step guide to running **DeepSeek V4 Flash** (640B parameters) on a single **NVIDIA DGX Spark (GB10)** using the **DwarfStar/ds4** custom CUDA inference engine.

> **Prerequisites:** A DGX Spark with 128 GB LPDDR5X Unified Memory and an installed NVIDIA CUDA Toolkit / Driver.

---

## 1. Hardware Check

```bash
# Check GPU type and CUDA capability
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Expected: NVIDIA GB10, compute_cap 12.1, 128 GB
# Driver version: ≥ 580.159.03
```

---

## 2. Clone and Build

```bash
git clone https://github.com/antirez/ds4.git
cd ds4

# Compile natively for GB10 (~8 seconds)
make cuda-spark
```

After the build, three binaries are in the working directory:

| Binary | Purpose |
|--------|---------|
| `ds4` | CLI tool for prompt processing |
| `ds4-server` | OpenAI-compatible API server |
| `ds4-bench` | Benchmark tool |

---

## 3. Download the Model

The model is **not included in the repository**. You need the quantized GGUF file from Hugging Face:

```bash
# Install Hugging Face CLI if not already available
pip install huggingface-hub

# Download the model
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /srv/models/deepseek-v4-flash
```

> The actual file loaded on our Spark is an **IQ2XXS-w2Q2K/AProjQ8/SExpQ8/OutQ8 GGUF** weighing **81 GB** — this is the trick that makes a 640B model fit in 128 GB RAM.

**Optional:** For Multi-Token Prediction (MTP), download the draft model separately:

```bash
cd ~/ds4
DS4_GGUF_DIR=/srv/models/deepseek-v4-flash ./download_model.sh mtp
```

This fetches `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (~3.8 GB).

---

## 4. Set the Symlink (Default Path)

The binaries expect `./ds4flash.gguf` in the working directory by default:

```bash
cd ~/ds4
ln -s /srv/models/deepseek-v4-flash/ds4flash.gguf ./ds4flash.gguf
```

Alternatively, pass the path at startup with `--model` (see step 5).

---

## 5. Start the Server

**Simple start:**

```bash
./ds4-server \
  --model /srv/models/deepseek-v4-flash/ds4flash.gguf \
  --host 127.0.0.1 \
  --port 8004
```

**With KV disk cache and larger context:**

```bash
./ds4-server \
  --model /srv/models/deepseek-v4-flash/ds4flash.gguf \
  --host 127.0.0.1 \
  --port 8004 \
  --ctx 131072 \
  --kv-disk-dir /srv/models/ds4/kv \
  --kv-disk-space-mb 32768 \
  --kv-cache-cold-max-tokens 120000 \
  --kv-cache-continued-interval-tokens 20000
```

**With MTP Draft (optional, higher throughput):**

```bash
./ds4-server \
  --model /srv/models/deepseek-v4-flash/ds4flash.gguf \
  --mtp /srv/models/deepseek-v4-flash/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --mtp-margin 3 \
  --host 127.0.0.1 \
  --port 8004
```

| Flag | Description |
|------|-------------|
| `--port` | API port (default: 8004) |
| `--ctx` | Context length in tokens (up to 131072) |
| `--kv-disk-dir` | Directory for disk-based KV cache |
| `--kv-disk-space-mb` | Max disk cache size in MB |
| `--mtp` | Path to the MTP draft model GGUF |
| `--mtp-draft` | Maximum autoregressive MTP draft tokens (start at 2) |
| `--mtp-margin` | Verifier confidence margin for fast MTP acceptance (default: 3) |

---

## 6. Smoke Test

```bash
# Fetch model info
curl -sS http://localhost:8004/v1/models | python3 -m json.tool

# Test a chat completion
curl -sS http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "What is 17 * 19?"}],
    "max_tokens": 120,
    "temperature": 0
  }' | python3 -m json.tool
```

Expected: the answer should contain `323`.

---

## 7. Systemd User Service (Production Setup)

Create a systemd user service so the server can be managed and started on demand:

```bash
cat > ~/.config/systemd/user/ds4-server.service << 'EOF'
[Unit]
Description=DwarfStar ds4 DeepSeek V4 Flash Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/stefan/ds4
ExecStart=/home/stefan/ds4/ds4-server --cuda --model /srv/models/ds4/gguf/ds4flash.gguf --host 127.0.0.1 --port 8004 --ctx 131072 --kv-disk-dir /srv/models/ds4/kv --kv-disk-space-mb 32768 --kv-cache-cold-max-tokens 120000 --kv-cache-continued-interval-tokens 20000
Restart=on-failure
RestartSec=10
TimeoutStartSec=1200

[Install]
WantedBy=default.target
EOF
```

> **⚠️ Important:** In a `systemd --user` service, **do not set `User=`**. Doing so causes `status=216/GROUP`.

Enable and test the service:

```bash
systemctl --user daemon-reload
systemctl --user start ds4-server
systemctl --user status ds4-server
journalctl --user -u ds4-server -f
```

**Recommendation:** Do not enable the service as an autostart — DS4 occupies ~80 GB RAM. Start it on demand via a router like llama-swap instead.

---

## 8. Integration with llama-swap (Optional)

If you use llama-swap as an API router, add DS4 as a route:

```yaml
# In spark.yaml
deepseek-v4-flash:
    name: Spark DeepSeek V4 Flash DwarfStar
    description: DGX Spark GB10, antirez/ds4 DwarfStar CUDA engine, DeepSeek V4 Flash IQ2XXS GGUF
    proxy: http://127.0.0.1:8004
    checkEndpoint: /v1/models
    ttl: 0
    useModelName: deepseek-v4-flash
    cmdStop: systemctl --user stop ds4-server >/dev/null 2>&1 || true
    cmd: /srv/llama-swap/start-ds4-deepseek-v4-flash.sh
```

Startup script for llama-swap (`/srv/llama-swap/start-ds4-deepseek-v4-flash.sh`):

```bash
#!/bin/bash
systemctl --user start ds4-server
# Wait for readiness
until curl -sS http://127.0.0.1:8004/v1/models >/dev/null 2>&1; do
  sleep 1
done
journalctl --user -u ds4-server -f --no-pager
```

---

## 9. Performance Tuning

**Check GPU cache usage:**
```bash
./ds4 --model /srv/models/deepseek-v4-flash/ds4flash.gguf --prompt "test" --verbose 2>&1 | grep gpu_cache
```

**Expected performance:**
- Steady-state decode: ~95 % of GB10 bandwidth (~273 GB/s)
- MTP can further increase throughput
- 1M token sessions are feasible (KV cache ~9.6 GB)

**If tokens/s are low:**
1. Check for thermal throttling — the GB10 downclocks under sustained load
2. Reduce context length (try 32k/64k first)
3. Adjust MTP draft tokens / margin

---

## 10. Important Notes

- **Not a generic GGUF runner:** DwarfStar/ds4 works **only** with DeepSeek V4 Flash (and PRO on very large machines). Other models produce garbage or won't start.
- **No `/health` endpoint:** DS4 does not provide a health endpoint. `checkEndpoint` must be `/v1/models`.
- **Port collision:** Assign DS4 to port **8004** (not 8000, which may be needed by other vLLM routes).
- **Disk space:** Plan for at least 120 GB free for model + cache. The Spark's 4 TB SSD has plenty of room.

---

## One-Command Community Installer

If you want to automate the entire process, the **Entrpi** community installer handles clone, build, download, and startup:

```bash
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh \
  | bash -s -- --with-mtp --start
```

Repo: [github.com/Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark)

---

## Sources

- [antirez/ds4 GitHub Repository](https://github.com/antirez/ds4)
- [Dre Dyson: Beginner Guide](https://dredyson.com/how-i-got-fully-custom-cuda-native-deepseek-4-flash-running-on-nvidia-dgx-spark-a-complete-beginners-step-by-step-guide-to-installation-benchmarks-roofline-analysis-and-common-misconcep/)
- [Dre Dyson: Advanced Guide](https://dredyson.com/how-i-mastered-fully-custom-cuda-native-deepseek-4-flash-dwarfstar-4-on-nvidia-dgx-spark-the-complete-advanced-configuration-guide-with-benchmarks-roofline-analysis-tensor-parallelism-a/)
- [NVIDIA Forums: ds4 on Spark](https://forums.developer.nvidia.com/t/fully-custom-cuda-native-deepseek-4-flash-optimized-for-1x-spark-antirez-ds4/369791)
- [Entrpi/ds4-on-spark Installer](https://github.com/Entrpi/ds4-on-spark)
