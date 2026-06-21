# DGX Spark: DeepSeek V4 Flash mit DwarfStar ds4 installieren

Eine Schritt-für-Schritt-Anleitung, um **DeepSeek V4 Flash** (640B Parameter) auf einem einzelnen **NVIDIA DGX Spark (GB10)** mit der **DwarfStar/ds4** Custom-CUDA-Engine zum Laufen zu bringen.

> **Voraussetzung:** Ein DGX Spark mit 128 GB LPDDR5X Unified Memory und installiertem NVIDIA CUDA Toolkit / Driver.

---

## 1. Hardware-Check

```bash
# GPU-Typ und CUDA-Fähigkeit prüfen
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Erwartet: NVIDIA GB10, compute_cap 12.1, 128 GB
# Driver-Version: ≥ 580.159.03
```

---

## 2. Repository klonen und bauen

```bash
git clone https://github.com/antirez/ds4.git
cd ds4

# Nativ für GB10 kompilieren (~8 Sekunden)
make cuda-spark
```

Nach dem Build liegen drei Binaries im Arbeitsverzeichnis:

| Binary | Zweck |
|--------|-------|
| `ds4` | CLI-Tool für Prompt-Processing |
| `ds4-server` | OpenAI-kompatibler API-Server |
| `ds4-bench` | Benchmark-Tool |

---

## 3. Modell herunterladen

Das Modell ist **nicht im Repository**. Du brauchst die quantisierte GGUF-Datei von Hugging Face:

```bash
# Hugging Face CLI installieren falls nicht vorhanden
pip install huggingface-hub

# Modell herunterladen
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /srv/models/deepseek-v4-flash
```

> Die konkrete Datei auf dem Spark ist eine **IQ2XXS-w2Q2K/AProjQ8/SExpQ8/OutQ8 GGUF** mit **81 GB** — das ist der Trick, warum ein 640B-Modell in 128 GB RAM passt.

**Optional:** Für Multi-Token Prediction (MTP) das Draft-Modell separat beziehen (siehe ds4-README).

---

## 4. Symlink für Default-Pfad setzen

Die Binaries erwarten standardmäßig `./ds4flash.gguf` im Arbeitsverzeichnis:

```bash
cd ~/ds4
ln -s /srv/models/deepseek-v4-flash/ds4flash.gguf ./ds4flash.gguf
```

Alternativ den Pfad beim Start mit `--model` übergeben (siehe Schritt 5).

---

## 5. Server starten

**Einfacher Start:**

```bash
./ds4-server \
  --model /srv/models/deepseek-v4-flash/ds4flash.gguf \
  --host 127.0.0.1 \
  --port 8004
```

**Mit KV-Disk-Cache und größerem Kontext:**

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

**Mit MTP Draft (optional, höherer Durchsatz):**

```bash
./ds4-server \
  --model /srv/models/deepseek-v4-flash/ds4flash.gguf \
  --mtp-draft /srv/models/deepseek-v4-flash/deepseek-v4-flash-mtp-draft.gguf \
  --mtp-layers 1 \
  --host 127.0.0.1 \
  --port 8004
```

| Flag | Bedeutung |
|------|-----------|
| `--port` | API-Port (Default: 8004) |
| `--ctx` | Kontextlänge in Tokens (bis 131072) |
| `--kv-disk-dir` | Verzeichnis für disk-basierten KV-Cache |
| `--kv-disk-space-mb` | Maximaler Disk-Cache in MB |
| `--mtp-draft` | Pfad zum MTP Draft-Modell |
| `--mtp-layers` | Anzahl MTP Layers (Startwert: 1) |

---

## 6. Smoke-Test

```bash
# Modell-Info abrufen
curl -sS http://localhost:8004/v1/models | python3 -m json.tool

# Chat-Completion testen
curl -sS http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "What is 17 * 19?"}],
    "max_tokens": 120,
    "temperature": 0
  }' | python3 -m json.tool
```

Erwartung: Die Antwort sollte `323` enthalten.

---

## 7. Systemd User Service (für Production)

Damit der Server verwaltet und bei Bedarf gestartet werden kann:

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

> **⚠️ Wichtig:** In einem `systemd --user` Service darf **kein `User=`** gesetzt werden. Das führt zu `status=216/GROUP`.

Service aktivieren und testen:

```bash
systemctl --user daemon-reload
systemctl --user start ds4-server
systemctl --user status ds4-server
journalctl --user -u ds4-server -f
```

**Empfehlung:** Den Service nicht als Autostart (`enable`) setzen — DS4 belegt ~80 GB RAM. Besser on-demand über einen Router wie llama-swap starten.

---

## 8. Integration mit llama-swap (optional)

Falls du llama-swap als API-Router verwendest, kannst du DS4 als Route eintragen:

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

Startscript für llama-swap (`/srv/llama-swap/start-ds4-deepseek-v4-flash.sh`):

```bash
#!/bin/bash
systemctl --user start ds4-server
# Auf Ready warten
until curl -sS http://127.0.0.1:8004/v1/models >/dev/null 2>&1; do
  sleep 1
done
journalctl --user -u ds4-server -f --no-pager
```

---

## 9. Performance-Tuning

**GPU-Cache auslesen:**
```bash
./ds4 --model /srv/models/deepseek-v4-flash/ds4flash.gguf --prompt "test" --verbose 2>&1 | grep gpu_cache
```

**Erwartete Performance:**
- Steady-State Decode: ~95 % der GB10-Bandbreite (~273 GB/s)
- MTP kann den Durchsatz zusätzlich steigern
- 1M Token Sessions sind möglich (KV-Cache ~9,6 GB)

**Bei niedrigen Tokens/s:**
1. Thermal Throttling prüfen — GB10 drosselt unter Dauerlast
2. Kontextlänge reduzieren (32k/64k testen)
3. MTP Layers anpassen

---

## 10. Wichtige Hinweise

- **Kein generischer GGUF-Runner:** DwarfStar/ds4 läuft **nur** mit DeepSeek V4 Flash (und PRO auf sehr großen Maschinen). Andere Modelle liefern Unsinn oder starten nicht.
- **Kein `/health`:** DS4 hat keinen Health-Endpoint. `checkEndpoint` muss `/v1/models` sein.
- **Port-Kollision:** DS4 auf Port **8004** setzen (nicht 8000, den brauchen evtl. andere vLLM-Routen).
- **Disk-Space:** Mindestens 120 GB frei für Modell + Cache einplanen. Die 4-TB-SSD im Spark reicht locker.

---

## One-Command-Installer (Community)

Falls du den ganzen Prozess automatisieren willst — der Community-Installer von **Entrpi** übernimmt Clone, Build, Download und Start:

```bash
curl -sSL https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/install.sh \
  | bash -s -- --with-mtp --start
```

Repo: [github.com/Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark)

---

## Quellen

- [antirez/ds4 GitHub Repository](https://github.com/antirez/ds4)
- [Dre Dyson: Beginner Guide](https://dredyson.com/how-i-got-fully-custom-cuda-native-deepseek-4-flash-running-on-nvidia-dgx-spark-a-complete-beginners-step-by-step-guide-to-installation-benchmarks-roofline-analysis-and-common-misconcep/)
- [Dre Dyson: Advanced Guide](https://dredyson.com/how-i-mastered-fully-custom-cuda-native-deepseek-4-flash-dwarfstar-4-on-nvidia-dgx-spark-the-complete-advanced-configuration-guide-with-benchmarks-roofline-analysis-tensor-parallelism-a/)
- [NVIDIA Forums: ds4 auf Spark](https://forums.developer.nvidia.com/t/fully-custom-cuda-native-deepseek-4-flash-optimized-for-1x-spark-antirez-ds4/369791)
- [Entrpi/ds4-on-spark Installer](https://github.com/Entrpi/ds4-on-spark)
