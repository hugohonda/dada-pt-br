# DADA-PT-BR

Translation and evaluation pipeline for multilingual AI safety datasets in Brazilian Portuguese.

## Quick Start

```bash
# Install (choose based on your needs)
uv pip install -e ".[light]"      # Basic: list, models, files, merge
uv pip install -e ".[translate]"  # + translate, review
uv pip install -e ".[evaluate]"   # + evaluate (needs GPU/CPU)
uv pip install -e ".[full]"       # All phases

# Pull models (first time only)
ollama pull gemma3:latest
ollama pull tibellium/towerinstruct-mistral:7b

# Run pipeline
dada run datasets/raw/m_alert.json --models tower,gemma3 --limit 100
```

## Requirements

- Python 3.10+
- Ollama with translation models
- (Optional) CUDA for evaluation phase

### Performance Optimization

**Multi-GPU:** Set `OLLAMA_SCHED_SPREAD=1` and `OLLAMA_NUM_PARALLEL=N` on Ollama server. `--workers auto` uses ~3 workers per GPU.

**CPU:** Connection pooling auto-configured based on `--workers` (reuses HTTP connections, reduces overhead).

## Usage

### Full Pipeline (Recommended)
```bash
dada run <input_file> --models tower,gemma3 [--limit N]
```

### Step-by-Step
```bash
# 1. Translate (pick one or multiple models)
dada translate <file> --model tower
dada translate <file> --model gemma3

# 2. Evaluate quality
dada evaluate <translated_file>

# 3. Merge (if using multiple models)
dada merge <eval_file1> <eval_file2>

# 4. Review and improve
dada review <merged_or_eval_file>
```

## Commands

| Command | Description | Dependencies |
|---------|-------------|--------------|
| `dada list` | Show available datasets | light |
| `dada models` | Show available models | light |
| `dada files` | List output files | light |
| `dada merge <file1> <file2>` | Merge best translations | light |
| `dada translate <file>` | Translate to PT-BR | translate |
| `dada review <file>` | LLM-based improvement | translate |
| `dada evaluate <file>` | Score translations (XCOMET) | evaluate |
| `dada run <file>` | Run full pipeline | full |

**Common Options:**
- `--limit N` - Process only N examples
- `--workers auto` - Parallel workers (GPU-aware, auto-detects)
- `--device auto` - Use CUDA if available
- `--output <path>` - Custom output path

## Models

- **tower** - TowerInstruct-Mistral-7B (default translator)
- **gemma3** - Gemma-3-4b (reviewer, second translator)
- **qwen3** - Qwen3-4B (third translator)
- **xcomet-xl** - XCOMET-XL (quality evaluator)

## Output Structure

```
output/
├── 01-translated/  # {timestamp}_{dataset}_{model}_translated.json
├── 02-evaluated/   # {timestamp}_{dataset}_evaluated.json
├── 03-merged/      # {timestamp}_{dataset}_merged.json
├── 04-reviewed/    # {timestamp}_{dataset}_reviewed.json
└── runs/{id}/      # manifest.json (pipeline artifacts)
```

## Docker Support

### Quick Start with Docker

```bash
# Build CPU version
docker build -t dada .

# Build GPU version
docker build -f Dockerfile.gpu -t dada-gpu .

# Basic commands
docker run dada list
docker run dada models
docker run dada files

# CPU: With data volumes
docker run --rm --network host \
           --user $(id -u):$(id -g) \
           -v $(pwd)/datasets:/app/datasets \
           -v $(pwd)/output:/app/output \
           -v $(pwd)/logs:/app/logs \
           dada translate datasets/felfri_M-ALERT_train.json --model gemma3

# GPU: For evaluation (requires NVIDIA runtime)
docker run --rm --gpus all --network host \
           --user $(id -u):$(id -g) \
           -v $(pwd)/datasets:/app/datasets \
           -v $(pwd)/output:/app/output \
           -v $(pwd)/logs:/app/logs \
           dada-gpu evaluate output/01-translated/translated_file.json
```

### Docker Compose

```bash
# Set your user ID for proper file permissions
export UID=$(id -u)
export GID=$(id -g)

# CPU version
docker-compose --profile cpu up

# GPU version (requires NVIDIA Docker runtime)
docker-compose --profile gpu up
```

### Available Dockerfiles

- **`Dockerfile`** - Multi-stage build (CPU-only, PyTorch CPU, secure, small)
- **`Dockerfile.gpu`** - Multi-stage build (CUDA 12.1, GPU-enabled, secure)

### Volume Mounts

The Docker containers use bind mounts for syncing data with the host:
- `./output:/app/output` - Pipeline outputs
- `./logs:/app/logs` - Log files
- `./datasets:/app/datasets` - Input datasets
- `./final-data:/app/final-data` - Final processed data

**Permissions:** Containers run as non-root user. Use `--user $(id -u):$(id -g)` with `docker run` or export `UID`/`GID` for docker-compose to match host permissions and avoid permission issues

## Dataset Support

Place raw datasets in `datasets/raw/`. Currently supported:
- M-ALERT (`felfri_M-ALERT_train.json`)
- ALERT (`Babelscape_ALERT_*.json`)
- AgentHarm (`ai-safety-institute_AgentHarm_*.json`)
