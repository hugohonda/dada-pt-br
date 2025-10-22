# DADA-PT-BR

Translation and evaluation pipeline for multilingual AI safety datasets in Brazilian Portuguese.

## Quick Start

```bash
# Install
uv pip install -e .

# Pull models (first time only)
ollama pull gemma3:latest
ollama pull tibellium/towerinstruct-mistral:7b

# Run pipeline
dada run datasets/raw/m_alert.json --models tower,gemma3 --limit 100
```

## Requirements

- Python 3.10+
- Ollama with translation models
- (Optional) CUDA for faster evaluation

## Usage

### Full Pipeline (Recommended)
```bash
dada run <input_file> --models tower,gemma3 [--limit N]
```

### Step-by-Step
```bash
# 1. Translate (pick one or both models)
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

| Command | Description |
|---------|-------------|
| `dada run <file>` | Run full pipeline |
| `dada translate <file>` | Translate to PT-BR |
| `dada evaluate <file>` | Score translations (XCOMET) |
| `dada merge <file1> <file2>` | Merge best translations |
| `dada review <file>` | LLM-based improvement |
| `dada list` | Show available datasets |
| `dada models` | Show available models |

**Common Options:**
- `--limit N` - Process only N examples
- `--workers auto` - Parallel workers (auto-detect)
- `--device auto` - Use CUDA if available
- `--output <path>` - Custom output path

## Models

- **tower** - TowerInstruct-Mistral-7B (default translator)
- **gemma3** - Gemma-3-4b (reviewer, second translator)
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

## Dataset Support

Place raw datasets in `datasets/raw/`. Currently supported:
- M-ALERT (`felfri_M-ALERT_train.json`)
- ALERT (`Babelscape_ALERT_*.json`)
- AgentHarm (`ai-safety-institute_AgentHarm_*.json`)
