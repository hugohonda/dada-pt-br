# DADA-PT-BR

Complete pipeline for translating and evaluating multilingual AI safety datasets in Brazilian Portuguese.

## Requirements

- Python 3.10+
- uv (Python package manager)
- Ollama with Gemma3 and TowerInstruct models

## Installation

```bash
git clone <repository-url>
cd dada-pt-br
uv pip install -e .
ollama pull gemma3:latest
```

## Workflow

```bash
# 1. Download dataset
dada download m_alert

# 2. Translate with multiple models
dada translate datasets/raw/file.json --model tower --limit=100
dada translate datasets/raw/file.json --model gemma3 --limit=100

# 3. Evaluate translations
dada evaluate output/translated/file_tower.json --limit=100
dada evaluate output/translated/file_gemma3.json --limit=100

# 4. Merge evaluations (keep best translations)
dada merge file1.json file2.json --limit=100

# 5. Review merged results
dada review output/evaluated/merged_file.json --limit=100
```

## Commands

```bash
# Download datasets/models
dada download <dataset_name|model_name>

# Translate datasets
dada translate <input_file> [--model=model_name] [--workers=N] [--limit=N]

# Evaluate translation quality
dada evaluate <input_file> [--limit=N]

# Merge evaluation results
dada merge <file1> <file2> [--limit=N]

# Review merged translations
dada review <input_file> [--limit=N]

# List datasets/models/files
dada list | dada models | dada files
```

## Available Datasets

- `m_alert`: M-ALERT multilingual dataset
- `agent_harm_chat`: AI Safety Institute AgentHarm chat dataset
- `alert`: Babelscape ALERT dataset

## Models

- `gemma3`: Google Gemma-3-4b-it (translation and revision - via Ollama)
- `tower`: Unbabel/TowerInstruct-Mistral-7B-v0.2 (translation - via Ollama)
- `xcomet-xl`: Unbabel/XCOMET-XL (evaluation)

## Output Structure

```
output/
├── 01-translated/           # Step 1: Translate English → Portuguese
│   └── {timestamp}_{dataset}_{model}_translated.json
├── 02-evaluated/            # Step 2: Evaluate translation quality
│   └── {timestamp}_{dataset}_evaluated.json
├── 03-merged/               # Step 3: Merge multiple evaluations
│   └── {timestamp}_{dataset}_merged.json
├── 04-reviewed/             # Step 4: Review and improve translations
│   └── {timestamp}_{dataset}_reviewed.json
└── logs/                    # System logs
```
