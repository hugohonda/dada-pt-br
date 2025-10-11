# DADA-PT-BR

This project provides a complete pipeline for downloading, translating, and evaluating multilingual AI safety datasets in Brazilian Portuguese.

## Requirements

- Python 3.10+
- uv (Python package manager)
- Ollama with Gemma3 model

## Installation

```bash
# Install dependencies
uv sync

# Install Ollama and pull model
ollama pull gemma3:latest
```

## Usage

```bash
# Main CLI interface (recommended)
uv run dada <command> [options]

# Direct script usage
uv run python <script.py> [options]
```

## Commands

### Main CLI

```bash
# Download datasets or models
uv run dada download <dataset_name|model_name>

# Translate datasets
uv run dada translate <input_file> [--output=file] [--workers=N]

# Evaluate translation quality
uv run dada evaluate <input_file> [--output=file] [--limit=N]

# List available datasets
uv run dada list

# List available models
uv run dada models

# List downloaded files
uv run dada files

# Show help
uv run dada --help
```

### Direct Script Usage

```bash
# Downloader (datasets and models)
uv run python downloader.py <dataset_name|model_name> [--list-datasets] [--list-models]

# Translator
uv run python translator.py <input_file> [--output=file] [--workers=N] [--tower] [--limit=N]

# Evaluator
uv run python evaluator.py <input_file> [--output=file] [--limit=N]
```

## Available Datasets

- `agent_harm_chat`: AI Safety Institute AgentHarm chat dataset
- `agent_harm_harmful`: AI Safety Institute AgentHarm harmful dataset
- `agent_harm_harmless`: AI Safety Institute AgentHarm harmless dataset
- `alert`: Babelscape ALERT dataset
- `alert_adversarial`: Babelscape ALERT adversarial dataset
- `m_alert`: M-ALERT multilingual dataset

## Available Models

- `xcomet-xl`: Unbabel/XCOMET-XL
- `tower-instruct-mistral`: Unbabel/TowerInstruct-Mistral-7B-v0.2

## Project Structure

```
dada-pt-br/
├── config/
│   ├── datasets.py          # Dataset and model configurations
│   └── logging.py           # Logging setup
├── data/
│   ├── translated/          # Translated datasets by model
│   ├── evaluated/           # Quality evaluation results
│   └── merged/              # Best translation selections
├── outputs/
│   ├── visualizations/      # Analysis charts and plots
│   ├── reports/             # Analysis and translation reports
│   └── logs/                # System logs
├── prompts/                 # Translation prompts
├── utils.py                 # Common utilities
├── llm_client.py            # LLM/Ollama client wrapper
├── main.py                  # Main CLI interface
├── downloader.py            # Dataset downloader
├── translator.py            # Dataset translator
├── evaluator.py             # Translation quality evaluator
├── analyser.py              # Analysis and visualization
├── merger.py                # Best translation merger
├── report_generator.py      # Translation reporting system
└── pyproject.toml           # Dependencies
```

## Workflow

```bash
# 1. Download dataset
uv run dada download m_alert

# 2. Download models (optional - will auto-download if needed)
uv run dada download xcomet-xl

# 3. Translate dataset
uv run dada translate data/m_alert_train.json --workers=4 --tower

# 4. Evaluate translation quality
uv run dada evaluate data/translated/m_alert/towerinstruct/m_alert_train_translated_20251006_123456.json

# 5. Check results
uv run dada files
```

## Output Files

### Translation
- `data/translated/dataset/model/dataset_translated_YYYYMMDD_HHMMSS.json` - Translated dataset
- `outputs/reports/dataset/translation/model/dataset_translation_YYYYMMDD_HHMMSS.json` - Translation report
- `outputs/logs/translation/dataset_translation_YYYYMMDD_HHMMSS.log` - Translation logs

### Evaluation
- `data/evaluated/dataset/model/dataset_evaluated_YYYYMMDD_HHMMSS.json` - Evaluated data
- `outputs/reports/dataset/evaluation/dataset_evaluation_YYYYMMDD_HHMMSS.json` - Evaluation report
- `outputs/logs/evaluation/dataset_evaluation_YYYYMMDD_HHMMSS.log` - Evaluation logs

### Analysis
- `outputs/visualizations/dataset/chart_name.png` - Analysis charts
- `outputs/reports/dataset/analysis/dataset_analysis_YYYYMMDD_HHMMSS.txt` - Analysis report

## Configuration

### Adding New Datasets

Edit `config/datasets.py`:

```python
DATASETS = {
    "your_dataset": "org/your-dataset:split",
    # Add more datasets here
}
```

### Adding New LLM Models

Edit `config/datasets.py`:

```python
LLM_MODELS = {
    "your_model": {
        "ollama_name": "your/model:tag",
        "display_name": "Your Model",
        "default": False,
    },
    # Add more models here
}
```

### Translation Prompts

- `prompts/translation_single.md` - Single language translation
- `prompts/translation_multilingual.md` - Multilingual translation
- `prompts/translation_safety.md` - AI safety translation
