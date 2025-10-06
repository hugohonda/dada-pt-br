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
# Main CLI interface
uv run main.py <command> [options]

# Direct script usage
uv run python <script.py> [options]
```

## Commands

### Main CLI

```bash
# Download datasets or models
uv run main.py download <dataset_name|model_name>

# Translate datasets
uv run main.py translate <input_file> [--output=file] [--workers=N]

# Evaluate translation quality
uv run main.py evaluate <input_file> [--output=file] [--limit=N]

# List available datasets
uv run main.py list

# List available models
uv run main.py models

# List downloaded files
uv run main.py files

# Show help
uv run main.py help
```

### Direct Script Usage

```bash
# Downloader (datasets and models)
uv run python downloader.py <dataset_name|model_name>

# Translator
uv run python translator.py <input_file> [--output=file] [--workers=N]

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
dadaptbr/
├── config/
│   ├── datasets.py          # Dataset configurations
│   └── logging.py           # Logging setup
├── datasets/
│   ├── raw/                 # Downloaded datasets
│   ├── processed/           # Translated datasets
│   └── evaluation/          # Translation quality evaluation
├── models/                  # Downloaded models cache
├── logs/                    # System logs
├── reports/                 # Analysis reports
├── prompts/                 # Translation prompts
├── utils.py                 # Common utilities
├── main.py                  # Main CLI interface
├── downloader.py            # Dataset downloader
├── translator.py            # Dataset translator
├── evaluator.py             # Translation quality evaluator
├── report_generator.py      # Translation reporting system
└── pyproject.toml           # Dependencies
```

## Workflow

```bash
# 1. Download dataset
uv run main.py download m_alert

# 2. Download models (optional - will auto-download if needed)
uv run main.py download xcomet-xl

# 3. Translate dataset
uv run main.py translate datasets/raw/m_alert_train.json --workers=4

# 4. Evaluate translation quality
uv run main.py evaluate datasets/processed/m_alert_train_translated_20251006_123456.json

# 5. Check results
uv run main.py files
```

## Output Files

### Translation
- `datasets/processed/dataset_translated_YYYYMMDD_HHMMSS.json` - Translated dataset
- `datasets/processed/dataset_translated_YYYYMMDD_HHMMSS.jsonl` - Line-by-line backup
- `reports/translation_report_YYYYMMDD_HHMMSS.json` - Translation report
- `logs/translation_YYYYMMDD_HHMMSS.log` - Translation logs

### Evaluation
- `datasets/evaluation/dataset_evaluated_YYYYMMDD_HHMMSS.json` - Evaluated data
- `datasets/evaluation/dataset_evaluated_YYYYMMDD_HHMMSS.jsonl` - Line-by-line backup
- `reports/dataset_evaluation_report_YYYYMMDD_HHMMSS.json` - Evaluation report
- `logs/evaluation_YYYYMMDD_HHMMSS.log` - Evaluation logs

## Configuration

### Adding New Datasets

Edit `config/datasets.py`:

```python
DATASETS = {
    "your_dataset": "org/your-dataset:split",
    # Add more datasets here
}
```

### Translation Prompts

- `prompts/translation_single.md` - Single language translation
- `prompts/translation_multilingual.md` - Multilingual translation
- `prompts/translation_safety.md` - AI safety translation
