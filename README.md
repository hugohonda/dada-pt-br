# DDADA-PT-BR

Dataset para Avaliação de Danos Agênticos em contexto linguístico Português Brasileiro (DADA-PT-BR), projetado para apoiar a validação de segurança de agentes de LLM em Português do Brasil.

## Quick Start

```bash
# Install dependencies
uv sync

# List available datasets
uv run main.py list

# Download a dataset
uv run main.py download agent_harm_chat

# Translate a dataset (with parallel processing)
uv run main.py translate datasets/raw/agent_harm_chat_test_public.json
uv run main.py translate datasets/raw/felfri_M-ALERT_train.json --workers=4

# Specify custom output location
uv run main.py translate dataset.json --output=custom_output.json --workers=8

# Resume interrupted translation
uv run main.py translate datasets/raw/large_dataset.json  # Automatically resumes from .jsonl file

# See what you have
uv run main.py files
```

## Available Datasets

- `agent_harm_chat`: AI Safety Institute AgentHarm chat dataset
- `agent_harm_harmful`: AI Safety Institute AgentHarm harmful dataset
- `agent_harm_harmless`: AI Safety Institute AgentHarm harmless dataset
- `alert`: Babelscape ALERT dataset
- `alert_adversarial`: Babelscape ALERT adversarial dataset
- `m_alert`: M-ALERT multilingual dataset

## Commands

```bash
# Download datasets
uv run main.py download <dataset_name>

# Translate datasets
uv run main.py translate <input_file> [output_file]

# List available datasets
uv run main.py list

# List downloaded files
uv run main.py files
```

## Translation

Uses Ollama with Gemma3:latest for Brazilian Portuguese translation.

**Setup:**
```bash
# Install Ollama and pull model
ollama pull gemma3:latest
```

**Features:**
- **Context-aware translation**: Automatically detects dataset type and uses appropriate prompt
- **Multilingual support**: Uses context from multiple languages for robust translation
- **Selective translation**: Only translates name and prompt fields, keeps categories in original language
- **Colloquial adaptation**: Brazilian Portuguese with proper slang and expressions
- **Multiple prompts**: Different prompts for single language, multilingual, and AI safety datasets
- **Comprehensive logging**: Detailed logs with timestamps saved to files for debugging
- **Performance reporting**: Automatic generation of detailed translation reports with system metrics
- **Crash resilience**: Line-by-line JSONL saving prevents data loss on interruption
- **Resume capability**: Automatically resumes from where it left off after crashes
- **Parallel processing**: Utilizes multiple CPU cores for faster translation (2-8 workers recommended)
- **Timestamped outputs**: All files use consistent timestamps for easy organization and tracking

## Project Structure

```
dadaptbr/
├── config/
│   ├── datasets.py          # Dataset configurations
│   └── logging.py           # Logging setup
├── datasets/
│   ├── raw/                 # Downloaded datasets
│   └── processed/           # Translated datasets
├── logs/                    # Translation logs
│   └── translation_YYYYMMDD_HHMMSS.log    # Timestamped log files
├── reports/                 # Translation reports
│   ├── translation_report_YYYYMMDD_HHMMSS.json    # Detailed JSON reports
│   └── translation_summary_YYYYMMDD_HHMMSS.txt    # Human-readable summaries
├── datasets/processed/      # Translated datasets
│   ├── dataset_translated_YYYYMMDD_HHMMSS.json    # Final JSON output
│   └── dataset_translated_YYYYMMDD_HHMMSS.jsonl   # Line-by-line backup
├── prompts/
│   ├── translation_single.md        # Single language translation prompt
│   ├── translation_multilingual.md  # Multilingual translation prompt
│   └── translation_safety.md        # AI safety translation prompt
├── downloader.py            # Dataset downloader
├── translator.py            # Dataset translator
├── report_generator.py      # Translation reporting system
├── main.py                  # Main CLI interface
└── pyproject.toml           # Dependencies
```

## Adding New Datasets

Edit `config/datasets.py`:

```python
DATASETS = {
    "your_dataset": "org/your-dataset:split",
    # Add more datasets here
}
```
