# DDADA-PT-BR

This project provides a complete pipeline for downloading, translating, and evaluating multilingual AI safety datasets in Brazilian Portuguese.

## Requirements

- **Python 3.10+**
- **uv** (Python package manager)
- **Ollama** with Gemma3 model
- **CUDA** (optional, for GPU acceleration)

## Quick Start

```bash
# Install dependencies
uv sync

# Install Ollama and pull model
ollama pull gemma3:latest

# List available datasets
uv run main.py list

# Download a dataset
uv run main.py download agent_harm_chat

# Translate a dataset (with parallel processing)
uv run main.py translate datasets/raw/agent_harm_chat_test_public.json
uv run main.py translate datasets/raw/felfri_M-ALERT_train.json --workers=4

# Evaluate translation quality
uv run main.py evaluate datasets/processed/dataset_translated_20251006_123456.json
uv run main.py evaluate datasets/processed/dataset_translated_20251006_123456.json --limit=100

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
uv run main.py translate <input_file> [output_file] [--workers=N]

# Evaluate translation quality
uv run main.py evaluate <input_file> [--limit=N] [--output=file.json]

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

## Translation Quality Evaluation

Uses XCOMET-XL for automatic translation quality assessment with explainable error detection.

**Features:**
- **XCOMET-XL model**: State-of-the-art translation quality evaluation with ~3.5B parameters
- **Explainable evaluation**: Provides detailed error spans and confidence scores
- **Batch processing**: Efficient evaluation of large datasets with batch size 8
- **Error detection**: Identifies specific translation errors with severity levels
- **Score explanation**: Detailed analysis of translation quality issues
- **Memory efficient**: Automatic CPU fallback for large models
- **Progress tracking**: Real-time progress bars during evaluation
- **Crash resilience**: Line-by-line JSONL saving prevents data loss
- **Comprehensive reporting**: Detailed evaluation reports with statistics

**Usage:**
```bash
# Evaluate all translations
uv run main.py evaluate datasets/processed/dataset_translated_20251006_123456.json

# Evaluate with limit
uv run main.py evaluate datasets/processed/dataset_translated_20251006_123456.json --limit=100

# Direct evaluation
uv run python evaluator.py datasets/processed/dataset_translated_20251006_123456.json --limit=50
```

**Output:**
- **Evaluated data**: `datasets/evaluation/dataset_evaluated_YYYYMMDD_HHMMSS.json`
- **Line-by-line backup**: `datasets/evaluation/dataset_evaluated_YYYYMMDD_HHMMSS.jsonl`
- **Evaluation report**: `reports/dataset_evaluation_report_YYYYMMDD_HHMMSS.json`
- **Evaluation logs**: `logs/evaluation_YYYYMMDD_HHMMSS.log`

**Evaluation Output Format:**
```json
{
  "index": 0,                    // Processing order
  "id": 1,                       // Original document ID
  "source": "Original text...",  // Source text
  "translation": "Translated...",// Translation text
  "score": 0.957,                // XCOMET quality score (0-1)
  "system_score": 0.826,         // System-level score
  "error_spans": [               // Detailed error analysis
    {
      "text": "error text",
      "confidence": 0.405,
      "severity": "minor",
      "start": 10,
      "end": 20
    }
  ]
}
```

## Project Structure

```
dadaptbr/
├── config/
│   ├── datasets.py          # Dataset configurations
│   └── logging.py           # Logging setup
├── datasets/
│   ├── raw/                 # Downloaded datasets
│   ├── processed/           # Translated datasets
│   │   ├── dataset_translated_YYYYMMDD_HHMMSS.json    # Final JSON output
│   │   └── dataset_translated_YYYYMMDD_HHMMSS.jsonl   # Line-by-line backup
│   └── evaluation/          # Translation quality evaluation
│       ├── dataset_evaluated_YYYYMMDD_HHMMSS.json     # Evaluated data
│       └── dataset_evaluated_YYYYMMDD_HHMMSS.jsonl    # Line-by-line backup
├── logs/                    # System logs
│   ├── translation_YYYYMMDD_HHMMSS.log    # Translation logs
│   ├── evaluation_YYYYMMDD_HHMMSS.log     # Evaluation logs
│   ├── report_YYYYMMDD_HHMMSS.log         # Report generation logs
│   └── download_YYYYMMDD_HHMMSS.log       # Download logs
├── reports/                 # Analysis reports
│   ├── translation_report_YYYYMMDD_HHMMSS.json    # Translation reports
│   ├── translation_summary_YYYYMMDD_HHMMSS.txt    # Translation summaries
│   └── dataset_evaluation_report_YYYYMMDD_HHMMSS.json  # Evaluation reports
├── prompts/
│   ├── translation_single.md        # Single language translation prompt
│   ├── translation_multilingual.md  # Multilingual translation prompt
│   └── translation_safety.md        # AI safety translation prompt
├── downloader.py            # Dataset downloader
├── translator.py            # Dataset translator
├── evaluator.py             # Translation quality evaluator
├── report_generator.py      # Translation reporting system
├── main.py                  # Main CLI interface
└── pyproject.toml           # Dependencies
```

## Complete Workflow

Here's a typical workflow for processing a dataset:

```bash
# 1. Download a dataset
uv run main.py download m_alert

# 2. Translate the dataset
uv run main.py translate datasets/raw/m_alert_train.json --workers=4

# 3. Evaluate translation quality
uv run main.py evaluate datasets/processed/m_alert_train_translated_20251006_123456.json

# 4. Check results
ls datasets/evaluation/
ls reports/
ls logs/
```

**Typical Output Files:**
- `datasets/processed/m_alert_train_translated_20251006_123456.json` - Translated dataset
- `datasets/evaluation/m_alert_train_evaluated_20251006_123456.json` - Quality evaluation
- `reports/m_alert_train_evaluation_report_20251006_123456.json` - Evaluation statistics
- `logs/translation_20251006_123456.log` - Translation process logs
- `logs/evaluation_20251006_123456.log` - Evaluation process logs

## Adding New Datasets

Edit `config/datasets.py`:

```python
DATASETS = {
    "your_dataset": "org/your-dataset:split",
    # Add more datasets here
}
```
