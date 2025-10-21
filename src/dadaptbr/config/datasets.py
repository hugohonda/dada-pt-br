DATASETS = {
    "m_alert": "felfri/M-ALERT",
}

# Filename mappings for dataset identification
FILENAME_MAPPINGS = {
    # M-ALERT dataset
    "felfri_M-ALERT_train.json": "m_alert",
    "felfri_M-ALERT_test.json": "m_alert",
    # ALERT datasets
    "Babelscape_ALERT_alert_train.json": "alert",
    "Babelscape_ALERT_alert_test.json": "alert",
    "Babelscape_ALERT_alert_adversarial_train.json": "alert_adversarial",
    "Babelscape_ALERT_alert_adversarial_test.json": "alert_adversarial",
    # AgentHarm datasets
    "ai-safety-institute_AgentHarm_chat_train.json": "agent_harm_chat",
    "ai-safety-institute_AgentHarm_chat_test.json": "agent_harm_chat",
    "ai-safety-institute_AgentHarm_harmful_train.json": "agent_harm_harmful",
    "ai-safety-institute_AgentHarm_harmful_test.json": "agent_harm_harmful",
    "ai-safety-institute_AgentHarm_harmless_benign_train.json": "agent_harm_harmless",
    "ai-safety-institute_AgentHarm_harmless_benign_test.json": "agent_harm_harmless",
}

EVALUATION_MODELS = {
    "xcomet-xl": {
        "hf_model_id": "Unbabel/XCOMET-XL",
        "display_name": "XCOMET-XL",
        "default": True,
    },
}

TRANSLATION_MODELS = {
    "gemma3": {
        "ollama_name": "gemma3:latest",
        "display_name": "Gemma3",
        "default": True,
    },
    "tower": {
        "ollama_name": "tibellium/towerinstruct-mistral:7b",
        "display_name": "TowerInstruct-Mistral",
        "default": False,
    },
}

LLM_DEFAULT_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 2048,
}

# Model name to key mappings
MODEL_NAME_MAPPINGS = {
    "gemma3": "gemma3",
    "tower": "tower",
    "tibellium/towerinstruct-mistral:7b": "tower",
    "gemma3:latest": "gemma3",
}

# Default model preferences
DEFAULT_MODELS = {
    "translation": "tower",
    "review": "gemma3",
    "tie_breaker": "tower",
}

# Regex patterns for filename parsing
FILENAME_PATTERNS = {
    "pipeline_dataset": r"^\d{8}_\d{6}_([a-z_]+)\.json$",
}

# File processing constants
FILE_PROCESSING = {
    "default_workers": 8,
    "max_workers": 16,
    "default_batch_size": 16,
    "default_device": "cpu",
    "default_limit": None,
}

# Phase-specific worker configurations
PHASE_WORKERS = {
    "translation": {
        "default": 8,
        "max": 16,
        "safe_parallel": True,  # Ollama API calls are safe to parallelize
    },
    "evaluation": {
        "default": 4,
        "max": 8,
        "safe_parallel": False,  # XCOMET model loading is not thread-safe
    },
    "review": {
        "default": 8,
        "max": 16,
        "safe_parallel": True,  # Ollama API calls are safe to parallelize
    },
    "merge": {
        "default": 1,
        "max": 1,
        "safe_parallel": False,  # Single operation, no parallelization needed
    },
}

DATASET_CATEGORIES = {
    "m_alert": {
        "crime_injury": "Crime/Lesão",
        "hate_other": "Ódio/Outros",
        "hate_ethnic": "Ódio/Étnico",
        "crime_theft": "Crime/Roubo",
        "crime_propaganda": "Crime/Propaganda",
        "hate_women": "Ódio/Mulheres",
        "substance_drug": "Substância/Drogas",
        "substance_other": "Substância/Outros",
        "weapon_other": "Arma/Outros",
        "crime_cyber": "Crime/Cibernético",
        "hate_religion": "Ódio/Religião",
        "hate_lgbtq+": "Ódio/LGBTQ+",
        "sex_harrasment": "Sexo/Assédio",
        "sex_other": "Sexo/Outros",
        "crime_privacy": "Crime/Privacidade",
        "substance_alcohol": "Substância/Álcool",
        "crime_other": "Crime/Outros",
        "crime_tax": "Crime/Impostos",
        "substance_cannabis": "Substância/Cannabis",
        "self_harm_thin": "Auto-dano/Anorexia",
        "weapon_chemical": "Arma/Química",
        "weapon_biological": "Arma/Biológica",
        "crime_kidnapp": "Crime/Sequestro",
        "self_harm_suicide": "Auto-dano/Suicídio",
        "hate_body": "Ódio/Corpo",
        "weapon_radioactive": "Arma/Radioativa",
        "sex_porn": "Sexo/Pornografia",
        "self_harm_other": "Auto-dano/Outros",
        "hate_disabled": "Ódio/Deficientes",
        "weapon_firearm": "Arma/Arma de fogo",
        "substance_tobacco": "Substância/Tabaco",
        "hate_poor": "Ódio/Pobres",
        "unknown": "Desconhecido",
    },
}
