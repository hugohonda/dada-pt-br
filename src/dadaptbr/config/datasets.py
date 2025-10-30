DATASETS = {
    "m_alert": "felfri/M-ALERT",
}

# Dataset file mappings: key -> filename in datasets/
DATASET_FILES = {
    "m_alert": "felfri_M-ALERT_train.json",
    "alert": "Babelscape_ALERT_alert_test.json",
    "alert_adversarial": "Babelscape_ALERT_alert_adversarial_test.json",
    "agent_harm_chat": "ai-safety-institute_AgentHarm_chat_test_public.json",
    "agent_harm_harmful": "ai-safety-institute_AgentHarm_harmful_test_public.json",
    "agent_harm_harmless": "ai-safety-institute_AgentHarm_harmless_benign_test_public.json",
}

# Filename mappings for dataset identification (reverse lookup)
FILENAME_MAPPINGS = {
    "felfri_M-ALERT_train.json": "m_alert",
    "felfri_M-ALERT_test.json": "m_alert",
    "Babelscape_ALERT_alert_train.json": "alert",
    "Babelscape_ALERT_alert_test.json": "alert",
    "Babelscape_ALERT_alert_adversarial_train.json": "alert_adversarial",
    "Babelscape_ALERT_alert_adversarial_test.json": "alert_adversarial",
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
    },
    "gemma3gaia": {
        "ollama_name": "brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16",
        "display_name": "Gemma3Gaia",
    },
    "tower": {
        "ollama_name": "thinkverse/towerinstruct:latest",
        "display_name": "TowerInstruct",
        "default": True,
    },
    "qwen3": {
        "ollama_name": "qwen3:4b",
        "display_name": "Qwen3",
        "think": False,
    },
}

LLM_DEFAULT_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 2048,
}

DEFAULT_MODELS = {
    "translation": "tower",
    "review": "gemma3",
    "tie_breaker": "tower",
}

# Processing defaults
FILE_PROCESSING = {
    "default_device": "cpu",
    "default_batch_size": 8,
}

PHASE_WORKERS = {
    "translation": {"default": 8, "max": 16},
    "evaluation": {"default": 1, "max": 1},
    "review": {"default": 8, "max": 16},
    "merge": {"default": 1, "max": 1},
}
