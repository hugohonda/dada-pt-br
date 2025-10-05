# Available datasets
DATASETS = {
    # AgentHarm datasets
    "agent_harm_chat": "ai-safety-institute/AgentHarm:chat",
    "agent_harm_harmful": "ai-safety-institute/AgentHarm:harmful",
    "agent_harm_harmless": "ai-safety-institute/AgentHarm:harmless_benign",
    # ALERT datasets
    "alert": "Babelscape/ALERT:alert",
    "alert_adversarial": "Babelscape/ALERT:alert_adversarial",
    # M-ALERT dataset
    "m_alert": "felfri/M-ALERT",
}

# Category translations for AgentHarm
AGENT_HARM_CATEGORIES = {
    "Disinformation": "Desinformação",
    "Malware": "Malware",
    "Phishing": "Phishing",
    "Social Engineering": "Engenharia Social",
    "Violence": "Violência",
    "Harassment": "Assédio",
    "Hate Speech": "Discurso de Ódio",
    "Self-Harm": "Autoagressão",
    "Illegal Activities": "Atividades Ilegais",
    "Privacy Violations": "Violações de Privacidade",
    "Fraud": "Fraude",
    "Terrorism": "Terrorismo",
    "Child Exploitation": "Exploração Infantil",
    "Drug Trafficking": "Tráfico de Drogas",
    "Money Laundering": "Lavagem de Dinheiro",
    "Weapons": "Armas",
    "Hacking": "Hacking",
    "Identity Theft": "Roubo de Identidade",
    "Scams": "Golpes",
    "Extortion": "Extorsão",
}
