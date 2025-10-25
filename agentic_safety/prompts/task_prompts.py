"""
Task execution prompts for the Agentic Safety Framework.

This module contains prompts for task planning, execution, and analysis
in Brazilian Portuguese contexts.
"""

from typing import Any


class TaskPrompts:
    """Collection of task execution prompts."""

    @staticmethod
    def get_task_planning_prompt(task: str, available_tools: list[str] = None) -> str:
        """Get prompt for task planning."""
        tools_str = ""
        if available_tools:
            tools_str = f"\nFerramentas disponíveis: {', '.join(available_tools)}"

        return f"""
        Crie um plano detalhado para executar a seguinte tarefa:

        Tarefa: {task}
        {tools_str}

        Considere:
        1. Quebrar a tarefa em passos lógicos
        2. Identificar dependências entre passos
        3. Considerar aspectos de segurança
        4. Incluir verificações e validações
        5. Usar ferramentas apropriadas quando necessário

        Responda em JSON:
        {{
            "plano": [
                {{
                    "passo": 1,
                    "descricao": "descrição do passo",
                    "tipo": "raciocinio/ferramenta/verificacao",
                    "ferramenta": "nome_da_ferramenta",
                    "argumentos": {{"arg1": "valor1"}},
                    "verificacoes_seguranca": ["verificação1", "verificação2"]
                }}
            ],
            "complexidade": "simples/moderada/complexa",
            "tempo_estimado": "X minutos",
            "nivel_risco": "baixo/medio/alto"
        }}
        """

    @staticmethod
    def get_step_execution_prompt(step: dict[str, Any], context: dict[str, Any] = None) -> str:
        """Get prompt for executing a single step."""
        context_str = ""
        if context:
            context_str = f"\nContexto: {context}"

        return f"""
        Execute o seguinte passo:

        Descrição: {step.get('descricao', '')}
        Tipo: {step.get('tipo', 'raciocinio')}
        {context_str}

        Instruções:
        1. Execute o passo conforme descrito
        2. Verifique a segurança antes de prosseguir
        3. Documente o resultado
        4. Identifique se há dependências para próximos passos

        Responda em JSON:
        {{
            "sucesso": true/false,
            "resultado": "resultado da execução",
            "verificacoes_realizadas": ["lista", "de", "verificações"],
            "proximos_passos": ["lista", "de", "próximos", "passos"],
            "observacoes": "observações adicionais"
        }}
        """

    @staticmethod
    def get_tool_usage_prompt(tool_name: str, args: dict[str, Any], context: dict[str, Any] = None) -> str:
        """Get prompt for using a specific tool."""
        context_str = ""
        if context:
            context_str = f"\nContexto: {context}"

        return f"""
        Use a ferramenta '{tool_name}' com os seguintes argumentos:

        Argumentos: {args}
        {context_str}

        Instruções:
        1. Valide os argumentos antes de usar
        2. Execute a ferramenta de forma segura
        3. Verifique o resultado
        4. Documente qualquer problema

        Responda em JSON:
        {{
            "ferramenta_usada": "{tool_name}",
            "argumentos": {args},
            "sucesso": true/false,
            "resultado": "resultado da ferramenta",
            "erro": "mensagem de erro se houver",
            "tempo_execucao": "X segundos"
        }}
        """

    @staticmethod
    def get_reasoning_prompt(question: str, context: dict[str, Any] = None) -> str:
        """Get prompt for reasoning tasks."""
        context_str = ""
        if context:
            context_str = f"\nContexto: {context}"

        return f"""
        Analise e responda à seguinte pergunta:

        Pergunta: {question}
        {context_str}

        Instruções:
        1. Analise cuidadosamente a pergunta
        2. Considere o contexto brasileiro
        3. Forneça uma resposta clara e precisa
        4. Seja transparente sobre limitações
        5. Mantenha o foco na segurança e ética

        Sua resposta deve ser:
        - Clara e objetiva
        - Culturalmente apropriada
        - Segura e ética
        - Em português brasileiro
        """

    @staticmethod
    def get_verification_prompt(step_result: dict[str, Any], expected_outcome: str) -> str:
        """Get prompt for verifying step results."""
        return f"""
        Verifique se o resultado do passo está correto:

        Resultado obtido: {step_result}
        Resultado esperado: {expected_outcome}

        Verifique:
        1. Se o resultado corresponde ao esperado
        2. Se há erros ou inconsistências
        3. Se a qualidade está adequada
        4. Se há riscos de segurança
        5. Se pode prosseguir para o próximo passo

        Responda em JSON:
        {{
            "verificacao_aprovada": true/false,
            "qualidade": "excelente/boa/regular/ruim",
            "problemas_detectados": ["lista", "de", "problemas"],
            "recomendacoes": ["lista", "de", "recomendações"],
            "pode_prosseguir": true/false
        }}
        """

    @staticmethod
    def get_error_handling_prompt(error: str, context: dict[str, Any] = None) -> str:
        """Get prompt for handling errors."""
        context_str = ""
        if context:
            context_str = f"\nContexto: {context}"

        return f"""
        Trate o seguinte erro:

        Erro: {error}
        {context_str}

        Instruções:
        1. Analise a causa do erro
        2. Determine se é recuperável
        3. Proponha soluções alternativas
        4. Considere implicações de segurança
        5. Documente a resolução

        Responda em JSON:
        {{
            "erro_analisado": true/false,
            "causa": "causa do erro",
            "recuperavel": true/false,
            "solucoes": ["lista", "de", "soluções"],
            "implicacoes_seguranca": "análise de segurança",
            "acao_recomendada": "ação recomendada"
        }}
        """

    @staticmethod
    def get_completion_analysis_prompt(task: str, results: list[dict[str, Any]]) -> str:
        """Get prompt for analyzing task completion."""
        results_str = "\n".join([
            f"- Passo {i+1}: {r.get('descricao', 'N/A')} - {r.get('status', 'N/A')}"
            for i, r in enumerate(results)
        ])

        return f"""
        Analise a conclusão da seguinte tarefa:

        Tarefa: {task}
        Resultados dos passos:
        {results_str}

        Avalie:
        1. Se a tarefa foi concluída com sucesso
        2. Qualidade geral da execução
        3. Problemas encontrados
        4. Aspectos de segurança
        5. Lições aprendidas

        Responda em JSON:
        {{
            "tarefa_concluida": true/false,
            "qualidade_execucao": "excelente/boa/regular/ruim",
            "problemas_encontrados": ["lista", "de", "problemas"],
            "aspectos_seguranca": "análise de segurança",
            "licoes_aprendidas": ["lista", "de", "lições"],
            "recomendacoes": ["lista", "de", "recomendações"]
        }}
        """
