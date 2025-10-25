# Agentic Safety Framework

A comprehensive framework for evaluating and ensuring safety in agentic AI systems operating in Brazilian Portuguese contexts.

## Overview

The Agentic Safety Framework is designed to address the "Agentic Harmfulness Curse" - the unique safety vulnerabilities that emerge when LLM-based agents operate in lower-resource languages like Brazilian Portuguese. Built with Pydantic AI and Context7, it provides type-safe, culturally-aware safety evaluation capabilities.

## Key Features

- **Type-Safe Agent Implementation**: Built on Pydantic AI for robust type checking
- **Brazilian Portuguese Focus**: Culturally-aware safety evaluation
- **Comprehensive Safety Metrics**: Multi-dimensional safety assessment
- **Tool Integration**: Safe tool usage with built-in safety checks
- **Cultural Analysis**: LGPD compliance and cultural appropriateness evaluation
- **Memory System**: Context-aware interaction tracking
- **Task Planning**: Multi-step task decomposition with safety considerations

## Installation

```bash
# Install the framework
pip install -r agentic_safety/requirements.txt

# Install in development mode
pip install -e agentic_safety/
```

## Quick Start

```python
import asyncio
from agentic_safety.core.agent import SafetyAgent, AgentConfig
from agentic_safety.core.tools import WebSearchTool, EmailTool
from agentic_safety.evaluation.safety_evaluator import SafetyEvaluator

async def main():
    # Create agent configuration
    config = AgentConfig(
        model_name="gpt-4o",
        language="pt-BR",
        safety_threshold=0.8
    )

    # Create tools
    tools = [WebSearchTool(), EmailTool()]

    # Create safety agent
    agent = SafetyAgent(config, tools)

    # Execute a task
    result = await agent.execute_task("Buscar informações sobre o clima em São Paulo")
    print(f"Status: {result['status']}")
    print(f"Safety Score: {result.get('safety_score', 'N/A')}")

# Run the example
asyncio.run(main())
```

## Architecture

### Core Components

- **SafetyAgent**: Main agent class with built-in safety evaluation
- **ToolRegistry**: Manages available tools with safety checks
- **AgentMemory**: Context-aware memory system
- **TaskPlanner**: Multi-step task planning with safety considerations

### Evaluation Framework

- **SafetyEvaluator**: Comprehensive safety assessment
- **SafetyMetrics**: Detailed safety metrics and scoring
- **CulturalAnalyzer**: Brazilian cultural appropriateness evaluation
- **HarmCategory**: Predefined harm categories for Brazilian context

### Prompts

- **SafetyPrompts**: Safety evaluation and refusal prompts
- **TaskPrompts**: Task planning and execution prompts
- **EvaluationPrompts**: Performance and safety evaluation prompts

## Safety Features

### Built-in Safety Checks

- **Content Safety**: Detection of harmful or inappropriate content
- **Cultural Safety**: Brazilian cultural appropriateness evaluation
- **Legal Compliance**: LGPD and Brazilian law compliance
- **Tool Safety**: Safe tool usage with validation

### Harm Categories

The framework includes predefined harm categories specific to Brazilian context:

- Violência Física
- Discriminação
- Fraude e Engano
- Violação de Privacidade (LGPD)
- Desinformação
- Cyberbullying
- Exploração
- Conteúdo Ilegal

## Usage Examples

### Basic Agent Usage

```python
from agentic_safety.core.agent import SafetyAgent, AgentConfig

# Create agent
config = AgentConfig(language="pt-BR", safety_threshold=0.8)
agent = SafetyAgent(config)

# Execute task
result = await agent.execute_task("Tarefa em português brasileiro")
```

### Safety Evaluation

```python
from agentic_safety.evaluation.safety_evaluator import SafetyEvaluator

# Create evaluator
evaluator = SafetyEvaluator()

# Evaluate interaction
evaluation = await evaluator.evaluate_interaction(
    task="Tarefa",
    response="Resposta do agente"
)

print(f"Safety Score: {evaluation.overall_score}")
print(f"Cultural Score: {evaluation.cultural_score}")
```

### Custom Tools

```python
from agentic_safety.core.tools import BaseTool, ToolResult

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="Ferramenta personalizada",
            safety_level="medium"
        )

    async def execute(self, args):
        # Implement tool logic
        return ToolResult.success("Resultado da ferramenta")

    def validate_args(self, args):
        # Implement argument validation
        return True, "Valid"
```

## Configuration

### Agent Configuration

```python
config = AgentConfig(
    model_name="gpt-4o",           # LLM model to use
    temperature=0.7,               # Model temperature
    language="pt-BR",              # Primary language
    safety_threshold=0.8,          # Safety threshold
    max_tool_calls=10,             # Maximum tool calls per task
    enable_memory=True,            # Enable memory system
    enable_planning=True           # Enable task planning
)
```

### Safety Evaluation Configuration

```python
evaluator = SafetyEvaluator(
    model_name="gpt-4o",           # Evaluation model
    temperature=0.3                # Evaluation temperature
)
```

## Research Applications

This framework is designed for research into:

1. **Agentic Harmfulness Curse**: How safety vulnerabilities manifest in non-English contexts
2. **Cultural Safety**: Brazilian-specific safety considerations
3. **Multilingual AI Safety**: Cross-linguistic safety evaluation
4. **Constitutional AI**: Culturally-adapted alignment techniques

## Contributing

Contributions are welcome! Please see the main project repository for contribution guidelines.

## License

This project is part of the DADA-PT-BR research initiative. Please refer to the main project for licensing information.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{agentic_safety_framework,
  title={Agentic Safety Framework for Brazilian Portuguese},
  author={DADA-PT-BR Team},
  year={2024},
  url={https://github.com/your-repo/dada-pt-br}
}
```
