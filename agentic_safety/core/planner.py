"""
Task planning system for the Agentic Safety Framework.

This module provides task planning capabilities using Pydantic AI
for breaking down complex tasks into executable steps.
"""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import OpenAIModel

from ..prompts.task_prompts import TaskPrompts
from .tools import ToolRegistry


class PlanStep(BaseModel):
    """Represents a single step in a task plan."""

    passo: int = Field(description="Step number")
    descricao: str = Field(description="Step description")
    tipo: str = Field(description="Step type (raciocinio/ferramenta/verificacao)")
    ferramenta: str | None = Field(None, description="Tool name if applicable")
    argumentos: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    verificacoes_seguranca: list[str] = Field(default_factory=list, description="Safety checks")


class TaskPlan(BaseModel):
    """Complete task execution plan."""

    plano: list[PlanStep] = Field(description="List of plan steps")
    complexidade: str = Field(description="Plan complexity")
    tempo_estimado: str = Field(description="Estimated execution time")
    nivel_risco: str = Field(description="Risk level")


class TaskPlanner:
    """
    Task planner for breaking down complex tasks into executable steps.

    This planner uses Pydantic AI to create structured, safe execution plans
    for agentic tasks in Brazilian Portuguese contexts.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize task planner.

        Args:
            model_name: Model name for planning
            temperature: Model temperature
        """
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

        # Initialize Pydantic AI agent for planning
        model = OpenAIModel(model_name, temperature=temperature)
        self.agent = Agent(
            model=model,
            result_type=TaskPlan,
            retries=2
        )

    async def create_plan(
        self,
        task: str,
        tool_registry: ToolRegistry,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create a detailed execution plan for a task.

        Args:
            task: Task description
            tool_registry: Available tools
            context: Optional context information

        Returns:
            Task execution plan
        """
        self.logger.info(f"Creating plan for task: {task[:100]}...")

        try:
            # Get available tools
            available_tools = tool_registry.list_tools()

            # Create planning prompt
            prompt = TaskPrompts.get_task_planning_prompt(task, available_tools)

            # Generate plan using Pydantic AI
            result = await self.agent.run(prompt)
            plan = result.data

            # Convert to dictionary format
            plan_dict = {
                "task": task,
                "steps": [step.dict() for step in plan.plano],
                "complexity": plan.complexidade,
                "estimated_time": plan.tempo_estimado,
                "risk_level": plan.nivel_risco,
                "created_at": datetime.now().isoformat(),
                "total_steps": len(plan.plano)
            }

            self.logger.info(f"Created plan with {len(plan.plano)} steps")
            return plan_dict

        except Exception as e:
            self.logger.error(f"Plan creation failed: {e}")
            # Return a simple fallback plan
            return {
                "task": task,
                "steps": [{
                    "passo": 1,
                    "descricao": task,
                    "tipo": "raciocinio",
                    "ferramenta": None,
                    "argumentos": {},
                    "verificacoes_seguranca": ["Verificar se a tarefa é apropriada"]
                }],
                "complexity": "simples",
                "estimated_time": "5 minutos",
                "risk_level": "baixo",
                "created_at": datetime.now().isoformat(),
                "total_steps": 1,
                "error": str(e)
            }

    async def optimize_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize an existing plan.

        Args:
            plan: Plan to optimize

        Returns:
            Optimized plan
        """
        self.logger.info("Optimizing plan...")

        try:
            # Add dependencies between steps
            optimized_steps = []

            for i, step in enumerate(plan.get("steps", [])):
                # Add dependencies based on step type
                dependencies = []

                if step.get("tipo") == "verificacao" and i > 0:
                    dependencies.append(i)  # Verification depends on previous step

                step["dependencias"] = dependencies
                optimized_steps.append(step)

            plan["steps"] = optimized_steps
            plan["optimized_at"] = datetime.now().isoformat()

            return plan

        except Exception as e:
            self.logger.error(f"Plan optimization failed: {e}")
            return plan

    def validate_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a task plan.

        Args:
            plan: Plan to validate

        Returns:
            Validation result
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        steps = plan.get("steps", [])

        if not steps:
            validation_result["valid"] = False
            validation_result["errors"].append("Plan has no steps")
            return validation_result

        # Validate each step
        for i, step in enumerate(steps):
            step_errors = []

            # Check required fields
            if not step.get("descricao"):
                step_errors.append(f"Step {i+1} missing description")

            if not step.get("tipo"):
                step_errors.append(f"Step {i+1} missing type")

            # Check tool usage
            if step.get("tipo") == "ferramenta" and not step.get("ferramenta"):
                step_errors.append(f"Step {i+1} is tool type but missing tool name")

            # Check safety checks
            if not step.get("verificacoes_seguranca"):
                validation_result["warnings"].append(f"Step {i+1} has no safety checks")

            if step_errors:
                validation_result["errors"].extend(step_errors)
                validation_result["valid"] = False

        # Check plan structure
        if plan.get("nivel_risco") == "alto" and len(steps) > 10:
            validation_result["warnings"].append("High-risk plan with many steps may be complex")

        return validation_result

    def get_plan_summary(self, plan: dict[str, Any]) -> dict[str, Any]:
        """
        Get a summary of a task plan.

        Args:
            plan: Plan to summarize

        Returns:
            Plan summary
        """
        steps = plan.get("steps", [])

        # Count step types
        step_types = {}
        for step in steps:
            step_type = step.get("tipo", "unknown")
            step_types[step_type] = step_types.get(step_type, 0) + 1

        # Count tool usage
        tool_usage = {}
        for step in steps:
            if step.get("tipo") == "ferramenta":
                tool_name = step.get("ferramenta")
                if tool_name:
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return {
            "total_steps": len(steps),
            "step_types": step_types,
            "tool_usage": tool_usage,
            "complexity": plan.get("complexity", "unknown"),
            "risk_level": plan.get("risk_level", "unknown"),
            "estimated_time": plan.get("estimated_time", "unknown"),
            "created_at": plan.get("created_at"),
            "has_safety_checks": any(
                step.get("verificacoes_seguranca")
                for step in steps
            )
        }
