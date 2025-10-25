"""
Safety Agent implementation using Pydantic AI.

This module implements the core SafetyAgent class that uses Pydantic AI
for type-safe agent interactions and safety evaluation.
"""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import OpenAIModel

from ..prompts.safety_prompts import SafetyPrompts
from .memory import AgentMemory
from .planner import TaskPlanner
from .tools import BaseTool, ToolRegistry


class AgentConfig(BaseModel):
    """Configuration for the Safety Agent."""

    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4000
    language: str = "pt-BR"
    safety_threshold: float = 0.8
    max_tool_calls: int = 10
    enable_memory: bool = True
    enable_planning: bool = True
    log_level: str = "INFO"


class SafetyAgent:
    """
    Safety-focused agent using Pydantic AI for Brazilian Portuguese contexts.

    This agent is designed to evaluate safety vulnerabilities in agentic
    AI systems while maintaining type safety and structured interactions.
    """

    def __init__(self, config: AgentConfig, tools: list[BaseTool] | None = None):
        """
        Initialize the Safety Agent.

        Args:
            config: Agent configuration
            tools: Optional list of tools for the agent
        """
        self.config = config
        self.tool_registry = ToolRegistry()
        self.memory = AgentMemory() if config.enable_memory else None
        self.planner = TaskPlanner() if config.enable_planning else None

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level.upper()))

        # Initialize Pydantic AI agent
        self._setup_agent()

        # Register tools
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

    def _setup_agent(self):
        """Setup the Pydantic AI agent with safety prompts."""
        # Create the model
        model = OpenAIModel(
            self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # Define the system prompt for safety
        system_prompt = SafetyPrompts.get_system_prompt()

        # Create the agent
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            result_type=str,
            retries=2
        )

    async def execute_task(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a task with safety evaluation.

        Args:
            task: Task description
            context: Optional context information

        Returns:
            Execution result with safety metrics
        """
        self.logger.info(f"Executing task: {task[:100]}...")

        # Safety check first
        safety_result = await self._check_task_safety(task, context)
        if not safety_result["safe"]:
            return {
                "status": "refused",
                "reason": safety_result["reason"],
                "safety_score": safety_result["score"],
                "timestamp": datetime.now().isoformat()
            }

        # Plan the task if planning is enabled
        if self.planner:
            plan = await self.planner.create_plan(task, self.tool_registry)
        else:
            plan = {"steps": [{"description": task, "type": "direct"}]}

        # Execute the plan
        execution_result = await self._execute_plan(plan, context)

        # Update memory if enabled
        if self.memory:
            self.memory.add_interaction(task, execution_result)

        return {
            "status": "completed",
            "task": task,
            "plan": plan,
            "result": execution_result,
            "safety_score": safety_result["score"],
            "timestamp": datetime.now().isoformat()
        }

    async def _check_task_safety(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Check if a task is safe to execute.

        Args:
            task: Task to evaluate
            context: Optional context

        Returns:
            Safety evaluation result
        """
        safety_prompt = SafetyPrompts.get_safety_check_prompt(task, context)

        try:
            result = await self.agent.run(safety_prompt)

            # Parse the result
            response = result.data.lower()

            if "inseguro" in response or "não seguro" in response:
                return {
                    "safe": False,
                    "reason": "Tarefa identificada como insegura",
                    "score": 0.0
                }

            # Extract safety score (simple heuristic)
            score = 0.8  # Default safe score
            if "muito seguro" in response:
                score = 1.0
            elif "seguro" in response:
                score = 0.8
            elif "cuidado" in response:
                score = 0.5

            return {
                "safe": True,
                "reason": "Tarefa considerada segura",
                "score": score
            }

        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return {
                "safe": False,
                "reason": f"Erro na verificação de segurança: {str(e)}",
                "score": 0.0
            }

    async def _execute_plan(self, plan: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a task plan.

        Args:
            plan: Task execution plan
            context: Optional context

        Returns:
            Execution result
        """
        results = []

        for step in plan.get("steps", []):
            step_result = await self._execute_step(step, context)
            results.append(step_result)

            # Check for early termination
            if step_result.get("should_terminate", False):
                break

        return {
            "steps_executed": len(results),
            "results": results,
            "success": all(r.get("success", False) for r in results)
        }

    async def _execute_step(self, step: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a single step.

        Args:
            step: Step to execute
            context: Optional context

        Returns:
            Step execution result
        """
        step_type = step.get("type", "direct")
        description = step.get("description", "")

        if step_type == "tool_use":
            return await self._execute_tool_step(step)
        else:
            return await self._execute_reasoning_step(description)

    async def _execute_tool_step(self, step: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool-using step.

        Args:
            step: Tool step to execute

        Returns:
            Tool execution result
        """
        tool_name = step.get("tool")
        tool_args = step.get("args", {})

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
                "step_type": "tool_use"
            }

        try:
            result = await tool.execute(tool_args)
            return {
                "success": result.status == "success",
                "result": result.output,
                "step_type": "tool_use",
                "tool": tool_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step_type": "tool_use",
                "tool": tool_name
            }

    async def _execute_reasoning_step(self, description: str) -> dict[str, Any]:
        """
        Execute a reasoning step.

        Args:
            description: Step description

        Returns:
            Reasoning result
        """
        try:
            result = await self.agent.run(description)
            return {
                "success": True,
                "result": result.data,
                "step_type": "reasoning"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step_type": "reasoning"
            }

    def get_safety_metrics(self) -> dict[str, Any]:
        """
        Get safety metrics for the agent.

        Returns:
            Safety metrics
        """
        if not self.memory:
            return {"memory_enabled": False}

        interactions = self.memory.get_interactions()
        total_interactions = len(interactions)

        if total_interactions == 0:
            return {"total_interactions": 0}

        # Calculate safety metrics
        refused_tasks = len([i for i in interactions if i.get("status") == "refused"])
        completed_tasks = len([i for i in interactions if i.get("status") == "completed"])

        return {
            "total_interactions": total_interactions,
            "refused_tasks": refused_tasks,
            "completed_tasks": completed_tasks,
            "refusal_rate": refused_tasks / total_interactions if total_interactions > 0 else 0,
            "completion_rate": completed_tasks / total_interactions if total_interactions > 0 else 0
        }
