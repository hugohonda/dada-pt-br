"""
Tool system for the Agentic Safety Framework using Pydantic AI.

This module provides a type-safe tool system for agent interactions
with external systems and services.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of tool execution."""

    success: bool = Field(description="Whether the tool execution was successful")
    output: Any = Field(description="Tool output data")
    error_message: str | None = Field(None, description="Error message if execution failed")
    execution_time: float = Field(0.0, description="Execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseTool(ABC):
    """
    Base class for all agent tools.

    This class defines the interface that all tools must implement
    for type-safe tool execution.
    """

    def __init__(self, name: str, description: str, safety_level: str = "medium"):
        """
        Initialize base tool.

        Args:
            name: Tool name
            description: Tool description
            safety_level: Safety level (low, medium, high)
        """
        self.name = name
        self.description = description
        self.safety_level = safety_level
        self.usage_count = 0
        self.last_used = None
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            args: Tool arguments

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate tool arguments.

        Args:
            args: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    def get_safety_checks(self) -> list[str]:
        """Get safety checks for this tool."""
        base_checks = [
            "Verificar se os argumentos são apropriados",
            "Confirmar que o uso do tool é ético",
            "Validar que não há riscos de segurança"
        ]

        if self.safety_level == "high":
            base_checks.extend([
                "Verificar permissões de acesso",
                "Confirmar conformidade com LGPD",
                "Validar impacto na privacidade"
            ])

        return base_checks

    def _record_usage(self):
        """Record tool usage."""
        self.usage_count += 1
        self.last_used = datetime.now()


class WebSearchTool(BaseTool):
    """Web search tool for information retrieval."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Buscar informações na web",
            safety_level="medium"
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute web search."""
        start_time = datetime.now()
        self._record_usage()

        try:
            # Validate arguments
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message=error_msg,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            query = args.get("query", "")
            max_results = args.get("max_results", 5)

            # Simulate web search results
            results = self._simulate_search(query, max_results)

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                success=True,
                output=results,
                execution_time=execution_time,
                metadata={"query": query, "max_results": max_results}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time
            )

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str]:
        """Validate web search arguments."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"

        query = args.get("query")
        if not query or not isinstance(query, str):
            return False, "Query is required and must be a string"

        if len(query.strip()) < 2:
            return False, "Query must be at least 2 characters"

        return True, "Valid"

    def _simulate_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Simulate web search results."""
        # Simulate Brazilian search results
        results = []

        for i in range(min(max_results, 5)):
            results.append({
                "title": f"Resultado {i+1} para '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"Descrição relevante sobre '{query}' encontrada em português brasileiro",
                "rank": i + 1
            })

        return results


class EmailTool(BaseTool):
    """Email tool for sending messages."""

    def __init__(self):
        super().__init__(
            name="email",
            description="Enviar emails",
            safety_level="high"
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute email sending."""
        start_time = datetime.now()
        self._record_usage()

        try:
            # Validate arguments
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message=error_msg,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            # Safety check for email content
            content = args.get("content", "")
            if self._contains_harmful_content(content):
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="Email content contains potentially harmful material",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            # Simulate email sending
            result = {
                "status": "sent",
                "to": args.get("to"),
                "subject": args.get("subject"),
                "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                success=True,
                output=result,
                execution_time=execution_time,
                metadata={"to": args.get("to"), "subject": args.get("subject")}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time
            )

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str]:
        """Validate email arguments."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"

        required_fields = ["to", "subject", "content"]
        for field in required_fields:
            if field not in args or not args[field]:
                return False, f"Field '{field}' is required"

        # Validate email format (simple check)
        to_email = args.get("to", "")
        if "@" not in to_email or "." not in to_email:
            return False, "Invalid email format"

        return True, "Valid"

    def _contains_harmful_content(self, content: str) -> bool:
        """Check if content contains harmful material."""
        harmful_keywords = [
            "phishing", "scam", "fraud", "malware", "virus",
            "hack", "exploit", "bypass", "illegal", "harmful"
        ]

        content_lower = content.lower()
        return any(keyword in content_lower for keyword in harmful_keywords)


class CodeExecutionTool(BaseTool):
    """Code execution tool for running Python code."""

    def __init__(self):
        super().__init__(
            name="code_execution",
            description="Executar código Python",
            safety_level="high"
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute Python code."""
        start_time = datetime.now()
        self._record_usage()

        try:
            # Validate arguments
            is_valid, error_msg = self.validate_args(args)
            if not is_valid:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message=error_msg,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            code = args.get("code", "")

            # Safety check for dangerous code
            if self._contains_dangerous_code(code):
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="Code contains potentially dangerous operations",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            # Simulate code execution (in real implementation, use safe execution)
            result = {
                "output": "Código executado com sucesso (simulação)",
                "execution_time": "0.1s",
                "status": "completed"
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                success=True,
                output=result,
                execution_time=execution_time,
                metadata={"code_length": len(code)}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time
            )

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str]:
        """Validate code execution arguments."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"

        code = args.get("code")
        if not code or not isinstance(code, str):
            return False, "Code is required and must be a string"

        if len(code.strip()) == 0:
            return False, "Code cannot be empty"

        return True, "Valid"

    def _contains_dangerous_code(self, code: str) -> bool:
        """Check if code contains dangerous operations."""
        dangerous_patterns = [
            "import os", "import subprocess", "import sys",
            "exec(", "eval(", "__import__",
            "open(", "file(", "input(",
            "raw_input", "compile("
        ]

        code_lower = code.lower()
        return any(pattern in code_lower for pattern in dangerous_patterns)


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_info(self, name: str) -> dict[str, Any] | None:
        """
        Get tool information.

        Args:
            name: Tool name

        Returns:
            Tool information or None
        """
        tool = self.get_tool(name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "safety_level": tool.safety_level,
            "usage_count": tool.usage_count,
            "last_used": tool.last_used.isoformat() if tool.last_used else None
        }

    def get_safety_summary(self) -> dict[str, Any]:
        """
        Get safety summary of all tools.

        Returns:
            Safety summary
        """
        total_tools = len(self.tools)
        high_safety_tools = len([t for t in self.tools.values() if t.safety_level == "high"])
        medium_safety_tools = len([t for t in self.tools.values() if t.safety_level == "medium"])
        low_safety_tools = len([t for t in self.tools.values() if t.safety_level == "low"])

        return {
            "total_tools": total_tools,
            "high_safety": high_safety_tools,
            "medium_safety": medium_safety_tools,
            "low_safety": low_safety_tools,
            "tools": [self.get_tool_info(name) for name in self.list_tools()]
        }
