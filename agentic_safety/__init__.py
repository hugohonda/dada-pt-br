"""
Agentic Safety Research Framework using Pydantic AI.

This module provides a modern, type-safe framework for evaluating agentic
AI safety in Brazilian Portuguese contexts using Pydantic AI and Context7.
"""

from .alignment.constitution import BrazilianConstitution
from .benchmarks.agentharm_pt import AgentHarmPTBenchmark
from .core.agent import SafetyAgent
from .core.tools import ToolRegistry
from .evaluation.safety_evaluator import SafetyEvaluator

__version__ = "0.1.0"
__all__ = [
    "SafetyAgent",
    "ToolRegistry",
    "SafetyEvaluator",
    "AgentHarmPTBenchmark",
    "BrazilianConstitution",
]
