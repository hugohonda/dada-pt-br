"""
Memory system for the Agentic Safety Framework.

This module provides memory management capabilities for agents to
maintain context and learn from interactions.
"""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Interaction(BaseModel):
    """Represents a single interaction in agent memory."""

    timestamp: datetime = Field(default_factory=datetime.now)
    task: str = Field(description="Task description")
    result: dict[str, Any] = Field(description="Task result")
    status: str = Field(description="Task status (completed, refused, failed)")
    safety_score: float = Field(description="Safety score for this interaction")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentMemory(BaseModel):
    """Memory system for agent interactions."""

    interactions: list[Interaction] = Field(default_factory=list)
    max_size: int = Field(default=1000, description="Maximum number of interactions to store")
    decay_factor: float = Field(default=0.95, description="Memory decay factor")

    def add_interaction(
        self,
        task: str,
        result: dict[str, Any],
        status: str = "completed",
        safety_score: float = 1.0,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Add a new interaction to memory.

        Args:
            task: Task description
            result: Task result
            status: Task status
            safety_score: Safety score
            metadata: Additional metadata
        """
        interaction = Interaction(
            task=task,
            result=result,
            status=status,
            safety_score=safety_score,
            metadata=metadata or {}
        )

        self.interactions.append(interaction)

        # Apply decay to existing interactions
        self._apply_decay()

        # Trim memory if necessary
        if len(self.interactions) > self.max_size:
            self._trim_memory()

    def _apply_decay(self) -> None:
        """Apply decay factor to existing interactions."""
        now = datetime.now()

        for interaction in self.interactions:
            # Calculate age in hours
            age_hours = (now - interaction.timestamp).total_seconds() / 3600

            # Apply exponential decay to safety score
            decay_multiplier = self.decay_factor ** age_hours
            interaction.safety_score *= decay_multiplier

    def _trim_memory(self) -> None:
        """Remove oldest interactions to stay within memory limit."""
        # Sort by timestamp (oldest first)
        self.interactions.sort(key=lambda x: x.timestamp)

        # Remove oldest interactions
        excess = len(self.interactions) - self.max_size
        self.interactions = self.interactions[excess:]

    def get_interactions(self) -> list[dict[str, Any]]:
        """
        Get all interactions.

        Returns:
            List of interaction dictionaries
        """
        return [interaction.dict() for interaction in self.interactions]

    def get_recent_interactions(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent interactions.

        Args:
            limit: Maximum number of interactions to return

        Returns:
            List of recent interaction dictionaries
        """
        recent = sorted(self.interactions, key=lambda x: x.timestamp, reverse=True)
        return [interaction.dict() for interaction in recent[:limit]]

    def get_safety_events(self) -> list[dict[str, Any]]:
        """
        Get safety-related events.

        Returns:
            List of safety event dictionaries
        """
        safety_events = []

        for interaction in self.interactions:
            if interaction.status == "refused" or interaction.safety_score < 0.5:
                safety_events.append({
                    "timestamp": interaction.timestamp.isoformat(),
                    "task": interaction.task,
                    "status": interaction.status,
                    "safety_score": interaction.safety_score,
                    "reason": interaction.result.get("reason", "Unknown")
                })

        return safety_events

    def get_failure_patterns(self) -> dict[str, Any]:
        """
        Analyze failure patterns in memory.

        Returns:
            Dictionary containing failure analysis
        """
        failed_interactions = [i for i in self.interactions if i.status == "failed"]

        if not failed_interactions:
            return {"total_failures": 0, "patterns": []}

        # Analyze failure types
        failure_types = {}
        for interaction in failed_interactions:
            error_type = interaction.result.get("error", "unknown")
            failure_types[error_type] = failure_types.get(error_type, 0) + 1

        # Calculate failure rate
        total_interactions = len(self.interactions)
        failure_rate = len(failed_interactions) / total_interactions if total_interactions > 0 else 0

        return {
            "total_failures": len(failed_interactions),
            "failure_rate": failure_rate,
            "failure_types": failure_types,
            "recent_failures": [i.dict() for i in failed_interactions[-5:]]
        }

    def get_safety_metrics(self) -> dict[str, Any]:
        """
        Get safety metrics from memory.

        Returns:
            Dictionary containing safety metrics
        """
        if not self.interactions:
            return {"total_interactions": 0}

        total_interactions = len(self.interactions)
        refused_interactions = len([i for i in self.interactions if i.status == "refused"])
        completed_interactions = len([i for i in self.interactions if i.status == "completed"])

        # Calculate average safety score
        avg_safety_score = sum(i.safety_score for i in self.interactions) / total_interactions

        # Calculate refusal rate
        refusal_rate = refused_interactions / total_interactions if total_interactions > 0 else 0

        return {
            "total_interactions": total_interactions,
            "refused_interactions": refused_interactions,
            "completed_interactions": completed_interactions,
            "refusal_rate": refusal_rate,
            "average_safety_score": avg_safety_score,
            "safety_events": self.get_safety_events(),
            "failure_patterns": self.get_failure_patterns()
        }

    def clear(self) -> None:
        """Clear all interactions from memory."""
        self.interactions.clear()

    def export_to_file(self, filepath: str) -> None:
        """
        Export memory to file.

        Args:
            filepath: Path to export file
        """
        memory_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_interactions": len(self.interactions),
                "max_size": self.max_size,
                "decay_factor": self.decay_factor
            },
            "interactions": [interaction.dict() for interaction in self.interactions],
            "safety_metrics": self.get_safety_metrics()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False, default=str)

    def load_from_file(self, filepath: str) -> None:
        """
        Load memory from file.

        Args:
            filepath: Path to load file from
        """
        with open(filepath, encoding='utf-8') as f:
            memory_data = json.load(f)

        # Load interactions
        self.interactions = []
        for interaction_data in memory_data.get("interactions", []):
            # Convert timestamp string back to datetime
            if isinstance(interaction_data.get("timestamp"), str):
                interaction_data["timestamp"] = datetime.fromisoformat(interaction_data["timestamp"])

            self.interactions.append(Interaction(**interaction_data))

        # Load metadata
        metadata = memory_data.get("metadata", {})
        self.max_size = metadata.get("max_size", 1000)
        self.decay_factor = metadata.get("decay_factor", 0.95)
