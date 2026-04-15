"""Machine learning ensemble engine v2 with Claude advisor."""

from .engine_v2 import MLEnsembleV2, PersistenceForecaster, ModelScoreboard
from .claude_advisor import ClaudeAdvisor
from .deep_backfill import DeepBackfill, BackfillThread

# Alias for backward compat
MLEngine = MLEnsembleV2

__all__ = ["MLEnsembleV2", "PersistenceForecaster", "ModelScoreboard",
           "ClaudeAdvisor", "MLEngine", "DeepBackfill", "BackfillThread"]
