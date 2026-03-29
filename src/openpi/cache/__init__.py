# Cache system for the Pi0/Pi0.5 inference pipeline.
# See docs/cache_system_architecture.md for the full design and roadmap.
#
# Public API exported from this package:
#
#   SystemTimer    — pluggable, hardware-aware timing system for all pipeline
#                    components (inference stages, cache checkpoints, etc.)
#
#   TaskLifecycle  — protocol implemented by InferenceInterceptor; allows
#                    WebsocketPolicyServer to call on_task_begin/on_task_end
#                    without importing InferenceInterceptor directly.
#
# Step 3+ will add: CacheEntry, CacheResult, VectorStore, CacheOrchestrator.

from openpi.cache.timing import SystemTimer, TaskLifecycle

__all__ = ["SystemTimer", "TaskLifecycle"]
