"""InferenceInterceptor: cache-aware drop-in replacement for Policy.

Overview
--------
``InferenceInterceptor`` wraps an existing ``Policy`` object and implements
the same ``BasePolicy`` interface, making it a transparent substitute for the
WebSocket server.  When ``--cache`` is passed to ``serve_policy.py``, the
server receives this interceptor instead of the raw ``Policy`` with zero
changes to ``WebsocketPolicyServer`` or any client code.

Current state (Step 2)
----------------------
The interceptor routes inference through the *staged* public API
(``run_stage1`` / ``run_stage2`` / ``run_stage3``) introduced in Step 1, and
times each stage using ``SystemTimer`` with CUDA Event backends.  No caching
logic is applied yet — the output is numerically identical to ``Policy.infer``.
Caching logic (``CacheOrchestrator``) will be injected at the TODO markers in
Step 4.

The manual ``time.monotonic()`` / ``torch.cuda.synchronize()`` timing from
Step 1 has been removed.  All timing now flows through ``SystemTimer``:

* Stage 1 (vision + token prep)  → probe ``"stage1_vision"`` (CUDA backend)
* Stage 2 (LLM backbone)         → probe ``"stage2_llm"``    (CUDA backend)
* Stage 3 (flow matching)        → probe ``"stage3_flow"``   (CUDA backend)
* End-to-end wall time           → probe ``"total_inference"`` (CPU backend)

Task lifecycle
--------------
``InferenceInterceptor`` implements the ``TaskLifecycle`` protocol.
``WebsocketPolicyServer._handler`` calls ``on_task_begin()`` / ``on_task_end()``
when a client connection opens / closes.  ``on_task_end()`` triggers
``SystemTimer.on_task_end()``, which prints a per-probe summary to the
terminal and (optionally) writes a CSV.

External contract (what the client / server sees)
-------------------------------------------------
``infer(obs)`` returns::

    {
        "actions": np.ndarray  [action_horizon, action_dim],
        "state":   np.ndarray  [...],
        "server_timing" is added by the server, not here,
    }

The ``stage_timing`` and ``policy_timing`` keys present in the Step 1 version
have been removed.  Timing output is now handled entirely by ``SystemTimer``
(printed at task end, optionally written to CSV).

Limitations
-----------
* Only PyTorch policies are supported.  JAX policies do not expose
  ``run_stage1 / run_stage2 / run_stage3``.
* ``SystemTimer`` is created with default settings.  To customise
  ``buffer_size`` or ``output_csv_dir``, pass a pre-configured
  ``SystemTimer`` instance via the ``timer`` argument.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.cache.timing import SystemTimer, TaskLifecycle
from openpi.models import model as _model
from openpi.policies import policy as _policy


class InferenceInterceptor(_base_policy.BasePolicy):
    """Drop-in Policy replacement that routes inference through the staged API.

    Wraps a ``Policy`` instance.  All input/output transforms are reused from
    the wrapped policy so normalisation, tokenisation, and action remapping are
    identical to the original path.

    This class implements ``TaskLifecycle`` so that ``WebsocketPolicyServer``
    can call ``on_task_begin`` / ``on_task_end`` without depending on this
    class directly (the server checks ``hasattr(policy, "on_task_begin")``).

    Args:
        policy: A fully initialised ``Policy`` object with
                ``is_pytorch=True``.  JAX policies are not supported.
        timer: Optional pre-configured ``SystemTimer``.  When ``None``
               (default), a timer with default settings is created
               (``enabled=True``, ``buffer_size=10_000``, no CSV output).
               Pass a custom instance to enable CSV export or adjust the
               buffer size::

                   timer = SystemTimer(enabled=True, output_csv_dir="/tmp/t")
                   interceptor = InferenceInterceptor(policy, timer=timer)

    Raises:
        ValueError: If the wrapped policy is not a PyTorch policy.
    """

    def __init__(
        self,
        policy: _policy.Policy,
        timer: Optional[SystemTimer] = None,
    ) -> None:
        if not policy._is_pytorch_model:  # noqa: SLF001
            raise ValueError(
                "InferenceInterceptor only supports PyTorch policies. "
                "The wrapped policy must be initialised with is_pytorch=True."
            )

        self._policy = policy
        # Borrow internals from the wrapped Policy — references only, no copy.
        self._model = policy._model                        # PI0Pytorch instance  # noqa: SLF001
        self._input_transform = policy._input_transform    # composed transform fn  # noqa: SLF001
        self._output_transform = policy._output_transform  # composed transform fn  # noqa: SLF001
        self._pytorch_device = policy._pytorch_device      # e.g. "cuda:0"          # noqa: SLF001

        # Set up SystemTimer.
        # Each probe corresponds to one pipeline component.  The backend
        # ("cuda" vs "cpu") determines the timing method:
        #   "cuda"  → torch.cuda.Event (accurate GPU execution time)
        #   "cpu"   → time.perf_counter_ns (wall time, for CPU-only ops)
        #
        # When Step 4 adds cache checkpoints (CP1 / CP2 / CP3), their probes
        # are registered here as well (using "cpu" or "cuda" as appropriate
        # for the checkpoint's dominant compute location).
        self._timer: SystemTimer = timer if timer is not None else SystemTimer()
        self._timer.register_probe("stage1_vision", backend="cuda")
        self._timer.register_probe("stage2_llm",    backend="cuda")
        self._timer.register_probe("stage3_flow",   backend="cuda")
        # CPU wall time wrapping all three stages; captures Python + sync
        # overhead in addition to pure GPU time.  Useful as the "felt" latency
        # from the robot's perspective.
        self._timer.register_probe("total_inference", backend="cpu")

    # -----------------------------------------------------------------------
    # TaskLifecycle interface  (called by WebsocketPolicyServer._handler)
    # -----------------------------------------------------------------------

    def on_task_begin(self) -> None:
        """Reset per-task state.  Called when a client connection opens.

        Forwards to ``SystemTimer.on_task_begin()``, which records the
        current record count as the task boundary.
        """
        self._timer.on_task_begin()

    def on_task_end(self) -> None:
        """Finalise and report timing for the completed task.

        Forwards to ``SystemTimer.on_task_end()``, which:
        1. Prints a per-probe summary table to the terminal.
        2. Writes a CSV file if ``output_csv_dir`` was configured.

        Called when a client WebSocket connection closes.
        """
        self._timer.on_task_end()

    # -----------------------------------------------------------------------
    # BasePolicy interface
    # -----------------------------------------------------------------------

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Cache-aware inference.

        Currently a pass-through (Step 2): calls run_stage1 → run_stage2 →
        run_stage3 sequentially and returns the same action dict as
        ``Policy.infer``.

        Timing is measured by ``SystemTimer``:
        * ``total_inference``: CPU wall time for the entire staged inference.
        * ``stage1_vision``, ``stage2_llm``, ``stage3_flow``: per-stage GPU
          execution time via CUDA Events.

        The ``stage_timing`` key is no longer included in the returned dict.
        Timing summaries are printed by ``SystemTimer.on_task_end()`` at the
        end of each connection.

        Args:
            obs: Observation dict in the format expected by the wrapped policy's
                 input transform.  Must contain at least the keys defined by
                 the robot-specific transform (e.g. ``AlohaInputs``).
            noise: Optional initial noise tensor for flow matching.  If given,
                   must have shape ``[action_horizon, action_dim]`` or
                   ``[1, action_horizon, action_dim]``.

        Returns:
            Dict with keys ``"actions"`` and ``"state"``.
        """
        # ---- 1. Input transforms (mirrors Policy.infer exactly) ----
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x))
                           .to(self._pytorch_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)

        # Optional noise forwarding (mirrors Policy.infer sample_kwargs).
        start_noise: torch.Tensor | None = None
        if noise is not None:
            start_noise = torch.from_numpy(noise).to(self._pytorch_device)
            if start_noise.ndim == 2:
                start_noise = start_noise[None, ...]

        # ---- 2. Staged inference with SystemTimer ----
        # total_inference uses a CPU (perf_counter) backend and wraps all
        # three stages.  It captures end-to-end wall time including Python
        # overhead and the per-stage CUDA event synchronizations.
        with self._timer.measure("total_inference"):
            with torch.no_grad():
                # Stage 1: SigLIP vision encoding + prefix embedding.
                # CUDA Event records GPU execution time on the default stream.
                with self._timer.measure("stage1_vision"):
                    stage1 = self._model.run_stage1(observation)

                # Stage 2: Gemma 2B backbone forward pass → KV cache.
                # CUDA Event records GPU execution time on the default stream.
                # TODO(Step 4): insert CP1 cache check between stage1 and stage2.
                with self._timer.measure("stage2_llm"):
                    stage2 = self._model.run_stage2(stage1)

                # Stage 3: Action Expert — 10-step Euler flow-matching loop.
                # CUDA Event records GPU execution time on the default stream.
                # TODO(Step 4): insert CP2 cache check between stage2 and stage3.
                with self._timer.measure("stage3_flow"):
                    stage3 = self._model.run_stage3(stage2, noise=start_noise)

                # TODO(Step 4): insert CP3 predictive cache check after stage3.

        # ---- 3. Build outputs ----
        # Output format matches Policy.infer so the server and client require
        # no changes.  stage_timing / policy_timing are intentionally omitted;
        # timing is reported by SystemTimer at task end instead.
        outputs: dict[str, Any] = {
            "state":   inputs["state"],
            "actions": stage3.action_chunk,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
        )
        outputs = self._output_transform(outputs)
        return outputs

    # -----------------------------------------------------------------------
    # BasePolicy metadata property
    # -----------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Forward metadata to the wrapped policy (robot type, action shape, etc.)."""
        return self._policy.metadata
