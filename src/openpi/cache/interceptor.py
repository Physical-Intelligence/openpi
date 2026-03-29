"""InferenceInterceptor: cache-aware drop-in replacement for Policy.

Wraps an existing Policy object and implements the same BasePolicy interface.
When cache=True is passed to serve_policy.py, the server receives this
interceptor instead of the raw Policy, with zero changes to the server itself.

Current state (Step 1): pass-through only — calls run_stage1/2/3 sequentially
with no caching logic.  Produces bit-identical results to Policy.infer().
Caching logic (CacheOrchestrator) will be injected in Step 4.

External contract (what the client/server sees):
  - infer(obs) -> dict with keys: actions, state, policy_timing, stage_timing
  - metadata property -> same as wrapped Policy
  - stage_timing keys: token_prep_ms, llm_backbone_ms, action_expert_ms, total_ms
"""

import time
from typing import Any

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.models import model as _model
from openpi.policies import policy as _policy


class InferenceInterceptor(_base_policy.BasePolicy):
    """Drop-in replacement for Policy that routes inference through the staged API.

    Wraps a Policy instance.  All input/output transforms are reused from the
    wrapped policy, so normalization, tokenization, and action remapping are
    identical to the original path.

    Args:
        policy: A fully initialised Policy object (PyTorch, is_pytorch=True).
                JAX policies are not supported by the staged API.

    Raises:
        ValueError: If the wrapped policy is not a PyTorch policy.
    """

    def __init__(self, policy: _policy.Policy) -> None:
        if not policy._is_pytorch_model:  # noqa: SLF001
            raise ValueError("InferenceInterceptor only supports PyTorch policies.")

        self._policy = policy
        # Borrow internals from the wrapped Policy — no copying, just references.
        self._model = policy._model                        # PI0Pytorch instance  # noqa: SLF001
        self._input_transform = policy._input_transform    # composed transform fn  # noqa: SLF001
        self._output_transform = policy._output_transform  # composed transform fn  # noqa: SLF001
        self._pytorch_device = policy._pytorch_device      # e.g. "cuda:0"  # noqa: SLF001

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Cache-aware inference.  Currently a pass-through (Step 1).

        Produces output identical to Policy.infer() in the staged path:
          {
            "actions":       np.ndarray [action_horizon, action_dim],
            "state":         np.ndarray [...],
            "policy_timing": {"infer_ms": float},
            "stage_timing":  {"token_prep_ms", "llm_backbone_ms",
                              "action_expert_ms", "total_ms"},
          }
        """
        # ---- 1. Input transforms (identical to Policy.infer) ----
        inputs = jax.tree.map(lambda x: x, obs)          # shallow copy
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x))
                           .to(self._pytorch_device)[None, ...],
            inputs,
        )

        observation = _model.Observation.from_dict(inputs)

        # Optional noise forwarding (mirrors Policy.infer sample_kwargs handling)
        start_noise: torch.Tensor | None = None
        if noise is not None:
            start_noise = torch.from_numpy(noise).to(self._pytorch_device)
            if start_noise.ndim == 2:
                start_noise = start_noise[None, ...]

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ---- 2. Staged inference through run_stage1/2/3 ----
        t_total = time.monotonic()

        with torch.no_grad():
            t0 = time.monotonic()
            stage1 = self._model.run_stage1(observation)
            _sync()
            token_prep_ms = (time.monotonic() - t0) * 1000

            t1 = time.monotonic()
            stage2 = self._model.run_stage2(stage1)
            _sync()
            llm_backbone_ms = (time.monotonic() - t1) * 1000

            t2 = time.monotonic()
            # TODO(Step 4): insert CP1 check after stage1, CP2 check after stage2.
            stage3 = self._model.run_stage3(stage2, noise=start_noise)
            _sync()
            action_expert_ms = (time.monotonic() - t2) * 1000

        total_ms = (time.monotonic() - t_total) * 1000

        # ---- 3. Build outputs (identical format to Policy.infer staged path) ----
        outputs: dict[str, Any] = {
            "state": inputs["state"],
            "actions": stage3.action_chunk,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
        )
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": total_ms}
        outputs["stage_timing"] = {
            "token_prep_ms": token_prep_ms,
            "llm_backbone_ms": llm_backbone_ms,
            "action_expert_ms": action_expert_ms,
            "total_ms": total_ms,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
