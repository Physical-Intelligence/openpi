# Sampling-equivalence baselines

These `.npz` files are the regression baseline for folding the four sampling entrypoints
(`sample_trace`, `sample_actions`, `sample_actions_and_completion`, `predict_completion`)
out of the per-variant model classes into `TraceVLABase`. They were produced by
`scripts/sampling_equivalence_test.py` from the three trace LoRA smoke checkpoints.

`*_preC.npz` = original per-variant sampling code; `*_postC.npz` = merged `TraceVLABase` code.
Both were captured **on CPU** (`JAX_PLATFORMS=cpu`), which is bit-deterministic — so an
equivalence-preserving merge yields `preC == postC` exactly (it does: worst max|Δ| = 0 across
all configs/skills/outputs). Captured with `--max-skills 3 --num-steps 10`.

Why CPU: on GPU the multi-step bf16 samplers are only reproducible to ~1e-2 across separate
runs/compilations (different HLO → different reduction order), which is too noisy for an
exact check. CPU removes that, giving a clean bit-for-bit verdict.

To re-verify after a future change to the sampling code:

    python scripts/sampling_equivalence_test.py <config> --out /tmp/new.npz \
        --max-skills 3 --num-steps 10            # run on a compute node, JAX_PLATFORMS=cpu
    python scripts/sampling_equivalence_test.py --compare \
        refactor_baselines/sampling/eq_<config>_postC.npz /tmp/new.npz --tol 1e-4
