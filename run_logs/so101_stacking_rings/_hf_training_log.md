# Training Log — pi0.5 SO101 Stacking Rings

## Mode
- run_type: replication
- objective: Verify pi0.5 fine-tuning on SO101 6D joint-space dataset with delta actions

## Config
- config: `pi05_so101_stacking_rings`
- exp_name: `so101_stacking_rings_run1`
- script: `slurm/train_so101_stacking_rings_gcloud.sh`
- dataset: [lorenzouttini/so101_stacking_rings](https://huggingface.co/datasets/lorenzouttini/so101_stacking_rings) (101 episodes, ~34k frames)
- key settings: pi0.5, action_horizon=30, batch_size=32, 50k steps, save_interval=5000, lr=2.5e-5 cosine (1k warmup), delta actions [T,T,T,T,T,F], per-timestep action norm, quantile norm, ema_decay=0.999, base weights `weights/pi05_base/params`

## Job
- vm: `openpi-so101-80g-2x` / us-central1-c
- hardware: 2x NVIDIA A100-SXM4-80GB (a2-ultragpu-2g)
- docker image: `openpi:latest`
- train log: `/home/ps/openpi/logs/pi05_so101_stacking_rings_20260602_071412.log`
- started: 2026-06-02 07:14 UTC

## Training Dynamics (through step 5,000)

| Step | Loss | Grad Norm |
|------|------|-----------|
| 0 | 0.2658 | 2.3902 |
| 1,000 | 0.0583 | 0.3301 |
| 2,000 | 0.0404 | 0.2397 |
| 3,000 | 0.0333 | 0.2198 |
| 4,000 | 0.0281 | 0.1959 |
| 5,000 | 0.0252 | 0.1817 |

- loss_one_liner: Steep drop in first 1k steps (0.27 → 0.06), then steady decline to 0.025 by 5k; stable grad norms (~0.18–0.33 after warmup).

## W&B
- local: `/workspace/repo/wandb/offline-run-20260602_071440-tmrsnyct` (VM docker)
- synced: https://wandb.ai/pravsels/so101_stacking_rings/runs/tmrsnyct (2026-06-02)

## Checkpoint Hashes

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | SHA-256 |
|------|---------|
| 5,000 | `3e772c819c5e0233b939e5f739f4de74b1ca8224e4fcf9499d59a9bf603cdb7c` |

## HF Publish
- repo: https://huggingface.co/pravsels/pi05-so101-stacking-rings
- revision: `main`
- published: `assets/`, `checkpoints/5000/params/` (params-only, no train_state)
- published: 2026-06-02 (training still running toward 50k on VM)

## Status
- 2026-06-02 07:14 UTC — Training launched
- 2026-06-02 ~10:12 UTC — Checkpoint 5000 saved (~15 min async save)
- 2026-06-02 — Boot disk resized 200G → 1TB on VM
- 2026-06-02 — Step 5000 uploaded to HF (see repo above)
