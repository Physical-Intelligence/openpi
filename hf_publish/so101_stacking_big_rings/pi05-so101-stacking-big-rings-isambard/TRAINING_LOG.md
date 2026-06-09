# Training Log — pi0.5 SO101 Stacking Big Rings (Isambard)

## Summary

- config: `pi05_so101_stacking_big_rings`
- exp_name: `so101_stacking_big_rings`
- objective: Fine-tune pi0.5 on SO101 6D joint-space big-rings stacking dataset with delta actions
- dataset: [lorenzouttini/so101_stacking_big_rings](https://huggingface.co/datasets/lorenzouttini/so101_stacking_big_rings)
- key settings: pi0.5, action_horizon=30, batch_size=32, 50k steps, save_interval=5000, lr=2.5e-5 cosine (1k warmup), delta actions [T,T,T,T,T,F], per-timestep action norm, quantile norm, ema_decay=0.999, base weights `pi05_base`
- hardware: Isambard GH200 (4× GPU node, arm64)
- container: `pravsels/openpi-isambard` — `openpi_arm64.sif`
- wandb_project: `so101_stacking_big_rings` (offline; to be synced after training)

## Training Dynamics

| Step | Loss | Grad Norm |
|------|------|-----------|
| _(to be filled)_ | | |

## Uploaded Checkpoints

_(to be filled as checkpoints are published)_

## Notes

- Sanitized for Hugging Face publication.
- Cluster-specific job IDs, node names, and scratch paths omitted.
