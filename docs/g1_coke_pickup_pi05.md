# G1 Coke pickup on pi0.5

This configuration keeps the pretrained pi0.5 tensor width at 32 while exposing the exact Coke task contract:

- `head_image`: rendered Isaac or real G1 RGB, HWC `uint8`
- `state`: 24 absolute upper-body joint positions
- `actions`: 10-step chunks of seven normalized right-hand pose/grip commands
- `prompt`: `pick up the Coke can and hold it upright`

The first six outputs are relative right-hand pose commands. The seventh is binary Dex3 grip intent. pi0.5 does not directly command G1 joints, locomotion, or balance.

## Convert successful Isaac demonstrations

```bash
python3 scripts/spark/convert_g1_coke_sim_demos.py \
  --raw-dir /data/g1-coke-pi05-raw \
  --repo-id YOUR_ORG/g1_coke_pickup \
  --push-to-hub
```

Conversion verifies the manifest format, every shard SHA-256, 50 Hz timestamps, RGB/state/action shapes and dtypes, finite values, normalized action range, and successful episode flags. It fails closed on unsuccessful demonstrations unless `--allow-failures` is explicitly supplied for debugging.

## Fine-tune on Spark

```bash
export OPENPI_G1_COKE_DATASET_REPO_ID=YOUR_ORG/g1_coke_pickup
scripts/spark/train_g1_coke.sh
```

The launcher computes dataset-specific quantile normalization statistics once, then trains `pi05_spark_g1_coke_pickup` in bfloat16 from `/openpi_assets/checkpoints/pi05_base_pytorch`.

## Serve the trained policy

```bash
python3 scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_spark_g1_coke_pickup \
  --policy.dir /openpi_assets/training/pi05_spark_g1_coke_pickup/EXPERIMENT/STEP
```

Use the Coke repository's `scripts/run_pi05_real.py` as the client. It is shadow-only unless explicitly enabled and leaves IK, joint limits, collision avoidance, controller ownership, and the hardware watchdog outside the VLA model.
