Next Steps (User Action Required):
Update your dataset repo_id in the config:
   # In config.py, find pi05_so101 and replace:   repo_id="your-hf-username/so101-dataset"  # Your actual HuggingFace dataset   default_prompt="your default task description"  # Your actual task
Verify your LeRobot dataset keys by running:
   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset   ds = LeRobotDataset("your-hf-username/so101-dataset")   print(ds[0].keys())
Then update the RepackTransform in LeRobotSO101DataConfig to match your actual keys.
Compute normalization stats:
   cd /home/junyuanxie/openpi   uv run scripts/compute_norm_stats.py --config-name=pi05_so101
Run training:
   uv run scripts/train.py --config-name=pi05_so101 --exp_name=so101_v1
