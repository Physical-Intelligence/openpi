"""Print the per-frame keys of allenai/24112025-yam-01.

The dataset is published in lerobot v3.0 format, which the lerobot revision
pinned by openpi can't load. We read the metadata + a single parquet row
directly via huggingface_hub instead.
"""

import json

import polars as pl
from huggingface_hub import hf_hub_download

REPO_ID = "allenai/24112025-yam-01"

info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset")
with open(info_path) as f:
    info = json.load(f)

print("codebase_version:", info["codebase_version"])
print("feature keys:", list(info["features"].keys()))

parquet_path = hf_hub_download(
    REPO_ID, "data/chunk-000/file-000.parquet", repo_type="dataset"
)
first_frame = pl.read_parquet(parquet_path, n_rows=1).row(0, named=True)
print("first frame keys:", list(first_frame.keys()))

"""
feature keys: [
'action', 
'observation.state', 
'observation.images.left', 
'observation.images.right', 
'observation.images.top', 
'timestamp', 
'frame_index', 
'episode_index', 
'index', 
'task_index']

first frame keys: [
'action', 
'observation.state', 
'timestamp', 
'frame_index', 
'episode_index', 
'index', 
'task_index']


  - observation.images.{left,right,top} — float32 tensors, shape (3, 360, 640) (CHW)
  - observation.state — float32(14,)
  - action — float32(14,)
  - timestamp, frame_index, episode_index, index, task_index — scalars
  - task — string
"""