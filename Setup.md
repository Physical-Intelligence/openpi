
## Compute Canada uv sync error
The issue is caused by `evdev` package.
Workaround was 
- disable lerobot in the `pyproject.toml` of openpi
- git clone lerobot with specific commit
- disable `pynput` in the `pyproject.toml` of lerobot
- install lerobot
- install openpi


## download partial behavior1k

Uses `scripts/download_behavior_tasks.py` with git sparse-checkout to download specific tasks only (avoids 1.5TB full dataset download and HuggingFace API rate limits).

Downloads to: `~/.cache/huggingface/datasets/behavior-1k/2025-challenge-demos`

```bash
# Downloads turning_on_radio and picking_up_trash by default
python scripts/download_behavior_tasks.py

# Or specify custom tasks
python scripts/download_behavior_tasks.py --tasks picking_up_trash turning_on_radio
```

This requires git lfs, so need to install `brew install git-lfs`. ComputeCanada doesn't seem like having lfs, so need to download locally and transfer by rsync would be a solution.


## lerobot patches

There could be an issue to load behavior dataset using lerobot_dataset, if that's the case, refer following patches for fix. It worked for my experiments but might be better to look into the details for specific situations.

```bash
./debug/apply_lerobot_patches.sh
```

### Patch Details

The lerobot package (commit `577cd10974b84bea1f06b6472eb9e5e74e07f77a`) requires 4 patches for BEHAVIOR-1K compatibility:

#### 1. `lerobot/datasets/lerobot_dataset.py` (3 patches)

**Patch 1: Prevent HuggingFace 429 errors** (lines 127, 582)
```python
# Commented out force_download to avoid rate limiting:
# force_download=True,  # Removed to avoid 429 errors - use only when needed
```
- Removes `force_download=True` from `snapshot_download()` calls
- Prevents HuggingFace API rate limiting errors

**Patch 2: Offline mode support** (lines 483-492)
```python
# Original:
self.revision = get_safe_version(self.repo_id, self.revision)
self.download_episodes(download_videos)
self.hf_dataset = self.load_hf_dataset()

# Patched:
import os
if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("SKIP_DOWNLOAD") == "1":
    # Try to load whatever files are available
    self.hf_dataset = self.load_hf_dataset()
else:
    self.revision = get_safe_version(self.repo_id, self.revision)
    self.download_episodes(download_videos)
    self.hf_dataset = self.load_hf_dataset()
```
- Enables offline dataset loading with `HF_HUB_OFFLINE=1` or `SKIP_DOWNLOAD=1`
- Allows working with partially downloaded datasets

**Patch 3: Episode filtering indexing bug fix** (lines 495-500, 663-668)
```python
# Lines 495-500: Create episode ID to array index mapping
# PATCH: Create mapping from episode ID to array index for filtered episodes
# See https://github.com/huggingface/lerobot/issues/959
if self.episodes is not None:
    self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(self.episodes)}
else:
    self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(self.meta.episodes.keys())}

# Lines 663-668: Use mapping in _get_query_indices
# PATCH: Convert episode ID to array index for filtered episodes
# See https://github.com/huggingface/lerobot/issues/959
arr_idx = self.ep_idx_to_arr_idx[ep_idx]
ep_start = self.episode_data_index["from"][arr_idx]
ep_end = self.episode_data_index["to"][arr_idx]
```
- **Critical for BEHAVIOR-1K**: Fixes indexing bug when filtering specific episodes
- Required for `task_filter=["picking_up_trash"]` to work correctly
- See GitHub issue [#959](https://github.com/huggingface/lerobot/issues/959)

#### 2. `lerobot/datasets/utils.py` (1 patch)

**Patch: Boundary check for timestamp validation** (lines 551-552)
```python
ignored_diffs = episode_data_index["to"][:-1] - 1  # indices at the end of each episode
# Filter out any indices that are out of bounds
ignored_diffs = ignored_diffs[ignored_diffs < len(diffs)]
mask[ignored_diffs] = False
```
- Prevents array out-of-bounds errors in `check_delta_timestamps()`
- Improves stability when checking timestamps at episode boundaries

### Importance for BEHAVIOR-1K

- **Patch 3 (episode filtering)** is **essential** - without it, task-specific filtering will cause indexing errors
- Patches 1 and 2 improve reliability and offline usage
- Patch 4 prevents edge-case crashes during dataset validation
