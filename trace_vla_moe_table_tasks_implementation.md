# TraceVLA-MoE on the physical-robot `table_tasks` dataset

This document describes the `trace_vla_moe_table_tasks` training pipeline: a faithful
re-targeting of the LIBERO `trace_vla_moe` combined-MoE TraceVLA model onto the
physical-robot `n5zhong/table_tasks` dataset. It records the full data flow, every
change made relative to `trace_vla_moe`, and the launch / norm-stat commands.

The guiding principle (per the task spec) was: **reuse the `trace_vla_moe` pipeline
unchanged wherever possible, and only make the changes that are genuinely required by
the new dataset's structure.** Three such changes were needed (2 skill experts instead
of 5; video-backed image decoding; new `TrainConfig`); everything else — augmentations,
trace generation, overlay rendering, completion head, losses, transforms — is inherited
verbatim.

---

## 1. Launch commands

Both commands run inside the project venv used for all training
(`mujoco_playground/.venv`, activated by `pace/.pace_python.sh`).

**Compute normalization stats** (writes `assets/n5zhong/table_tasks/{state,actions}/...`;
must be run once before training):

```bash
python pace/openpi/scripts/compute_norm_stats.py --config-name trace_vla_moe_table_tasks
```

`compute_norm_stats.py` decodes every frame's video to run `__getitem__`; to cap the
work you may pass `--max-frames <N>` (e.g. `--max-frames 20000`) — norm stats only need
`state` / `actions`, which are unaffected by the image/trace augmentations.

**Train** (full finetune from `pi05_base`):

```bash
python pace/openpi/scripts/train_trace_vla_moe.py trace_vla_moe_table_tasks --exp-name trace_vla_moe_table_tasks
```

The same `scripts/train_trace_vla_moe.py` entry point used by `trace_vla_moe` is reused
unchanged — only the config name differs.

---

## 2. Dataset structure: `table_tasks` vs LIBERO

`meta/info.json` comparison (the source of the one structural discrepancy):

| field            | LIBERO (`yilin-wu/libero-100`)   | table_tasks (`n5zhong/table_tasks`) |
|------------------|----------------------------------|-------------------------------------|
| codebase_version | v2.0                             | v2.1                                |
| robot_type       | panda                            | yam                                 |
| total_episodes   | 4338                             | 299                                 |
| total_frames     | 676070                           | 98158                               |
| fps              | 10                               | 30                                  |
| `image` dtype    | **`image`** (in-parquet)         | **`video`** (MP4, AV1)              |
| `image` shape    | `[256,256,3]` (HWC)              | `[3,480,640]` (CHW)                 |
| `wrist_image`    | `image`, `[256,256,3]`           | `video`, `[3,480,640]`              |
| `state` shape    | `[8]`                            | `[8]` (`q_1..q_4,x,y,z,gripper`)    |
| `actions` shape  | `[7]`                            | `[7]` (`rx,ry,rz,x,y,z,gripper`)    |
| `videos/` dir    | absent (`total_videos: 0`)       | present (`total_videos: 598`)       |

**The only structural discrepancy that affects training is image storage.** LIBERO
stores images as in-parquet `image` features, so `LeRobotDataset[idx]` returns the
decoded frame directly. table_tasks stores them as `video` features (MP4 per
`{video_key}` per episode), so the per-frame parquet row carries *no* image — frames
must be decoded on demand from the episode video at the frame's timestamp.

Non-issues that look like discrepancies but need no special handling:

- **State / action dims (both 8 / 7).** `LiberoTraceDataset` splits state as
  `[:3] / [3:6] / [6:]` and immediately re-concatenates it
  ([libero_trace_dataset.py:124-127](src/openpi/policies/libero_trace_dataset.py#L124-L127),
  [libero_trace_dataset.py:399-406](src/openpi/policies/libero_trace_dataset.py#L399-L406)),
  so the 8-D state is reconstructed verbatim regardless of per-axis semantics.
- **Action representation.** table_tasks actions are 6 deltas + 1 absolute gripper
  (verified: first-6 std ≈ 0.01–0.07, centered near 0; gripper ∈ [0.03, 0.99]) — the
  same convention as LIBERO. So the skill-end zero-padding
  ([libero_reason_dataset.py:21-43](src/openpi/policies/libero_reason_dataset.py#L21-L43))
  ("zero pose deltas + hold last gripper") is correct, and no delta transform is added.
- **Image resolution 480×640.** Handled by the existing resize/normalize path (see §6).

---

## 3. Annotation health verification

Annotation files:
`pace/openpi/data/table-tasks/tabletask_skill_annotations.json` and
`pace/openpi/data/table-tasks/tabletask_skill_target_traces.json`.

They were validated against the dataset (299 episodes / 98158 frames) and the exact
fields `LiberoTraceDataset` reads. All checks passed:

- **Episode coverage:** both files key all 299 episodes (0–298); none missing/extra.
- **Timestep coverage:** every episode's `segments` tile `[0, length)` with **no gaps
  and no overlaps** (matches `_segment_index_for_step`,
  [libero_trace_dataset.py:62-66](src/openpi/policies/libero_trace_dataset.py#L62-L66)).
- **Skills:** only `PICKUP_FROM` (650), `PLACE_ON` (525), `PLACE_IN` (125) — exactly the
  three expected skills.
- **Trace ↔ segment match:** every segment index has a `target_traces` entry with the
  matching `skill_index` and identical skill name (matches the join at
  [libero_trace_dataset.py:229-234](src/openpi/policies/libero_trace_dataset.py#L229-L234)).
- **EE-trace adequacy:** every `end_effector_trace.trace` has length **exactly equal** to
  its segment length (so `ee_full.shape[0] >= seg_end - seg_start` always holds and
  `has_trace` is never silently dropped — see
  [libero_trace_dataset.py:259](src/openpi/policies/libero_trace_dataset.py#L259)).
- **Statuses:** every `end_effector_trace.status` and `semantic_target.status` is `"OK"`.
- **Coordinate space:** every per-episode trace dict carries `image_width=640`,
  `image_height=480`, and all trace / semantic points lie inside `[0,640]×[0,480]`. This
  is the coordinate space `LiberoTraceDataset` reads at
  [libero_trace_dataset.py:151-153](src/openpi/policies/libero_trace_dataset.py#L151-L153)
  and normalizes against — so the 640×480 trace space is handled correctly even though
  the model resizes images to 224×224.

(The validation script is reproducible; it cross-checks `meta/episodes.jsonl` lengths,
segment tiling, trace lengths, statuses, skills, and coordinate ranges.)

---

## 4. Skill → expert routing (2 experts)

The three table-task skills route onto **2** experts, per the mapping the user pinned in
`embed_sigma()` ([tokenizer.py:122-129](src/openpi/models/tokenizer.py#L122-L129)):

```
PICKUP_FROM -> expert 0
PLACE_ON    -> expert 1
PLACE_IN    -> expert 1
```

The trace pipeline computes the routing id from `trace_utils.skill_to_expert_id`
([libero_trace_dataset.py:236](src/openpi/policies/libero_trace_dataset.py#L236),
[trace_utils.py:29-48](src/openpi/models/trace_utils.py#L29-L48)), which yields the
**identical** mapping for these three skills (`PICKUP_FROM→0`, `PLACE_ON→1`,
`PLACE_IN→1`) — verified programmatically against `embed_sigma`. The id is carried as
`atomic_token` ([libero_trace_dataset.py:432](src/openpi/policies/libero_trace_dataset.py#L432)),
flows through `LiberoTraceInputs`
([libero_trace_policy.py:71-72](src/openpi/policies/libero_trace_policy.py#L71-L72))
and `TraceObservation` into the model, where it becomes a one-hot of width `K=2`
driving both hard-routed MoE streams
([pi0_trace_vla_moe.py:338-348](src/openpi/models/pi0_trace_vla_moe.py#L338-L348),
[pi0_trace_vla_moe.py:424-426](src/openpi/models/pi0_trace_vla_moe.py#L424-L426)).

Because every annotated skill is one of the three expected skills (verified in §3), the
routing never falls through to the default-0 branch of `skill_to_expert_id`.

---

## 5. End-to-end data flow

```
train_trace_vla_moe.py  trace_vla_moe_table_tasks
  └─ main()                                              scripts/train_trace_vla_moe.py:354
     └─ _create_trace_data_loader()                      scripts/train_trace_vla_moe.py:305
        ├─ isinstance(config.data, LeRobotTraceVLAMoeDataConfig)  ✓
        ├─ data_config = config.data.create(...)         -> LiberoTraceDataConfig
        │    (LeRobotTraceVLAMoeDataConfig.create        config.py:743)
        │     repack = Group()  (empty)
        │     data_transforms.inputs  = [LiberoTraceInputs]
        │     data_transforms.outputs = [LiberoTraceOutputs]
        │     model_transforms.inputs = [TraceResizeImages(224,224),
        │                                TraceTokenizePrompt(PaligemmaTokenizer(200)),
        │                                PadStatesAndActions(32)]
        ├─ dataset = LiberoTraceDataset(data_config, action_horizon=10)
        │    ├─ root = _resolve_dataset_root("n5zhong/table_tasks", skill_json)
        │    │         -> HF-cache snapshot (libero_reason_dataset.py:46)
        │    ├─ super().__init__(repo_id, root, revision="main")  (no delta_timestamps)
        │    ├─ state/actions stacked from parquet columns        (lines 124-128)
        │    └─ skill + trace annotations loaded; 640×480 read    (lines 138-153)
        ├─ transform_dataset(dataset, data_config, skip_norm_stats=False)
        │    -> repack + data_transforms.inputs + Normalize + model_transforms.inputs
        └─ TorchDataLoader(..., num_workers=config.num_workers)   data decoded in workers

LiberoTraceDataset.__getitem__(idx):                     libero_trace_dataset.py:172
   item = hf_dataset[idx]                                # parquet row (state/actions/idx)
   if meta.video_keys:                                   # table_tasks -> ["image","wrist_image"]
       decode current-ts frames via _query_videos        # NEW guarded block, lines 178-191
   segment lookup, anchor-age aug, trace target, overlay  # unchanged trace logic
   returns dict: observation/{image,wrist_image,overlay_image},
                 state(8), actions(10,7), atomic_token, semantic_target_xy,
                 current_ee_xy, future_trace_xy(20,2), has_trace, has_overlay, progress, ...

LiberoTraceInputs -> TraceResizeImages(224) -> Normalize -> TraceTokenizePrompt
   -> PadStatesAndActions(32) -> TraceObservation.from_dict -> Pi0TraceVLAMoe.compute_loss
```

The model forward (`Pi0TraceVLAMoe`, unchanged): 3-stream Gemma trunk
(PaliGemma-2B VLM + action MoE + trace MoE); trace planning stream
([pi0_trace_vla_moe.py:361](src/openpi/models/pi0_trace_vla_moe.py#L361)) and action
execution stream ([pi0_trace_vla_moe.py:457](src/openpi/models/pi0_trace_vla_moe.py#L457)),
plus the per-skill completion head
([pi0_trace_vla_moe.py:311-333](src/openpi/models/pi0_trace_vla_moe.py#L311-L333)).
Losses = action FM + trace FM + completion regression
([pi0_trace_vla_moe.py:563-567](src/openpi/models/pi0_trace_vla_moe.py#L563-L567)).

---

## 6. Changes made (and why each is essential)

### 6.1 Two 2-expert MoE variants — `src/openpi/models/gemmoe_trace.py`

`Pi0TraceVLAMoe.__init__` asserts each stream's variant `num_local_experts` equals the
configured expert count
([pi0_trace_vla_moe.py:128-137](src/openpi/models/pi0_trace_vla_moe.py#L128-L137)), and
the existing variants hard-code 5 experts. Two new variants were added with
`num_local_experts=2`, otherwise identical in shape to their 5-expert siblings (so the
joint-attention head/width/depth contracts against `gemma_2b` still hold):

- `trace_moe_gemma_300m_2e` — action MoE, width 1024 / mlp 4096 / depth 18 / **K=2**
  ([gemmoe_trace.py:122-139](src/openpi/models/gemmoe_trace.py#L122-L139)).
- `trace_moe_small_2e` — trace MoE (shrunk), width 512 / mlp 2048 / depth 18 / **K=2**
  ([gemmoe_trace.py:155-169](src/openpi/models/gemmoe_trace.py#L155-L169)).

Both are added to the `Variant` literal
([gemmoe_trace.py:69-77](src/openpi/models/gemmoe_trace.py#L69-L77)). The MoE machinery
itself is K-agnostic: `HardMoeBlock` builds `K` experts named `expert_{e}`
([gemmoe_trace.py:261-273](src/openpi/models/gemmoe_trace.py#L261-L273)) and the
`pi05_base` warm-start fans the dense `mlp_1` FFN into `K` action experts in
`scripts/train_trace_vla_moe.py`
([train_trace_vla_moe.py:135-151](scripts/train_trace_vla_moe.py#L135-L151)) — so K=2
needs no loader change. The trace MoE (stream 2) is randomly initialized exactly as in
`trace_vla_moe`.

### 6.2 Video-aware frame loading — `src/openpi/policies/libero_trace_dataset.py`

A single guarded block was added at the top of `__getitem__`
([libero_trace_dataset.py:178-191](src/openpi/policies/libero_trace_dataset.py#L178-L191)):

```python
if len(self.meta.video_keys) > 0:
    current_ts = item["timestamp"].item()
    query_timestamps = self._get_query_timestamps(current_ts, query_indices=None)
    video_frames = self._query_videos(query_timestamps, ep_idx)
    item = {**video_frames, **item}
```

This mirrors `LeRobotDataset.__getitem__` and `Atomic_Dataset` (the proven loader for
this exact dataset). Properties:

- **No-op for LIBERO:** `self.meta.video_keys` is empty for image-backed datasets, so the
  block is skipped and `item["image"]` / `item["wrist_image"]` come from the parquet row
  exactly as before — the `trace_vla_moe` path is byte-for-byte unchanged.
- **Single-frame query:** `LiberoTraceDataset` passes no `delta_timestamps`, so
  `self.delta_indices is None` and `_get_query_timestamps(..., query_indices=None)` returns
  one timestamp per video key — the current frame.
- **No segfault risk:** `_query_videos` runs only inside `__getitem__`, i.e. in the
  DataLoader worker processes; the main process never decodes video.
- **Format handoff is automatic:** decoded frames are CHW float[0,1] torch tensors;
  the existing `_ensure_hwc_uint8`
  ([libero_trace_dataset.py:459-468](src/openpi/policies/libero_trace_dataset.py#L459-L468))
  already converts CHW→HWC and float→uint8, so the rest of `__getitem__` is unchanged.

This one change fixes **both** the training path (`train_trace_vla_moe.py` imports
`LiberoTraceDataset` directly) and the norm-stats path (`compute_norm_stats.py` →
`create_torch_dataset` selects `LiberoTraceDataset` for any `LiberoTraceDataConfig`,
[data_loader.py:149-154](src/openpi/training/data_loader.py#L149-L154)).

### 6.3 New `TrainConfig` — `src/openpi/training/config.py`

`trace_vla_moe_table_tasks`
([config.py:2426](src/openpi/training/config.py#L2426)) is a copy of `trace_vla_moe`
([config.py:2305](src/openpi/training/config.py#L2305)) with only:

- `action_expert_variant="trace_moe_gemma_300m_2e"`, `trace_expert_variant="trace_moe_small_2e"`,
  `num_action_experts=2`, `num_trace_experts=2`;
- `data.repo_id="n5zhong/table_tasks"` and the table-task annotation paths
  (`tabletask_skill_annotations.json`, `tabletask_skill_target_traces.json`).

It reuses `LeRobotTraceVLAMoeDataConfig` + `LiberoTraceDataConfig` unchanged (so the data
factory's `Pi0TraceVLAMoeConfig` type check at
[config.py:748](src/openpi/training/config.py#L748) is satisfied), is a full finetune (no
`freeze_filter`), warm-starts from `pi05_base`, and keeps `trace_vla_moe`'s optimizer /
LR schedule / EMA / horizons / `max_token_len` and all `LiberoTraceDataConfig` augmentation
defaults (anchor-age `h_train_max=15`, `scene_dropout_rate=0.15`, `overlay_dropout_rate=0.10`,
`trace_perturb_max_sigma=0.03`, overlay color/thickness, `trace_horizon=20`). It also sets
`model.image_source_hw=(480, 640)` — see §6.4.

### 6.4 Letterbox-aware keypoint augmentation — `src/openpi/models/trace_observation.py`

The table-tasks camera is non-square (480×640), so `resize_with_pad` letterboxes each
frame into the 224×224 model input (content 224×168, with 28 px zero-pad top and bottom;
x fills the frame). The trace/EE/semantic-target coordinates are normalized in the
640×480 camera frame. Train-time geometric augmentation
([trace_observation.py](src/openpi/models/trace_observation.py)) applies the *same* random
crop/resize/rotate to the image **and** to these image-space keypoints so they stay in
correspondence (label-preserving augmentation — the standard requirement for a model that
generates 2-D points on an image). Previously the keypoints were placed onto the full
`[0, W-1]×[0, H-1]` frame, which is exact for LIBERO's square 256×256 images but, for a
letterboxed source, would put them off the visible content band (a y-misalignment that the
crop/rotate then made inconsistent).

The fix makes the keypoint↔pixel mapping letterbox-aware:

- A new optional model-config field `image_source_hw`
  ([pi0_trace_vla_moe_config.py:77-83](src/openpi/models/pi0_trace_vla_moe_config.py#L77-L83))
  carries the original camera `(H, W)`; `trace_vla_moe_table_tasks` sets `(480, 640)`,
  `trace_vla_moe` (and every other model) leaves it `None`.
- `preprocess_trace_observation` now takes `image_source_hw`
  ([trace_observation.py:102-109](src/openpi/models/trace_observation.py#L102-L109)) and
  computes the placement `px = xy·kp_scale + kp_offset` from the same `resize_with_pad`
  geometry ([trace_observation.py:193-211](src/openpi/models/trace_observation.py#L193-L211)):
  keypoints are placed on the letterboxed content rectangle, augmented in lockstep with the
  image, and inverted with the **same** fixed rectangle
  ([trace_observation.py:235-237, 257-260](src/openpi/models/trace_observation.py#L235-L260)).
- The model passes the field through only on the `train=True` path in `compute_loss`
  ([pi0_trace_vla_moe.py:537-548](src/openpi/models/pi0_trace_vla_moe.py#L537-L548)); the
  inference (`sample_*`) paths use `train=False`, which never transforms keypoints.

This keeps the coordinate convention in the 640×480 camera frame end-to-end — overlay
rendering, the trace supervision target, and inference are all unchanged, so **no inference
change is required**. **It is provably a no-op for square sources:** when `image_source_hw`
is `None`, `kp_scale=[W-1, H-1]` and `kp_offset=[0, 0]`, i.e. exactly the prior mapping
(verified: `None` forward/inverse are bit-identical to the old `×[W-1,H-1]` / `÷[W-1,H-1]`),
so the LIBERO `trace_vla_moe` path is untouched.

---

## 7. Behaviors inherited unchanged from `trace_vla_moe`

All of the following are reused verbatim (no table-task-specific code):

- Anchor-age augmentation for receding-horizon training
  ([libero_trace_dataset.py:242-306](src/openpi/policies/libero_trace_dataset.py#L242-L306)).
- Scene dropout, overlay dropout, smooth low-frequency trace perturbation
  ([libero_trace_dataset.py:330-396](src/openpi/policies/libero_trace_dataset.py#L330-L396)).
- Trace overlay rendering (cyan polyline, same color/thickness/endpoints)
  ([trace_utils.py:207-276](src/openpi/models/trace_utils.py#L207-L276)).
- Train-time geometric+color image augmentation applied jointly to base/overlay/keypoints
  ([trace_observation.py](src/openpi/models/trace_observation.py)) — same chain as
  `trace_vla_moe`, with the keypoint placement now letterbox-aware for the non-square
  camera (see §6.4; a no-op for square LIBERO).
- Trace flow-matching with EE row-0 inpainting clamp + appended semantic-target anchor;
  action flow-matching on the overlay image; per-skill completion regression
  ([pi0_trace_vla_moe.py:361-574](src/openpi/models/pi0_trace_vla_moe.py#L361-L574)).
- Prompt construction (`"Plan: ... Current step: K. <skill_text>"`)
  ([libero_trace_policy.py:196-205](src/openpi/policies/libero_trace_policy.py#L196-L205)).

---

## 8. 480×640 letterboxing and the keypoint-augmentation fix

table_tasks frames are 480×640 (4:3); `resize_with_pad`
([image_tools.py:38-58](packages/openpi-client/src/openpi_client/image_tools.py#L38-L58))
scales to 224×168 and pads 28 px top/bottom to reach 224×224. The base image **and** the
trace overlay go through the *same* `resize_with_pad`, so the rendered overlay stays
pixel-aligned with the image content, and the trace/EE/semantic coordinates remain in the
640×480 camera frame throughout (normalized → rendered → resized identically). LIBERO's
256×256 is square, so no padding occurs there.

The one place the letterbox needed explicit handling is **train-time geometric
augmentation**: it transforms the image-space keypoints (trace/EE/semantic target) jointly
with the image, and that requires placing them on the letterboxed content rectangle rather
than the full square frame. This is implemented as the letterbox-aware mapping in §6.4 and
gated by `model.image_source_hw` (set to `(480, 640)` here, `None`/no-op for LIBERO). With
that fix, the augmentation is label-preserving and the coordinate convention stays in the
640×480 camera frame end-to-end — so **the inference pipeline simply normalizes EE /
semantic-target points in the 640×480 camera frame** (the same convention the annotations
use), with no inference-side coordinate change needed.
