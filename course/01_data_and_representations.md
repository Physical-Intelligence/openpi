# Module 01 — Data & Representations

> "A model is a function of its data representation." Before a single weight matters,
> you must know exactly what tensors enter the model and how a raw robot episode became
> them. This is the module people skip and then spend a week debugging.

## Learning objectives

- Define the canonical `Observation` and `Actions` structures and their shapes.
- Explain image preprocessing (resize-with-pad, [-1,1] range, train-time augmentation).
- Explain **why** action/state normalization matters for flow matching, and the
  difference between z-score and quantile normalization.
- Understand the `openpi` data-transform pipeline as a composable list of functions.
- Build a fake dataset that produces valid model inputs.

---

## 1. The model's I/O contract: `Observation` and `Actions`

Everything funnels into two structures defined in
[`model.py:81`](../src/openpi/models/model.py).

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    images:      dict[str, Float["*b h w c"]]   # in [-1, 1]
    image_masks: dict[str, Bool["*b"]]          # True = view present
    state:       Float["*b s"]                  # proprioception
    tokenized_prompt:      Int["*b l"]  | None
    tokenized_prompt_mask: Bool["*b l"] | None
    # FAST-only fields:
    token_ar_mask, token_loss_mask
```

`Actions` is simply `Float["*b ah ad"]` — `ah` = action horizon, `ad` = action dim
([`model.py:141`](../src/openpi/models/model.py)).

Fixed conventions you must respect (they're enforced or assumed downstream):

- **Three image keys**: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
  ([`model.py:39`](../src/openpi/models/model.py)). Fewer cameras? You still pass three
  keys and use `image_masks` to mark the absent ones as `False` (their tokens get
  masked out of attention). This keeps the token layout static for `jit`.
- **Image resolution 224×224**, channels-last, float32 in `[-1, 1]`
  ([`model.py:47`](../src/openpi/models/model.py)). `from_dict` converts uint8 `[0,255]`
  → `[-1,1]` for you ([`model.py:116`](../src/openpi/models/model.py)).
- **State and actions are padded to `action_dim` (32)**. A 7-DoF arm uses 7 dims; the
  rest are zeros. This lets one model serve many robot morphologies.

> **Design insight — static shapes.** Every "optional" thing (missing camera, short
> prompt, low-DoF robot) is handled by *masking a fixed-size tensor*, never by changing
> tensor shapes. JAX `jit` compiles for fixed shapes; this is a hard constraint that
> shapes (pun intended) the entire design. Internalize it.

The `from_dict` / `to_dict` methods ([`model.py:109`](../src/openpi/models/model.py))
define the bridge between the unstructured dict the data pipeline emits and the typed
struct the model consumes.

---

## 2. Image preprocessing

`preprocess_observation` ([`model.py:144`](../src/openpi/models/model.py)) runs *inside*
the model (so it's part of the compiled graph and identical at train/inference):

1. **Resize-with-pad** to 224×224 if needed (`image_tools.resize_with_pad`) — preserves
   aspect ratio by padding rather than distorting. Aspect-ratio distortion silently
   degrades manipulation, where geometry matters.
2. **Train-time augmentation** ([`model.py:168`](../src/openpi/models/model.py)) via
   `augmax`: random crop (95%) + resize + small rotation (±5°) + color jitter — but
   **only for non-wrist cameras** (`if "wrist" not in key`). Why? Wrist cameras are
   rigidly mounted; spatial augmentation would teach the model false hand-eye geometry.
   Color jitter is applied to all.
3. **Default masks**: any view without an explicit mask defaults to "present".

> Note the augmentation happens in `[0,1]` space then converts back to `[-1,1]`
> ([`model.py:169`](../src/openpi/models/model.py),
> [`model.py:186`](../src/openpi/models/model.py)) — augmax expects `[0,1]`.

---

## 3. Normalization: the unglamorous keystone

Robot actions live on wildly different scales (a wrist-roll joint in radians vs. a
gripper in meters). Flow matching adds Gaussian noise of unit scale to the *target*
actions ([`pi0.py:196`](../src/openpi/models/pi0.py)). If your action dimensions span
orders of magnitude, unit-scale noise is enormous on one axis and negligible on
another, and the learned vector field is garbage. **Normalization makes every action
dimension roughly unit-scaled so the flow-matching noise is well-conditioned.**

`openpi` supports two schemes ([`transforms.py:114`](../src/openpi/transforms.py)):

**Z-score** (used by π₀): `x_norm = (x - mean) / (std + 1e-6)`.

**Quantile** (used by π₀.₅ / π₀-FAST): map the 1st–99th percentile range to `[-1, 1]`:
```python
x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
```
([`transforms.py:141`](../src/openpi/transforms.py))

Quantile normalization is robust to outliers and bounds the data to roughly `[-1,1]`,
which matters for π₀.₅ because the **discrete state tokenizer assumes inputs are in
`[-1,1]`** before binning ([`tokenizer.py:26`](../src/openpi/models/tokenizer.py)). The
switch is automatic: `use_quantile_norm = model_type != PI0`
([`config.py:188`](../src/openpi/training/config.py)).

The norm stats (`mean/std/q01/q99`) are **precomputed per dataset** by
`scripts/compute_norm_stats.py` and stored as assets, then loaded into the
`Normalize`/`Unnormalize` transforms. At inference, the policy `Unnormalize`s the
model's output back to physical units ([`transforms.py:148`](../src/openpi/transforms.py)).

> **Mental model:** the model only ever sees normalized actions in roughly `[-1,1]`.
> Normalization in, un-normalization out, is a hard shell around the network.

---

## 4. The transform pipeline

`openpi` represents data preprocessing as an ordered **list of pure functions**
`DataDict -> DataDict` ([`transforms.py:23`](../src/openpi/transforms.py)), grouped and
composed via `Group` / `CompositeTransform`. There are two layers:

1. **Data transforms** (dataset-specific): `RepackTransform` renames keys, robot-specific
   input transforms reshape raw data, `Normalize`, `DeltaActions`, etc.
2. **Model transforms** (model-specific, built by `ModelTransformFactory`,
   [`config.py:108`](../src/openpi/training/config.py)): for π₀.₅,
   - `InjectDefaultPrompt`
   - `ResizeImages(224,224)`
   - `TokenizePrompt(PaligemmaTokenizer, discrete_state_input=...)`
   - `PadStatesAndActions(action_dim)`

This factory is exactly where π₀ and π₀.₅ diverge in the pipeline: π₀.₅ passes
`discrete_state_input=True` so the tokenizer folds the state into the prompt string
([`config.py:135`](../src/openpi/training/config.py)). You'll implement that tokenizer
in Module 06.

The full journey of one sample:
```
LeRobot dataset record
  → repack (rename keys to canonical)
  → robot input transform (e.g. LiberoInputs: stack images, build state vector)
  → Normalize (quantile, using dataset stats)
  → [model transforms] inject prompt, resize, tokenize, pad
  → dict → Observation.from_dict
  → batched by data_loader → model
```

You don't need to memorize every transform class, but you must understand the
**pattern**: a static, inspectable, composable list — not a tangle of imperative
preprocessing. This is what makes the same model serve ALOHA, DROID, LIBERO, and your
own robot by swapping only the dataset-specific transforms.

---

## 5. Action chunking and horizons

A dataset stores per-timestep actions. The data loader slices a window of `action_horizon`
future actions to form each training target `a_{t:t+H}`. `SubsampleActions`
([`transforms.py:195`](../src/openpi/transforms.py)) can stride this for low-frequency
control; `DeltaActions` ([`transforms.py:204`](../src/openpi/transforms.py)) can convert
absolute joint targets into deltas relative to current state (common for arms, applied
to a subset of dims via a mask).

`action_horizon` is a real knob: 50 for base π₀, 10 for `pi05_libero`, 15 for
`pi05_droid` ([`config.py:716`](../src/openpi/training/config.py),
[`config.py:893`](../src/openpi/training/config.py)). Longer horizons = more open-loop
commitment, fewer VLM calls, but harder prediction.

---

## Self-check

1. A robot has 2 cameras and a 6-DoF arm. What does its `Observation` look like
   (keys, shapes, masks), given `action_dim=32`?
2. Why is augmentation disabled for wrist cameras?
3. Why does π₀.₅ use quantile normalization while π₀ uses z-score? What downstream
   component *requires* `[-1,1]` inputs?
4. Where, physically in the code path, does normalization happen vs. image
   preprocessing? (One is a data transform, one is inside the model — which and why?)
5. What does `image_masks` accomplish, and why not just change the tensor shape?

## Lab 01

Open [`labs/lab01_data.py`](labs/lab01_data.py). You will:
1. Implement quantile normalize/unnormalize and verify they invert each other and match
   `openpi`'s `Normalize`/`Unnormalize`.
2. Write a `make_fake_episode()` that emits a valid raw dict, and run it through a small
   transform list into a real `Observation`.
3. Confirm your `Observation` passes openpi's type checks by feeding it to a `dummy`
   model's `compute_loss`.

Next: [Module 02 — Vision-language backbone](02_vision_language_backbone.md).
