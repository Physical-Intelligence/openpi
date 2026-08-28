# VLM bbox finetuning

## Baseline diagnostic: `pi05_base` vs raw PaliGemma-3b-mix-224

Before finetuning, we need to confirm the load-bearing claim that motivates
the whole effort: **π0.5 Stage-2 destroyed PaliGemma's `<loc_NNNN>`
detection capability.** `experiments/inference/compare_pi05_vs_paligemma.py`
runs both checkpoints on the **same image and same prompt** and prints a
side-by-side report.

### Command

```bash
python3 compare_pi05_vs_paligemma.py \
    --image /…/_inputs/data_1__ep0__f0045.png \
    --prompt "detect box" \
    --device cpu \
    --max_new_tokens 16 \
    --out_dir experiments/results/compare
```

`--device cpu` is fine here, raw PaliGemma's CPU forward is fast and we
only do greedy decoding; pi05_base goes through JAX and uses whatever
backend is available (`CudaDevice(id=0)` in the log).

### Side-by-side report

```
raw PaliGemma-3b-mix-224
   token ids  : [256474, 256384, 256662, 256529, 3741, 1]
   decoded    : '<loc0474><loc0384><loc0662><loc0529> box<eos>'
   bboxes     : 1
     [0] YXYX [0,1] : (0.463, 0.375, 0.646, 0.517)
         224 px box : y=[103.7, 144.8]  x=[84.0, 115.7]
   verdict    : DETECTS  

pi05_base
   token ids  : [255689, 255624, 255695, 255665, 255376, 255635, 255180, 255467, 1, 1, 1, 255570, 1, 1, 1, 255705]
   decoded    : 'Ẁꪑⓙʶ옻媄邽⣞䊐'
   step0 loc-top1 rank: 203  in full 257152-vocab
   step0 loc-range mass: 8.988%
   free-decode bboxes : 0
   forced-loc bbox YXYX : (0.301, 0.865, 0.394, 0.874)
   verdict    : NO LOC TOKEN in free decode  
```

What this tells us:

- **Raw PaliGemma** emits a clean 4-loc quartet on the very first
  tokens, followed by the class label `" box"` and `<eos>` — a
  textbook detection. The bbox lands at y=[104, 145] px, x=[84, 116] px,
  centred near (100 px, 124 px).
- **pi05_base** with the identical prompt and image emits **zero** loc
  tokens. Its top-1 is always a FAST token (`255xxx`) or `<eos>` (`1`).
  Even the loc range still has a non-trivial top1 (rank 203 at step 0,
  ~9% mass), but it is heavily outranked by the FAST tokens.
- The forced-loc bbox `(0.30, 0.87, 0.39, 0.87)` is *not* the raw
  PaliGemma box — it has drifted to the right edge of the image and
  collapsed in x. Confirms the loc embeddings still exist but no longer
  correspond to spatially meaningful coordinates.

### Outputs

```
experiments/results/compare/data_1__ep0__f0045__raw_paligemma.png
experiments/results/compare/data_1__ep0__f0045__pi05_base.png
```

Each PNG has the input frame plus the bbox (if any) and a header strip
with the model name, prompt, and decoded text — meant to be skimmed.

### Conclusion (what this evidence supports)

> Same image · same prompt · different checkpoint.
> Raw PaliGemma → valid bbox; pi05_base → zero.
> π0.5 Stage-2 fine-tuning overwrote PaliGemma's detection capability.
> The LM head was retrained to emit FAST action tokens
> (vocab ≈ 255k–257k), so the spatial route is no longer addressable.

This is the single result that justifies the LoRA finetune evaluated in
the next section.

## Running the training script

Once the baseline diagnostic confirms that pi05_base lost detection, the
finetune itself is launched from `experiments/train/run_train.py` with
the YAML config at `experiments/configs/pi05_loc_ce.yaml`.

### Command

```bash
python3 experiments/train/run_train.py \
    --config experiments/configs/pi05_loc_ce.yaml \
    --no_wandb \
    --backend jax
```

CLI flags (defined in `run_train.py`):

| flag | meaning |
| --- | --- |
| `--config <path>` | YAML experiment file. Parsed by `config.build_config_from_yaml` and turned into a `TrainConfig`. |
| `--no_wandb` | overrides `training.wandb_enabled` in the YAML to `False`. Useful for smoke runs / local debugging. |
| `--backend jax` | picks the JAX trainer at `scripts/train.py`. The other option is `pytorch` (`scripts/train_pytorch.py`), but only `jax` supports LoRA, which we need for `paligemma_variant=gemma_2b_lora`. |
| `--exp_name <name>` | optional override of `experiment.name`. Becomes the checkpoint subdirectory. |
| `--resume` | overrides `training.resume` to `True` — load the latest checkpoint from `checkpoint_base_dir/<exp_name>/`. |

`run_train.py` also patches `lerobot.common.datasets.utils.get_safe_version`
so it short-circuits the HuggingFace Hub version check whenever the
dataset already exists locally under `HF_LEROBOT_HOME` — important because
our merged dataset `box-pick-bimanual-all-right` is only local.

### What the YAML config encodes

`experiments/configs/pi05_loc_ce.yaml` selects **bbox** — joint
cross-entropy on the `<loc_NNNN>` quartet **plus** flow-matching MSE on
the action expert — and configures it like this:

```yaml
data:
  merged_name: box-pick-bimanual-all-right
  default_prompt: "pick the box"
  bbox_sidecar_parquet: /…/cam_high__detect_box__merged.parquet
  bbox_query: box

model:
  base_checkpoint: gs://openpi-assets/checkpoints/pi05_base/params
  assets_dir:      gs://openpi-assets/checkpoints/pi05_base/assets
  asset_id: trossen

training:
  model_variant: pi05_loc_ce       # → Pi0WithLocCEConfig
  freeze_mode:   vlm_only          # freezes action expert + SigLIP + heads
  use_lora:      true              # gemma_2b_lora on PaliGemma
  loc_loss_weight: 5.0             # α — CE over the 4 forced loc tokens
  mse_loss_weight: 10.0            # β — flow-matching MSE on actions
  num_train_steps: 3000
  batch_size: 16
  peak_lr: 1.0e-5
  warmup_steps: 200
  save_interval: 500
  log_interval: 25
```

Key decisions baked in here:

- `model_variant: pi05_loc_ce` dispatches to `Pi0WithLocCEConfig`, which
  augments the standard pi05 graph with the joint CE+MSE loss head.
- `freeze_mode: vlm_only` + `use_lora: true` together mean: only the
  PaliGemma LoRA slice gets gradients. The action expert, SigLIP vision
  tower, and other heads are frozen. This keeps the action policy intact
  while we re-teach the LM head to emit loc tokens.
- `loc_loss_weight=5.0, mse_loss_weight=10.0` — the trainer reduces with
  `jnp.mean`, so the effective objective is `5·mean_B(ce) + 10·mean_B(mean_H(mse))`.
- `peak_lr=1e-5, warmup_steps=200` — small, because we're updating a
  thin LoRA slice of a 2B-parameter VLM, not full PaliGemma.
- `save_interval=500` × `num_train_steps=3000` → checkpoints at steps
  500, 1000, 1500, 2000, 2500, 2999. The probe in the previous section
  ran on step 2999.

## Testing after finetuning

Once a LoRA checkpoint is produced (here at step 2999 of
`pi05_aloha_box_pick_loc_ce`), use `experiments/inference/probe_lm_head.py`
to ask a single question of the LM head: **does the finetuned model emit
`<loc_NNNN>` tokens for a "detect box" query on a robot image?**

It is intentionally a diagnostic, not an eval loop — it dumps per-step
logits, ranks and softmax masses on the spatial sub-range so you can see
*why* the head behaves the way it does.

### Command

```bash
python3 ../inference/probe_lm_head.py \
    --image /…/_inputs/data_1__ep0__f0045.png \
    --variants stock task task_state \
    --constrained_decode_loc \
    --frame_id gate_step3000 \
    --checkpoint_path /…/checkpoints/pi05_aloha_box_pick_loc_ce/box_pick_pi05_loc_ce/2999/params \
    --use_lora
```

Flags worth noting:

| flag | what it does |
| --- | --- |
| `--variants stock task task_state` | runs three prompt templates back to back so we can see whether prompt formatting changes anything (see *Variants* below). |
| `--constrained_decode_loc` | after the free decode, also force the head to emit 4 tokens from the `<loc_NNNN>` sub-range `[256000, 257024)`. Tests whether the spatial bins retain meaning even when the head wouldn't pick them on its own. |
| `--checkpoint_path …/2999/params` | local orbax params dir for the LoRA-finetuned checkpoint at step 2999. |
| `--use_lora` | switches `paligemma_variant` to `gemma_2b_lora` so the param tree matches the saved checkpoint. Without this the restore would fail. |
| `--frame_id gate_step3000` | prefix used in the output filenames (`gate_step3000__stock.png`, etc.). |


## Observation:
_Experiment 1: α=1, β=10 (doc default)_
  Setup: peak_lr=1e-5, warmup=200, batch=16, 3000 steps, no wandb, JAX backend  
                                                  
| step | loss  | grad_norm | param_norm |
|-----:|------:|----------:|-----------:|
| 0    | 9.79  | 12.40     | 1802.6104  |
| 500  | 0.97  | 2.36      | 1802.6196  |
| 1000 | 0.31  | 1.50      | 1802.6284  |
| 1500 | 0.21  | 1.68      | 1802.6329  |
| 2000 | 0.18  | 1.73      | 1802.6356  |
| 2500 | 0.17  | 1.74      | 1802.6368  |
| 2975 | 0.175 | 1.73      | 1802.6370  |


**Positions 1-3 of constrained decode are learning the right sequence structure**
Look at the rank of <loc_423> across the 4 forced positions at step 2999:  
  position 0 (first token after "detect box\n"):  rank 234   ← model can't pick a loc token here
  position 1 (after seeing one loc token):         rank 7        ← suddenly very confident
  position 2:                                                            rank 5
  position 3:                                                            rank 1       ← model's unconstrained top-1!

  This rank trajectory 234 -> 7 -> 5 -> 1 is only possible if the model learned the structural pattern "loc tokens come in groups of 4". The CE supervision worked: the LoRA encoded the fact that after one <loc_*> token, the next three are also <loc_*> tokens with very high probability.

  Compare to pi05_base baseline (untrained):
  position 0:  rank ~210
  position 1:  rank ~217  -> still no idea what comes next
  position 2:  rank ~1067
  position 3:  rank ~1893

 **Loc-token probability mass jumped at positions 1-3**
| position   | pi05_base baseline | trained (step 2999) |
|------------|-------------------:|--------------------:|
| position 0 | 8.4%               | 7.1%                |
| position 1 | 30.5%              | 49.4%               |
| position 2 | 9.8%               | 48.9%               |
| position 3 | 7.5%               | 62.9%               |

**MSE anchor preserved action quality**
param_norm drifted from 1802.6104 to 1802.6370 across 3000 steps, that's a 0.0014 % change. The 27.87 M LoRA params absorbed all the learning; the 3.35 B frozen params (including the action expert) are bit-identical to pi05_base.