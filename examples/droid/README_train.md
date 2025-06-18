# Training on DROID

Here we describe how to fine-tune the pi0-FAST model on the DROID dataset. This is an approximate open-source reproduction of the pi0-FAST-DROID training pipeline.
(small differences in data loading and the used action space)

In contrast to the rest of openpi, which uses LeRobot for data loading, we need to use RLDS as the data format for DROID training (since atm LeRobot isn't scalable enough 
for larger datasets like DROID -- they are working on improving it though). Below, we provide instructions for updating your openpi environment for RLDS data loading and where to download the DROID dataset.

## Install

We need a few additional dependencies for RLDS data loading. Run:
```bash
uv sync --group rlds
```

## Download DROID dataset

You can download a (slightly outdated) version of DROID with the following command (after installing the `gsutil` google cloud CLI):
```
gsutil -m cp -r gs://gresearch/robotics/droid <your_download_path>
```

Note that this version of DROID is slightly outdated: it only contains a partial set of language annotations (~30k episodes).
Please email [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com) to get access to the most up-to-date version of the DROID RLDS dataset (with language annotations on 75k episodes)!
(sorry, we are working on updating the version on the official bucket).

You will need 1.8TB of disk storage to download the DROID RLDS dataset.

## Run

First, change the `rlds_data_dir` path in your `TrainConfig` to the directory that you downloaded the `droid` dataset into (see [src/openpi/training/config.py](src/openpi/training/config.py)).

Then, compute normalization statistics (this will take ~10 minutes):
```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi0_fast_droid_finetune --max-frames 10_000_000
```

Run training:
```bash
uv run --group rlds scripts/train.py pi0_fast_droid_finetune --exp-name=my_experiment --overwrite
```

**Note**: The original pi0-FAST-DROID model was trained with joint velocity actions.
Joint velocity actions are not compatible with simulated evaluation environments (much harder to simulate). 
Thus, we do not recommend training with joint velocity actions and instead use joint position actions here.


## Compute Requirements

Our DROID training config requires approximately 2 days on 8x H100 GPUs for convergence (100k iterations, bs256, approx. 1 epoch).
If you start from PaliGemma instead of pi0 initialization, plan with ~5 days on 8x H100s (240k iterations, i.e. 3 epochs).

We have experimented with LoRA for cheaper finetuning, but haven't found the policies to perform well so far.
