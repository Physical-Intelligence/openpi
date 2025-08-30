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

You can download the DROID dataset with the following command (after installing the `gsutil` google cloud CLI):
```
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1
```

Note that downloading version 1.0.1 is important (not v1.0.0): it contains the complete set of language annotations (~75k episodes) while v1.0.0 only has annotations for 30k episodes. If for some reason you would like to use another version, modify the line `version="1.0.1"` in the `DroidRldsDataset` object [here](src/openpi/training/droid_rlds_dataset.py).

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

By default, training uses a simple filtering scheme that removes any frames that have little-to-no movement in the first half of the action chunk. Alternatively, you can use a custom filtering scheme by providing a json that maps from episode keys to a list of time step ranges (denoted as a tuple of start and end time step indicies) in that episode you wish to keep. The episode key is a unique ID defined as `f"{recording_folderpath}--{file_path}"`. We choose this convention because both paths are easily accessible in the DROID RLDS episodes' metadata.

We provide an example of such a filtering scheme in [filtering/compute_droid_nonidle_ranges.py](examples/droid/filtering/compute_droid_nonidle_ranges.py), which is significantly more aggressive than the default (and thus leads to policies that take significantly fewer idle actions). We recommend using the filter produced by this script, and have also provided a copy of the filter [here](https://huggingface.co/KarlP/droid#filtering-data) specifically for `droid/1.0.1`.

The filter json you wish to use can be specified by modifying the line `filter_dict_path="<path_to_filter_dict>"` in [src/openpi/training/config.py](src/openpi/training/config.py).

**Note**: The original pi0-FAST-DROID model was trained with joint velocity actions.
Joint velocity actions are not compatible with simulated evaluation environments (much harder to simulate). 
Thus, we do not recommend training with joint velocity actions and instead use joint position actions here.


## Compute Requirements

Our DROID training config requires approximately 2 days on 8x H100 GPUs for convergence (100k iterations, bs256, approx. 1 epoch).
If you start from PaliGemma instead of pi0 initialization, plan with ~5 days on 8x H100s (240k iterations, i.e. 3 epochs).

We have experimented with LoRA for cheaper finetuning, but haven't found the policies to perform well so far.


## RoboArena

Consider submitting your DROID policies to the [RoboArena benchmark](https://robo-arena.github.io/), which allows you to evaluate your policies on diverse tasks & scenes, **in the real world**! :)

If you have questions about RoboArena, please email [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com).
