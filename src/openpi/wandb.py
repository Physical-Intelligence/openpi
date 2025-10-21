import logging
import os
from typing import Any, SupportsFloat

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb
from wandb import Image

_use_tensorboard: bool = os.environ.get("OPENPI_USE_TENSORBOARD", "0") == "1"


def use_tensorboard():
    return _use_tensorboard


summary_writer: SummaryWriter | None = None


def init(*args, summary_dir: str = "", flush_secs: int = 30, max_queue: int = 10, **kwargs):
    if kwargs.get("mode") == "disabled" or not use_tensorboard():
        wandb.init(*args, **kwargs)
    else:
        global summary_writer  # noqa: PLW0603
        assert summary_dir != "", "summary_dir must be specified when not using wandb"
        summary_writer = SummaryWriter(log_dir=summary_dir, flush_secs=flush_secs, max_queue=max_queue)


def log(
    data: dict[str, Any],
    step: int | None = None,
    commit: bool | None = None,
    sync: bool | None = None,
):
    if not use_tensorboard():
        wandb.log(data, step=step, commit=commit, sync=sync)
    else:
        for k, v in data.items():
            if isinstance(v, Image):
                summary_writer.add_image(k, np.array(v.image), global_step=step)
            elif isinstance(v, list) and all(isinstance(x, Image) for x in v):
                imgs = [np.array(x.image) for x in v]
                summary_writer.add_images(k, np.array(imgs), global_step=step, dataformats="NHWC")
            elif isinstance(v, SupportsFloat):
                summary_writer.add_scalar(k, v, global_step=step)
            else:
                logging.warning(f"Unsupported type for logging: {k}: {type(v)}")


def finish():
    if use_tensorboard():
        wandb.finish()
    else:
        summary_writer.close()
