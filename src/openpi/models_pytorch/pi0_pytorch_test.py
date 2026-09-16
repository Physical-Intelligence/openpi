import pytest
import torch

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models_pytorch import gemma_pytorch as _gemma_pytorch
from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing


def _transformers_replace_installed() -> bool:
    try:
        from transformers.models.siglip import check
    except ImportError:
        return False
    return check.check_whether_transformers_replace_is_installed_correctly()


# transformers_replace is copied into site-packages by a manual step documented in the README,
# so it is absent in a plain `uv sync` environment.
requires_transformers_replace = pytest.mark.skipif(
    not _transformers_replace_installed(),
    reason="transformers_replace is not installed into site-packages",
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_observation(config: pi0_config.Pi0Config, batch_size: int, *, channels_first: bool) -> _model.Observation:
    """Builds an observation the way FakeDataset does: float32 images already in [-1, 1]."""
    shape = (batch_size, 3, *_model.IMAGE_RESOLUTION) if channels_first else (batch_size, *_model.IMAGE_RESOLUTION, 3)
    generator = torch.Generator().manual_seed(0)

    def rand(*size: int) -> torch.Tensor:
        return (torch.rand(size, generator=generator) * 2.0 - 1.0).to(_DEVICE)

    keys = _preprocessing.IMAGE_KEYS
    return _model.Observation(
        images={key: rand(*shape) for key in keys},
        image_masks={key: torch.ones(batch_size, dtype=torch.bool, device=_DEVICE) for key in keys},
        state=rand(batch_size, config.action_dim),
        tokenized_prompt=torch.zeros(batch_size, config.max_token_len, dtype=torch.int32, device=_DEVICE),
        tokenized_prompt_mask=torch.ones(batch_size, config.max_token_len, dtype=torch.bool, device=_DEVICE),
    )


@pytest.mark.parametrize("layout", ["nhwc", "nchw"])
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_preprocess_returns_channels_first(layout: str, mode: str):
    """SigLIP consumes [B, C, H, W], so preprocessing must emit it whatever the input layout is."""
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    observation = _make_observation(config, batch_size=2, channels_first=layout == "nchw")

    processed = _preprocessing.preprocess_observation_pytorch(observation, train=mode == "train")

    for key, image in processed.images.items():
        assert image.shape == (2, 3, *_model.IMAGE_RESOLUTION), f"{key} has shape {tuple(image.shape)}"


@requires_transformers_replace
@pytest.mark.parametrize("variant", ["dummy", "gemma_300m"])
def test_vision_projection_matches_text_width(variant: str):
    """The vision tower projects into the text embedding space, so the two widths must agree."""
    vlm_config = _gemma.get_config(variant)

    model = _gemma_pytorch.PaliGemmaWithExpertModel(vlm_config, _gemma.get_config("dummy"), precision="float32")

    assert model.paligemma.config.vision_config.projection_dim == vlm_config.width


@requires_transformers_replace
def test_pi0_pytorch_forward_with_fake_data():
    """Covers `train_pytorch.py debug`, which feeds float32 NHWC images from FakeDataset."""
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    observation = _make_observation(config, batch_size=1, channels_first=False)
    actions = torch.zeros(1, config.action_horizon, config.action_dim, device=_DEVICE)

    model = _pi0_pytorch.PI0Pytorch(config).to(_DEVICE)
    losses = model(observation, actions)

    assert losses.shape == (1, config.action_horizon, config.action_dim)
