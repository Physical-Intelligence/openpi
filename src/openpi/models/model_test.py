from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils
from openpi.training import config as _config


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.parametrize(
    ("model_dtype", "training_precision"),
    [
        ("bfloat16", "float32"),
        ("float32", "bfloat16"),
    ],
)
def test_load_pytorch_uses_training_precision(monkeypatch, model_dtype, training_precision):
    created_configs = []
    loaded = []

    class DummyPytorchModel:
        pass

    def fake_pi0_pytorch(config):
        created_configs.append(config)
        return DummyPytorchModel()

    def fake_load_model(model, weight_path):
        loaded.append((model, weight_path))

    monkeypatch.setattr(_model.pi0_pytorch, "PI0Pytorch", fake_pi0_pytorch)
    monkeypatch.setattr(_model.safetensors.torch, "load_model", fake_load_model)

    train_config = _config.TrainConfig(
        name="test_config",
        exp_name="test_run",
        model=pi0_config.Pi0Config(dtype=model_dtype),
        pytorch_training_precision=training_precision,
    )

    model = train_config.model.load_pytorch(train_config, "dummy.safetensors")

    assert created_configs[0].dtype == training_precision
    assert train_config.model.dtype == model_dtype
    assert loaded == [(model, "dummy.safetensors")]


def test_load_pytorch_rejects_unsupported_model_type(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("PI0Pytorch and load_model should not be called for unsupported model types")

    monkeypatch.setattr(_model.pi0_pytorch, "PI0Pytorch", fail_if_called)
    monkeypatch.setattr(_model.safetensors.torch, "load_model", fail_if_called)

    train_config = _config.TrainConfig(
        name="test_config",
        exp_name="test_run",
        model=pi0_fast.Pi0FASTConfig(),
    )

    with pytest.raises(ValueError, match="PI0/PI05"):
        train_config.model.load_pytorch(train_config, "dummy.safetensors")


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
