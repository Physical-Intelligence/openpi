import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

import openpi.models.gemma as gemma
import openpi.models.lora as lora


def get_annotation_to_dim_size() -> dict[str, int]:
    return {
        "B": 8,
        "T": 13,
        "S": 7,
        "N": 4,
        "M": 4,
        "K": 2,
        "H": 48,
        "D": 64,
    }


def eqn_to_shape(eqn: str, annotation_to_dim_size: dict[str, int]) -> tuple[tuple[int, ...], ...]:
    (lhs_part_0, lhs_part_1), _ = gemma._parse_einops_eqn(eqn)  # noqa: SLF001
    return tuple(int(ann) if ann.isdigit() else annotation_to_dim_size[ann] for ann in lhs_part_0), tuple(
        int(ann) if ann.isdigit() else annotation_to_dim_size[ann] for ann in lhs_part_1
    )


@pytest.mark.parametrize(
    ("eqn", "lora_annotation"),
    [
        ("BSD,3KDH->3BSKH", "3KDL,3KLKH->3KDH"),
        ("BTD,NDH->BTNH", "NDL,NLNH->NDH"),
        ("BSD,2KDH->2BSKH", "2KDL,2KLKH->2KDH"),
        ("BTNH,NHD->BTD", "NHNL,NLD->NHD"),
    ],
)
def test_lora_einsum_equivalent_to_original(eqn: str, lora_annotation: str):
    annotation_to_dim_size = get_annotation_to_dim_size()
    x_shape, w_shape = eqn_to_shape(eqn, annotation_to_dim_size)
    einsum = gemma.Einsum(shape=w_shape, name="einsum", init_fn=nn.initializers.lecun_normal())
    lora_einsum = gemma.LoRAEinsum(
        einsum,
        gemma.LoRAConfig(rank=4, alpha=4.0),
        lora_annotation,
        nn.initializers.zeros_init(),
        nn.initializers.zeros_init(),
    )

    x = jax.random.normal(jax.random.key(0), x_shape)

    def module_call(instance, x):
        return instance(eqn, x)

    einsum_variables = einsum.init(jax.random.key(0), x, method=module_call)
    lora_einsum_variables = lora_einsum.init(jax.random.key(0), x, method=module_call)
    # Copy over the weights from the original einsum to the lora einsum since the initialization order is
    # not the same.
    lora_einsum_variables["params"]["w"] = einsum_variables["params"]["w"]

    y = einsum.apply(einsum_variables, x, rngs={}, method=module_call)
    y_lora = lora_einsum.apply(lora_einsum_variables, x, rngs={}, method=module_call)
    chex.assert_trees_all_close(y, y_lora)


@pytest.mark.parametrize(
    ("eqn", "lora_annotation"),
    [
        ("BSD,3KDH->3BSKH", "3KDL,3KLKH->3KDH"),
        ("BTD,NDH->BTNH", "NDL,NLNH->NDH"),
        ("BSD,2KDH->2BSKH", "2KDL,2KLKH->2KDH"),
        ("BTNH,NHD->BTD", "NHNL,NLD->NHD"),
    ],
)
def test_lora_einsum_param_merge_works(eqn: str, lora_annotation: str):
    annotation_to_dim_size = get_annotation_to_dim_size()
    x_shape, w_shape = eqn_to_shape(eqn, annotation_to_dim_size)
    einsum = gemma.Einsum(shape=w_shape, name="einsum", init_fn=nn.initializers.lecun_normal())
    lora_einsum = gemma.LoRAEinsum(
        einsum,
        gemma.LoRAConfig(rank=4, alpha=4.0),
        lora_annotation,
        nn.initializers.lecun_normal(),
        nn.initializers.lecun_normal(),
    )

    x = jax.random.uniform(jax.random.key(0), x_shape)

    def module_call(instance, x):
        return instance(eqn, x)

    lora_einsum_variables = lora_einsum.init(jax.random.key(0), x, method=module_call)
    einsum_variables = gemma.merge_lora_params(lora_einsum_variables, lambda x: lora_annotation)

    y = einsum.apply(einsum_variables, x, rngs={}, method=module_call)
    y_lora = lora_einsum.apply(lora_einsum_variables, x, rngs={}, method=module_call)
    chex.assert_trees_all_close(y, y_lora, atol=0.001)


def test_lora_einsum_params_shape():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    lora0 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2))
    lora1 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, axes=(1, 2)))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    # Ensure that lora parameters are not initialized when LoRA is not used.
    params = einsum.init(key, eqn, x)
    assert "lora_a" not in params["params"]
    assert "lora_b" not in params["params"]

    # Check that default axes work.
    params_lora0 = lora0.init(key, eqn, x)
    assert params_lora0["params"]["lora_a"].shape == (3, 8, 32, 2)
    assert params_lora0["params"]["lora_b"].shape == (3, 8, 2, 4)

    # Check that user provided axes work.
    params_lora1 = lora1.init(key, eqn, x)
    assert params_lora1["params"]["lora_a"].shape == (3, 8, 2, 4)
    assert params_lora1["params"]["lora_b"].shape == (3, 2, 32, 4)


def test_lora_einsum_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    einsum_lora = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    params = einsum.init(key, eqn, x)
    output = einsum.apply(params, eqn, x)

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    # Results are the same since the LoRA parameters are initialized to zeros.
    assert jnp.allclose(output, output_lora)


def test_lora_einsum_param_merge_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    lora_config = lora.LoRAConfig(rank=2, init_fn=nn.initializers.normal(stddev=0.01))
    einsum_lora = lora.Einsum(shape, lora_config=lora_config)

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"
    merge_eqn = "3KDL,3KLH->3KDH"

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    params = {
        "params": {
            "w": params_lora["params"]["w"]
            + jnp.einsum(merge_eqn, params_lora["params"]["lora_a"], params_lora["params"]["lora_b"])
            * lora_config.scaling_value
        }
    }
    output = jnp.einsum(eqn, x, params["params"]["w"])

    # Results are the same since the LoRA parameters are initialized to zeros.
    assert jnp.allclose(output, output_lora)
