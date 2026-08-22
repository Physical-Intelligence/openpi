import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge

from openpi.models import gemma


def test_module_initializes_with_dropout(monkeypatch):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    config = dataclasses.replace(gemma.get_config("dummy"), depth=1)
    module = nnx_bridge.ToNNX(
        gemma.Module(
            configs=[config],
            embed_dtype="float32",
            dropout=0.1,
        )
    )

    module.lazy_init(
        rngs=nnx.Rngs(params=0, dropout=1),
        method="init",
        use_adarms=[False],
    )
