import numpy as np

from openpi.models import tokenizer as _tokenizer


class _FakeSentencePiece:
    def encode(self, text, **kwargs):
        tokens = [sum(map(ord, piece)) % 97 + 2 for piece in text.split()]
        if kwargs.get("add_bos", False):
            tokens = [0, *tokens]
        if kwargs.get("add_eos", False):
            tokens = [*tokens, _tokenizer.PALIGEMMA_EOS_TOKEN]
        return tokens

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def _make_fake_paligemma_tokenizer(max_len=64):
    tokenizer = _tokenizer.PaligemmaTokenizer.__new__(_tokenizer.PaligemmaTokenizer)
    tokenizer._max_len = max_len  # noqa: SLF001
    tokenizer._tokenizer = _FakeSentencePiece()  # noqa: SLF001
    return tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_subtask_tokenize_training():
    tokenizer = _make_fake_paligemma_tokenizer(max_len=64)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize_subtask_training(
        "Put apple on plate", "reach for the apple", np.zeros(3, dtype=np.float32)
    )

    assert tokens.shape == (64,)
    assert token_masks.shape == (64,)
    assert ar_masks.shape == (64,)
    assert loss_masks.shape == (64,)
    assert np.sum(loss_masks) > 0
    assert np.all(ar_masks[loss_masks])


def test_subtask_tokenize_inference():
    tokenizer = _make_fake_paligemma_tokenizer(max_len=64)
    tokens, token_masks, ar_masks, loss_masks, suffix, suffix_masks = tokenizer.tokenize_subtask_inference(
        "Put apple on plate", np.zeros(3, dtype=np.float32)
    )

    assert tokens.shape == (64,)
    assert token_masks.shape == (64,)
    assert ar_masks.shape == (64,)
    assert loss_masks.shape == (64,)
    assert suffix.shape == (64,)
    assert suffix_masks.shape == (64,)
    assert not np.any(loss_masks)
    assert np.sum(suffix_masks) > 0


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
