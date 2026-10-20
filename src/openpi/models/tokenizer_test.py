# ruff: noqa: SLF001

import numpy as np
import pytest
from scipy.fft import idct

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    rng = np.random.default_rng(0)
    state = rng.random(5).astype(np.float32)
    action = rng.random((3, 2)).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
    codec = tokenizer._fast_tokenizer
    np.testing.assert_allclose(act, codec.decode(codec(action[None]), time_horizon=3, action_dim=2)[0])


@pytest.fixture(scope="module")
def fast_tokenizer():
    return _tokenizer.FASTTokenizer(max_len=256)


@pytest.mark.parametrize("shape", [(1, 4), (20, 7)])
@pytest.mark.parametrize("suffix_only", [False, True])
def test_fast_extract_preserves_whitespace_tokens(fast_tokenizer, shape, suffix_only):
    # The final four DCT coefficients encode to a FAST token whose PaliGemma
    # representation is 28 tabs. Stripping decoded text silently loses it.
    coeff = np.zeros(shape)
    coeff.reshape(-1)[-4:] = [-1, 1, -1, 1]
    actions = idct(coeff / 10, axis=0, norm="ortho").astype(np.float32)
    codec = fast_tokenizer._fast_tokenizer
    expected = codec.decode(codec(actions[None]), time_horizon=shape[0], action_dim=shape[1])[0]
    assert np.any(expected != 0)
    tokens, _, ar_mask, _ = fast_tokenizer.tokenize("test", np.zeros(shape[1]), actions)
    if suffix_only:
        tokens = tokens[ar_mask.astype(bool)]
    np.testing.assert_allclose(fast_tokenizer.extract_actions(tokens, *shape), expected)


@pytest.mark.parametrize("ending", ["delimiter", "eos", "padding", "none"])
def test_fast_extract_preserves_adjacent_whitespace_tokens(fast_tokenizer, ending):
    # These FAST IDs map to 15 and 21 tabs. Even without strip(), a SentencePiece
    # text round trip resegments them into 31 and 5 tabs (different FAST IDs).
    codec = fast_tokenizer._fast_tokenizer
    pg = fast_tokenizer._paligemma_tokenizer
    action_ids = [1041, 1035]
    payload = fast_tokenizer._act_tokens_to_paligemma_tokens(action_ids).tolist()
    assert pg.encode(pg.decode(payload)) != payload
    action_dim = len(codec.bpe_tokenizer.decode(action_ids))
    expected = codec.decode([action_ids], time_horizon=1, action_dim=action_dim)[0]
    assert np.any(expected != 0)
    endings = {
        "delimiter": pg.encode("|") + pg.encode("ignored Action: ") + payload + [pg.eos_id()],
        "eos": [pg.eos_id()] + payload + [pg.pad_id()] * 3,
        "padding": [pg.pad_id()] * 3,
        "none": [],
    }
    tokens = np.array(pg.encode("Action: ") + payload + endings[ending])
    np.testing.assert_allclose(fast_tokenizer.extract_actions(tokens, 1, action_dim), expected)


@pytest.mark.parametrize(
    "case", ["empty", "no_marker", "partial_marker", "empty_payload", "wrong_shape", "invalid_id", "after_eos"]
)
def test_fast_extract_invalid_output_returns_zeros(fast_tokenizer, case):
    pg = fast_tokenizer._paligemma_tokenizer
    start = pg.encode("Action: ")
    end = pg.encode("|", add_eos=True)
    cases = {
        "empty": [],
        "no_marker": pg.encode("Task: test"),
        "partial_marker": start[:-1],
        "empty_payload": start + end,
        "wrong_shape": start + fast_tokenizer._act_tokens_to_paligemma_tokens([1028]).tolist() + end,
        "invalid_id": start + pg.encode("invalid") + end,
        "after_eos": [pg.eos_id(), *start, *fast_tokenizer._act_tokens_to_paligemma_tokens([1028]).tolist(), *end],
    }
    shape = (1, 4) if case == "after_eos" else (20, 7)
    actual = fast_tokenizer.extract_actions(np.array(cases[case], dtype=np.int32), *shape)
    np.testing.assert_array_equal(actual, np.zeros(shape))
