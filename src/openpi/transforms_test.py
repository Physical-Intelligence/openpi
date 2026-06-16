import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


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


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_pi05_subtask_training():
    tokenizer = _make_fake_paligemma_tokenizer(max_len=64)
    transform = _transforms.TokenizePi05SubtaskInputs(
        tokenizer,
        discrete_state_input=True,
        train_subtask_prediction=True,
    )

    data = transform(
        {
            "prompt": "Put apple on plate",
            "subtask": "reach for the apple",
            "state": np.zeros(3, dtype=np.float32),
            "actions": np.zeros((2, 3), dtype=np.float32),
        }
    )

    assert data["tokenized_prompt"].shape == (64,)
    assert data["tokenized_prompt_mask"].shape == (64,)
    assert data["token_ar_mask"].shape == (64,)
    assert data["token_loss_mask"].shape == (64,)
    assert "tokenized_action_suffix" not in data
    assert np.sum(data["token_loss_mask"]) > 0


def test_tokenize_pi05_subtask_inference():
    tokenizer = _make_fake_paligemma_tokenizer(max_len=64)
    transform = _transforms.TokenizePi05SubtaskInputs(
        tokenizer,
        discrete_state_input=True,
        sample_subtask_prediction=True,
    )

    data = transform(
        {
            "prompt": "Put apple on plate",
            "state": np.zeros(3, dtype=np.float32),
        }
    )

    assert data["tokenized_prompt"].shape == (64,)
    assert data["token_ar_mask"].shape == (64,)
    assert data["token_loss_mask"].shape == (64,)
    assert data["tokenized_action_suffix"].shape == (64,)
    assert data["tokenized_action_suffix_mask"].shape == (64,)
    assert not np.any(data["token_loss_mask"])


def test_tokenize_pi05_subtask_training_requires_subtask():
    tokenizer = _make_fake_paligemma_tokenizer(max_len=64)
    transform = _transforms.TokenizePi05SubtaskInputs(
        tokenizer,
        train_subtask_prediction=True,
    )

    with pytest.raises(ValueError, match="Subtask is required"):
        transform(
            {
                "prompt": "Put apple on plate",
                "state": np.zeros(3, dtype=np.float32),
                "actions": np.zeros((2, 3), dtype=np.float32),
            }
        )


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
