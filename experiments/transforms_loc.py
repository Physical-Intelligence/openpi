"""transforms_loc.py — bbox-sidecar join + dual-prompt tokenization for bbox.

Two DataTransformFn classes:

  LocTargetsBuilder    Loads the merged bbox sidecar parquet (output of
                       merge_bbox_sidecars.py) and, per sample, joins
                       `target_loc_tokens` + `has_bbox` + `bbox_query` by
                       (episode_index, frame_index). Placed in the
                       data_transforms.inputs group BEFORE the embodiment
                       transform so the bbox fields ride along through
                       normalization etc.

  DualPromptTokenize   Replaces the stock openpi TokenizePrompt. Emits both
                       prompts per sample:
                         tokenized_prompt           — π0.5 template
                                                      "Task: {prompt}, State: {bins}; Action: "
                                                      (for the flow-matching MSE loss).
                         tokenized_prompt_detect    — PaliGemma-native detect
                                                      "detect {bbox_query}\\n" + 4 target loc tokens
                                                      (for the loc CE loss).
                       Plus `target_loc_positions: int32[4]` indicating where
                       the 4 target tokens land in the detect prompt (the loss
                       reads logits at those positions), and `target_loc_mask:
                       bool` (a copy of `has_bbox`) so the loss can be masked
                       per-sample.

Both transforms are stateless except for the parquet/tokenizer they hold;
they are picklable and safe to use in torch dataloader workers.

Why two prompts: proved in Step 0 / probe_raw_paligemma.py that raw PaliGemma
cannot emit `<loc_NNNN>` tokens under the π0.5 template (it just echoes
"Box"). The CE supervision must therefore use PaliGemma's native `detect`
format. The two forward passes share image tokens but not language
conditioning — see Pi0WithLocCE.compute_loss.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer

log = logging.getLogger("transforms_loc")


# ---------------------------------------------------------------------------
# LocTargetsBuilder
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LocTargetsBuilder(_transforms.DataTransformFn):
    """Joins per-sample bbox supervision from the merged sidecar parquet.

    Expects the merged parquet schema from merge_bbox_sidecars.py:
      episode_index, frame_index, has_bbox, target_loc_tokens (list[int32][4]),
      query (string)

    Lookup key: (int(data['episode_index']), int(data['frame_index'])).
    These two columns are written by LeRobotDataset for every sample.

    If a sample's (ep, frame) isn't in the sidecar, treats it as has_bbox=False
    with sentinel target_loc_tokens — the loss masks it out. This is the
    safe-default behaviour for any frame whose annotation parquet doesn't
    cover it (e.g. you ran annotation on cam_high only but training pulls a
    sample whose cam_high was somehow missing).
    """

    sidecar_parquet: Path
    default_query: str = "box"

    # populated lazily on first call (after dataclass instantiation)
    _lookup: dict[tuple[int, int], dict] = dataclasses.field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._lookup is not None:
            return
        log.info("LocTargetsBuilder: loading sidecar %s", self.sidecar_parquet)
        df = pq.read_table(self.sidecar_parquet).to_pandas()
        # Build the keyed lookup. pandas iterrows is fine here — this runs once
        # per worker at process startup.
        lookup: dict[tuple[int, int], dict] = {}
        for _, row in df.iterrows():
            key = (int(row["episode_index"]), int(row["frame_index"]))
            lookup[key] = {
                "has_bbox": bool(row["has_bbox"]),
                "target_loc_tokens": np.asarray(row["target_loc_tokens"], dtype=np.int32),
                "query": str(row.get("query", self.default_query) or self.default_query),
            }
        log.info("LocTargetsBuilder: loaded %d rows (hits=%d, rate=%.1f%%)",
                 len(lookup), sum(1 for v in lookup.values() if v["has_bbox"]),
                 100.0 * sum(1 for v in lookup.values() if v["has_bbox"]) / max(len(lookup), 1))
        object.__setattr__(self, "_lookup", lookup)

    def __call__(self, data: dict) -> dict:
        self._ensure_loaded()

        # LeRobotDataset returns scalars as 0-dim arrays / torch tensors.
        # Coerce to plain Python ints for the lookup.
        ep = data.get("episode_index")
        fr = data.get("frame_index")
        if ep is None or fr is None:
            raise KeyError(
                "LocTargetsBuilder needs `episode_index` and `frame_index` in the sample dict. "
                "Make sure repack_transforms doesn't strip them."
            )
        try:
            key = (int(np.asarray(ep).item()), int(np.asarray(fr).item()))
        except (TypeError, ValueError) as e:
            raise ValueError(f"LocTargetsBuilder: bad ep/frame in sample: ep={ep!r} fr={fr!r}") from e

        hit = self._lookup.get(key)
        if hit is None:
            data["target_loc_tokens"] = np.array([-1, -1, -1, -1], dtype=np.int32)
            data["has_bbox"] = np.False_
            data["bbox_query"] = self.default_query
        else:
            data["target_loc_tokens"] = hit["target_loc_tokens"]
            data["has_bbox"] = np.asarray(hit["has_bbox"])
            data["bbox_query"] = hit["query"]
        return data


# ---------------------------------------------------------------------------
# PreserveAuxFields
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PreserveAuxFields(_transforms.DataTransformFn):
    """Wrap another transform, carrying named auxiliary fields across it.

    The colleague's AlohaSingleArmInputs (experiments/transforms.py:37) returns
    a fresh dict and drops every key that isn't in its canonical set
    (`state`, `image`, `image_mask`, `actions`, `prompt`). For bbox we
    need `target_loc_tokens`, `has_bbox`, `bbox_query` etc. to survive across
    the embodiment transform without modifying the colleague's code. This
    wrapper does that: capture the named keys before, restore them after.

    Default keys cover both directions:
      - LocTargetsBuilder outputs: target_loc_tokens, has_bbox, bbox_query
      - Original LeRobot keys that LocTargetsBuilder needs as its lookup:
        episode_index, frame_index  (only relevant if you wrap a transform
        that drops them too)
    """

    wrapped: _transforms.DataTransformFn
    keys: tuple[str, ...] = (
        "target_loc_tokens",
        "has_bbox",
        "bbox_query",
        "episode_index",
        "frame_index",
    )

    def __call__(self, data: dict) -> dict:
        carried = {k: data[k] for k in self.keys if k in data}
        out = self.wrapped(data)
        # Restore aux fields that the wrapped transform may have dropped.
        # Don't overwrite if the wrapped transform set them (it shouldn't, but be safe).
        for k, v in carried.items():
            if k not in out:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# DualPromptTokenize
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DualPromptTokenize(_transforms.DataTransformFn):
    """Tokenize BOTH the π0.5 template AND PaliGemma-native detect prompt.

    Inputs (in data dict):
      prompt              str       (popped) — used for the π0.5 template.
      state               array     (kept)   — digitized by pi05 tokenizer.
      bbox_query          str       (kept)   — used for "detect {query}\\n".
      target_loc_tokens   int32[4]  (kept)   — appended to detect prompt.
      has_bbox            bool      (kept)   — copied to target_loc_mask.

    Outputs added:
      tokenized_prompt              int32[L1]   π0.5 template
      tokenized_prompt_mask         bool[L1]
      tokenized_prompt_detect       int32[L2]   detect prompt + 4 targets
      tokenized_prompt_detect_mask  bool[L2]
      target_loc_positions          int32[4]    indices of the 4 target tokens
      target_loc_mask               bool        loss mask (= has_bbox)
    """

    pi05_tokenizer: _tokenizer.PaligemmaTokenizer
    spm: sentencepiece.SentencePieceProcessor
    detect_max_len: int = 32  # "detect box\\n" tokens (~5) + 4 target tokens = ~9, pad to 32

    def _tokenize_detect(self, query: str, target_loc_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build detect_prompt + 4-target sequence, padded to detect_max_len.
        Returns (tokens, mask, target_positions[4])."""
        cleaned = str(query).strip().lower().replace("_", " ")
        detect_text = f"detect {cleaned}\n"
        # Match PaligemmaTokenizer.tokenize's pi0 branch: BOS + text-with-\n encoded together
        # (the SPM encoder treats trailing \n as a separate token).
        prompt_ids = self.spm.encode(detect_text, add_bos=True)
        # Append the 4 target loc-token IDs verbatim (they're already PaliGemma vocab ids
        # in [256000, 257024), or [-1, -1, -1, -1] when has_bbox=False).
        # When has_bbox=False we use a benign in-range placeholder so the embedder doesn't
        # see a negative id — the loss mask blocks gradient anyway.
        targets = target_loc_tokens.tolist()
        if any(t < 0 for t in targets):
            # placeholder: <loc0000>..<loc0003>; harmless because target_loc_mask masks the loss
            targets = [256000, 256000, 256000, 256000]
        full = prompt_ids + targets

        n = min(len(full), self.detect_max_len)
        if len(full) > self.detect_max_len:
            log.warning("detect prompt %d tokens > detect_max_len=%d, truncating", len(full), self.detect_max_len)
            full = full[: self.detect_max_len]

        tokens = np.zeros(self.detect_max_len, dtype=np.int32)
        tokens[:n] = np.asarray(full[:n], dtype=np.int32)
        mask = np.zeros(self.detect_max_len, dtype=bool)
        mask[:n] = True

        # The 4 target tokens are the LAST 4 of the unpadded prefix.
        target_positions = np.array([n - 4, n - 3, n - 2, n - 1], dtype=np.int32)
        return tokens, mask, target_positions

    def __call__(self, data: dict) -> dict:
        # ----- π0.5 template branch -----
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("DualPromptTokenize: 'prompt' is required")
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        state = data.get("state")
        if state is None:
            raise ValueError("DualPromptTokenize: 'state' is required for the pi05 template")
        tokens, mask = self.pi05_tokenizer.tokenize(prompt, state)

        # ----- PaliGemma-native detect branch -----
        # Pop bbox_query (string) so it doesn't survive into the JAX batch — JAX
        # array conversion can't handle non-numeric dtypes.
        query = data.pop("bbox_query", "box")
        target_loc_tokens = data.get("target_loc_tokens")
        if target_loc_tokens is None:
            raise KeyError(
                "DualPromptTokenize: 'target_loc_tokens' missing — did LocTargetsBuilder run before this transform?"
            )
        target_loc_tokens = np.asarray(target_loc_tokens, dtype=np.int32)
        det_tokens, det_mask, target_positions = self._tokenize_detect(query, target_loc_tokens)

        # ----- loss mask -----
        target_loc_mask = np.asarray(data.get("has_bbox", False), dtype=bool)

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "tokenized_prompt_detect": det_tokens,
            "tokenized_prompt_detect_mask": det_mask,
            "target_loc_positions": target_positions,
            "target_loc_mask": target_loc_mask,
        }
