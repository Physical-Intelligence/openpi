"""Tests for the MobileBench v1.2 modules. Each checks a property the design doc requires."""

import dataclasses

import pytest
import torch

from openpi.models_pytorch.mobilebench import losses
from openpi.models_pytorch.mobilebench import rotation
from openpi.models_pytorch.mobilebench.conditioner import SRC
from openpi.models_pytorch.mobilebench.conditioner import MobileBenchConditioner
from openpi.models_pytorch.mobilebench.conditioner import StepInputs
from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.layers import RelationalCrossAttention

CFG = MobileBenchConfig(
    vlm_width=64,
    d_model=32,
    num_heads=4,
    num_fast_slots=8,
    num_slow_slots=4,
    state_dim=10,
    exec_dim=6,
    gripper_dim=4,
    base_desc_dim=3,
    num_end_effectors=1,
    trace_lags=3,
    phase_text_dim=16,
    slow_write_every=4,
)
N_H, N_W = 12, 16


def random_rotations(*shape):
    q = torch.nn.functional.normalize(torch.randn(*shape, 4), dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def make_inputs(b=2, *, armless=False, new_episode=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ws_mask = torch.zeros(b, N_W, dtype=torch.bool) if armless else torch.ones(b, N_W, dtype=torch.bool)
    return StepInputs(
        h=r(b, N_H, CFG.vlm_width),
        h_mask=torch.ones(b, N_H, dtype=torch.bool),
        state=r(b, CFG.state_dim),
        state_mask=torch.ones(b, CFG.state_dim, dtype=torch.bool),
        exec_increment=r(b, CFG.exec_dim),
        exec_mask=torch.ones(b, CFG.exec_dim, dtype=torch.bool),
        dt=torch.full((b,), 0.5),
        gripper_desc=r(b, CFG.gripper_dim),
        base_desc=r(b, CFG.base_desc_dim),
        eef_pos=r(b, 3),
        eef_rot=random_rotations(b),
        lib_pos=r(b, N_W, 3),
        lib_rot=random_rotations(b, N_W),
        lib_cost=r(b, N_W).abs(),
        lib_margin=torch.rand(b, N_W, generator=g),
        ws_mask=ws_mask,
        ee_mask=torch.full((b, CFG.num_end_effectors), not armless),
        new_episode=torch.zeros(b, dtype=torch.bool) if new_episode is None else new_episode,
    )


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MobileBenchConditioner(CFG)


# --- rotations ----------------------------------------------------------------------------
def test_rot6d_round_trip_and_degenerate_inputs_stay_on_so3():
    rots = random_rotations(64)
    assert torch.allclose(rotation.rot6d_to_matrix(rotation.matrix_to_rot6d(rots)), rots, atol=1e-5)
    bad = torch.tensor([[0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 2, 0, 0], [1e-9, 0, 0, 0, 1e-9, 0]])
    m = rotation.rot6d_to_matrix(bad)
    assert torch.isfinite(m).all()
    eye = torch.eye(3).expand(3, 3, 3)
    assert torch.allclose(m.transpose(-1, -2) @ m, eye, atol=1e-4)
    assert torch.allclose(torch.linalg.det(m), torch.ones(3), atol=1e-4)


# --- one update ---------------------------------------------------------------------------
def test_step_shapes_and_condition_stream_layout(model):
    inp = make_inputs()
    out = model(inp, model.init_memory(2), with_readouts=True)
    e = CFG.num_end_effectors
    expected = CFG.num_fast_slots + CFG.num_slow_slots + CFG.state_tokens + N_W + e + 2 + e + 1 + 1 + 1
    assert out.cond_tokens.shape == (2, expected, CFG.d_model)
    assert out.cond_mask.shape == (2, expected) and out.cond_types.shape == (expected,)
    assert out.a_obs.points.shape == out.a_use.points.shape == (2, 2, 3)
    assert out.goal.pos.shape == (2, e, 3) and out.goal.rot.shape == (2, e, 3, 3)
    assert out.mode_logits.shape == (2, CFG.num_modes)
    assert out.trace[0].shape == (2, CFG.trace_lags, 3)
    assert out.phase_z.shape == (2, CFG.phase_text_dim)
    assert torch.isfinite(out.cond_tokens).all()


def test_slow_group_written_every_kth_update_fast_every_update(model):
    mem = model.init_memory(2)
    written = []
    for t in range(9):
        before_fast, before_slow = mem.fast.clone(), mem.slow.clone()
        out = model(make_inputs(seed=t), mem)
        mem = out.memory
        written.append(bool(out.slow_written[0]))
        assert not torch.allclose(mem.fast, before_fast), "fast group must change on every update"
        assert torch.allclose(mem.slow, before_slow) != out.slow_written[0].item()
    assert written == [t % CFG.slow_write_every == 0 for t in range(9)]


def test_new_episode_resets_only_that_slot(model):
    mem = model.init_memory(2)
    for t in range(3):
        mem = model(make_inputs(seed=t), mem).memory
    reset = model.memory.reset(mem, torch.tensor([True, False]))
    assert torch.allclose(reset.fast[0], model.memory.fast_init[0])
    assert torch.allclose(reset.slow[0], model.memory.slow_init[0])
    assert reset.step.tolist() == [0, 3]
    assert torch.allclose(reset.fast[1], mem.fast[1]) and torch.allclose(reset.slow[1], mem.slow[1])


def test_armless_robot_masks_workspace_and_nulls_goal_without_nan(model):
    model.eval()
    out = model(make_inputs(armless=True), model.init_memory(2))
    assert torch.isfinite(out.cond_tokens).all()
    ws = out.cond_types == SRC["workspace"]
    assert not out.cond_mask[:, ws].any(), "an arm-less robot exposes no workspace tokens"
    assert torch.allclose(out.capability, model.goal_ws.c_null.expand_as(out.capability))
    goal_tok = out.cond_tokens[:, out.cond_types == SRC["goal"]]
    null = model.goal_enc.slot_emb + model.goal_enc.null_emb + model.cond_type_emb.weight[SRC["goal"]]
    assert torch.allclose(goal_tok, null.expand_as(goal_tok), atol=1e-6)


def test_hard_null_replaces_invalid_point_at_inference(model):
    model.eval()
    enc = model.use_encoder
    tok = enc(torch.randn(3, 2, 3), torch.zeros(3, 2))  # gate 0 = invalid
    assert torch.allclose(tok, (enc.type_emb + enc.null_emb).expand_as(tok))


# --- training-time properties ---------------------------------------------------------------
def test_gradients_reach_both_writers_through_two_updates(model):
    mem = model.init_memory(2)
    out1 = model(make_inputs(seed=1), mem)  # step 0: slow written
    out2 = model(make_inputs(seed=2), out1.memory)  # step 1: slow not written
    out2.cond_tokens.square().mean().backward()
    for name in ("fast_writer", "slow_writer"):
        grads = [p.grad for p in getattr(model.memory, name).parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), f"no gradient reached {name}"


def test_detach_at_tbptt_boundary_cuts_the_graph_but_keeps_memory(model):
    out1 = model(make_inputs(seed=1), model.init_memory(2))
    detached = out1.memory.detach()
    assert torch.equal(detached.fast, out1.memory.fast) and not detached.fast.requires_grad
    out2 = model(make_inputs(seed=2), detached)
    model.zero_grad()
    out2.cond_tokens.square().mean().backward()
    # The slow writer only ran at step 0, which is now on the far side of the boundary.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.memory.slow_writer.parameters())


def test_trace_and_phase_readouts_read_only_their_memory_group(model):
    out = model(make_inputs(), model.init_memory(2), with_readouts=True)
    # Reproducible from the memory group alone -> no hidden path from H_t or the other group.
    assert all(torch.allclose(a, b) for a, b in zip(out.trace, model.trace_readout(out.memory.fast), strict=True))
    assert torch.allclose(out.phase_z, model.phase_readout(out.memory.slow))


def test_relational_attention_values_carry_distance():
    torch.manual_seed(0)
    attn = RelationalCrossAttention(16, 4, rel_dim=3)
    q, kv = torch.randn(1, 1, 16), torch.randn(1, 5, 16).expand(1, 5, 16)
    near = torch.zeros(1, 1, 5, 3)
    far = torch.full((1, 1, 5, 3), 5.0)
    # Same keys, same relation for every key -> identical attention weights; only the
    # relation-dependent values can tell "near" from "far".
    assert not torch.allclose(attn(q, kv, near), attn(q, kv, far))


# --- losses -----------------------------------------------------------------------------------
def test_missing_label_is_not_treated_as_invalid():
    logits = torch.tensor([[5.0, 5.0]])
    present = torch.tensor([[True, False]])
    # The absent label would be a large error if it were counted as "invalid".
    assert losses.validity_loss(logits, torch.tensor([[True, False]]), present) < 0.01


def test_point_loss_only_counts_present_and_valid_targets():
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor([[[1.0, 1, 1], [9.0, 9, 9]]])
    mask_valid = torch.tensor([[True, False]])
    present = torch.tensor([[True, True]])
    full = losses.point_loss(pred, target, torch.tensor([[True, True]]), present)
    only_nav = losses.point_loss(pred, target, mask_valid, present)
    assert only_nav < full


def test_phase_text_synonyms_are_positives_and_empty_rows_are_skipped():
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    cands = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0]]]), dim=-1)
    both_pos = torch.tensor([[True, True, False]])
    one_pos = torch.tensor([[True, False, False]])
    m = torch.ones(1, 3, dtype=torch.bool)
    on = torch.tensor([True])
    # Treating the synonym as a negative must cost more than treating it as a positive.
    assert losses.phase_text_loss(z, cands, both_pos, m, on) < losses.phase_text_loss(z, cands, one_pos, m, on)
    none = losses.phase_text_loss(z, cands, torch.zeros(1, 3, dtype=torch.bool), m, on)
    assert torch.isfinite(none) and none == 0


def test_goal_loss_is_zero_for_perfect_prediction_and_masks_absent_effectors():
    rots = random_rotations(2, 1)
    pos = torch.randn(2, 1, 3)
    r6 = rotation.matrix_to_rot6d(rots)
    t = torch.ones(2, 1, dtype=torch.bool)
    assert losses.goal_loss(pos, r6, pos, rots, t, t, t) < 1e-4
    absent = torch.zeros(2, 1, dtype=torch.bool)
    assert losses.goal_loss(pos + 3.0, r6, pos, rots, t, t, absent) == 0


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CFG.d_model = 64  # type: ignore[misc]
