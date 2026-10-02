"""Auxiliary losses of design doc sec. 9.3-9.4 (the flow-matching action loss lives with the action heads).

Two masks are kept apart everywhere:
  label_present  a label exists for this sample / slot
  valid_target   the label says the quantity is valid (e.g. the point is knowable)
"No label" and "truly invalid" are different states: validity BCE uses only
label_present; geometric errors use label_present & valid_target. Predicted confidence
never multiplies a geometric loss, so the model cannot dodge it by being unsure (sec. 9.3).
"""

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.mobilebench.rotation import geodesic_distance
from openpi.models_pytorch.mobilebench.rotation import rot6d_to_matrix


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over masked entries; an exact 0 that keeps the graph when the mask is empty."""
    m = mask.to(values.dtype)
    denom = m.sum()
    return (values * m).sum() / denom.clamp_min(1.0)


def point_loss(pred: Tensor, target: Tensor, valid_target: Tensor, label_present: Tensor, delta: float = 0.1) -> Tensor:
    """Huber point error per affordance type, summed over types.

    pred/target [B, T, 3] (metres, current body frame); valid_target/label_present [B, T].
    Used for both L_aff^obs (current-observation labels) and L_aff^use (task labels):
    when a target is invisible but historically known, the CURRENT branch gets
    valid_target = False there while the JOINT branch keeps supervising the point.
    """
    err = F.huber_loss(pred, target, delta=delta, reduction="none").sum(-1)  # [B, T]
    mask = valid_target & label_present
    return sum(masked_mean(err[:, j], mask[:, j]) for j in range(err.shape[1]))


def validity_loss(logits: Tensor, valid_target: Tensor, label_present: Tensor) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, valid_target.to(logits.dtype), reduction="none")
    return masked_mean(bce, label_present)


def goal_loss(
    pred_pos: Tensor,
    pred_rot6d: Tensor,
    target_pos: Tensor,
    target_rot: Tensor,
    valid_target: Tensor,
    label_present: Tensor,
    ee_mask: Tensor,
    lambda_p: float = 1.0,
    lambda_r: float = 1.0,
    delta: float = 0.05,
) -> Tensor:
    """lambda_p Huber(p - p*) + lambda_R d_SO3(R, R*)^2, masked by v^G* (sec. 9.4).

    Valid goals are supervised on NAV frames too, expressed in the CURRENT body frame --
    never pre-transformed into a future near-target body frame.
    """
    pos = F.huber_loss(pred_pos, target_pos, delta=delta, reduction="none").sum(-1)
    rot = geodesic_distance(rot6d_to_matrix(pred_rot6d), target_rot) ** 2
    mask = valid_target & label_present & ee_mask
    return lambda_p * masked_mean(pos, mask) + lambda_r * masked_mean(rot, mask)


def trace_loss(
    pred_pos: Tensor,
    pred_rot6d: Tensor,
    pred_grip: Tensor,
    target_pos: Tensor,
    target_rot: Tensor,
    target_grip: Tensor,
    available: Tensor,
    delta: float = 0.05,
) -> Tensor:
    """Past-EEF recall error, masked where not enough history exists (never placeholder pasts)."""
    pos = F.huber_loss(pred_pos, target_pos, delta=delta, reduction="none").sum(-1)
    rot = geodesic_distance(rot6d_to_matrix(pred_rot6d), target_rot) ** 2
    grip = (pred_grip - target_grip).abs()
    return masked_mean(pos, available) + masked_mean(rot, available) + masked_mean(grip, available)


def phase_text_loss(
    z: Tensor, candidates: Tensor, positive: Tensor, candidate_mask: Tensor, active: Tensor, tau: float = 0.07
) -> Tensor:
    """Multi-positive contrastive alignment of the slow readout to frozen text teachers (sec. 7).

        L = -log( sum_{j in P_t} exp(<z, e_j>/tau) / sum_{j in C_t} exp(<z, e_j>/tau) )

    z [B, D] normalised; candidates [B, C, D] normalised teacher embeddings (stop-grad:
    the text encoder is frozen); positive [B, C] marks semantically equivalent
    descriptions -- synonyms are NOT negatives of each other; candidate_mask [B, C]
    valid candidates; active [B] slots to supervise (normally: slow group written this
    update, and at least one positive present). Hard negatives should change task facts
    (object, holding state, destination), not just wording.
    """
    logits = torch.einsum("bd,bcd->bc", z, candidates.detach()) / tau
    neg_inf = torch.finfo(logits.dtype).min
    all_l = logits.masked_fill(~candidate_mask, neg_inf)
    pos_l = logits.masked_fill(~(positive & candidate_mask), neg_inf)
    per = torch.logsumexp(all_l, dim=-1) - torch.logsumexp(pos_l, dim=-1)
    ok = active & (positive & candidate_mask).any(dim=-1)
    # Rows with no positive are huge (not inf, since neg_inf is finite) -- zero them with
    # `where` rather than relying on value * 0, which is NaN if anything overflowed.
    return masked_mean(torch.where(ok, per, torch.zeros_like(per)), ok)


def mode_loss(logits: Tensor, target: Tensor, label_present: Tensor) -> Tensor:
    ce = F.cross_entropy(logits, target.clamp_min(0), reduction="none")
    return masked_mean(ce, label_present)
