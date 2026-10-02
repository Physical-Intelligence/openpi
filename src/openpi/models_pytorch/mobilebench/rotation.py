"""6D rotation representation (Zhou et al., CVPR 2019) and SO(3) geodesic distance."""

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812

_EPS = 1e-6


def rot6d_to_matrix(r6: Tensor) -> Tensor:
    """[..., 6] -> [..., 3, 3] via Gram-Schmidt. Columns are the orthonormalised basis.

    Degenerate inputs (near-parallel or near-zero vectors) are handled by the eps in
    normalize and by falling back to a perpendicular helper axis, so the output is
    always a valid rotation (sec. 8.3: "stable normalisation").
    """
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    # A (near-)zero first vector has no direction: fall back to the x axis instead of
    # normalising it to a zero vector (which would yield a non-rotation).
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=r6.device, dtype=r6.dtype).expand_as(a1)
    a1 = torch.where(a1.norm(dim=-1, keepdim=True) < _EPS, x_axis, a1)
    b1 = F.normalize(a1, dim=-1, eps=_EPS)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # If a2 collapsed onto b1, substitute any axis not parallel to b1.
    helper = torch.where(
        b1[..., :1].abs() < 0.9,
        torch.tensor([1.0, 0.0, 0.0], device=r6.device, dtype=r6.dtype).expand_as(b1),
        torch.tensor([0.0, 1.0, 0.0], device=r6.device, dtype=r6.dtype).expand_as(b1),
    )
    helper = helper - (b1 * helper).sum(-1, keepdim=True) * b1
    degenerate = a2.norm(dim=-1, keepdim=True) < _EPS
    b2 = F.normalize(torch.where(degenerate, helper, a2), dim=-1, eps=_EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(m: Tensor) -> Tensor:
    """[..., 3, 3] -> [..., 6]: the first two columns."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def geodesic_distance(r1: Tensor, r2: Tensor) -> Tensor:
    """Rotation angle (rad) of r1^T r2, [..., 3, 3] x2 -> [...]."""
    rel = r1.transpose(-1, -2) @ r2
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0 + _EPS, 1.0 - _EPS)
    return torch.acos(cos)
