"""The batched make_pretext_batch must be bit-identical to the original per-sample version (same outputs, same
labels, same RNG state afterwards), so CPU results of every experiment stay unchanged."""
import pytest
import torch
import torch.nn.functional as F

from whssl.ssl.multitask import TASKS, make_pretext_batch


# ---- the original per-sample implementation (commit 0d7ee21), kept verbatim as the reference ----
def _permute(x, n_seg=4):
    segs = list(torch.tensor_split(x, n_seg, dim=-1))
    order = torch.randperm(n_seg)
    while torch.equal(order, torch.arange(n_seg)):
        order = torch.randperm(n_seg)
    return torch.cat([segs[i] for i in order], dim=-1)


def _time_warp(x, strength=0.35):
    T = x.shape[-1]
    knots = torch.rand(6) * 2 * strength + (1 - strength)
    speed = F.interpolate(knots.view(1, 1, -1), size=T, mode="linear", align_corners=True).view(-1)
    pos = torch.cumsum(speed, 0)
    pos = (pos - pos[0]) / (pos[-1] - pos[0]) * (T - 1)
    lo = pos.floor().long().clamp(0, T - 2)
    w = (pos - lo).unsqueeze(0)
    return x[:, lo] * (1 - w) + x[:, lo + 1] * w


def reference_make_pretext_batch(xb):
    B = xb.shape[0]
    labels = {t: (torch.rand(B) < 0.5) for t in TASKS}
    out = xb.clone()
    for i in range(B):
        x = out[i]
        if labels["aot"][i]:
            x = torch.flip(x, dims=[-1])
        if labels["perm"][i]:
            x = _permute(x)
        if labels["warp"][i]:
            x = _time_warp(x)
        out[i] = x
    return out, {t: v.float() for t, v in labels.items()}
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("B,T", [(1, 150), (7, 150), (256, 150), (64, 151), (32, 300)])
def test_batched_pretext_is_bit_identical(B, T):
    for seed in range(25):
        xb = torch.randn(B, 3, T, generator=torch.Generator().manual_seed(1000 + seed))
        torch.manual_seed(seed)
        ref, ref_lab = reference_make_pretext_batch(xb)
        ref_state = torch.get_rng_state()
        torch.manual_seed(seed)
        new, new_lab = make_pretext_batch(xb)
        assert torch.equal(new, ref), f"outputs differ (seed {seed}, B {B}, T {T})"
        assert all(torch.equal(new_lab[t], ref_lab[t]) for t in TASKS)
        assert torch.equal(torch.get_rng_state(), ref_state), "RNG stream consumed differently"
