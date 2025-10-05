import torch
from src import fluid

def test_projection_reduces_divergence():
    torch.manual_seed(0)
    H, W = 64, 64
    v = torch.randn(2, H, W) * 0.5
    div_before = torch.mean(torch.abs(fluid.divergence(v))).item()
    v_proj = fluid.project(v, max_iters=120)
    div_after = torch.mean(torch.abs(fluid.divergence(v_proj))).item()
    assert div_after <= div_before / 10.0


def test_advect_constant_field_stability():
    H, W = 64, 64
    C = 3
    field = torch.ones(C, H, W)
    v = torch.randn(2, H, W) * 2.0
    out = fluid.advect(field, v, dt=0.5)
    err = torch.mean(torch.abs(out - field)).item()
    assert err < 1e-5
