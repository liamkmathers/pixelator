import torch
import torch.nn.functional as F


def device_auto():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def divergence(v):
    u = v[0]
    w = v[1]
    u_left = F.pad(u, (1, 0), mode="replicate")[:, :, :-1]
    w_up = F.pad(w, (0, 0, 1, 0), mode="replicate")[:, :-1, :]
    du_dx = u - u_left
    dw_dy = w - w_up
    return du_dx + dw_dy


def gradient(p):
    p_left = F.pad(p, (1, 0), mode="replicate")[:, :, :-1]
    p_up = F.pad(p, (0, 0, 1, 0), mode="replicate")[:, :-1, :]
    dp_dx = p - p_left
    dp_dy = p - p_up
    return torch.stack([dp_dx, dp_dy], dim=0)


def laplace_jacobi_solve(b, max_iters=80, tol=0.0):
    device = b.device
    H, W = b.shape[-2:]
    p = torch.zeros((H, W), device=device, dtype=b.dtype)
    for _ in range(max_iters):
        p_pad = F.pad(p, (1, 1, 1, 1), mode="replicate")
        p_new = (p_pad[1:-1, 2:] + p_pad[1:-1, :-2] + p_pad[2:, 1:-1] + p_pad[:-2, 1:-1] - b) * 0.25
        if tol > 0.0:
            if torch.mean(torch.abs(p_new - p)) < tol:
                p = p_new
                break
        p = p_new
    return p


def project(v, max_iters=80, tol=0.0):
    div = divergence(v)
    p = laplace_jacobi_solve(div, max_iters=max_iters, tol=tol)
    grad_p = gradient(p)
    return v - grad_p


def curl(v):
    u = v[0]
    w = v[1]
    w_left = F.pad(w, (1, 0), mode="replicate")[:, :, :-1]
    u_up = F.pad(u, (0, 0, 1, 0), mode="replicate")[:, :-1, :]
    dw_dx = w - w_left
    du_dy = u - u_up
    return dw_dx - du_dy


def vorticity_confinement(v, alpha=0.1):
    w = curl(v)
    abs_w = torch.abs(w)
    gx = abs_w - F.pad(abs_w, (1, 0), mode="replicate")[:, :, :-1]
    gy = abs_w - F.pad(abs_w, (0, 0, 1, 0), mode="replicate")[:, :-1, :]
    mag = torch.sqrt(gx * gx + gy * gy) + 1e-6
    nx = gx / mag
    ny = gy / mag
    f_x = ny * w
    f_y = -nx * w
    f = torch.stack([f_x, f_y], dim=0)
    return alpha * f


def _make_base_grid(H, W, device, dtype):
    ys = torch.linspace(0.0, H - 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(0.0, W - 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def _to_normalized(xx, yy, H, W):
    x_norm = 2.0 * (xx / (W - 1.0)) - 1.0
    y_norm = 2.0 * (yy / (H - 1.0)) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def advect(field, v, dt):
    if field.dim() == 3:
        C, H, W = field.shape
        field_b = field.unsqueeze(0)
    else:
        _, C, H, W = field.shape
        field_b = field
    device = field_b.device
    dtype = field_b.dtype
    xx, yy = _make_base_grid(H, W, device, dtype)
    vx = v[0]
    vy = v[1]
    px = xx - vx * dt
    py = yy - vy * dt
    px = torch.clamp(px, 0.0, W - 1.0)
    py = torch.clamp(py, 0.0, H - 1.0)
    grid = _to_normalized(px, py, H, W)
    grid = grid.unsqueeze(0)
    out = F.grid_sample(field_b, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(0)


def step_velocity(v, alpha=0.0, dt=1.0, max_iters=60):
    if alpha != 0.0:
        v = v + vorticity_confinement(v, alpha) * dt
    v = project(v, max_iters=max_iters)
    return v
