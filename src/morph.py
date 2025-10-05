import os
import time
import numpy as np
from PIL import Image
import requests
import io
import torch
from . import ot as ot_utils
from . import fluid
import imageio


def load_image(source, size_hw):
    H, W = size_hw
    if isinstance(source, np.ndarray):
        img = Image.fromarray((np.clip(source, 0, 1) * 255).astype(np.uint8)) if source.dtype != np.uint8 else Image.fromarray(source)
    elif isinstance(source, (str, bytes)):
        s = str(source)
        if s.startswith('http://') or s.startswith('https://'):
            r = requests.get(s, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert('RGB')
        else:
            img = Image.open(s).convert('RGB')
    else:
        raise ValueError('Unsupported image source')
    img = img.convert('RGB').resize((W, H), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def to_torch_img(img_np, device):
    t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    return t.to(device)


def to_numpy_img(img_t):
    arr = img_t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def compute_masses(img_s, img_t, balanced=True):
    rho_s = ot_utils.luminance(img_s)
    rho_t = ot_utils.luminance(img_t)
    rho_s = np.clip(rho_s, 1e-6, 1.0)
    rho_t = np.clip(rho_t, 1e-6, 1.0)
    if balanced:
        rho_s = ot_utils.normalize_mass(rho_s)
        rho_t = ot_utils.normalize_mass(rho_t)
    return rho_s, rho_t


def compute_velocity_from_ot(img_s, img_t, duration, epsilon=0.05, lam=0.3, unbalanced=True, coarse_max=64, smooth_sigma=1.0, device=None):
    if device is None:
        device = fluid.device_auto()
    H, W, _ = img_s.shape
    rho_s, rho_t = compute_masses(img_s, img_t, balanced=not unbalanced)
    disp = ot_utils.compute_displacement_from_masses(rho_s, rho_t, epsilon=epsilon, lam=lam, unbalanced=unbalanced, coarse_max_side=coarse_max, smooth_sigma=smooth_sigma)
    v0 = disp / max(1e-6, float(duration))
    vx = torch.from_numpy(v0[..., 0]).to(device=device, dtype=torch.float32)
    vy = torch.from_numpy(v0[..., 1]).to(device=device, dtype=torch.float32)
    v = torch.stack([vx, vy], dim=0)
    v = fluid.project(v, max_iters=80)
    return v


def render_morph(img_s, img_t, duration=2.0, fps=30, epsilon=0.05, lam=0.3, unbalanced=True, alpha=0.1, coarse_max=64, smooth_sigma=1.0, device=None, save_mp4_path=None, save_gif_path=None, preview_cb=None):
    if device is None:
        device = fluid.device_auto()
    H, W, _ = img_s.shape
    v = compute_velocity_from_ot(img_s, img_t, duration=duration, epsilon=epsilon, lam=lam, unbalanced=unbalanced, coarse_max=coarse_max, smooth_sigma=smooth_sigma, device=device)
    dt = 1.0 / float(fps)
    frames = []
    color = to_torch_img(img_s, device=device)
    num_frames = max(1, int(round(duration * fps)))
    v_step = v.clone()
    for i in range(num_frames):
        v_step = fluid.step_velocity(v_step, alpha=alpha, dt=dt, max_iters=60)
        color = fluid.advect(color, v_step, dt)
        frame = to_numpy_img(color)
        frames.append(frame)
        if preview_cb is not None:
            preview_cb(i + 1, num_frames, frame)
    mp4_written = None
    gif_written = None
    if save_mp4_path is not None:
        writer = imageio.get_writer(save_mp4_path, fps=fps, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        mp4_written = save_mp4_path
    if save_gif_path is not None:
        imageio.mimsave(save_gif_path, frames, fps=fps)
        gif_written = save_gif_path
    return frames, mp4_written, gif_written


def run_pipeline(source_s, source_t, H=512, W=512, duration=2.0, fps=30, epsilon=0.05, lam=0.3, unbalanced=True, alpha=0.1, coarse_max=64, smooth_sigma=1.0, out_dir="outputs", make_gif=False, device=None, preview_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    img_s = load_image(source_s, (H, W))
    img_t = load_image(source_t, (H, W))
    ts = int(time.time())
    mp4_path = os.path.join(out_dir, f"morph_{ts}.mp4")
    gif_path = os.path.join(out_dir, f"morph_{ts}.gif") if make_gif else None
    frames, mp4_written, gif_written = render_morph(
        img_s,
        img_t,
        duration=duration,
        fps=fps,
        epsilon=epsilon,
        lam=lam,
        unbalanced=unbalanced,
        alpha=alpha,
        coarse_max=coarse_max,
        smooth_sigma=smooth_sigma,
        device=device,
        save_mp4_path=mp4_path,
        save_gif_path=gif_path,
        preview_cb=preview_cb,
    )
    return {
        'frames': frames,
        'mp4': mp4_written,
        'gif': gif_written,
        'H': H,
        'W': W,
        'fps': fps,
        'duration': duration,
    }
