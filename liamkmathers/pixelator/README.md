# Pixelator — OT Morph with Stable Fluids

A Streamlit app that morphs a source image S into a target image T by transporting pixel mass with entropic optimal transport (OT) and advecting it through an incompressible flow (Stable Fluids). The OT stage computes a coarse barycentric map between luminance mass of S and T; the resulting displacement initializes a divergence-free velocity field that drives a semi-Lagrangian morph of RGB colors. GPU acceleration is used for the fluid/advection when available; CPU fallback is provided.

## Quick start

1) Install dependencies (Python 3.10+ recommended):

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the app:

```
streamlit run app.py
```

3) Upload images or use the demo URLs, adjust parameters, click Render. The app shows a live preview and exports MP4 (and optional GIF) to `outputs/`.

## Controls

- Resolution preset: output H×W (default 512×512)
- Duration (s): total animation length
- FPS: frames per second (default 30)
- OT entropic ε: Sinkhorn regularization (default 0.05)
- Unbalanced OT: enable KL-relaxed unbalanced formulation
- Unbalanced mass penalty λ: KL penalty for mass change (default 0.3)
- Vorticity strength α: vorticity confinement strength to add subtle swirls (default 0.1)
- OT coarse grid max side: downsampled grid for OT (default 64). Larger is sharper but slower/more memory.
- Render button

## Pipeline

1. Preprocess: load S,T → RGB, resize to H×W, convert to [0,1]
2. Mass fields: compute luminance ρ_S, ρ_T. Balanced renormalizes to sum=1; unbalanced keeps raw and uses λ.
3. OT (POT): downsample masses to a coarse grid (default up to 64×64). Run entropic Sinkhorn (balanced) or unbalanced Sinkhorn with KL penalty. Compute coupling Γ and barycentric map; derive coarse displacement d(x) and upsample to full-res with smoothing.
4. Velocity: v₀ = d/τ where τ is the duration. Project to divergence-free via pressure solve (Jacobi). Optionally add vorticity confinement scaled by α.
5. Animation: semi-Lagrangian advection of RGB image each frame, with per-frame projection if forces are applied. Render and encode.

## Performance

- CUDA autodetect: if `torch.cuda.is_available()` is true, the fluid/advection runs on GPU; OT remains on CPU (coarse grid keeps it fast).
- Typical runtimes for 512×512, 2.0s, 30 FPS (varies by hardware):
  - CPU (modern laptop): ~10–25s
  - GPU (mid-range): ~3–8s

## Testing

Basic correctness tests:

```
pytest -q
```

- Projection reduces mean |div(v)| by at least 10× on random fields
- Advecting a constant field returns the same field within a small tolerance

## Notes and caveats

- OT memory/time scales with the coarse grid size. The default max side 64 keeps the cost manageable (≈16.7M entries). Increasing to 96 or 128 gives sharper displacement but will be significantly heavier.
- λ acts as a mass-change penalty in the unbalanced OT. Larger λ more strongly penalizes mass variation (approaching balanced behavior).
- For very dissimilar S and T, expect visually pleasing but not exact reconstruction. The method biases toward liquid-like flows.

## Demo image attribution

The default URLs are for demo only. Please replace with owned/licensed images.

- https://headshots-inc.com/wp-content/uploads/2024/09/professional-corporate-headshots-for-executive-844x1080.jpg
- https://b1688923.smushcdn.com/1688923/wp-content/uploads/2022/05/Melbourne-Corporate-Headshots-Julia-Nance-Portraits2.jpg?lossy=2&strip=1&webp=1

© respective owners. Used here solely for demonstration and testing purposes.
