Title: Add Streamlit OT + Stable Fluids morph app (GPU-accelerated) with tests and docs

Summary (why)
- Deliver a robust, efficient image-morphing pipeline using optimal transport for structure alignment and incompressible Stable Fluids for visually pleasing motion; GPU acceleration where available.
- Provide a friendly Streamlit UI to make the method accessible with sensible defaults, live preview, and MP4/GIF export.
- Establish a modular codebase (OT, fluid, morph) with tests to ensure stability and maintainability.

Changes
- New app: app.py — Streamlit UI with uploads/URLs, parameter controls (resolution, duration, fps, ε, λ, α), progress/preview, MP4/GIF export.
- OT utilities: src/ot.py — luminance mass, coarse-grid entropic Sinkhorn (POT) for balanced/unbalanced OT, barycentric map → displacement, upsample/smooth.
- Fluid core: src/fluid.py — PyTorch Stable Fluids (divergence, gradient, Jacobi pressure solve, projection), semi-Lagrangian advection, optional vorticity confinement; GPU autodetect.
- Morph pipeline: src/morph.py — end-to-end orchestration from images to frames/video; CPU fallback and progress callbacks.
- Tests: tests/test_fluid.py — projection reduces divergence ≥10×; advecting constant field remains unchanged within tolerance.
- Dependencies/docs: requirements.txt and README.md — setup, parameters guide, performance notes, CUDA autodetect, demo image attribution.

Nature of change
- New feature; adds dependencies (torch, pot, streamlit, imageio, opencv, etc.). No breaking changes to existing code.

Impact
- Introduces a complete GPU-accelerated morphing capability that can be extended (e.g., higher-res OT, alternative solvers). CPU path remains viable for 512×512×2s.

Usage
- streamlit run app.py, then use demo URLs or uploads; adjust ε, λ, α, duration, fps; Render to preview and save MP4/GIF.

Notes
- OT runs on CPU (coarse grid keeps it fast); fluid/advection runs on CUDA when available.
- Demo images are for demonstration only; users should replace with owned/licensed images.
