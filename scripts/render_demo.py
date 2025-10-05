import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src import morph

DEMO_S = "https://headshots-inc.com/wp-content/uploads/2024/09/professional-corporate-headshots-for-executive-844x1080.jpg"
DEMO_T = "https://b1688923.smushcdn.com/1688923/wp-content/uploads/2022/05/Melbourne-Corporate-Headshots-Julia-Nance-Portraits2.jpg?lossy=2&strip=1&webp=1"

def main():
    H, W = 512, 512
    duration = 2.0
    fps = 30
    epsilon = 0.05
    lam = 0.3
    unbalanced = True
    alpha = 0.1
    coarse_max = 64

    result = morph.run_pipeline(
        DEMO_S,
        DEMO_T,
        H=H,
        W=W,
        duration=duration,
        fps=fps,
        epsilon=epsilon,
        lam=lam,
        unbalanced=unbalanced,
        alpha=alpha,
        coarse_max=coarse_max,
        smooth_sigma=1.0,
        out_dir=os.path.join(ROOT, "outputs"),
        make_gif=False,
        device=None,
        preview_cb=None,
    )
    print(json.dumps({"mp4": result["mp4"], "fps": result["fps"], "H": result["H"], "W": result["W"], "duration": result["duration"]}))


if __name__ == "__main__":
    main()
