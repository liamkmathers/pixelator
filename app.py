import os
import streamlit as st
import numpy as np
from src import morph

st.set_page_config(page_title="Pixelator OT Morph", layout="wide")
st.title("Pixelator: OT + Stable Fluids Morph")

with st.sidebar:
    st.header("Inputs")
    demo_urls = [
        "https://headshots-inc.com/wp-content/uploads/2024/09/professional-corporate-headshots-for-executive-844x1080.jpg",
        "https://b1688923.smushcdn.com/1688923/wp-content/uploads/2022/05/Melbourne-Corporate-Headshots-Julia-Nance-Portraits2.jpg?lossy=2&strip=1&webp=1",
    ]
    use_demo = st.checkbox("Use demo URLs", value=True)
    url_s = st.text_input("Source URL S", value=demo_urls[0])
    url_t = st.text_input("Target URL T", value=demo_urls[1])
    up_s = st.file_uploader("Or upload Source S", type=["png","jpg","jpeg","webp"]) 
    up_t = st.file_uploader("Or upload Target T", type=["png","jpg","jpeg","webp"]) 

    st.header("Parameters")
    res_preset = st.selectbox("Resolution", options=["256x256","384x384","512x512","768x768"], index=2)
    W = int(res_preset.split("x")[0])
    H = int(res_preset.split("x")[1])
    duration = st.slider("Duration (s)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    fps = st.slider("FPS", min_value=12, max_value=60, value=30, step=1)
    eps = st.number_input("OT entropic ε", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    unbalanced = st.checkbox("Unbalanced OT", value=True)
    lam = st.number_input("Unbalanced mass penalty λ", min_value=0.01, max_value=5.0, value=0.3, step=0.05, format="%.2f")
    alpha = st.number_input("Vorticity strength α", min_value=0.0, max_value=1.0, value=0.1, step=0.05, format="%.2f")
    coarse_max = st.select_slider("OT coarse grid max side", options=[32,48,64,96], value=64)
    make_gif = st.checkbox("Also export GIF", value=False)
    render = st.button("Render")

col_l, col_r = st.columns([1,1])
preview_slot = st.empty()
progress_bar = st.progress(0)
status_txt = st.empty()

@st.cache_data(show_spinner=False)
def read_upload(file):
    import PIL.Image
    import numpy as np
    img = PIL.Image.open(file).convert('RGB')
    return np.asarray(img)

if render:
    if up_s is not None and up_t is not None:
        source_s = read_upload(up_s)
        source_t = read_upload(up_t)
    else:
        source_s = url_s if use_demo or up_s is None else read_upload(up_s)
        source_t = url_t if use_demo or up_t is None else read_upload(up_t)

    def on_preview(i, n, frame):
        progress_bar.progress(int((i / n) * 100))
        status_txt.info(f"Rendering frame {i}/{n}")
        preview_slot.image(frame, channels="RGB")

    try:
        result = morph.run_pipeline(
            source_s,
            source_t,
            H=H,
            W=W,
            duration=duration,
            fps=fps,
            epsilon=eps,
            lam=lam,
            unbalanced=unbalanced,
            alpha=alpha,
            coarse_max=coarse_max,
            smooth_sigma=1.0,
            out_dir="outputs",
            make_gif=make_gif,
            device=None,
            preview_cb=on_preview,
        )
        progress_bar.progress(100)
        st.success("Done")
        st.video(result['mp4'])
        with open(result['mp4'], 'rb') as f:
            st.download_button("Download MP4", data=f, file_name=os.path.basename(result['mp4']), mime="video/mp4")
        if result['gif']:
            with open(result['gif'], 'rb') as f:
                st.download_button("Download GIF", data=f, file_name=os.path.basename(result['gif']), mime="image/gif")
    except Exception as e:
        st.error(f"Error: {e}")
