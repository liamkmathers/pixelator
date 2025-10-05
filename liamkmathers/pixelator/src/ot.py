import numpy as np
import cv2
import ot


def luminance(img):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(y, 0.0, 1.0)


def normalize_mass(m):
    s = m.sum()
    if s <= 0:
        return np.ones_like(m) / np.prod(m.shape)
    return m / s


def downsample_avg(arr, size_hw):
    H, W = size_hw
    return cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)


def gaussian_blur(arr, k=5, sigma=1.0):
    return cv2.GaussianBlur(arr, (k, k), sigma)


def choose_coarse(H, W, max_side=64):
    if max(H, W) <= max_side:
        return H, W
    if H >= W:
        Hc = max_side
        Wc = max(2, int(round(W * (max_side / H))))
    else:
        Wc = max_side
        Hc = max(2, int(round(H * (max_side / W))))
    return Hc, Wc


def compute_displacement_from_masses(rho_s_full, rho_t_full, epsilon=0.05, lam=0.3, unbalanced=True, coarse_max_side=64, smooth_sigma=1.0):
    H, W = rho_s_full.shape
    Hc, Wc = choose_coarse(H, W, max_side=coarse_max_side)
    a = downsample_avg(rho_s_full, (Hc, Wc)).astype(np.float64)
    b = downsample_avg(rho_t_full, (Hc, Wc)).astype(np.float64)
    a = np.maximum(a, 1e-8)
    b = np.maximum(b, 1e-8)
    if not unbalanced:
        a = normalize_mass(a)
        b = normalize_mass(b)
    xs = np.linspace(0, Wc - 1, Wc)
    ys = np.linspace(0, Hc - 1, Hc)
    xx, yy = np.meshgrid(xs, ys)
    X = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    Y = X.copy()
    a_vec = a.reshape(-1)
    b_vec = b.reshape(-1)
    M = ot.utils.dist(X, Y, metric='sqeuclidean')
    M /= M.max() + 1e-8
    if unbalanced:
        gamma = ot.unbalanced.sinkhorn_unbalanced(a_vec, b_vec, M, reg=epsilon, reg_m=lam)
    else:
        gamma = ot.bregman.sinkhorn(a_vec, b_vec, M, reg=epsilon)
    Y_mat = Y
    num = gamma @ Y_mat
    den = gamma.sum(axis=1, keepdims=True) + 1e-12
    T_map = num / den
    d_coarse = T_map - X
    d_coarse = d_coarse.reshape(Hc, Wc, 2)
    dx = cv2.resize(d_coarse[..., 0], (W, H), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(d_coarse[..., 1], (W, H), interpolation=cv2.INTER_CUBIC)
    if smooth_sigma and smooth_sigma > 0.0:
        dx = gaussian_blur(dx, k=5, sigma=smooth_sigma)
        dy = gaussian_blur(dy, k=5, sigma=smooth_sigma)
    sx = W / float(Wc)
    sy = H / float(Hc)
    dx *= sx
    dy *= sy
    disp = np.stack([dx, dy], axis=-1).astype(np.float32)
    return disp
