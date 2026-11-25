"""讲义：第 43 章 现代优化器（notes/lessons/43-modern-optimizers.md）
运行：uv run python ch43_modern_optimizers.py

例子：病态二次函数 f = ½(100x² + y²) 上看“对角预条件”的作用；一个 4×3 的梯度矩阵上看 Shampoo 的 L^{−1/4} G R^{−1/4}
和 Muon 的 Newton-Schulz 正交化把奇异值拉到 1 附近；Lion 的符号更新；WSD 学习率调度；线性回归上的梯度噪声尺度。
里程碑 ME-M27 的“小 MLP 上对比 AdamW / Lion / Muon / SOAP 的 loss 曲线”留给自己做。
"""

import numpy as np

H = np.array([100.0, 1.0])   # 二次函数的曲率


def run(method: str, steps: int = 60) -> list[np.ndarray]:
    """从 (1, 1) 出发。gd：x ← x − 0.019·∇f（稳定上限是 2/100）；adam：β1 = 0.9、β2 = 0.999、lr = 0.05，
    每个坐标除以自己梯度的均方根——对角预条件；newton：x ← x − H⁻¹∇f。返回每步的位置。"""
    x = np.array([1.0, 1.0])
    m, v, hist = np.zeros(2), np.zeros(2), []
    for t in range(1, steps + 1):
        g = H * x
        if method == "gd":
            x = x - 0.019 * g
        elif method == "adam":
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            x = x - 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        else:
            x = x - g / H
        hist.append(x.copy())
    return hist


def mat_pow(m: np.ndarray, p: float, eps: float = 1e-12) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return (v * np.maximum(w, eps) ** p) @ v.T


def shampoo_direction(g: np.ndarray) -> np.ndarray:
    """单步 Shampoo（累积矩阵只有这一个梯度）：L = GGᵀ、R = GᵀG，方向 L^{−1/4} G R^{−1/4}。"""
    return mat_pow(g @ g.T, -0.25) @ g @ mat_pow(g.T @ g, -0.25)


def newton_schulz(g: np.ndarray, steps: int = 5) -> np.ndarray:
    """Muon 用的五次 Newton-Schulz 迭代（系数 a, b, c = 3.4445, −4.7750, 2.0315），把 G/‖G‖_F 推向 UVᵀ（奇异值都变成约 1）。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (np.linalg.norm(g) + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return x.T if tall else x


def lion_step(g: np.ndarray, m: np.ndarray, beta1: float = 0.9, beta2: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """Lion：更新方向 sign(β1·m + (1 − β1)·g)，动量 m ← β2·m + (1 − β2)·g。每个坐标的更新幅度都是 lr。"""
    u = np.sign(beta1 * m + (1 - beta1) * g)
    return u, beta2 * m + (1 - beta2) * g


def wsd(t: int, total: int = 1000, warm: int = 50, decay_frac: float = 0.1, peak: float = 1.0) -> float:
    """Warmup-Stable-Decay：线性预热 → 常数 → 最后 10% 线性降到 0。"""
    dstart = int(total * (1 - decay_frac))
    if t < warm:
        return peak * (t + 1) / warm
    if t < dstart:
        return peak
    return peak * max(0.0, (total - t) / (total - dstart))


def noise_scale(n: int = 10000, d: int = 20, seed: int = 0) -> float:
    """梯度噪声尺度 B_simple = tr(Σ)/|G|²（McCandlish 2018）：Σ 是单样本梯度的协方差，G 是平均梯度。
    线性回归 y = xᵀw* + 噪声，在 w = 0 处算。批量远小于 B 时加倍批量几乎等于加倍步数；远大于 B 时收益递减。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = x @ np.ones(d) * 0.1 + rng.standard_normal(n)
    gi = -x * y[:, None]                       # 每个样本在 w = 0 处的梯度
    g = gi.mean(0)
    return float(np.trace(np.cov(gi.T)) / (g @ g))


def main() -> None:
    print("\n第一步 Adam 是对角预条件：每个坐标按自己的梯度尺度走")
    for m in ("gd", "adam", "newton"):
        h = run(m)
        print(f"  {m}：第 10 步 (x, y) = {np.round(h[9], 3).tolist()}，第 60 步 {np.round(h[59], 3).tolist()}")

    print("\n第二步 Shampoo：左右两个矩阵预条件，等于把梯度的奇异值都变成 1")
    g = np.array([[3.0, 0.5, 0.0], [1.0, 2.0, 0.5], [0.0, 0.2, 0.1], [0.5, 0.0, 1.0]])
    print(f"  G 的奇异值 {np.round(np.linalg.svd(g, compute_uv=False), 4).tolist()}")
    print(f"  L^(−1/4) G R^(−1/4) 的奇异值 {np.round(np.linalg.svd(shampoo_direction(g), compute_uv=False), 4).tolist()}")

    print("\n第三步 Muon：不求矩阵根，用 Newton-Schulz 迭代把 G 的奇异值都推到 1 附近（0.7–1.2，不追求精确）")
    for k in (0, 1, 3, 5):
        sv = np.linalg.svd(newton_schulz(g, k), compute_uv=False)
        print(f"  {k} 步：奇异值 {np.round(sv, 4).tolist()}")

    print("\n第四步 Lion：只要符号")
    u, _ = lion_step(np.array([0.003, -2.0, 50.0]), np.zeros(3))
    print(f"  梯度 [0.003, −2, 50] → 更新方向 {u.tolist()}（每个坐标一样大）")

    print("\n第五步 WSD 调度与临界批量")
    print(f"  WSD（1000 步）：第 0、49、500、900、950、999 步 = {[round(wsd(t), 3) for t in (0, 49, 500, 900, 950, 999)]}")
    print(f"  线性回归在 w = 0 处的梯度噪声尺度 B_simple ≈ {noise_scale():.0f}")


if __name__ == "__main__":
    main()
