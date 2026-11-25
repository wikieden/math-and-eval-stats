"""讲义：第 49 章 量化的数学（notes/lessons/49-quantization.md）
运行：uv run python ch49_quantization.py

例子：一个 8 维的小向量做 int4 非对称量化；舍入误差的方差 Δ²/12；一个离群值把整段的步长撑大；SmoothQuant 的等价变换；
两个相关权重上的 OBS / GPTQ 式误差补偿；FP4（E2M1）能表示的数。
里程碑 ME-M30 的“对一个权重矩阵比 per-tensor / per-channel / group int4 的误差”和“实现简化 GPTQ”留给自己做。
"""

import numpy as np

V = np.array([-0.62, -0.31, -0.05, 0.0, 0.12, 0.40, 0.55, 0.91])


def quant_asym(x: np.ndarray, bits: int = 4) -> tuple[np.ndarray, float, int]:
    """非对称均匀量化：scale = (max − min)/(2^b − 1)，zero_point = round(−min/scale)，q = clip(round(x/scale) + zp)。返回 (反量化值, scale, zp)。"""
    qmax = 2 ** bits - 1
    scale = (x.max() - x.min()) / qmax
    zp = int(round(-x.min() / scale))
    q = np.clip(np.round(x / scale) + zp, 0, qmax)
    return (q - zp) * scale, float(scale), zp


def quant_sym(x: np.ndarray, bits: int = 4) -> np.ndarray:
    """对称量化：scale = max|x|/(2^{b−1} − 1)，没有零点。"""
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max() / qmax
    return np.clip(np.round(x / s), -qmax - 1, qmax) * s


def rounding_noise(bits: int = 8, n: int = 200000, seed: int = 0) -> tuple[float, float]:
    """均匀量化的舍入误差近似 U(−Δ/2, Δ/2)，方差 Δ²/12。返回 (实测方差, Δ²/12)。"""
    x = np.random.default_rng(seed).uniform(-1, 1, n)
    s = 2 / (2 ** bits - 1)
    err = np.round(x / s) * s - x
    return float(err.var()), s * s / 12


def outlier_effect(seed: int = 0) -> tuple[float, float]:
    """1024 个 N(0, 1) 的数，int8 对称量化；再把其中一个改成 60（离群值）。返回两种情况下其余数的均方误差。"""
    x = np.random.default_rng(seed).standard_normal(1024)
    e0 = ((quant_sym(x, 8) - x) ** 2)[1:].mean()
    y = x.copy()
    y[0] = 60.0
    e1 = ((quant_sym(y, 8) - y) ** 2)[1:].mean()
    return float(e0), float(e1)


def smoothquant(alpha: float = 0.5, seed: int = 0) -> tuple[float, float, float, float]:
    """X 的第 0 列（通道）是离群通道。s_j = max|X_j|^α / max|W_j|^{1−α}，X' = X/s、W' = s·W：X'W' = XW 精确相等，
    但激活的最大值被压下来（难度转移给权重）。返回 (变换前后输出最大差, X 的最大值, X' 的最大值, W' 的最大值)。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((16, 8))
    x[:, 0] *= 50
    w = rng.standard_normal((8, 4)) * 0.1
    s = np.abs(x).max(0) ** alpha / np.abs(w).max(1) ** (1 - alpha)
    x2, w2 = x / s, w * s[:, None]
    return float(np.abs(x @ w - x2 @ w2).max()), float(np.abs(x).max()), float(np.abs(x2).max()), float(np.abs(w2).max())


def obs_two_weights() -> tuple[float, float]:
    """输出误差 E = (Δw)ᵀ H (Δw)，H = XXᵀ（两个输入高度相关）。先把 w1 量化（取整到 0.5 的倍数），
    不补偿：w2 不动；OBS / GPTQ 补偿：δw = −(w1 − q1)/H⁻¹[0,0]·H⁻¹[:,0]，即 δw2 = Δw1·H⁻¹[1,0]/H⁻¹[0,0]，让剩下的权重吸收误差
    （等价于在 Δw1 固定时对 Δw2 求 ΔwᵀHΔw 的最小值：Δw2 = −H[1,0]·Δw1/H[1,1]）。返回 (不补偿, 补偿) 的 E。"""
    h = np.array([[1.0, 0.9], [0.9, 1.0]])
    w = np.array([0.3, 0.7])
    w1q = np.round(w[0] / 0.5) * 0.5
    d1 = w1q - w[0]
    hinv = np.linalg.inv(h)
    dw_no = np.array([d1, 0.0])
    dw_obs = np.array([d1, d1 * hinv[1, 0] / hinv[0, 0]])
    return float(dw_no @ h @ dw_no), float(dw_obs @ h @ dw_obs)


def fp4_e2m1_values() -> list[float]:
    """E2M1：1 位符号、2 位指数（偏置 1）、1 位尾数。正数部分：次正规 0、0.5，正规 1、1.5、2、3、4、6。"""
    vals = {0.0, 0.5}
    for e in (1, 2, 3):
        for m in (0, 1):
            vals.add(2 ** (e - 1) * (1 + m / 2))
    return sorted(vals)


def main() -> None:
    print("\n第一步 均匀量化：scale 和 zero point")
    deq, s, zp = quant_asym(V)
    print(f"  x = {V.tolist()}")
    print(f"  int4 非对称：scale = {s:.4f}，zero point = {zp}；反量化 {np.round(deq, 4).tolist()}")

    print("\n第二步 舍入误差的方差是 Δ²/12")
    for b in (4, 8):
        m, t = rounding_noise(b)
        print(f"  {b} bit（[−1, 1]）：实测 {m:.3e}，Δ²/12 = {t:.3e}")

    print("\n第三步 离群值把步长撑大")
    e0, e1 = outlier_effect()
    print(f"  int8 对称量化 1024 个 N(0, 1)：其余数的均方误差 {e0:.2e}；混进一个 60 后 {e1:.2e}（{e1 / e0:.0f} 倍）")

    print("\n第四步 SmoothQuant：把激活的难度挪给权重")
    d, xm, x2m, w2m = smoothquant()
    print(f"  输出最大差 {d:.1e}（数学上等价）；激活最大值 {xm:.1f} → {x2m:.2f}，权重最大值变成 {w2m:.2f}")

    print("\n第五步 GPTQ 的核心：量化一个权重，让其余权重补偿")
    no, obs = obs_two_weights()
    print(f"  两个高度相关的输入（H = [[1, 0.9], [0.9, 1]]）：只量化 w1 不补偿，输出误差 {no:.4f}；按 H⁻¹ 补偿 w2 后 {obs:.4f}")
    print(f"  FP4（E2M1）的正数：{fp4_e2m1_values()}")


if __name__ == "__main__":
    main()
