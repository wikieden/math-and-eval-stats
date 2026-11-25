"""讲义：第 41 章 Scaling law（notes/lessons/41-scaling-law.md）
运行：uv run python ch41_scaling_law.py

例子：用一个已知的幂律 L(C) = 2.0 + 30·C^{−0.1} 造“实验点”（加 1% 噪声），只用小算力的点拟合，再往大算力外推；
Chinchilla 参数化形式 L(N, D) = E + A/N^α + B/D^β 的系数取论文 Approach 3 的拟合值；6ND 计算账。
里程碑 ME-M26 的“用 Chinchilla 表里的点做拟合、给定 FLOPs 求最优 N / D、估 7B 的 GPU 小时”留给自己做。
"""

import numpy as np

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28   # Chinchilla Approach 3（Hoffmann 2022 式 (10)）


def true_loss(c):
    return 2.0 + 30.0 * np.asarray(c, float) ** -0.1


def make_points(seed: int = 0, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """算力 1e15 到 1e18 之间取 n 个点，loss 加 1% 的乘性噪声。"""
    rng = np.random.default_rng(seed)
    c = np.logspace(15, 18, n)
    return c, true_loss(c) * (1 + 0.01 * rng.standard_normal(n))


def fit_power(c, l, irreducible: float = 0.0) -> tuple[float, float]:
    """在 log-log 上拟合 L − E0 = a·C^{−b}（E0 是假定的不可约 loss）。返回 (a, b)。"""
    slope, inter = np.polyfit(np.log(c), np.log(np.asarray(l) - irreducible), 1)
    return float(np.exp(inter)), float(-slope)


def extrapolate(a: float, b: float, c, irreducible: float = 0.0):
    return irreducible + a * np.asarray(c, float) ** -b


def chinchilla_loss(n, d):
    return E + A / np.asarray(n, float) ** ALPHA + B / np.asarray(d, float) ** BETA


def opt_exponents() -> tuple[float, float]:
    """C = 6ND 固定时最小化 L(N, D)：N_opt ∝ C^{β/(α+β)}，D_opt ∝ C^{α/(α+β)}。"""
    return BETA / (ALPHA + BETA), ALPHA / (ALPHA + BETA)


def flops_6nd(n: float, d: float) -> float:
    """训练 FLOPs ≈ 6·N·D（前向 2ND、反向 4ND）。"""
    return 6 * n * d


def main() -> None:
    print("\n第一步 幂律在 log-log 上是直线")
    c, l = make_points()
    a, b = fit_power(c, l, irreducible=2.0)
    print(f"  已知不可约 loss 2.0：拟合 a = {a:.2f}、b = {b:.4f}（真值 30、0.1）")
    a0, b0 = fit_power(c, l, irreducible=0.0)
    print(f"  忘了减不可约项：拟合指数 b = {b0:.4f}（被严重低估）")

    print("\n第二步 外推：差一点的指数，放大到 1000 倍算力")
    for cc in (1e18, 1e20, 1e22):
        print(f"  C = {cc:.0e}：真值 {true_loss(cc):.4f}，正确拟合 {extrapolate(a, b, cc, 2.0):.4f}，忘减常数 {extrapolate(a0, b0, cc):.4f}")

    print("\n第三步 Chinchilla 的参数化形式 L(N, D) = E + A/N^α + B/D^β")
    for n, d in ((70e9, 1.4e12), (280e9, 300e9)):
        print(f"  N = {n / 1e9:.0f}B、D = {d / 1e9:.0f}B：L = {chinchilla_loss(n, d):.4f}，FLOPs 6ND = {flops_6nd(n, d):.2e}")
    an, ad = opt_exponents()
    print(f"  算力固定时 N_opt ∝ C^{an:.3f}，D_opt ∝ C^{ad:.3f}（Kaplan 2020 给的是 N ∝ C^0.73）")

    print("\n第四步 6ND 计算账")
    for n, d in ((1e9, 20e9), (1e9, 1e12)):
        print(f"  N = {n / 1e9:.0f}B、D = {d / 1e9:.0f}B：6ND = {flops_6nd(n, d):.2e} FLOPs，D/N = {d / n:.0f}")


if __name__ == "__main__":
    main()
