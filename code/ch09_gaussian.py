"""讲义：第 9 章 高斯分布（notes/lessons/09-gaussian.md）
运行：uv run python ch09_gaussian.py
"""

import math

import numpy as np
from scipy.stats import norm

SIGMA = np.array([[2.0, 1.2], [1.2, 1.0]])  # 讲义里的 2×2 协方差（编的）
MU = np.array([0.0, 0.0])


def within_k_sigma(k: float) -> float:
    """第一步：一维标准正态落在 ±kσ 内的概率。"""
    return float(norm.cdf(k) - norm.cdf(-k))


def correlation(sigma: np.ndarray) -> float:
    """第二步：相关系数 ρ = Σ12 / sqrt(Σ11·Σ22)。"""
    return float(sigma[0, 1] / math.sqrt(sigma[0, 0] * sigma[1, 1]))


def ellipse_axes(sigma: np.ndarray, k: float = 1.0):
    """第二步：协方差椭圆 = 特征分解。返回 (半轴长 k·sqrt(λ), 主轴方向)。"""
    lam, vec = np.linalg.eigh(sigma)
    vec = vec * np.sign(vec[0])  # 方向向量的正负号是任意的，统一让 x 分量为正
    return k * np.sqrt(lam), vec


def sample_via_cholesky(mu, sigma, n, seed=0):
    """第三步：x = μ + L z，L = cholesky(Σ)，z ~ N(0, I)。L Lᵀ = Σ。"""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(sigma)
    z = rng.standard_normal((n, len(mu)))
    return mu + z @ L.T


def diffusion_forward(x0: float, alpha_bar: float):
    """第三步：扩散前向 x_t = sqrt(ᾱ)·x0 + sqrt(1-ᾱ)·ε 的均值和方差。"""
    return math.sqrt(alpha_bar) * x0, 1 - alpha_bar


def conditional(mu, sigma, x2: float):
    """第四步：x1 | x2 的均值 μ1 + Σ12/Σ22·(x2-μ2) 和方差 Σ11 - Σ12²/Σ22。"""
    m = mu[0] + sigma[0, 1] / sigma[1, 1] * (x2 - mu[1])
    v = sigma[0, 0] - sigma[0, 1] ** 2 / sigma[1, 1]
    return float(m), float(v)


def gaussian_log_prob(a: float, mu: float, sigma: float) -> float:
    """第五步：高斯动作头的 log 概率 = -(a-μ)²/(2σ²) - log σ - ½ log 2π。"""
    return -((a - mu) ** 2) / (2 * sigma**2) - math.log(sigma) - 0.5 * math.log(2 * math.pi)


def main() -> None:
    print("第一步 一维高斯：±1σ、±2σ、±3σ 内的概率")
    print("  " + "、".join(f"{within_k_sigma(k):.4f}" for k in (1, 2, 3)))
    print("\n第二步 二维协方差 Σ = [[2, 1.2], [1.2, 1]]")
    print(f"  ρ = {correlation(SIGMA):.3f}")
    axes, vec = ellipse_axes(SIGMA)
    print(f"  特征值 = {np.round(axes**2, 4)}，1σ 椭圆半轴 = {np.round(axes, 3)}")
    print(f"  长轴方向 = {np.round(vec[:, 1], 3)}（和 x 轴夹角 {math.degrees(math.atan2(vec[1, 1], vec[0, 1])):.1f}°）")
    print("\n第三步 线性变换与采样")
    L = np.linalg.cholesky(SIGMA)
    print(f"  L = {np.round(L, 4).tolist()}")
    x = sample_via_cholesky(MU, SIGMA, 200_000)
    print(f"  20 万个样本的协方差 ≈ {np.round(np.cov(x.T), 3).tolist()}")
    print(f"  扩散前向 x0 = 2，ᾱ = 0.36：均值 {diffusion_forward(2, 0.36)[0]:.2f}，方差 {diffusion_forward(2, 0.36)[1]:.2f}")
    print("\n第四步 条件分布：已知 x2 = 1.5")
    m, v = conditional(MU, SIGMA, 1.5)
    print(f"  x1 | x2 = 1.5 ~ N({m:.2f}, {v:.2f})  ← 方差从 2 降到 {v:.2f}")
    print("\n第五步 高斯动作头的 log 概率：μ = 0.2，σ = 0.1")
    for a in (0.2, 0.3, 0.5):
        print(f"  a = {a}: log π = {gaussian_log_prob(a, 0.2, 0.1):.3f}")


if __name__ == "__main__":
    main()
