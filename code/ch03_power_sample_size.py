"""讲义：第 3 章 要测多少次（notes/lessons/03-power-sample-size.md）
运行：uv run python ch03_power_sample_size.py

例子：真成功率 0.70 vs 0.85（编的）；Miller 2024 式 (9) 的 969 题。
里程碑 ME-M2 的 0.6 / 0.7 功效模拟表留给自己做，这里不算。
"""

import math

import numpy as np
from scipy.stats import norm


def z_of(p: float) -> float:
    """标准正态的上 p 分位数 z_p（z_0.025 = 1.96，z_0.20 = 0.84）。"""
    return float(norm.ppf(1 - p))


def n_per_group(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """两组独立比较、每组要多少次：n = (z_α/2 + z_β)² · (p1(1−p1) + p2(1−p2)) / δ²，向上取整。"""
    delta = abs(p1 - p2)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    return math.ceil((z_of(alpha / 2) + z_of(1 - power)) ** 2 * var / delta**2)


def power_at(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    """每组 n 次时的功效（正态近似）：Φ(δ / SE − z_α/2)，只算"往对的方向"那一侧。"""
    se = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return float(norm.cdf(abs(p1 - p2) / se - z_of(alpha / 2)))


def mde(p: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """每组 n 次、两组成功率都在 p 附近时，能可靠检出的最小差值 δ = (z_α/2 + z_β)·√(2p(1−p)/n)。"""
    return (z_of(alpha / 2) + z_of(1 - power)) * math.sqrt(2 * p * (1 - p) / n)


def miller_n(omega2: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Miller 2024 式 (9)（σ² 项取 0）：n = (z_α/2 + z_β)² · ω² / δ²。"""
    return (z_of(alpha / 2) + z_of(1 - power)) ** 2 * omega2 / delta**2


def simulate_power(p1: float, p2: float, n: int, reps: int = 20000, seed: int = 0) -> float:
    """模拟：重做 reps 次实验，双比例 z 检验 p < 0.05 的比例。"""
    rng = np.random.default_rng(seed)
    k1, k2 = rng.binomial(n, p1, reps), rng.binomial(n, p2, reps)
    pbar = (k1 + k2) / (2 * n)
    se = np.sqrt(pbar * (1 - pbar) * 2 / n)
    z = np.divide(k2 / n - k1 / n, se, out=np.zeros(reps), where=se > 0)
    return float(np.mean(np.abs(z) > z_of(0.025)))


def main() -> None:
    print("\n第一步 两类错误：α 管误报，β 管漏报")
    print(f"  z_0.025 = {z_of(0.025):.3f}，z_0.20 = {z_of(0.20):.3f}，z_0.10 = {z_of(0.10):.3f}")

    print("\n第二步 每组 30 次，0.70 vs 0.85 的功效")
    print(f"  功效 = {power_at(0.70, 0.85, 30):.3f}（公式），模拟 {simulate_power(0.70, 0.85, 30):.3f}")

    print("\n第三步 想要 80% 功效，每组要多少次")
    for power in (0.8, 0.9):
        print(f"  功效 {power:.0%}：每组 {n_per_group(0.70, 0.85, power=power)} 次")
    print(f"  只想检出 5 个点（0.70 vs 0.75）：每组 {n_per_group(0.70, 0.75)} 次")

    print("\n第四步 次数定了，能检出多小的差（MDE）")
    for n in (30, 100, 300):
        print(f"  每组 {n:3d} 次、成功率约 0.7：MDE = {mde(0.7, n):.3f}")

    print("\n第五步 Miller 2024 的例子：ω² = 1/9、δ = 0.03")
    print(f"  n = {miller_n(1 / 9, 0.03):.0f} 道独立题")


if __name__ == "__main__":
    main()
