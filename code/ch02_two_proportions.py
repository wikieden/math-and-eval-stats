"""讲义：第 2 章 两个成功率比一比（notes/lessons/02-two-proportions.md）
运行：uv run python ch02_two_proportions.py

只算讲义里的例子：14/20 vs 13/20（Kress-Gazit 2024 的真实案例）和编的 40/50 vs 30/50。
锚点题 21/30 vs 18/30 是练习，这里不算。
"""

import math

import numpy as np
from scipy.stats import fisher_exact, norm

from ch01_success_rate_ci import wilson_ci

Z95 = 1.959964


def diff_se(k1: int, n1: int, k2: int, n2: int) -> float:
    """第一步：差值的标准误 sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)（两组独立，方差相加）。"""
    p1, p2 = k1 / n1, k2 / n2
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def diff_wald_ci(k1, n1, k2, n2, z=Z95) -> tuple[float, float]:
    """第一步：差值的 Wald 区间 (p1 - p2) ± z·SE。"""
    d = k1 / n1 - k2 / n2
    se = diff_se(k1, n1, k2, n2)
    return d - z * se, d + z * se


def newcombe_ci(k1, n1, k2, n2, z=Z95) -> tuple[float, float]:
    """第一步：Newcombe 区间（用两个 Wilson 区间拼出差值区间，小样本更稳）。"""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def two_proportion_z(k1, n1, k2, n2) -> tuple[float, float]:
    """第二步：双比例 z 检验。零假设下两组成功率相同，用合并的 p 算标准误。返回 (z, 双侧 p 值)。"""
    pooled = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se0
    return z, 2 * norm.sf(abs(z))


def fisher_p(k1, n1, k2, n2) -> float:
    """第三步：Fisher 精确检验的双侧 p 值（不靠正态近似，小样本用它）。"""
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def simulate_null_p(k1, n1, k2, n2, reps=200_000, seed=0) -> float:
    """第二步的模拟版：假设两组真成功率都等于合并成功率，重复做实验，数 |差值| ≥ 观测值的比例。"""
    rng = np.random.default_rng(seed)
    pooled = (k1 + k2) / (n1 + n2)
    d_obs = abs(k1 / n1 - k2 / n2)
    d = rng.binomial(n1, pooled, reps) / n1 - rng.binomial(n2, pooled, reps) / n2
    return float(np.mean(np.abs(d) >= d_obs - 1e-12))


def main() -> None:
    cases = [("Kress-Gazit 真实案例：B 14/20 vs A 13/20", 14, 20, 13, 20), ("编的例子：40/50 vs 30/50", 40, 50, 30, 50)]
    for title, k1, n1, k2, n2 in cases:
        print(f"\n== {title}")
        print("第一步 差值与区间")
        print(f"  差值 = {k1 / n1 - k2 / n2:+.3f}，SE = {diff_se(k1, n1, k2, n2):.4f}")
        lo, hi = diff_wald_ci(k1, n1, k2, n2)
        print(f"  Wald 区间 = [{lo:+.3f}, {hi:+.3f}]")
        lo, hi = newcombe_ci(k1, n1, k2, n2)
        print(f"  Newcombe 区间 = [{lo:+.3f}, {hi:+.3f}]")
        print("第二步 双比例 z 检验")
        z, p = two_proportion_z(k1, n1, k2, n2)
        print(f"  z = {z:.3f}，双侧 p = {p:.3f}；模拟 p ≈ {simulate_null_p(k1, n1, k2, n2):.3f}")
        print("第三步 Fisher 精确检验")
        print(f"  p = {fisher_p(k1, n1, k2, n2):.3f}")
    print("\n第四步 两个区间重叠 ≠ 差异不显著（40/50 vs 30/50）")
    print(f"  40/50 Wilson = [{wilson_ci(40, 50)[0]:.3f}, {wilson_ci(40, 50)[1]:.3f}]")
    print(f"  30/50 Wilson = [{wilson_ci(30, 50)[0]:.3f}, {wilson_ci(30, 50)[1]:.3f}]  ← 有重叠，但 z 检验 p < 0.05")


if __name__ == "__main__":
    main()
