"""讲义：第 1 章 一个成功率有多准（notes/lessons/01-success-rate-ci.md）
运行：uv run python ch01_success_rate_ci.py

只算讲义里讲过的例子（21/30、30/30、覆盖率模拟），不算练习的数。
"""

import math

import numpy as np

Z95 = 1.959964  # 标准正态分布 97.5% 分位数，95% 双侧区间用它（常写成 1.96）


def point_estimate(k: int, n: int) -> float:
    """第一步：成功率的点估计 p̂ = k / n。"""
    return k / n


def standard_error(k: int, n: int) -> float:
    """第二步：标准误 SE = sqrt(p̂(1-p̂)/n)。"""
    p = k / n
    return math.sqrt(p * (1 - p) / n)


def wald_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """第二步：Wald 区间 p̂ ± z·SE（教科书最常见、小样本最不靠谱的那个）。"""
    p, se = k / n, standard_error(k, n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """第三步：Wilson 区间。中心往 0.5 拉一点，宽度里多一项 z²/4n²，所以 p̂=0 或 1 时也不会缩成一个点。"""
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return center - half, center + half


def n_for_half_width(p: float, half_width: float, z: float = Z95) -> int:
    """第四步：想让区间半宽 ≤ half_width，大约要多少次试验（按 Wald 公式反解，向上取整）。"""
    return math.ceil(z * z * p * (1 - p) / half_width**2)


def coverage(method, p_true: float, n: int, reps: int = 20000, seed: int = 0) -> float:
    """第五步：重复做 reps 次「测 n 次」的实验，数一数区间盖住真值 p_true 的比例。"""
    rng = np.random.default_rng(seed)
    ks = rng.binomial(n, p_true, size=reps)
    hit = 0
    for k in ks:
        lo, hi = method(int(k), n)
        hit += lo <= p_true <= hi
    return hit / reps


def main() -> None:
    print("第一步 点估计：21/30")
    print(f"  p̂ = {point_estimate(21, 30):.2f}")

    print("\n第二步 标准误与 Wald 区间：21/30")
    se = standard_error(21, 30)
    lo, hi = wald_ci(21, 30)
    print(f"  SE = sqrt(0.7×0.3/30) = {se:.4f}，1.96·SE = {Z95 * se:.3f}")
    print(f"  Wald 95% 区间 = [{lo:.3f}, {hi:.3f}]")
    lo, hi = wald_ci(30, 30)
    print(f"  30/30 的 Wald 区间 = [{lo:.3f}, {hi:.3f}]  ← 宽度 0，显然不对")

    print("\n第三步 Wilson 区间")
    for k, n in [(21, 30), (30, 30)]:
        lo, hi = wilson_ci(k, n)
        print(f"  {k}/{n}: Wilson 95% 区间 = [{lo:.3f}, {hi:.3f}]")

    print("\n第四步 区间宽度随 n 缩小（p̂ = 0.7，Wilson 半宽）")
    for n in [10, 30, 100, 300, 1000]:
        lo, hi = wilson_ci(round(0.7 * n), n)
        print(f"  n = {n:4d}: 半宽 ≈ ±{(hi - lo) / 2 * 100:.1f} 个点")
    print(f"  想要 ±5 个点：n ≈ {n_for_half_width(0.7, 0.05)}")

    print("\n第五步 覆盖率模拟：真值 p = 0.9，n = 30，重复 20000 次")
    print(f"  Wald  盖住真值的比例 = {coverage(wald_ci, 0.9, 30):.3f}")
    print(f"  Wilson 盖住真值的比例 = {coverage(wilson_ci, 0.9, 30):.3f}")


if __name__ == "__main__":
    main()
