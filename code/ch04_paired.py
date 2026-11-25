"""讲义：第 4 章 配对比较（notes/lessons/04-paired-comparison.md）
运行：uv run python ch04_paired.py

例子：30 个初始条件，A、B 在同一批上各跑一次（编的四格表）：
都成功 14、A 成 B 败 8、A 败 B 成 1、都失败 7。
"""

import math

import numpy as np
from scipy.stats import binomtest, chi2, norm

TABLE = {"both": 14, "a_only": 8, "b_only": 1, "neither": 7}


def unpaired_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """不配对：双比例 z 检验（第 2 章），返回 (z, 双侧 p)。"""
    pbar = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    return z, float(2 * norm.sf(abs(z)))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验：只看不一致的 b + c 对，H₀ 下 b ~ Binomial(b + c, 0.5)。"""
    return float(binomtest(b, b + c, 0.5).pvalue)


def mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    """McNemar 卡方（连续性校正）：(|b − c| − 1)² / (b + c)，自由度 1。"""
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return stat, float(chi2.sf(stat, 1))


def paired_se(diffs: np.ndarray) -> float:
    """Miller 2024 式 (7)：配对差值的标准误 √(Var(d)/n)，Var 用 n − 1。"""
    return float(np.sqrt(diffs.var(ddof=1) / len(diffs)))


def table_to_diffs(t: dict) -> np.ndarray:
    """四格表 → 每个初始条件的差值 d = A − B（1、0、−1）。"""
    return np.array([0] * t["both"] + [1] * t["a_only"] + [-1] * t["b_only"] + [0] * t["neither"], dtype=float)


def variance_ratio(rho: float) -> float:
    """两组方差相同、相关系数 ρ 时：配对方差 / 不配对方差 = 1 − ρ。"""
    return 1 - rho


def simulate_difficulty(n: int = 200000, seed: int = 0) -> tuple[float, float]:
    """模拟"初始条件有难有易"：每个 IC 难度 u，A 成功率 0.95 − 0.9u，B 成功率 0.85 − 0.9u。返回 (A、B 结果的相关系数, 配对 / 不配对方差比)。"""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    a = (rng.random(n) < 0.95 - 0.9 * u).astype(float)
    b = (rng.random(n) < 0.85 - 0.9 * u).astype(float)
    rho = float(np.corrcoef(a, b)[0, 1])
    return rho, float((a - b).var() / (a.var() + b.var()))


def main() -> None:
    t = TABLE
    ka, kb, n = t["both"] + t["a_only"], t["both"] + t["b_only"], sum(t.values())
    print("\n第一步 同一批初始条件上的四格表")
    print(f"  A 成功 {ka}/{n}，B 成功 {kb}/{n}；不一致的对：A 成 B 败 {t['a_only']}，A 败 B 成 {t['b_only']}")

    print("\n第二步 当成两组独立数据（不配对）")
    z, p = unpaired_z(ka, n, kb, n)
    print(f"  z = {z:.3f}，p = {p:.3f}")

    print("\n第三步 McNemar：只看不一致的那几对")
    print(f"  精确检验 p = {mcnemar_exact(t['a_only'], t['b_only']):.3f}")
    s, p2 = mcnemar_chi2(t["a_only"], t["b_only"])
    print(f"  卡方（连续性校正）= {s:.2f}，p = {p2:.3f}")

    print("\n第四步 配对的标准误（Miller 式 7）")
    d = table_to_diffs(t)
    print(f"  差值均值 {d.mean():.3f}，配对 SE = {paired_se(d):.4f}")
    se_unpaired = math.sqrt(ka / n * (1 - ka / n) / n + kb / n * (1 - kb / n) / n)
    print(f"  不配对 SE = {se_unpaired:.4f}")

    print("\n第五步 配对为什么省：方差里减掉 2Cov")
    for rho in (0.0, 0.3, 0.5):
        print(f"  相关 ρ = {rho}：配对方差是不配对的 {variance_ratio(rho):.0%}")
    rho, ratio = simulate_difficulty()
    print(f"  模拟（IC 有难有易）：A、B 结果相关 {rho:.3f}，方差比 {ratio:.3f}")


if __name__ == "__main__":
    main()
