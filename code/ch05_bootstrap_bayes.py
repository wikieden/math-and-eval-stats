"""讲义：第 5 章 Bootstrap 与 Beta 后验（notes/lessons/05-bootstrap-bayes.md）
运行：uv run python ch05_bootstrap_bayes.py

例子：第 2 章的 40/50 vs 30/50；分层 bootstrap 用编的"3 个任务 × 5 个种子"。
里程碑 ME-M4 的 13/20 vs 14/20 的 P(B > A) 留给自己算。
"""

import numpy as np
from scipy.stats import beta

SCORES = np.array([[0.62, 0.70, 0.58, 0.66, 0.64],   # 任务 1，5 个种子（编的）
                   [0.31, 0.45, 0.38, 0.29, 0.40],   # 任务 2
                   [0.88, 0.91, 0.85, 0.93, 0.90]])  # 任务 3


def bootstrap_ci(k: int, n: int, reps: int = 20000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """百分位 bootstrap：把 n 次结果（k 个 1、n − k 个 0）有放回重抽 reps 次，取成功率的分位数。"""
    rng = np.random.default_rng(seed)
    data = np.r_[np.ones(k), np.zeros(n - k)]
    means = rng.choice(data, (reps, n), replace=True).mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def beta_posterior(k: int, n: int, a0: float = 1, b0: float = 1):
    """均匀先验 Beta(1, 1) + k 次成功、n − k 次失败 → 后验 Beta(1 + k, 1 + n − k)。"""
    return beta(a0 + k, b0 + n - k)


def credible_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """后验的中间 95%（等尾可信区间）。"""
    d = beta_posterior(k, n)
    a = (1 - level) / 2
    return float(d.ppf(a)), float(d.ppf(1 - a))


def prob_b_better(kb: int, nb: int, ka: int, na: int, draws: int = 400000, seed: int = 0) -> float:
    """P(p_B > p_A)：两个后验各抽一个数比大小，数 B 赢的比例。"""
    rng = np.random.default_rng(seed)
    pb = rng.beta(1 + kb, 1 + nb - kb, draws)
    pa = rng.beta(1 + ka, 1 + na - ka, draws)
    return float(np.mean(pb > pa))


def diff_bootstrap(kb: int, nb: int, ka: int, na: int, reps: int = 20000, seed: int = 0) -> tuple[float, float]:
    """差值的 bootstrap 区间：两组各自有放回重抽，算 p_B − p_A 的 2.5% 和 97.5% 分位。"""
    rng = np.random.default_rng(seed)
    b = rng.binomial(nb, kb / nb, reps) / nb
    a = rng.binomial(na, ka / na, reps) / na
    d = b - a
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def stratified_bootstrap_mean(scores: np.ndarray, reps: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    """Agarwal 2021 的分层 bootstrap：每个任务内部对种子有放回重抽，再对任务取平均。返回 (点估计, 下界, 上界)。"""
    rng = np.random.default_rng(seed)
    t, s = scores.shape
    idx = rng.integers(0, s, (reps, t, s))
    res = np.take_along_axis(np.broadcast_to(scores, (reps, t, s)), idx, 2).mean((1, 2))
    return float(scores.mean()), float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


def main() -> None:
    print("\n第一步 Bootstrap：把手里的数据当成总体，重抽")
    for k, n in ((40, 50), (30, 50)):
        lo, hi = bootstrap_ci(k, n)
        print(f"  {k}/{n}：bootstrap 95% 区间 [{lo:.3f}, {hi:.3f}]")

    print("\n第二步 Beta 后验：均匀先验 + 数据")
    for k, n in ((40, 50), (30, 50)):
        d = beta_posterior(k, n)
        lo, hi = credible_interval(k, n)
        print(f"  {k}/{n} → Beta({1 + k}, {1 + n - k})：均值 {d.mean():.3f}，95% 可信区间 [{lo:.3f}, {hi:.3f}]")

    print("\n第三步 B 比 A 好的概率")
    print(f"  P(p_40/50 > p_30/50) = {prob_b_better(40, 50, 30, 50):.3f}")
    lo, hi = diff_bootstrap(40, 50, 30, 50)
    print(f"  差值 bootstrap 区间 [{lo:+.3f}, {hi:+.3f}]")

    print("\n第四步 多任务、多种子：分层 bootstrap")
    m, lo, hi = stratified_bootstrap_mean(SCORES)
    print(f"  3 个任务 × 5 个种子：平均 {m:.3f}，95% 区间 [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
