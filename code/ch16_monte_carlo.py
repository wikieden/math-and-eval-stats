"""讲义：第 16 章 蒙特卡洛与采样（notes/lessons/16-monte-carlo-sampling.md）
运行：uv run python ch16_monte_carlo.py

例子：估计 E[X²]（X ~ N(0, 1)，真值 1）；估计尾部概率 P(X > 3)（真值 0.00135）；PPO 的比率；一维 CEM 与 MPPI。
里程碑 ME-M12 的二维代价函数对比留给自己做。
"""

import math

import numpy as np
from scipy.stats import norm

TRUE_TAIL = float(norm.sf(3.0))


def mc_mean_x2(n: int, seed: int = 0) -> float:
    """普通蒙特卡洛：抽 n 个 X，平均 X²。"""
    x = np.random.default_rng(seed).standard_normal(n)
    return float((x**2).mean())


def mc_error(n: int, reps: int = 2000) -> float:
    """重复 reps 次，看估计值的标准差（理论值 √(Var(X²)/n) = √(2/n)）。"""
    rng = np.random.default_rng(1)
    return float((rng.standard_normal((reps, n)) ** 2).mean(1).std())


def tail_naive(n: int, seed: int = 0) -> float:
    """直接数 X > 3 的比例。"""
    return float((np.random.default_rng(seed).standard_normal(n) > 3).mean())


def tail_importance(n: int, shift: float, seed: int = 0) -> float:
    """重要性采样：从 N(shift, 1) 抽，权重 w = p(x)/q(x)，估计 E_q[1(x>3)·w]。shift = 0 就是直接数。"""
    x = np.random.default_rng(seed).normal(shift, 1.0, n)
    w = np.exp(norm.logpdf(x) - norm.logpdf(x, shift))
    return float(((x > 3) * w).mean())


def relative_error(n: int, shift: float, reps: int = 500) -> float:
    """重复 reps 次，估计值的标准差 / 真值。"""
    est = np.array([tail_importance(n, shift, seed=s) for s in range(reps)])
    return float(est.std() / TRUE_TAIL)


def ppo_ratio(p_new: float, p_old: float) -> float:
    """PPO 的比率 r = π_new(a|s)/π_old(a|s)：就是用旧策略的样本估新策略期望时的重要性权重。"""
    return p_new / p_old


def cost(x: np.ndarray) -> np.ndarray:
    """一维代价：(x − 2)²，最小点 x = 2。"""
    return (x - 2.0) ** 2


def cem(iters: int = 6, n: int = 50, elite: int = 10, seed: int = 0) -> list[tuple[float, float]]:
    """交叉熵方法：从 N(μ, σ²) 抽 n 个，留代价最低的 elite 个，用它们的均值和标准差更新。返回每轮 (μ, σ)。"""
    rng = np.random.default_rng(seed)
    mu, sd, hist = 0.0, 2.0, [(0.0, 2.0)]
    for _ in range(iters):
        x = rng.normal(mu, sd, n)
        e = x[np.argsort(cost(x))[:elite]]
        mu, sd = float(e.mean()), float(e.std() + 1e-6)
        hist.append((mu, sd))
    return hist


def mppi_weights(costs: np.ndarray, lam: float) -> np.ndarray:
    """MPPI 的指数加权：w ∝ exp(−(c − c_min)/λ)，λ 越小越只信最好的那几个。"""
    w = np.exp(-(costs - costs.min()) / lam)
    return w / w.sum()


def main() -> None:
    print("\n第一步 蒙特卡洛：用样本平均代替积分，误差 ∝ 1/√n")
    for n in (100, 10000):
        print(f"  n = {n:5d}：E[X²] ≈ {mc_mean_x2(n):.4f}；重复 2000 次的标准差 {mc_error(n):.4f}（理论 {math.sqrt(2 / n):.4f}）")

    print("\n第二步 重要性采样：估计罕见事件 P(X > 3)")
    print(f"  真值 {TRUE_TAIL:.5f}")
    for shift, name in ((0.0, "直接数（提议 = 原分布 N(0, 1)）"), (3.0, "提议 N(3, 1)：样本集中在尾部"), (-1.0, "提议 N(−1, 1)：偏到反方向")):
        print(f"  {name}：1000 个样本估计 {tail_importance(1000, shift):.5f}，相对误差 {relative_error(1000, shift):.2f}")

    print("\n第三步 PPO 的比率就是重要性权重")
    print(f"  旧策略选这个动作的概率 0.20，新策略 0.30：r = {ppo_ratio(0.30, 0.20):.2f}；截断到 [0.8, 1.2] 后用 1.2")

    print("\n第四步 CEM：抽样 → 留精英 → 更新分布")
    for k, (mu, sd) in enumerate(cem()):
        print(f"  第 {k} 轮：μ = {mu:.3f}，σ = {sd:.3f}")

    print("\n第五步 MPPI：按代价做指数加权，λ 是温度")
    c = np.array([0.1, 0.5, 1.0, 4.0])
    for lam in (0.1, 1.0, 10.0):
        print(f"  代价 {c.tolist()}，λ = {lam}：权重 {np.round(mppi_weights(c, lam), 3).tolist()}")


if __name__ == "__main__":
    main()
