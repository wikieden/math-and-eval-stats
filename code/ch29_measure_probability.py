"""讲义：第 29 章 测度论概率入门（notes/lessons/29-measure-probability.md）
运行：uv run python ch29_measure_probability.py

例子都很小：一颗骰子、一个指数分布、两个高斯、一个“几乎处处收敛但期望不收敛”的序列。
测度论的定理本身不是算出来的；这里只把每个定义落到能算的数上。
"""

from itertools import combinations

import numpy as np
from scipy import integrate, stats

DIE = np.arange(1, 7)


def sigma_algebra(blocks: list[set]) -> list[frozenset]:
    """由一个划分生成的 σ-代数：所有块的并（含空集和全集），共 2^块数 个事件。"""
    out = []
    for k in range(len(blocks) + 1):
        for c in combinations(blocks, k):
            out.append(frozenset().union(*c))
    return out


def layer_cake(dist, upper: float = np.inf) -> float:
    """非负随机变量 E[X] = ∫_0^∞ P(X > t) dt（“分层蛋糕”，Lebesgue 积分按值域切片的直觉）。"""
    return integrate.quad(dist.sf, 0, upper)[0]


def die_layer_cake() -> float:
    """骰子：E[X] = Σ_{t=0}^{5} P(X > t)，每层宽 1。"""
    return float(sum((DIE > t).mean() for t in range(6)))


def rn_derivative(n: int = 200000, seed: int = 0) -> tuple[float, float]:
    """P = N(1, 1)、Q = N(0, 1)：dP/dQ (x) = exp(x − 1/2)。用 Q 的样本检查 E_Q[dP/dQ] = 1、E_Q[X·dP/dQ] = E_P[X] = 1。"""
    x = np.random.default_rng(seed).standard_normal(n)
    w = np.exp(x - 0.5)
    return float(w.mean()), float((x * w).mean())


def cond_exp_die(blocks=({1, 2}, {3, 4}, {5, 6})) -> dict[int, float]:
    """E[X | G]：G 由划分生成时，就是在每块上取平均（X 在 G 可测函数里的最佳均方逼近）。"""
    out = {}
    for b in blocks:
        m = float(np.mean(sorted(b)))
        for w in b:
            out[w] = m
    return out


def orthogonality(blocks=({1, 2}, {3, 4}, {5, 6})) -> list[float]:
    """残差 X − E[X|G] 和 G 里每个事件的指示函数正交：E[(X − E[X|G])·1_B] = 0。"""
    ce = cond_exp_die(blocks)
    res = np.array([w - ce[w] for w in DIE])
    return [float(np.mean(res * np.isin(DIE, sorted(b)))) for b in blocks]


def spike_sequence(n: int) -> tuple[float, float]:
    """X_n = n·1{U < 1/n}（U ~ 均匀）：P(X_n ≠ 0) = 1/n → 0（依概率收敛到 0），但 E[X_n] = 1 一直不变。返回 (P(X_n ≠ 0), E[X_n])。"""
    return 1 / n, n * (1 / n)


def lln_clt(n: int, reps: int = 20000, seed: int = 1) -> tuple[float, float]:
    """骰子平均：P(|均值 − 3.5| > 0.1)（大数定律：→ 0）和 √n(均值 − 3.5)/σ 的标准差（中心极限定理：→ 1）。"""
    rng = np.random.default_rng(seed)
    m = rng.integers(1, 7, (reps, n)).mean(axis=1)
    sigma = np.sqrt(35 / 12)
    return float((np.abs(m - 3.5) > 0.1).mean()), float((np.sqrt(n) * (m - 3.5) / sigma).std())


def main() -> None:
    print("\n第一步 σ-代数：能问概率的事件的集合")
    sa = sigma_algebra([{1, 2}, {3, 4}, {5, 6}])
    print(f"  只看“掷出几点在哪一对里”：{len(sa)} 个事件（2³）；完整的骰子是 2⁶ = 64 个")
    print(f"  {{1, 2}} 在里面：{frozenset({1, 2}) in sa}；{{1}} 不在：{frozenset({1}) not in sa}（这个信息层次分不出 1 和 2）")

    print("\n第二步 Lebesgue 积分：E[X] = ∫ P(X > t) dt")
    print(f"  骰子：Σ P(X > t) = {die_layer_cake():.4f}；指数分布 Exp(1)：∫ P(X > t) dt = {layer_cake(stats.expon()):.4f}")

    print("\n第三步 密度 = Radon-Nikodym 导数：dP/dQ")
    a, b = rn_derivative()
    print(f"  P = N(1, 1)、Q = N(0, 1)：dP/dQ = e^(x − 1/2)；E_Q[dP/dQ] ≈ {a:.4f}，E_Q[X·dP/dQ] ≈ {b:.4f}（= E_P[X] = 1）")

    print("\n第四步 条件期望：在信息允许的范围里做最好的猜测")
    ce = cond_exp_die()
    print(f"  E[X | 哪一对] = {[ce[w] for w in DIE]}；E[E[X|G]] = {np.mean([ce[w] for w in DIE]):.4f}")
    print(f"  残差和每个块的指示函数的内积：{orthogonality()}（全是 0：条件期望是投影）")

    print("\n第五步 收敛方式：依概率收敛不等于期望收敛")
    for n in (10, 1000, 100000):
        p, e = spike_sequence(n)
        print(f"  n = {n}：P(X_n ≠ 0) = {p:g}，E[X_n] = {e:g}")
    for n in (10, 100, 1000):
        p, s = lln_clt(n)
        print(f"  骰子 n = {n}：P(|均值 − 3.5| > 0.1) = {p:.4f}，标准化后的标准差 {s:.4f}")


if __name__ == "__main__":
    main()
