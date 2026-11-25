"""讲义：第 33 章 最优传输（notes/lessons/33-optimal-transport.md）
运行：uv run python ch33_optimal_transport.py

例子：一维两个高斯 N(0, 1) → N(2, 0.5²)（排序就是最优配对）；一个 3 个源、3 个汇的小运输问题（线性规划精确解 vs Sinkhorn）；
两个点质量之间的 KL 和 Wasserstein；位移插值。
里程碑 ME-M22 的“ε 从大到小扫一遍、看传输计划变尖”留给自己做；这里 Sinkhorn 只跑一个 ε。
"""

import numpy as np
from scipy.optimize import linprog

XS = np.array([0.0, 1.0, 2.0])
YS = np.array([0.5, 1.5, 2.5])
A_W = np.array([0.5, 0.3, 0.2])
B_W = np.array([0.2, 0.4, 0.4])
COST = (XS[:, None] - YS[None, :]) ** 2


def w2_gauss(m0, s0, m1, s1) -> float:
    """一维高斯之间的 W2 有闭式：√((m0 − m1)² + (s0 − s1)²)。"""
    return float(np.sqrt((m0 - m1) ** 2 + (s0 - s1) ** 2))


def w2_empirical(n: int = 100000, seed: int = 0) -> float:
    """一维时最优配对 = 两边各自排序后一一对应（Monge 映射单调）。"""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.standard_normal(n))
    y = np.sort(2 + 0.5 * rng.standard_normal(n))
    return float(np.sqrt(((x - y) ** 2).mean()))


def kantorovich_lp(a=A_W, b=B_W, c=COST) -> tuple[np.ndarray, float]:
    """Kantorovich：min Σ C_ij P_ij，行和 = a、列和 = b、P ≥ 0。一个线性规划。"""
    m, n = c.shape
    aeq = []
    for i in range(m):
        row = np.zeros((m, n))
        row[i] = 1
        aeq.append(row.ravel())
    for j in range(n):
        col = np.zeros((m, n))
        col[:, j] = 1
        aeq.append(col.ravel())
    res = linprog(c.ravel(), A_eq=np.array(aeq), b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    return res.x.reshape(m, n), float(res.fun)


def sinkhorn(eps: float, a=A_W, b=B_W, c=COST, iters: int = 2000) -> tuple[np.ndarray, int]:
    """熵正则 OT：P = diag(u) K diag(v)，K = e^{−C/ε}；交替缩放让行和 = a、列和 = b。返回 (P, 收敛用的迭代次数)。"""
    k = np.exp(-c / eps)
    u, v = np.ones_like(a), np.ones_like(b)
    for it in range(1, iters + 1):
        u = a / (k @ v)
        v = b / (k.T @ u)
        p = u[:, None] * k * v[None, :]
        if np.abs(p.sum(axis=1) - a).max() < 1e-10:
            return p, it
    return p, iters


def kl_vs_w(theta: float) -> tuple[float, float]:
    """δ_0 和 δ_θ：θ ≠ 0 时支撑不重叠，KL = ∞；W1 = |θ|，随 θ 光滑变化、有梯度。"""
    return (np.inf if theta != 0 else 0.0), abs(theta)


def displacement(t: float, m0=0.0, s0=1.0, m1=2.0, s1=0.5) -> tuple[float, float]:
    """位移插值（动态 OT 的测地线）：每个粒子沿直线匀速走，t 时刻的分布是 N((1−t)m0 + t m1, ((1−t)s0 + t s1)²)。"""
    return (1 - t) * m0 + t * m1, (1 - t) * s0 + t * s1


def mixture_sd(t: float, m0=0.0, s0=1.0, m1=2.0, s1=0.5) -> float:
    """对比：密度直接线性混合 (1−t)p0 + t p1 的标准差（中间会变成两个峰）。"""
    mean = (1 - t) * m0 + t * m1
    second = (1 - t) * (s0 ** 2 + m0 ** 2) + t * (s1 ** 2 + m1 ** 2)
    return float(np.sqrt(second - mean ** 2))


def main() -> None:
    print("\n第一步 Monge：一维时排序就是最优运输")
    print(f"  N(0, 1) → N(2, 0.5²)：W2 闭式 {w2_gauss(0, 1, 2, 0.5):.4f}，排序配对的样本估计 {w2_empirical():.4f}")

    print("\n第二步 Kantorovich：允许把一份质量拆开运，变成线性规划")
    p, cost = kantorovich_lp()
    print(f"  源 {XS.tolist()}（质量 {A_W.tolist()}）→ 汇 {YS.tolist()}（质量 {B_W.tolist()}），代价 = 距离²")
    print(f"  最优总代价 {cost:.4f}，传输计划：")
    for row in p:
        print("   ", np.round(row, 4).tolist())

    print("\n第三步 熵正则与 Sinkhorn：交替缩放行、列")
    ps, it = sinkhorn(1.0)
    print(f"  ε = 1：{it} 次迭代收敛，运输代价 Σ C·P = {(COST * ps).sum():.4f}（精确 {cost:.4f}）")
    print(f"  计划（大多数格子都 > 0，是“模糊”版的最优计划）：")
    for row in ps:
        print("   ", np.round(row, 4).tolist())

    print("\n第四步 为什么生成模型喜欢 Wasserstein：支撑不重叠时 KL 没有梯度")
    for th in (0.0, 0.5, 2.0):
        kl, w = kl_vs_w(th)
        print(f"  δ_0 vs δ_{th}：KL = {kl}，W1 = {w}")

    print("\n第五步 动态 OT：位移插值，每个粒子走直线（和 flow matching 的直线路径同一个想法）")
    for t in (0.0, 0.5, 1.0):
        m, s = displacement(t)
        print(f"  t = {t}：位移插值 N({m:.2f}, {s:.2f}²)；密度直接混合的标准差 {mixture_sd(t):.4f}")


if __name__ == "__main__":
    main()
