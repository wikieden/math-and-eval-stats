"""讲义：第 25 章 LQR 与 Riccati（notes/lessons/25-lqr-riccati.md）
运行：uv run python ch25_lqr_riccati.py

例子：标量系统 x_{k+1} = a·x_k + b·u_k（a = 1.2 不稳定，b = 1），代价 Σ q·x² + r·u²（q = r = 1），终端代价 q_f·x_N²。
用标量是为了每一步都能手算；矩阵版就是把乘法换成矩阵乘法。
里程碑 ME-M20 的"矩阵系统的有限时域 Riccati 和 scipy.linalg.solve_discrete_are 对数"留给自己做。
"""

import numpy as np

A = 1.2
B = 1.0
Q = 1.0
R = 1.0


def riccati_step(p: float, a: float = A, b: float = B, q: float = Q, r: float = R) -> tuple[float, float]:
    """动态规划往回推一步：已知下一步的值函数 p·x²，求这一步的 (p, K)。
    min_u [q x² + r u² + p (a x + b u)²] → u = −K x，K = a b p / (r + b² p)，新 p = q + a² p − (a b p)² / (r + b² p)。"""
    k = a * b * p / (r + b * b * p)
    return q + a * a * p - (a * b * p) ** 2 / (r + b * b * p), k


def finite_horizon(n: int, qf: float = 1.0, **kw) -> tuple[list[float], list[float]]:
    """从终端 p_N = q_f 往回推 n 步，返回按时间顺序的 [p_0, …, p_N] 和 [K_0, …, K_{N−1}]。"""
    ps, ks, p = [qf], [], qf
    for _ in range(n):
        p, k = riccati_step(p, **kw)
        ps.append(p)
        ks.append(k)
    return ps[::-1], ks[::-1]


def dare_scalar(a: float = A, b: float = B, q: float = Q, r: float = R) -> float:
    """无限时域：p = q + a²p − a²b²p²/(r + b²p) 化成 b²p² + (r − a²r − q b²)p − q r = 0，取正根。"""
    c2, c1, c0 = b * b, r - a * a * r - q * b * b, -q * r
    return (-c1 + np.sqrt(c1 * c1 - 4 * c2 * c0)) / (2 * c2)


def gain(p: float, a: float = A, b: float = B, r: float = R) -> float:
    return a * b * p / (r + b * b * p)


def rollout_cost(x0: float, ks: list[float], qf: float = 1.0) -> float:
    """按增益序列闭环跑一遍，把每一步代价加起来（含终端）。"""
    x, c = x0, 0.0
    for k in ks:
        u = -k * x
        c += Q * x * x + R * u * u
        x = A * x + B * u
    return c + qf * x * x


def ctrb_rank(a: np.ndarray, b: np.ndarray) -> int:
    """可控性矩阵 [B, AB, …, A^{n−1}B] 的秩；等于 n 才能把状态推到任意地方。"""
    n = a.shape[0]
    cols = [b]
    for _ in range(n - 1):
        cols.append(a @ cols[-1])
    return int(np.linalg.matrix_rank(np.hstack(cols)))


def main() -> None:
    print("\n第一步 动态规划往回推：值函数一直是 p_k·x²")
    ps, ks = finite_horizon(10)
    print(f"  N = 10，p_N = 1：p_9 = {ps[9]:.4f}，p_8 = {ps[8]:.4f}，p_5 = {ps[5]:.4f}，p_0 = {ps[0]:.4f}")
    print(f"  增益 K_9 = {ks[9]:.4f}，K_0 = {ks[0]:.4f}")

    print("\n第二步 无限时域：p 收敛到代数 Riccati 方程的正根")
    p = dare_scalar()
    print(f"  p∞ = {p:.4f}，K∞ = {gain(p):.4f}，闭环 a − bK = {A - B * gain(p):.4f}（|·| < 1，稳了）")
    for n in (1, 2, 5, 10):
        print(f"  往回推 {n} 步：p_0 = {finite_horizon(n)[0][0]:.4f}")

    print("\n第三步 p_0·x0² 就是最优总代价")
    x0 = 2.0
    print(f"  x0 = {x0}：p_0·x0² = {ps[0] * x0 * x0:.4f}，按 K_k 闭环跑一遍加起来 = {rollout_cost(x0, ks):.4f}")
    for k in (0.6, 1.0, 1.2):
        print(f"  换成固定增益 K = {k}：{rollout_cost(x0, [k] * 10):.4f}（都更大）")

    print("\n第四步 r 的作用")
    for r in (0.01, 1.0, 100.0):
        pr = dare_scalar(r=r)
        kr = gain(pr, r=r)
        print(f"  r = {r}：p∞ = {pr:.4f}，K = {kr:.4f}，闭环极点 {A - B * kr:.4f}")

    print("\n第五步 可控性：控制量够不着的不稳定模态，LQR 也救不了")
    a = np.array([[1.0, 0.1], [0.0, 1.0]])
    print(f"  双积分器，B = [0, 0.1]：秩 {ctrb_rank(a, np.array([[0.0], [0.1]]))}（可控）")
    a2 = np.diag([1.2, 0.5])
    print(f"  A = diag(1.2, 0.5)，B = [0, 1]：秩 {ctrb_rank(a2, np.array([[0.0], [1.0]]))}（1.2 那个模态推不动，会一直发散）")


if __name__ == "__main__":
    main()
