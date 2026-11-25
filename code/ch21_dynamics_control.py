"""讲义：第 21 章 动力系统与控制的数学（notes/lessons/21-dynamics-control.md）
运行：uv run python ch21_dynamics_control.py

例子：单摆在下垂点和倒立点线性化；双积分器（推一个小车）用 ZOH 离散化、求 LQR；一个 2 状态的 MDP 当 Bellman 不动点。
里程碑 ME-M17 的倒立摆 LQR 闭环仿真和 4×4 网格值迭代留给自己做。
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

G_L = 9.81  # g/l，摆长 1 米
DT = 0.1


def pendulum_A(theta0: float) -> np.ndarray:
    """单摆 θ'' = −(g/l) sin θ，状态 [θ, θ']，在 θ0 处线性化：A = [[0, 1], [−(g/l)cos θ0, 0]]。"""
    return np.array([[0.0, 1.0], [-G_L * np.cos(theta0), 0.0]])


def zoh(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """零阶保持离散化：A_d = e^{A dt}，B_d = ∫₀^dt e^{A s} ds · B（用一个扩展矩阵一次算出）。"""
    n, m = a.shape[0], b.shape[1]
    big = np.zeros((n + m, n + m))
    big[:n, :n], big[:n, n:] = a, b
    e = expm(big * dt)
    return e[:n, :n], e[:n, n:]


def lqr_gain(ad, bd, q, r) -> tuple[np.ndarray, np.ndarray]:
    """离散 LQR：解代数 Riccati 方程得 P，K = (R + BᵀPB)⁻¹BᵀPA，控制 u = −Kx。"""
    p = solve_discrete_are(ad, bd, q, r)
    k = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    return k, p


def bellman_policy_value(gamma: float = 0.9) -> tuple[np.ndarray, np.ndarray, int]:
    """2 状态 MDP（编的）：固定策略下转移 P = [[0.8, 0.2], [0.3, 0.7]]，奖励 r = [1, 0]。
    精确解 V = (I − γP)⁻¹r，和反复套 V ← r + γPV 的结果对比。返回 (精确 V, 迭代 V, 迭代次数)。"""
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    r = np.array([1.0, 0.0])
    exact = np.linalg.solve(np.eye(2) - gamma * p, r)
    v, k = np.zeros(2), 0
    while np.abs(v - exact).max() > 1e-6:
        v, k = r + gamma * p @ v, k + 1
    return exact, v, k


def fmt_eigs(a: np.ndarray) -> str:
    """把特征值打印成 "±3.132i" 或 "3.132, −3.132" 这样好读的样子。"""
    out = []
    for z in np.linalg.eigvals(a):
        if abs(z.imag) < 1e-12:
            out.append(f"{z.real:.3f}")
        else:
            out.append(f"{z.real:.3f}{z.imag:+.3f}i" if abs(z.real) > 1e-12 else f"{z.imag:+.3f}i")
    return "、".join(out)


def main() -> None:
    print("\n第一步 线性化：在平衡点附近，非线性系统看起来像 ẋ = Ax")
    for name, th in (("下垂点 θ = 0", 0.0), ("倒立点 θ = π", np.pi)):
        a = pendulum_A(th)
        print(f"  {name}：A = {np.round(a, 2).tolist()}，特征值 {fmt_eigs(a)}")

    print("\n第二步 稳定性看特征值：连续时间看实部 < 0，离散时间看模 < 1")
    ad, _ = zoh(pendulum_A(np.pi), np.array([[0.0], [1.0]]), DT)
    print(f"  倒立点离散化（dt = 0.1）后特征值的模 {np.round(np.abs(np.linalg.eigvals(ad)), 4).tolist()}：有一个 > 1，不稳定")

    print("\n第三步 ZOH 离散化：双积分器 ẍ = u")
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0], [1.0]])
    ad, bd = zoh(a, b, DT)
    print(f"  A_d = {np.round(ad, 4).tolist()}，B_d = {np.round(bd, 4).ravel().tolist()}（= [dt²/2, dt]）")

    print("\n第四步 LQR：Q = I、R = 1 时的反馈增益")
    k, p = lqr_gain(ad, bd, np.eye(2), np.array([[1.0]]))
    cl = ad - bd @ k
    print(f"  K = {np.round(k, 4).ravel().tolist()}；闭环特征值的模 {np.round(np.abs(np.linalg.eigvals(cl)), 4).tolist()}（都 < 1，稳了）")
    k2, _ = lqr_gain(ad, bd, np.eye(2), np.array([[0.01]]))
    print(f"  R 从 1 降到 0.01（控制便宜）：K = {np.round(k2, 4).ravel().tolist()}，更激进")

    print("\n第五步 Bellman 方程是不动点方程")
    exact, v, n = bellman_policy_value()
    print(f"  精确解 V = {np.round(exact, 4).tolist()}；反复套 V ← r + γPV，{n} 次误差 < 1e-6：{np.round(v, 4).tolist()}")


if __name__ == "__main__":
    main()
