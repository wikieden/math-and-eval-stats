"""讲义：第 26 章 轨迹优化与 MPC（notes/lessons/26-trajopt-mpc.md）
运行：uv run python ch26_trajopt_mpc.py

例子：第 25 章的标量系统 x_{k+1} = 1.2·x_k + u_k，q = r = 1，终端代价 1，N = 10 步，x0 = 2。
iLQR 用一个编的非线性标量系统 x_{k+1} = x_k + 0.1·(sin x_k + u_k)；MPPI 用一步的标量问题，最优分布有闭式。
里程碑 ME-M20 的"一维小车 MPC 的稀疏 QP"和"一维小车上的 iLQR"留给自己做。
"""

import numpy as np
from scipy.optimize import minimize

from ch25_lqr_riccati import A, B, finite_horizon

N, X0 = 10, 2.0


def dense_qp(n: int = N, a: float = A, b: float = B, q: float = 1.0, r: float = 1.0, qf: float = 1.0):
    """把状态消掉：X = S_x x0 + S_u U，代价 = Uᵀ H U + 2 x0 fᵀ U + 常数。返回 (H, f, S_x, S_u)。"""
    sx = np.array([a ** (k + 1) for k in range(n)])
    su = np.zeros((n, n))
    for k in range(n):
        for j in range(k + 1):
            su[k, j] = a ** (k - j) * b
    w = np.full(n, q)
    w[-1] = qf
    h = su.T @ (w[:, None] * su) + r * np.eye(n)
    f = su.T @ (w * sx)
    return h, f, sx, su


def solve_unconstrained(x0: float = X0) -> np.ndarray:
    h, f, _, _ = dense_qp()
    return np.linalg.solve(h, -f * x0)


def solve_box(umax: float, x0: float = X0) -> np.ndarray:
    """加上 |u| ≤ umax 的盒约束，用 L-BFGS-B 解这个 QP。"""
    h, f, _, _ = dense_qp()
    res = minimize(lambda u: u @ h @ u + 2 * x0 * f @ u, np.zeros(N), jac=lambda u: 2 * h @ u + 2 * x0 * f,
                   bounds=[(-umax, umax)] * N, method="L-BFGS-B", options={"ftol": 1e-14, "gtol": 1e-12})
    return res.x


def nnz_counts(n: int) -> tuple[int, int]:
    """标量系统：稠密形式 H 是 n×n 满矩阵；稀疏形式变量 [x_1..x_n, u_0..u_{n−1}]，
    Hessian 对角 2n 个非零，动力学等式 x_{k+1} − a x_k − b u_k = 0 每行最多 3 个非零（第一行 x_0 已知，只有 2 个）。"""
    return n * n, 2 * n + (3 * n - 1)


# ---------- iLQR ----------
DT = 0.1


def f_nl(x, u):
    return x + DT * (np.sin(x) + u)


def ilqr(x0: float = 2.0, n: int = 30, q: float = 1.0, r: float = 1.0, iters: int = 8, trajs: list | None = None) -> list[float]:
    """iLQR：沿当前轨迹线性化（A_k = 1 + 0.1 cos x_k，B = 0.1）、二次化代价，往回推得到修正量，带线搜索往前滚。返回每轮的总代价；传入 trajs 列表时顺便把每轮的状态轨迹存进去（画图用）。"""
    u = np.zeros(n)

    def roll(us):
        xs = [x0]
        for v in us:
            xs.append(f_nl(xs[-1], v))
        return np.array(xs)

    def cost(xs, us):
        return float(q * (xs ** 2).sum() + r * (us ** 2).sum())

    xs = roll(u)
    hist = [cost(xs, u)]
    if trajs is not None:
        trajs.append(xs)
    for _ in range(iters):
        vx, vxx = 2 * q * xs[-1], 2 * q
        kff, kfb = np.zeros(n), np.zeros(n)
        for k in range(n - 1, -1, -1):
            ak, bk = 1 + DT * np.cos(xs[k]), DT
            qx = 2 * q * xs[k] + ak * vx
            qu = 2 * r * u[k] + bk * vx
            qxx = 2 * q + ak * vxx * ak
            quu = 2 * r + bk * vxx * bk
            qux = bk * vxx * ak
            kff[k], kfb[k] = -qu / quu, -qux / quu
            vx = qx + kfb[k] * quu * kff[k] + kfb[k] * qu + qux * kff[k]
            vxx = qxx + kfb[k] * quu * kfb[k] + 2 * kfb[k] * qux
        best = None
        for alpha in (1.0, 0.5, 0.25, 0.1):
            xn, un = [x0], np.zeros(n)
            for k in range(n):
                un[k] = u[k] + alpha * kff[k] + kfb[k] * (xn[-1] - xs[k])
                xn.append(f_nl(xn[-1], un[k]))
            xn = np.array(xn)
            if cost(xn, un) < hist[-1]:
                best = (xn, un)
                break
        if best is None:
            break
        xs, u = best
        hist.append(cost(xs, u))
        if trajs is not None:
            trajs.append(xs)
    return hist


# ---------- MPPI ----------
def mppi_one_step(x0: float = 1.0, sigma: float = 1.0, lam: float = 1.0, n: int = 100000, seed: int = 0):
    """一步问题：u ~ N(0, σ²) 是“先验”控制分布，代价 S(u) = (x0 + u)²。
    最优分布 q*(u) ∝ N(u; 0, σ²)·exp(−S(u)/λ) 还是高斯：精度 1/σ² + 2/λ，均值 −(2x0/λ)/(1/σ² + 2/λ)。
    MPPI 用样本加权 w ∝ exp(−S/λ) 估计这个均值。返回 (闭式均值, 加权估计, 有效样本数)。"""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, sigma, n)
    s = (x0 + u) ** 2
    w = np.exp(-(s - s.min()) / lam)
    w /= w.sum()
    exact = -(2 * x0 / lam) / (1 / sigma ** 2 + 2 / lam)
    return exact, float((w * u).sum()), float(1 / (w ** 2).sum())


def main() -> None:
    print("\n第一步 稠密 QP：把状态消掉，只剩 N 个控制量")
    u = solve_unconstrained()
    _, ks = finite_horizon(N)
    print(f"  QP 解出的 u_0 = {u[0]:.4f}；有限时域 Riccati 的 −K_0·x0 = {-ks[0] * X0:.4f}（同一个最优解）")
    h, f, _, _ = dense_qp()
    print(f"  H 是 {h.shape[0]}×{h.shape[0]} 满矩阵，条件数 {np.linalg.cond(h):.1f}")

    print("\n第二步 加约束 |u| ≤ 1：Riccati 写不出来，QP 照样解")
    ub = solve_box(1.0)
    print(f"  无约束 u_0..u_2 = {np.round(u[:3], 4).tolist()}；有约束 = {np.round(ub[:3], 4).tolist()}")
    print("  MPC：每个控制周期只执行 u_0，下一周期用新测量的 x 再解一次")

    print("\n第三步 稀疏形式：保留状态当变量，矩阵变大但几乎全是 0")
    for n in (10, 50, 200):
        d, s = nnz_counts(n)
        print(f"  N = {n}：稠密 H 非零元 {d}，稀疏 Hessian + 等式约束非零元 {s}")

    print("\n第四步 iLQR：非线性系统，反复“线性化 + 二次化 + 往回推 + 往前滚”")
    hist = ilqr()
    print(f"  每轮总代价：{[round(c, 4) for c in hist]}")

    print("\n第五步 MPPI：最优控制分布 ∝ 先验 × exp(−代价/λ)")
    ex, est, ess = mppi_one_step()
    print(f"  x0 = 1、σ = 1、λ = 1：闭式最优均值 {ex:.4f}，10 万样本加权估计 {est:.4f}，有效样本数 {ess:.0f}")
    ex2, est2, ess2 = mppi_one_step(lam=0.1)
    print(f"  λ = 0.1：闭式 {ex2:.4f}，估计 {est2:.4f}，有效样本数 {ess2:.0f}（λ 越小越贪心、有效样本越少）")


if __name__ == "__main__":
    main()
