"""讲义：第 55 章 SLAM 的优化（notes/lessons/55-slam-optimization.md）
运行：uv run python ch55_slam_optimization.py

例子：一个 3D 点在一台相机里的重投影误差和雅可比的形状；10 台相机、1000 个点的 BA 的 Hessian 规模与 Schur 补；
带一个外点的直线拟合上 Huber / Cauchy 鲁棒核（IRLS）；两个变量的高斯里边缘化掉一个（信息矩阵的 Schur 补）。
里程碑 ME-M33 的“用 scipy.optimize.least_squares 做小型 BA、看稀疏 Jacobian”留给自己做。
"""

import numpy as np

from ch52_camera_geometry import K


def reproj_residual(p_world: np.ndarray, uv_obs: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """重投影误差 e = π(K(R p + t)) − u_obs（2 维）。"""
    pc = r @ p_world + t
    uvw = K @ pc
    return uvw[:2] / uvw[2] - uv_obs


def numeric_jac(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    fx = f(x)
    return np.stack([(f(x + h * e) - fx) / h for e in np.eye(len(x))], 1)


def ba_sizes(ncam: int, npt: int, obs_per_pt: int) -> dict:
    """BA：每台相机 6 个参数、每个点 3 个。Hessian 是 (6c + 3p)²；相机-点块只在“点被这台相机看到”时非零。
    相机-相机、点-点之间的非对角块都是 0。Schur 补把点消掉，只解 6c × 6c 的“缩减相机系统”。"""
    n = 6 * ncam + 3 * npt
    nnz = 36 * ncam + 9 * npt + 2 * 18 * npt * obs_per_pt   # 相机对角块 + 点对角块 + 相机-点块（上下对称两份）
    return {"变量数": n, "满矩阵元素": n * n, "非零元（约）": nnz, "缩减系统": (6 * ncam) ** 2}


def robust_weights(r: np.ndarray, delta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """IRLS 的权重 w(r) = ρ'(r)/r：Huber 在 |r| > δ 时 δ/|r|；Cauchy 是 1/(1 + (r/δ)²)。"""
    a = np.abs(r)
    huber = np.where(a <= delta, 1.0, delta / a)
    cauchy = 1.0 / (1.0 + (r / delta) ** 2)
    return huber, cauchy


def line_fit(kind: str, iters: int = 20) -> tuple[float, float]:
    """y = 2x + 1 上 10 个点加小噪声，再加一个外点 (9, 0)。最小二乘 vs IRLS（Huber / Cauchy）。返回 (斜率, 截距)。"""
    rng = np.random.default_rng(0)
    x = np.arange(10.0)
    y = 2 * x + 1 + 0.1 * rng.standard_normal(10)
    x, y = np.r_[x, 9.0], np.r_[y, 0.0]
    a = np.c_[x, np.ones_like(x)]
    w = np.ones_like(x)
    for _ in range(iters if kind != "ls" else 1):
        sol = np.linalg.solve(a.T @ (w[:, None] * a), a.T @ (w * y))
        r = y - a @ sol
        if kind == "ls":
            break
        h, c = robust_weights(r, 1.0)
        w = h if kind == "huber" else c
    return float(sol[0]), float(sol[1])


def marginalize() -> tuple[np.ndarray, float]:
    """两个变量的高斯，信息矩阵 Λ = [[a, b], [b, c]]。边缘化掉 x1：x0 的信息 = a − b²/c（Schur 补），
    和直接从协方差 Σ = Λ⁻¹ 取 Σ00 再求逆一样。返回 (Λ, Schur 补)。"""
    lam = np.array([[4.0, -2.0], [-2.0, 3.0]])
    return lam, lam[0, 0] - lam[0, 1] ** 2 / lam[1, 1]


def main() -> None:
    print("\n第一步 重投影误差和雅可比的形状")
    p = np.array([0.5, -0.2, 4.0])
    r, t = np.eye(3), np.zeros(3)
    obs = np.array([385.0, 214.0])
    e = reproj_residual(p, obs, r, t)
    jp = numeric_jac(lambda q: reproj_residual(q, obs, r, t), p)
    print(f"  点 {p.tolist()}、观测 {obs.tolist()}：误差 {np.round(e, 3).tolist()} 像素")
    print(f"  对点的雅可比（2×3）：{np.round(jp, 2).tolist()}；对相机位姿的雅可比是 2×6")

    print("\n第二步 BA 的 Hessian：稀疏 + Schur 补")
    s = ba_sizes(10, 1000, 5)
    for k, v in s.items():
        print(f"  {k}：{v:,}")

    print("\n第三步 鲁棒核：外点的权重被压下去")
    rr = np.array([0.5, 2.0, 10.0])
    h, c = robust_weights(rr)
    print(f"  残差 {rr.tolist()}（δ = 1）：Huber 权重 {np.round(h, 3).tolist()}，Cauchy 权重 {np.round(c, 3).tolist()}")
    for k in ("ls", "huber", "cauchy"):
        print(f"  直线 y = 2x + 1 加一个外点，{k}：斜率 {line_fit(k)[0]:.4f}、截距 {line_fit(k)[1]:.4f}")

    print("\n第四步 边缘化 = 信息矩阵的 Schur 补")
    lam, sc = marginalize()
    print(f"  Λ = {lam.tolist()}：边缘化 x1 后 x0 的信息 {sc:.4f}；从协方差算 1/Σ00 = {1 / np.linalg.inv(lam)[0, 0]:.4f}")


if __name__ == "__main__":
    main()
