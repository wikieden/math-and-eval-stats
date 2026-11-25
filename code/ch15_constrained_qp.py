"""讲义：第 15 章 约束与凸优化（notes/lessons/15-constrained-convex-qp.md）
运行：uv run python ch15_constrained_qp.py

例子：x² + y² 在 x + y = 1 上的最小值；(x − 2)² 在 x ≤ 1 上的最小值；一个 2 维 QP（目标点被关节限位挡住）。
里程碑 ME-M11 的带限位 IK 与 MPC 留给自己做。
"""

import numpy as np
from scipy.optimize import minimize

P_QP = np.diag([1.0, 1.0])
Q_QP = np.array([-3.0, -1.0])  # min ½‖x‖² − 3x₁ − x₂：不受限时最优点 (3, 1)
G_QP = np.array([[1.0, 0.0], [0.0, 1.0]])
H_QP = np.array([1.5, 2.0])  # 限位 x₁ ≤ 1.5、x₂ ≤ 2


def equality_kkt():
    """min x² + y²  s.t. x + y = 1：拉格朗日 L = x² + y² + λ(x + y − 1)，∇L = 0 与约束连立成线性方程组。返回 (x, y, λ)。"""
    k = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    rhs = np.array([0.0, 0.0, 1.0])
    return tuple(np.linalg.solve(k, rhs))


def inequality_kkt() -> tuple[float, float]:
    """min (x − 2)²  s.t. x ≤ 1：约束起作用，x = 1；驻点条件 2(x − 2) + μ = 0 → μ = 2 ≥ 0。返回 (x, μ)。"""
    x = 1.0
    return x, -2 * (x - 2)


def is_convex_quadratic(m: np.ndarray) -> bool:
    """二次函数 ½xᵀMx 凸 ⇔ M 半正定（特征值都 ≥ 0）。"""
    return bool(np.all(np.linalg.eigvalsh(m) >= -1e-12))


def qp_active_set() -> tuple[np.ndarray, np.ndarray]:
    """小 QP 的手算解：先不管约束得 (3, 1)，x₁ 超了限位 → 把 x₁ ≤ 1.5 当等式解 KKT。返回 (x, 乘子)。"""
    k = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rhs = np.array([3.0, 1.0, 1.5])
    sol = np.linalg.solve(k, rhs)
    return sol[:2], np.array([sol[2], 0.0])


def qp_scipy() -> np.ndarray:
    """同一个 QP 交给 scipy 的 SLSQP 求解，用来核对。"""
    f = lambda x: 0.5 * x @ P_QP @ x + Q_QP @ x
    cons = {"type": "ineq", "fun": lambda x: H_QP - G_QP @ x}
    return minimize(f, np.zeros(2), constraints=[cons], method="SLSQP").x


def main() -> None:
    print("\n第一步 等式约束：拉格朗日乘子")
    x, y, lam = equality_kkt()
    print(f"  min x² + y²，x + y = 1：x = {x:.2f}，y = {y:.2f}，λ = {lam:.2f}（f 的梯度 [{2 * x:.1f}, {2 * y:.1f}] 和约束法向 [1, 1] 平行）")

    print("\n第二步 不等式约束：KKT 与互补松弛")
    x, mu = inequality_kkt()
    print(f"  min (x − 2)²，x ≤ 1：x = {x}，μ = {mu}（μ ≥ 0、约束取等号、μ·(x − 1) = 0）")

    print("\n第三步 凸性：Hessian 半正定")
    for m in (np.diag([1.0, 10.0]), np.array([[1.0, 2.0], [2.0, 1.0]])):
        print(f"  {m.tolist()}：特征值 {np.linalg.eigvalsh(m).tolist()}，凸 = {is_convex_quadratic(m)}")

    print("\n第四步 QP 标准形：min ½xᵀPx + qᵀx  s.t. Gx ≤ h")
    xs, lam = qp_active_set()
    print(f"  不受限的最优点 (3, 1)；加限位 x₁ ≤ 1.5、x₂ ≤ 2 → 手算 x = {xs.tolist()}，乘子 {lam.tolist()}")
    print(f"  scipy SLSQP：x = {np.round(qp_scipy(), 4).tolist()}")


if __name__ == "__main__":
    main()
