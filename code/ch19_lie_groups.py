"""讲义：第 19 章 李群入门（notes/lessons/19-lie-groups.md）
运行：uv run python ch19_lie_groups.py

例子：绕 z 轴转 90°、绕 x 轴转 90° 的两个旋转；切空间里的小扰动；SE(3) 的 exp；两个姿态的 SLERP。
里程碑 ME-M15 的"切空间梯度下降求旋转平均"留给自己做。
"""

import numpy as np

PHI_Z = np.array([0.0, 0.0, np.pi / 2])  # 绕 z 转 90°（轴角向量 = 轴 × 角）
PHI_X = np.array([np.pi / 2, 0.0, 0.0])  # 绕 x 转 90°


def hat(w: np.ndarray) -> np.ndarray:
    """向量 → 反对称矩阵 [w]×：[w]× v = w × v。"""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(m: np.ndarray) -> np.ndarray:
    """反对称矩阵 → 向量。"""
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """切空间 → SO(3)：Rodrigues 公式 R = I + sinθ/θ·[φ]× + (1 − cosθ)/θ²·[φ]×²。"""
    th = np.linalg.norm(phi)
    k = hat(phi)
    if th < 1e-9:
        return np.eye(3) + k
    return np.eye(3) + np.sin(th) / th * k + (1 - np.cos(th)) / th**2 * k @ k


def so3_log(r: np.ndarray) -> np.ndarray:
    """SO(3) → 切空间：θ = arccos((tr R − 1)/2)，φ = θ/(2 sinθ)·vee(R − Rᵀ)（θ 不接近 0、π 时）。"""
    th = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    if th < 1e-9:
        return vee(r - r.T) / 2
    return th / (2 * np.sin(th)) * vee(r - r.T)


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """SO(3) 的右雅可比 J_r(φ)：exp(φ + δ) ≈ exp(φ)·exp(J_r δ)。"""
    th = np.linalg.norm(phi)
    k = hat(phi)
    if th < 1e-9:
        return np.eye(3) - 0.5 * k
    return np.eye(3) - (1 - np.cos(th)) / th**2 * k + (th - np.sin(th)) / th**3 * k @ k


def numeric_right_jacobian(phi: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """数值核对：第 j 列 = log(exp(φ)ᵀ exp(φ + h e_j)) / h。"""
    r0 = so3_exp(phi)
    return np.column_stack([so3_log(r0.T @ so3_exp(phi + h * e)) / h for e in np.eye(3)])


def se3_exp(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """SE(3) 的 exp：旋转部分 exp(φ)，平移部分 V(φ)·ρ（不是直接 ρ），V = I + (1 − cosθ)/θ²[φ]× + (θ − sinθ)/θ³[φ]×²。"""
    th = np.linalg.norm(phi)
    k = hat(phi)
    v = np.eye(3) + (1 - np.cos(th)) / th**2 * k + (th - np.sin(th)) / th**3 * k @ k if th > 1e-9 else np.eye(3)
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = so3_exp(phi), v @ rho
    return t


def slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    """流形上的插值：R(s) = R0 · exp(s · log(R0ᵀ R1))。"""
    return r0 @ so3_exp(s * so3_log(r0.T @ r1))


def main() -> None:
    print("\n第一步 切空间 ↔ 群：exp 和 log")
    r = so3_exp(PHI_Z)
    print(f"  exp([0, 0, π/2]) = {np.round(r, 4).tolist()}")
    print(f"  log 回来：{np.round(so3_log(r), 4).tolist()}（π/2 = {np.pi / 2:.4f}）")

    print("\n第二步 群上不能直接加：exp(a)·exp(b) ≠ exp(a + b)")
    lhs = so3_exp(PHI_Z) @ so3_exp(PHI_X)
    rhs = so3_exp(PHI_Z + PHI_X)
    print(f"  两者最大差 {np.abs(lhs - rhs).max():.3f}；顺序反过来也不同：{np.abs(lhs - so3_exp(PHI_X) @ so3_exp(PHI_Z)).max():.3f}")
    print(f"  真正的组合 log(exp(a)exp(b)) = {np.round(so3_log(lhs), 4).tolist()}，而 a + b = {np.round(PHI_Z + PHI_X, 4).tolist()}")

    print("\n第三步 扰动与雅可比：exp(φ + δ) ≈ exp(φ)·exp(J_r δ)")
    phi = np.array([0.3, -0.5, 0.8])
    jr, jn = right_jacobian(phi), numeric_right_jacobian(phi)
    print(f"  闭式 J_r 和数值差分最大差 {np.abs(jr - jn).max():.1e}")
    print(f"  φ = 0 时 J_r = I；‖φ‖ = {np.linalg.norm(phi):.3f} 时对角元 {np.round(np.diag(jr), 4).tolist()}")

    print("\n第四步 SE(3) 的 exp：平移要乘 V(φ)")
    t = se3_exp(np.array([1.0, 0.0, 0.0]), PHI_Z)
    print(f"  ρ = [1, 0, 0]、φ = [0, 0, π/2]：平移 = {np.round(t[:3, 3], 4).tolist()}（不是 [1, 0, 0]）")

    print("\n第五步 SLERP：在流形上走直线")
    r0, r1 = np.eye(3), so3_exp(PHI_Z)
    for s in (0.0, 0.5, 1.0):
        print(f"  s = {s}：转角 {np.degrees(np.linalg.norm(so3_log(slerp(r0, r1, s)))):.1f}°")
    naive = 0.5 * (r0 + r1)
    print(f"  直接平均两个矩阵：det = {np.linalg.det(naive):.3f}（不再是旋转矩阵）")


if __name__ == "__main__":
    main()
