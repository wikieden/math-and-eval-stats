"""讲义：第 12 章 线代二（notes/lessons/12-svd-least-squares.md）
运行：uv run python ch12_svd_least_squares.py

例子：一个 3×2 矩阵做 SVD；4096×4096 的层加 LoRA；5 个点拟合直线；一个几乎共线的 2×2 方程组。
里程碑 ME-M8 的"权重矩阵 秩 → 误差"曲线留给自己画。
"""

import numpy as np

A = np.array([[3.0, 2.0], [2.0, 3.0], [2.0, -2.0]])
XS = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
YS = np.array([1.1, 2.9, 5.2, 7.1, 8.8])  # 编的：大约 y = 2x + 1
ILL = np.array([[1.0, 1.0], [1.0, 1.0001]])


def svd_parts(m: np.ndarray):
    """m = U Σ Vᵀ，返回 (U, 奇异值, Vᵀ)，奇异值从大到小。"""
    return np.linalg.svd(m, full_matrices=False)


def rank_k(m: np.ndarray, k: int) -> np.ndarray:
    """截断 SVD：只留前 k 个奇异值，是 Frobenius 范数下最好的秩 k 近似（Eckart–Young）。"""
    u, s, vt = svd_parts(m)
    return (u[:, :k] * s[:k]) @ vt[:k]


def lora_params(d: int, k: int, r: int) -> tuple[int, int]:
    """LoRA：ΔW = B A，B 是 d×r、A 是 r×k。返回 (LoRA 参数量, 原矩阵参数量)。"""
    return r * (d + k), d * k


def design(xs: np.ndarray) -> np.ndarray:
    """直线拟合的设计矩阵 [x, 1]。"""
    return np.column_stack([xs, np.ones_like(xs)])


def normal_equations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小二乘的正规方程：θ = (XᵀX)⁻¹ Xᵀ y。"""
    return np.linalg.solve(x.T @ x, x.T @ y)


def main() -> None:
    print("\n第一步 SVD：任何矩阵 = 旋转 · 拉伸 · 旋转")
    u, s, vt = svd_parts(A)
    print(f"  A 的奇异值 {np.round(s, 4).tolist()}；UΣVᵀ 还原误差 {np.abs(u @ np.diag(s) @ vt - A).max():.1e}")
    print(f"  奇异值的平方 = AᵀA 的特征值：{np.round(s**2, 4).tolist()} vs {np.round(np.linalg.eigvalsh(A.T @ A)[::-1], 4).tolist()}")

    print("\n第二步 低秩近似：只留最大的奇异值")
    a1 = rank_k(A, 1)
    print(f"  秩 1 近似的误差 ‖A − A₁‖_F = {np.linalg.norm(A - a1):.4f} = 第二个奇异值 {s[1]:.4f}")
    print(f"  保留的能量 σ₁²/Σσ² = {s[0] ** 2 / (s**2).sum():.3f}")
    lp, full = lora_params(4096, 4096, 8)
    print(f"  LoRA r = 8 加在 4096×4096 上：{lp:,} 个参数，原矩阵 {full:,} 个，占 {lp / full:.2%}")

    print("\n第三步 最小二乘：拟合 5 个点")
    x = design(XS)
    th = normal_equations(x, YS)
    print(f"  正规方程：斜率 {th[0]:.4f}，截距 {th[1]:.4f}")
    print(f"  伪逆：{np.round(np.linalg.pinv(x) @ YS, 4).tolist()}；lstsq：{np.round(np.linalg.lstsq(x, YS, rcond=None)[0], 4).tolist()}")
    r = YS - x @ th
    print(f"  残差和每一列正交：Xᵀr = {np.round(x.T @ r, 10).tolist()}")

    print("\n第四步 条件数：输入抖一点，答案抖多少")
    print(f"  ILL 的条件数 {np.linalg.cond(ILL):,.0f}")
    b1, b2 = np.array([2.0, 2.0001]), np.array([2.0, 2.0002])
    print(f"  b = [2, 2.0001] → x = {np.round(np.linalg.solve(ILL, b1), 4).tolist()}；b 的第二个数只多 0.0001 → x = {np.round(np.linalg.solve(ILL, b2), 4).tolist()}")


if __name__ == "__main__":
    main()
