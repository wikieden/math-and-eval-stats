"""讲义：第 11 章 线代一（notes/lessons/11-linear-algebra-basics.md）
运行：uv run python ch11_linear_algebra.py

例子：A = [[1, 2], [3, 4]]、B = [[5, 6], [7, 8]]；一个秩为 2 的 3×3 矩阵；对称矩阵 S = [[2, 1], [1, 2]]。
"""

import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
M3 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])  # 第 2 行 = 2 × 第 1 行
S = np.array([[2.0, 1.0], [1.0, 2.0]])


def matmul_dot(a, b):
    """看法一：C[i, j] = A 的第 i 行 · B 的第 j 列。"""
    return np.array([[a[i] @ b[:, j] for j in range(b.shape[1])] for i in range(a.shape[0])])


def matmul_columns(a, b):
    """看法二：C 的第 j 列 = A 的各列按 B[:, j] 加权求和。"""
    return np.column_stack([sum(b[k, j] * a[:, k] for k in range(a.shape[1])) for j in range(b.shape[1])])


def matmul_outer(a, b):
    """看法三：C = Σ_k (A 的第 k 列) ⊗ (B 的第 k 行)，k 个秩 1 矩阵相加。"""
    return sum(np.outer(a[:, k], b[k]) for k in range(a.shape[1]))


def rotation(deg: float) -> np.ndarray:
    """平面旋转矩阵。"""
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def null_vector(m: np.ndarray) -> np.ndarray:
    """零空间的一个单位向量：SVD 里最小奇异值对应的右奇异向量。"""
    return np.linalg.svd(m)[2][-1]


def power_by_eig(s: np.ndarray, k: int) -> np.ndarray:
    """S^k = Q Λ^k Qᵀ（对称矩阵的特征分解）。"""
    lam, q = np.linalg.eigh(s)
    return q @ np.diag(lam**k) @ q.T


def quadratic_form(m: np.ndarray, x: np.ndarray) -> float:
    """xᵀ M x。"""
    return float(x @ m @ x)


def main() -> None:
    print("\n第一步 矩阵乘法的三种看法，结果一样")
    for name, f in (("行·列", matmul_dot), ("列的组合", matmul_columns), ("外积之和", matmul_outer)):
        print(f"  {name}：{f(A, B).tolist()}")

    print("\n第二步 矩阵 = 线性映射：看它把基向量送到哪")
    print(f"  旋转 90°：e1 → {np.round(rotation(90) @ [1, 0], 3).tolist()}，e2 → {np.round(rotation(90) @ [0, 1], 3).tolist()}")
    print(f"  A 把 e1 送到 {(A @ [1, 0]).tolist()}（A 的第 1 列），e2 送到 {(A @ [0, 1]).tolist()}（第 2 列）；det(A) = {np.linalg.det(A):.1f}")

    print("\n第三步 秩：有几个方向是独立的")
    v = null_vector(M3)
    print(f"  M3 的秩 = {np.linalg.matrix_rank(M3)}；零空间方向 {np.round(v / v[np.argmax(abs(v))], 3).tolist()}，M3·v = {np.round(M3 @ v, 10).tolist()}")

    print("\n第四步 特征分解：对称矩阵 = 旋转 · 拉伸 · 转回来")
    lam, q = np.linalg.eigh(S)
    print(f"  S 的特征值 {lam.tolist()}，特征向量（列）{np.round(q, 4).tolist()}")
    print(f"  S^10 用特征分解算 {power_by_eig(S, 10).round(1).tolist()}，直接乘 {np.linalg.matrix_power(S, 10).tolist()}")

    print("\n第五步 正定与二次型")
    for name, m in (("S = [[2,1],[1,2]]", S), ("[[1,2],[2,1]]", np.array([[1.0, 2.0], [2.0, 1.0]]))):
        print(f"  {name}：特征值 {np.linalg.eigvalsh(m).tolist()}；x = [1, −1] 时 xᵀMx = {quadratic_form(m, np.array([1.0, -1.0])):.1f}")


if __name__ == "__main__":
    main()
