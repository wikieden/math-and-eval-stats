"""讲义：第 31 章 微分几何与流形优化（notes/lessons/31-manifold-optimization.md）
运行：uv run python ch31_manifold_optimization.py

例子：单位球面 S² ⊂ R³（最简单的弯曲流形）。在球面上最大化 Rayleigh 商 xᵀAx，答案是 A 的最大特征向量。
里程碑 ME-M22 的“Stiefel / SO(3) 上的黎曼梯度下降”留给自己做；这里只在球面上演示。
"""

import numpy as np

A_MAT = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])


def proj(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """切空间投影 P_x v = (I − xxᵀ)v：去掉沿法向（x 方向）的分量。"""
    return v - x * (x @ v)


def retract(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """收缩映射：先在切平面上走 v，再归一化拉回球面。"""
    y = x + v
    return y / np.linalg.norm(y)


def exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """指数映射：沿大圆（测地线）走长度 ‖v‖。"""
    t = np.linalg.norm(v)
    return x if t == 0 else np.cos(t) * x + np.sin(t) * v / t


def geodesic(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.arccos(np.clip(x @ y, -1, 1)))


def rgd(x0: np.ndarray, lr: float = 0.2, iters: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, int, list[float]]:
    """黎曼梯度上升：欧氏梯度 2Ax → 投影到切空间 → 沿它走一步 → 收缩回球面。返回 (x, 迭代次数, 每次的 Rayleigh 商)。"""
    x = x0 / np.linalg.norm(x0)
    vals = [float(x @ A_MAT @ x)]
    for k in range(1, iters + 1):
        g = proj(x, 2 * A_MAT @ x)
        if np.linalg.norm(g) < tol:
            return x, k - 1, vals
        x = retract(x, lr * g)
        vals.append(float(x @ A_MAT @ x))
    return x, iters, vals


def naive_gd(x0: np.ndarray, lr: float = 0.2, iters: int = 20) -> float:
    """不投影、不收缩，直接在 R³ 里做梯度上升：‖x‖ 会一路变大（xᵀAx 无上界），离开球面。返回最后的 ‖x‖。"""
    x = x0 / np.linalg.norm(x0)
    for _ in range(iters):
        x = x + lr * 2 * A_MAT @ x
    return float(np.linalg.norm(x))


def so3_tangent(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    """SO(3) 上的黎曼梯度（嵌入度量）：把欧氏梯度 G 投影成 R·skew(RᵀG)，skew(M) = (M − Mᵀ)/2。"""
    m = r.T @ g
    return r @ ((m - m.T) / 2)


def main() -> None:
    x = np.array([0.0, 0.0, 1.0])
    v = np.array([0.3, 0.0, 0.0])

    print("\n第一步 切空间与投影：球面在 x 处的切平面 = 和 x 垂直的向量")
    w = np.array([1.0, 2.0, 3.0])
    print(f"  x = [0, 0, 1]，w = [1, 2, 3]：投影 P_x w = {proj(x, w).tolist()}，和 x 的内积 {x @ proj(x, w):.1f}")

    print("\n第二步 收缩映射 vs 指数映射：两种“把切向量拉回流形”的办法，一阶相同")
    for t in (0.3, 0.1, 0.01):
        vv = v / 0.3 * t
        d = np.linalg.norm(retract(x, vv) - exp_map(x, vv))
        print(f"  ‖v‖ = {t}：两者相差 {d:.3e}")

    print("\n第三步 测地线距离 vs 弦长")
    for ang in (10, 90, 180):
        y = np.array([np.sin(np.radians(ang)), 0, np.cos(np.radians(ang))])
        print(f"  夹角 {ang}°：测地线 {geodesic(x, y):.4f}，弦长 {np.linalg.norm(x - y):.4f}")

    print("\n第四步 黎曼梯度上升：在球面上最大化 xᵀAx")
    xs, k, vals = rgd(np.array([1.0, 1.0, 1.0]))
    lam, vec = np.linalg.eigh(A_MAT)
    print(f"  {k} 次收敛到 x = {np.round(xs, 4).tolist()}，xᵀAx = {vals[-1]:.6f}；最大特征值 {lam[-1]:.6f}")
    print(f"  前几次 xᵀAx：{[round(v_, 4) for v_ in vals[:5]]}；每次都在球面上：‖x‖ = {np.linalg.norm(xs):.6f}")
    print(f"  不投影不收缩，直接在 R³ 里走 20 步：‖x‖ = {naive_gd(np.array([1.0, 1.0, 1.0])):.3g}")

    print("\n第五步 和第 19 章接上：SO(3) 也是流形，黎曼梯度在 R·so(3) 里")
    r = np.eye(3)
    g = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    t = so3_tangent(r, g)
    print(f"  R = I，欧氏梯度 G：黎曼梯度 = skew(G) = {t.tolist()}（反对称 ↔ 轴角向量）")


if __name__ == "__main__":
    main()
