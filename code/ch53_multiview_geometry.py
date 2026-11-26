"""讲义：第 53 章 多视几何（notes/lessons/53-multiview-geometry.md）
运行：uv run python ch53_multiview_geometry.py

例子：两台相机（第二台在第一台右边 0.5 m、绕竖直轴转 5°），看同一组 3D 点；本质矩阵 E = [t]× R 的对极约束；
一个平面上的点之间的单应；两视图三角化（DLT）；RANSAC 的迭代次数公式。
里程碑 ME-M32 的“八点法估本质矩阵 + RANSAC（带外点），对照 OpenCV”留给自己做；这里 E 直接由已知的 R、t 构造。
"""

import numpy as np

from ch52_camera_geometry import K


def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def two_cams():
    """第一台相机在原点；第二台：p2 = R p1 + t。"""
    th = np.radians(5)
    r = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    t = np.array([-0.5, 0.0, 0.0])
    return r, t


PTS = np.array([[0.0, 0.0, 4.0], [1.0, -0.5, 5.0], [-1.0, 0.3, 6.0], [0.5, 0.8, 3.0], [-0.4, -0.6, 8.0]])


def normalized(p: np.ndarray) -> np.ndarray:
    return p / p[2]


def epipolar_residuals(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """x2ᵀ E x1（归一化坐标）：真对应点应该为 0；返回每个点的值。"""
    r, t = two_cams()
    e = skew(t) @ r
    rng = np.random.default_rng(seed)
    out = []
    for p in PTS:
        x1 = normalized(p)
        x2 = normalized(r @ p + t)
        x2 = x2 + np.r_[noise * rng.standard_normal(2), 0]
        out.append(float(x2 @ e @ x1))
    return np.array(out)


def fundamental() -> np.ndarray:
    """像素坐标下 F = K⁻ᵀ E K⁻¹，约束 u2ᵀ F u1 = 0；秩 2。"""
    r, t = two_cams()
    ki = np.linalg.inv(K)
    return ki.T @ skew(t) @ r @ ki


def plane_homography(n: np.ndarray, d: float) -> np.ndarray:
    """平面 nᵀp = d 上的点在两视图之间满足 x2 ∝ H x1，H = R + t nᵀ/d（归一化坐标）。"""
    r, t = two_cams()
    return r + np.outer(t, n) / d


def triangulate(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """DLT 三角化：每个视图给两条线性方程 x × (P X) = 0，SVD 取最小奇异向量。"""
    r, t = two_cams()
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = np.hstack([r, t[:, None]])
    a = np.vstack([x1[0] * p1[2] - p1[0], x1[1] * p1[2] - p1[1], x2[0] * p2[2] - p2[0], x2[1] * p2[2] - p2[1]])
    x = np.linalg.svd(a)[2][-1]
    return x[:3] / x[3]


def ransac_iters(inlier: float, s: int, p: float = 0.99) -> int:
    """要以概率 p 至少抽到一次全是内点的样本（每次抽 s 个）：N = log(1 − p) / log(1 − w^s)。"""
    return int(np.ceil(np.log(1 - p) / np.log(1 - inlier ** s)))


def main() -> None:
    print("\n第一步 对极约束：x2ᵀ E x1 = 0，E = [t]× R")
    print(f"  5 个真对应点：{np.round(epipolar_residuals(), 12).tolist()}")
    print(f"  第二个视图加 0.002 的噪声：{np.round(epipolar_residuals(0.002), 5).tolist()}")
    r, t = two_cams()
    e = skew(t) @ r
    print(f"  E 的奇异值 {np.round(np.linalg.svd(e, compute_uv=False), 4).tolist()}（两个相等、一个为 0）")

    print("\n第二步 基础矩阵：换成像素坐标")
    f = fundamental()
    u1 = K @ normalized(PTS[1])
    u2 = K @ normalized(r @ PTS[1] + t)
    print(f"  u2ᵀ F u1 = {u2 @ f @ u1:.2e}；F 的秩 {np.linalg.matrix_rank(f, tol=1e-10)}")
    line = f @ u1
    print(f"  对极线 l2 = F u1 = {np.round(line / np.linalg.norm(line[:2]), 4).tolist()}；u2 到它的距离 {abs(u2 @ line) / np.linalg.norm(line[:2]):.2e} 像素")

    print("\n第三步 单应：平面上的点一一对应")
    n, d = np.array([0.0, 0.0, 1.0]), 6.0
    h = plane_homography(n, d)
    p = np.array([0.7, -0.4, 6.0])
    x2 = normalized(h @ normalized(p))
    print(f"  平面 Z = 6 上的点 {p.tolist()}：H x1 → {np.round(x2, 5).tolist()}，真值 {np.round(normalized(r @ p + t), 5).tolist()}")

    print("\n第四步 三角化")
    for pt in PTS[:3]:
        x1, x2 = normalized(pt), normalized(r @ pt + t)
        print(f"  真值 {pt.tolist()} → 三角化 {np.round(triangulate(x1, x2), 6).tolist()}")

    print("\n第五步 RANSAC 要抽多少次")
    for w, s in ((0.5, 8), (0.5, 5), (0.8, 8), (0.5, 3)):
        print(f"  内点率 {w}、每次抽 {s} 个、成功率 99%：{ransac_iters(w, s)} 次")


if __name__ == "__main__":
    main()
