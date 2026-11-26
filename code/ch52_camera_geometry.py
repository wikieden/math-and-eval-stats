"""讲义：第 52 章 相机几何（notes/lessons/52-camera-geometry.md）
运行：uv run python ch52_camera_geometry.py

例子：一台 640×480 的相机（f = 500 像素，主点 (320, 240)），装在离地 1.5 米、水平朝前的位置；径向畸变 k1 = −0.2；
地面点的逆透视变换（IPM）；Lift-Splat-Shoot 的“深度分布 × 特征”外积。
里程碑 ME-M32 的“投影 / 反投影往返测试”留给自己做。
"""

import numpy as np

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
H_CAM = 1.5   # 相机离地高度（米）


def project(p_cam: np.ndarray, k: np.ndarray = K) -> np.ndarray:
    """针孔模型：相机坐标 (X, Y, Z) → 像素 (u, v) = (f X/Z + cx, f Y/Z + cy)。"""
    uvw = k @ p_cam
    return uvw[:2] / uvw[2]


def world_to_cam(p_world: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """外参：p_cam = R p_world + t。"""
    return r @ p_world + t


def distort(xn: np.ndarray, k1: float = -0.2, k2: float = 0.0) -> np.ndarray:
    """径向畸变（归一化坐标 x = X/Z）：x_d = x(1 + k1 r² + k2 r⁴)。k1 < 0 是桶形畸变。"""
    r2 = (xn ** 2).sum()
    return xn * (1 + k1 * r2 + k2 * r2 * r2)


def ground_from_pixel(u: float, v: float, k: np.ndarray = K, h: float = H_CAM) -> np.ndarray:
    """逆透视：相机水平朝前（y 轴向下），地面是 Y = h 的平面。像素反投影成射线 K⁻¹[u, v, 1]，和地面求交。
    返回地面点在相机坐标系里的 (X, Z)（横向、前向）。"""
    ray = np.linalg.inv(k) @ np.array([u, v, 1.0])
    s = h / ray[1]
    return np.array([s * ray[0], s * ray[2]])


def lss_lift(depth_logits: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Lift-Splat-Shoot 的 lift：每个像素一个深度分布 α = softmax(logits)（D 个深度格）和一个 C 维特征 c，
    外积 α cᵀ（D×C）就是这条射线上每个深度格分到的特征。"""
    a = np.exp(depth_logits - depth_logits.max())
    a /= a.sum()
    return np.outer(a, feat)


def main() -> None:
    print("\n第一步 针孔投影：相机坐标 → 像素")
    for p in ([0.0, 0.0, 5.0], [1.0, 0.5, 5.0], [1.0, 0.5, 10.0]):
        print(f"  相机坐标 {p} → 像素 {np.round(project(np.array(p)), 2).tolist()}")

    print("\n第二步 外参：世界 → 相机")
    th = np.radians(10)
    r = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
    pc = world_to_cam(np.array([0.0, 0.0, 5.0]), r, np.array([0.2, 0.0, 0.0]))
    print(f"  相机绕竖直轴转 10°、平移 0.2 m：世界点 (0, 0, 5) → 相机坐标 {np.round(pc, 4).tolist()} → 像素 {np.round(project(pc), 2).tolist()}")

    print("\n第三步 径向畸变：离中心越远偏得越多")
    for xn in ([0.1, 0.0], [0.4, 0.3], [0.6, 0.45]):
        xd = distort(np.array(xn))
        print(f"  归一化坐标 {xn}（r = {np.hypot(*xn):.2f}）→ {np.round(xd, 4).tolist()}，像素偏移 {500 * np.linalg.norm(xd - xn):.1f}")

    print("\n第四步 逆透视：地面上的像素能直接反算出距离")
    for v in (300, 340, 400, 480):
        g = ground_from_pixel(320, v)
        print(f"  像素 (320, {v})：地面前方 {g[1]:.2f} m")
    print(f"  像素 (420, 340)：地面点 (横向, 前向) = {np.round(ground_from_pixel(420, 340), 3).tolist()} m")

    print("\n第五步 Lift-Splat-Shoot：没有深度就预测一个深度分布")
    out = lss_lift(np.array([0.0, 2.0, 1.0, -1.0]), np.array([1.0, -0.5]))
    print(f"  4 个深度格的 logits [0, 2, 1, −1]、特征 [1, −0.5]：每格分到的特征\n{np.round(out, 4)}")


if __name__ == "__main__":
    main()
