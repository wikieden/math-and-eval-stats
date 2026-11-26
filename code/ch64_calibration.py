"""讲义：第 64 章 多传感器标定与对齐（notes/lessons/64-calibration.md）
运行：uv run python ch64_calibration.py

例子：手眼标定 AX = XB（旋转用对数映射 + Kabsch、平移用最小二乘），无噪声和有噪声各一次；
两个 IMU 位姿之间按时间戳插值（位置线性、旋转 SLERP）；时间偏差和外参误差各造成多大的投影误差；BEVFusion 的点数账。
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def make_T(rotvec, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


X_TRUE = make_T([0.1, -0.2, 1.5], [0.05, 0.02, 0.10])   # 末端 → 相机（要标定的量）


def motions(n: int = 3, noise: float = 0.0, seed: int = 0):
    """机械臂末端的 n 次相对运动 A_i；相机看到的相对运动 B_i = X⁻¹ A_i X（可加噪声）。"""
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        a = make_T(rng.normal(0, 0.6, 3), rng.normal(0, 0.2, 3))
        b = np.linalg.inv(X_TRUE) @ a @ X_TRUE
        if noise:
            b = make_T(Rot.from_matrix(b[:3, :3]).as_rotvec() + rng.normal(0, noise, 3), b[:3, 3] + rng.normal(0, noise * 0.1, 3))
        A.append(a)
        B.append(b)
    return A, B


def hand_eye(A, B) -> np.ndarray:
    """R_A R_X = R_X R_B ⇒ log(R_A) = R_X log(R_B)：旋转轴之间是一个旋转，用 Kabsch 求（第 54 章）。
    平移：(R_A − I) t_X = R_X t_B − t_A，叠起来最小二乘。"""
    al = np.array([Rot.from_matrix(a[:3, :3]).as_rotvec() for a in A])
    be = np.array([Rot.from_matrix(b[:3, :3]).as_rotvec() for b in B])
    U, _, Vt = np.linalg.svd(be.T @ al)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    M = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    v = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    t, *_ = np.linalg.lstsq(M, v, rcond=None)
    return make_T(Rot.from_matrix(R).as_rotvec(), t)


def calib_error(X) -> tuple[float, float]:
    """(旋转误差°, 平移误差 mm)。"""
    dR = X[:3, :3].T @ X_TRUE[:3, :3]
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec()))), float(1000 * np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3]))


def interp_pose(t: float, t0: float = 0.100, t1: float = 0.110):
    """IMU 在 t0、t1 的位姿：位置 (0, 0, 0) → (0.2, 0, 0)，偏航 0° → 10°。返回 (位置, 偏航°, 线性插值四元数的模长)。"""
    s = (t - t0) / (t1 - t0)
    p = (1 - s) * np.zeros(3) + s * np.array([0.2, 0, 0])
    r = Rot.from_euler("z", [[0], [10]], degrees=True)
    yaw = Slerp([t0, t1], r)([t]).as_euler("zyx", degrees=True)[0, 0]
    q = (1 - s) * r[0].as_quat() + s * r[1].as_quat()
    return p, float(yaw), float(np.linalg.norm(q))


def time_offset_error(omega: float, dt: float, dist: float) -> float:
    """车以角速度 ω（rad/s）转弯，传感器时间差 dt 秒：距离 dist 处的点横向错开约 ω·dt·dist 米。"""
    return omega * dt * dist


KCAM = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
T_CAM_LIDAR = np.array([[0, -1, 0, 0], [0, 0, -1, -0.3], [1, 0, 0, -0.5], [0, 0, 0, 1.0]])  # 激光雷达 x 前 y 左 z 上 → 相机 x 右 y 下 z 前


def project(p_lidar, T=T_CAM_LIDAR, K=KCAM):
    pc = T @ np.append(p_lidar, 1.0)
    uv = K @ pc[:3]
    return uv[:2] / uv[2], pc[2]


def yaw_error_pixels(deg: float, p_lidar=(20.0, 2.0, 0.0)) -> float:
    """外参多了 deg 度的偏航误差，20 m 外的点在图像上偏多少像素。"""
    E = np.eye(4)
    E[:3, :3] = Rot.from_euler("z", deg, degrees=True).as_matrix()
    uv0, _ = project(np.array(p_lidar))
    uv1, _ = project(np.array(p_lidar), T=T_CAM_LIDAR @ E)
    return float(np.linalg.norm(uv1 - uv0))


def bevfusion_points(cams: int = 6, depth_bins: int = 118, h: int = 32, w: int = 88) -> int:
    """每个相机特征图像素 × 每个深度格 → 一个带特征的 3D 点（Lift-Splat 的“lift”）。"""
    return cams * depth_bins * h * w


def main() -> None:
    print("\n第一步 手眼标定 AX = XB")
    for n, noise in ((2, 0.0), (3, 0.0), (3, 0.01), (20, 0.01)):
        e = calib_error(hand_eye(*motions(n, noise)))
        print(f"  {n} 次运动、噪声 {noise}：旋转误差 {e[0]:.4f}°，平移误差 {e[1]:.3f} mm")

    print("\n第二步 时间同步与插值")
    p, yaw, qn = interp_pose(0.105)
    print(f"  激光雷达帧 t = 0.105 s，在 IMU 的 0.100 s 和 0.110 s 之间：位置 {p.round(3).tolist()}，SLERP 偏航 {yaw:.2f}°，直接线性插值四元数的模长 {qn:.6f}")
    for dt in (0.01, 0.05):
        print(f"  转弯 ω = 0.5 rad/s、时间差 {dt * 1000:.0f} ms：30 m 处错开 {time_offset_error(0.5, dt, 30):.2f} m")

    print("\n第三步 相机-激光雷达投影")
    uv, z = project(np.array([20.0, 2.0, 0.0]))
    print(f"  激光雷达点 (20, 2, 0) → 相机深度 {z:.1f} m，像素 ({uv[0]:.1f}, {uv[1]:.1f})")
    for deg in (0.1, 0.5, 1.0):
        print(f"  外参偏航误差 {deg}°：这个点偏 {yaw_error_pixels(deg):.1f} 像素")

    print("\n第四步 BEVFusion：统一到俯视图")
    print(f"  6 个相机 × 118 个深度格 × 32 × 88 的特征图 = {bevfusion_points():,} 个点要汇聚进 BEV 栅格")


if __name__ == "__main__":
    main()
