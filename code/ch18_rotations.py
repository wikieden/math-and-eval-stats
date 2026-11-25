"""讲义：第 18 章 3D 旋转与刚体变换（notes/lessons/18-rotations-rigid-transforms.md）
运行：uv run python ch18_rotations.py

约定：四元数写成 (w, x, y, z)；欧拉角用 ZYX（先绕 z 转 yaw，再绕 y 转 pitch，再绕 x 转 roll），R = Rz·Ry·Rx。
"""

import math

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyx(yaw, pitch, roll):
    """第二步：R = Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def skew(u):
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


def rodrigues(axis, angle):
    """第三步：R = I + sin θ·[u]× + (1 - cos θ)·[u]×²。"""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    K = skew(u)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def quat_from_axis_angle(axis, angle):
    """第四步：q = (cos θ/2, sin θ/2 · u)。"""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * u])


def quat_mul(a, b):
    """第四步：四元数乘法（Hamilton 约定），q_a ⊗ q_b 表示先转 b 再转 a。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotate(q, v):
    """第四步：v' = q ⊗ (0, v) ⊗ q*。"""
    qc = q * np.array([1, -1, -1, -1])
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), qc)[1:]


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_from_6d(a1, a2):
    """第五步：6D 表示 → 旋转矩阵（Gram-Schmidt：b1 = a1 归一化，b2 = a2 去掉 b1 分量再归一化，b3 = b1 × b2）。"""
    b1 = np.asarray(a1, float) / np.linalg.norm(a1)
    b2 = np.asarray(a2, float) - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def se3(R, t):
    """第六步：4×4 齐次变换 T = [[R, t], [0, 1]]。"""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def se3_inv(T):
    """第六步：T⁻¹ = [[Rᵀ, -Rᵀt], [0, 1]]。"""
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def main() -> None:
    R = rot_z(math.radians(90))
    print("第一步 旋转矩阵：绕 z 转 90°")
    print(f"  R·(1,0,0) = {np.round(R @ [1, 0, 0], 3).tolist()}；RᵀR = I：{np.allclose(R.T @ R, np.eye(3))}；det = {np.linalg.det(R):.1f}")
    A, B = rot_z(math.radians(90)), rot_x(math.radians(90))
    print(f"  顺序有关：Rz·Rx 和 Rx·Rz 相同吗？{np.allclose(A @ B, B @ A)}")
    print("\n第二步 万向锁：pitch = 90° 时，yaw 和 roll 同时加 20° 得到同一个姿态（ZYX 顺序）")
    R1 = euler_zyx(math.radians(30), math.radians(90), math.radians(10))
    R2 = euler_zyx(math.radians(50), math.radians(90), math.radians(30))
    print(f"  yaw30 roll10 与 yaw50 roll30 相同：{np.allclose(R1, R2)}（只有 yaw − roll 起作用）")
    print("\n第三步 轴角（Rodrigues）：绕 z 转 90°")
    print(f"  与 rot_z(90°) 相同：{np.allclose(rodrigues([0, 0, 1], math.radians(90)), R)}")
    print("\n第四步 四元数")
    q = quat_from_axis_angle([0, 0, 1], math.radians(90))
    print(f"  q = {np.round(q, 4).tolist()}；转 (1,0,0) → {np.round(quat_rotate(q, np.array([1.0, 0, 0])), 3).tolist()}")
    print(f"  q 和 -q 是同一个旋转：{np.allclose(quat_to_matrix(q), quat_to_matrix(-q))}")
    qx = quat_from_axis_angle([1, 0, 0], math.radians(90))
    print(f"  先绕 x 再绕 z = q_z ⊗ q_x，对应矩阵 Rz·Rx：{np.allclose(quat_to_matrix(quat_mul(q, qx)), A @ B)}")
    print("\n第五步 6D 表示：网络随便输出两列，Gram-Schmidt 后一定是旋转矩阵")
    M = matrix_from_6d([1.0, 0.2, 0.0], [0.3, 1.0, 0.1])
    print(f"  MᵀM = I：{np.allclose(M.T @ M, np.eye(3))}，det = {np.linalg.det(M):.3f}")
    print("\n第六步 SE(3)：相机在世界坐标系 绕 z 转 90°、平移 (1, 2, 0)；点在相机系 (1, 0, 0)")
    T_wc = se3(R, [1, 2, 0])
    p_w = T_wc @ [1, 0, 0, 1]
    print(f"  世界坐标 = {np.round(p_w[:3], 3).tolist()}；再变回相机系 = {np.round((se3_inv(T_wc) @ p_w)[:3], 3).tolist()}")


if __name__ == "__main__":
    main()
