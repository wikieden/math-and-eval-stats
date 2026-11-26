"""讲义：第 56 章 VIO（notes/lessons/56-vio.md）
运行：uv run python ch56_vio.py

例子：200 Hz 的 IMU、1 秒内恒定角速度和比力；预积分量 ΔR、Δv、Δp 不依赖起点的速度和姿态；
陀螺 / 加速度计偏置造成的误差随时间的增长（解析）；角度随机游走；偏置变化时的一阶修正。
里程碑 ME-M34 的“IMU 积分漂移模拟”留给自己做；这里漂移只用公式算。
"""

import numpy as np

from ch19_lie_groups import right_jacobian, so3_exp, so3_log

G = np.array([0.0, 0.0, -9.81])
DT = 1 / 200


def preintegrate(omega: np.ndarray, acc: np.ndarray, n: int = 200, bg: np.ndarray | None = None):
    """在 i 时刻的机体系里累积：ΔR ← ΔR·Exp((ω − b_g)dt)，Δv ← Δv + ΔR a dt，Δp ← Δp + Δv dt + ½ΔR a dt²（Forster 2015 式 (33)）。"""
    bg = np.zeros(3) if bg is None else bg
    dr, dv, dp = np.eye(3), np.zeros(3), np.zeros(3)
    for _ in range(n):
        dp = dp + dv * DT + 0.5 * dr @ acc * DT * DT
        dv = dv + dr @ acc * DT
        dr = dr @ so3_exp((omega - bg) * DT)
    return dr, dv, dp


def integrate_world(r0: np.ndarray, v0: np.ndarray, p0: np.ndarray, omega: np.ndarray, acc: np.ndarray, n: int = 200):
    """直接在世界系里逐步积分：a_world = R a + g。"""
    r, v, p = r0.copy(), v0.copy(), p0.copy()
    for _ in range(n):
        aw = r @ acc + G
        p = p + v * DT + 0.5 * aw * DT * DT
        v = v + aw * DT
        r = r @ so3_exp(omega * DT)
    return r, v, p


def drift_from_bias(t: float, bg: float = 0.01, ba: float = 0.05) -> tuple[float, float, float]:
    """恒定偏置的影响：陀螺偏置 → 角度误差 b_g·t；加速度计偏置 → 位置误差 ½b_a t²；
    陀螺偏置造成的倾斜让重力漏进水平方向：位置误差约 (1/6)·g·b_g·t³。"""
    return bg * t, 0.5 * ba * t * t, 9.81 * bg * t ** 3 / 6


def angle_random_walk(t: float, sigma: float = 0.005) -> float:
    """白噪声积分：角度标准差 σ_g·√t（σ_g 单位 rad/s/√Hz）。"""
    return sigma * np.sqrt(t)


def bias_update_error(dbg: np.ndarray, omega: np.ndarray) -> tuple[float, float]:
    """偏置改了 δb_g：重新积分 vs 一阶修正 ΔR(b + δb) ≈ ΔR(b)·Exp(J δb)，J = −Σ ΔR_{k+1→j}ᵀ J_r(ω_k dt) dt。返回 (不修正的误差, 一阶修正后的误差)（弧度）。"""
    dr0, _, _ = preintegrate(omega, np.zeros(3))
    dr_true, _, _ = preintegrate(omega, np.zeros(3), bg=dbg)
    jac = np.zeros((3, 3))
    rs = [np.eye(3)]
    for _ in range(200):
        rs.append(rs[-1] @ so3_exp(omega * DT))
    for k in range(200):
        r_kj = rs[k + 1].T @ rs[200]
        jac -= r_kj.T @ right_jacobian(omega * DT) * DT
    approx = dr0 @ so3_exp(jac @ dbg)
    return float(np.linalg.norm(so3_log(dr0.T @ dr_true))), float(np.linalg.norm(so3_log(approx.T @ dr_true)))


def main() -> None:
    omega, acc = np.array([0.0, 0.1, 0.5]), np.array([0.5, 0.0, 9.81])
    print("\n第一步 预积分：Δ 量只依赖这段时间的 IMU 读数")
    dr, dv, dp = preintegrate(omega, acc)
    print(f"  1 秒、ω = {omega.tolist()}、a = {acc.tolist()}：ΔR 的转角 {np.degrees(np.linalg.norm(so3_log(dr))):.2f}°，Δv = {np.round(dv, 4).tolist()}，Δp = {np.round(dp, 4).tolist()}")
    for v0 in (np.zeros(3), np.array([2.0, -1.0, 0.5])):
        r0 = so3_exp(np.array([0.1, 0.0, 0.3]))
        rj, vj, pj = integrate_world(r0, v0, np.zeros(3), omega, acc)
        pred = v0 * 1.0 + 0.5 * G + r0 @ dp
        print(f"  起始速度 {v0.tolist()}：直接积分 p_j = {np.round(pj, 4).tolist()}；用预积分拼 = {np.round(pred, 4).tolist()}")

    print("\n第二步 偏置：误差按 t、t²、t³ 增长")
    for t in (1, 10, 60):
        a, b, c = drift_from_bias(t)
        print(f"  t = {t} s：陀螺偏置 0.01 rad/s → 角度 {a:.2f} rad；加速度计偏置 0.05 → 位置 {b:.2f} m；倾斜漏重力 → 位置 {c:.1f} m")

    print("\n第三步 噪声：随机游走按 √t 增长")
    for t in (1, 100, 3600):
        print(f"  t = {t} s：角度随机游走标准差 {angle_random_walk(t):.3f} rad")

    print("\n第四步 偏置变了，不用重新积分：一阶修正")
    for db in (0.001, 0.01):
        e0, e1 = bias_update_error(np.array([db, 0, 0]), omega)
        print(f"  陀螺偏置改 {db} rad/s：不修正误差 {e0:.2e} rad，一阶修正后 {e1:.2e} rad")


if __name__ == "__main__":
    main()
