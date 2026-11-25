"""讲义：第 20 章 状态估计（notes/lessons/20-state-estimation.md）
运行：uv run python ch20_state_estimation.py

例子：一维 Kalman（位置不动，测量噪声 0.5²）；EKF 的线性化（测距 h(x) = √(x² + 1)）；一维粒子滤波；一条 4 个位姿的链加一个回环，当成最小二乘解。
里程碑 ME-M16 的二维跟踪和平面机器人 EKF 留给自己做。
"""

import numpy as np

Z = np.array([1.2, 0.8, 1.1, 0.9, 1.0])  # 编的 5 次测量（真位置 1.0）


def kf_step(m: float, p: float, z: float, q: float = 0.0, r: float = 0.25) -> tuple[float, float, float]:
    """一维 Kalman 一步：预测 m⁻ = m、P⁻ = P + Q；更新 K = P⁻/(P⁻ + R)，m = m⁻ + K(z − m⁻)，P = (1 − K)P⁻。返回 (m, P, K)。"""
    pp = p + q
    k = pp / (pp + r)
    return m + k * (z - m), (1 - k) * pp, k


def run_kf(m0: float = 0.0, p0: float = 1.0, q: float = 0.0) -> list[tuple[float, float, float]]:
    """把 Z 一个个喂进去。"""
    out, m, p = [], m0, p0
    for z in Z:
        m, p, k = kf_step(m, p, z, q)
        out.append((m, p, k))
    return out


def steady_gain(q: float, r: float = 0.25) -> float:
    """过程噪声 Q > 0 时，P 收敛到一个稳态，K 也收敛：解 P = (P + Q)R/(P + Q + R)。"""
    p = 1.0
    for _ in range(1000):
        _, p, k = kf_step(0.0, p, 0.0, q, r)
    return k


def ekf_update(m: float, p: float, z: float, r: float = 0.01) -> tuple[float, float, float]:
    """EKF 更新：测量 h(x) = √(x² + 1)，在 m 处线性化 H = m/√(m² + 1)。返回 (新 m, 新 P, H)。"""
    h = np.sqrt(m * m + 1)
    hj = m / h
    s = hj * p * hj + r
    k = p * hj / s
    return m + k * (z - h), (1 - k * hj) * p, hj


def particle_filter(z: float, n: int = 1000, r: float = 0.25, seed: int = 0) -> tuple[float, float, float]:
    """一维粒子滤波一步：粒子来自先验 N(0, 1)，按似然 N(z; x, r) 加权，再重采样。返回 (加权均值, 重采样前的有效样本数, 重采样后的均值)。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    w = np.exp(-0.5 * (z - x) ** 2 / r)
    w /= w.sum()
    ess = 1 / (w**2).sum()
    xr = x[rng.choice(n, n, p=w)]
    return float((w * x).sum()), float(ess), float(xr.mean())


def pose_graph() -> tuple[np.ndarray, np.ndarray]:
    """一维位姿链 x0..x3：x0 固定为 0；里程计说每步 +1.0、+1.1、+0.9；回环说 x3 − x0 = 2.7。当成线性最小二乘解。返回 (只用里程计, 加上回环)。"""
    a_odo = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1]], float)  # 未知 x1..x3
    b_odo = np.array([1.0, 1.1, 0.9])
    x_odo = np.linalg.solve(a_odo, b_odo)
    a = np.vstack([a_odo, [0, 0, 1]])
    b = np.r_[b_odo, 2.7]
    return x_odo, np.linalg.lstsq(a, b, rcond=None)[0]


def main() -> None:
    print("\n第一步 Kalman：预测一步、用测量修正一步")
    for i, (m, p, k) in enumerate(run_kf(), 1):
        print(f"  第 {i} 次测量 z = {Z[i - 1]}：K = {k:.3f}，估计 m = {m:.4f}，方差 P = {p:.4f}")
    print(f"  5 次测量的平均 = {Z.mean():.4f}（先验 N(0, 1) 把估计往 0 拉了一点）")

    print("\n第二步 过程噪声 Q：估计不会无限变准，K 收敛到一个稳态")
    for q in (0.0, 0.01, 0.1):
        print(f"  Q = {q}：稳态 K = {steady_gain(q):.3f}" if q > 0 else "  Q = 0：K 一直变小（第 n 次 K = 4/(4n + 1)），越来越相信旧估计")

    print("\n第三步 EKF：在当前估计处把测量线性化")
    m, p, hj = ekf_update(1.0, 0.5, 1.6)
    print(f"  先验 m = 1.0、P = 0.5，测到 z = 1.6（h(1.0) = {np.sqrt(2):.4f}）：H = {hj:.4f}，更新后 m = {m:.4f}，P = {p:.4f}")

    print("\n第四步 粒子滤波：加权、算有效样本数、重采样")
    wm, ess, rm = particle_filter(1.0)
    print(f"  先验 N(0, 1)、测量 z = 1.0：加权均值 {wm:.3f}（精确后验均值 0.8），有效样本数 {ess:.0f} / 1000，重采样后均值 {rm:.3f}")

    print("\n第五步 因子图 = 非线性最小二乘（这里是线性的）")
    xo, xl = pose_graph()
    print(f"  只用里程计：x = {np.round(xo, 3).tolist()}（x3 = 3.0，和回环说的 2.7 差 0.3）")
    print(f"  加上回环：x = {np.round(xl, 3).tolist()}（4 条约束各让 0.075：里程计每步少走一点，回环也松一点）")


if __name__ == "__main__":
    main()
