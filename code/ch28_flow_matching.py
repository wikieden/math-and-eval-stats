"""讲义：第 28 章 Flow matching 推导（notes/lessons/28-flow-matching.md）
运行：uv run python ch28_flow_matching.py

例子：沿用第 17 章的线性插值路径 x_t = (1 − t)·x_0 + t·x_1，x_0 ~ N(0, 1)（噪声），x_1 ~ N(2, 0.5²)（数据）。
两端都是高斯，边际速度场有闭式，可以拿来检查回归、采样和 reflow。
里程碑 ME-M21 的数值验证（前向闭式、Tweedie、OU 过程）不在这里。
"""

import numpy as np

M1, S1 = 2.0, 0.5


def marginal_velocity(x, t: float):
    """u_t(x) = E[x_1 − x_0 | x_t = x]。(x_1 − x_0, x_t) 联合高斯：
    E[x_1 − x_0] = m，Var(x_t) = (1 − t)² + t² s²，Cov(x_1 − x_0, x_t) = t s² − (1 − t)。"""
    var = (1 - t) ** 2 + t * t * S1 ** 2
    cov = t * S1 ** 2 - (1 - t)
    return M1 + cov / var * (x - t * M1)


def regress_cfm(t: float = 0.5, n: int = 200000, seed: int = 0) -> tuple[float, float, float, float]:
    """在条件目标 E‖v(x_t) − (x_1 − x_0)‖² 下，最优 v 是条件期望。用线性函数 v = α + β x 拟合样本，
    和闭式比。返回 (拟合 α, 拟合 β, 闭式 α, 闭式 β)。"""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = M1 + S1 * rng.standard_normal(n)
    xt = (1 - t) * x0 + t * x1
    beta, alpha = np.polyfit(xt, x1 - x0, 1)
    var = (1 - t) ** 2 + t * t * S1 ** 2
    cov = t * S1 ** 2 - (1 - t)
    b = cov / var
    return float(alpha), float(beta), M1 - b * t * M1, b


def euler_sample(steps: int, vel=marginal_velocity, n: int = 20000, seed: int = 1) -> tuple[float, float]:
    """从 x_0 ~ N(0, 1) 用 Euler 积分 dx/dt = v(x, t) 到 t = 1，返回 (均值, 标准差)。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    for i in range(steps):
        x = x + vel(x, i / steps) / steps
    return float(x.mean()), float(x.std())


def reflow_velocity(x, t: float):
    """把 ODE 的起点、终点重新配对（x_0 ↔ m + s·x_0，一维时 ODE 映射是单调的这一条），新路径是直线，
    速度 = x_1 − x_0 = m + (s − 1)x_0，沿直线不变；用 x_t 反解出 x_0 = (x − t m)/(1 + t(s − 1))。"""
    x0 = (x - t * M1) / (1 + t * (S1 - 1))
    return M1 + (S1 - 1) * x0


def path_energy(coupling: str, n: int = 200000, seed: int = 2) -> float:
    """E‖x_1 − x_0‖²：独立配对 vs 排序配对（一维的最优传输配对）。"""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = M1 + S1 * rng.standard_normal(n)
    if coupling == "ot":
        x0, x1 = np.sort(x0), np.sort(x1)
    return float(((x1 - x0) ** 2).mean())


def pi0_time_mean(s: float = 0.999, n: int = 400000, seed: int = 3) -> tuple[float, float]:
    """π0 训练时的时间采样：(s − τ)/s ~ Beta(1.5, 1)，即偏向 τ 小（噪声多）的一端。返回 (E[τ], P(τ < 0.5))。"""
    rng = np.random.default_rng(seed)
    tau = s * (1 - rng.beta(1.5, 1.0, n))
    return float(tau.mean()), float((tau < 0.5).mean())


def main() -> None:
    print("\n第一步 概率路径与边际速度场")
    for t in (0.0, 0.5, 0.9):
        b = (t * S1 ** 2 - (1 - t)) / ((1 - t) ** 2 + t * t * S1 ** 2)
        print(f"  t = {t}：u_t(x) = {M1 - b * t * M1:.4f} {"+" if b >= 0 else "−"} {abs(b):.4f}·x")

    print("\n第二步 条件目标的最优解就是条件期望：回归样本得到的就是边际速度场")
    a, b, ea, eb = regress_cfm()
    print(f"  t = 0.5：拟合 v = {a:.4f} − {-b:.4f}·x；闭式 {ea:.4f} − {-eb:.4f}·x")

    print("\n第三步 用 Euler 采样：边际路径是弯的，步数少会偏")
    for k in (1, 2, 10, 100):
        m, s = euler_sample(k)
        print(f"  {k} 步：均值 {m:.4f}、标准差 {s:.4f}（目标 2、0.5）")

    print("\n第四步 Rectified flow：重新配对后路径是直线，1 步就准")
    for k in (1, 2):
        m, s = euler_sample(k, reflow_velocity)
        print(f"  reflow 后 {k} 步：均值 {m:.4f}、标准差 {s:.4f}")
    print(f"  E‖x_1 − x_0‖²：独立配对 {path_energy('ind'):.4f}，排序（最优传输）配对 {path_energy('ot'):.4f}")

    print("\n第五步 π0 的动作头：时间采样偏向噪声多的一端")
    mt, pl = pi0_time_mean()
    print(f"  τ = s·(1 − Beta(1.5, 1))：E[τ] = {mt:.4f}，P(τ < 0.5) = {pl:.4f}（均匀采样是 0.5、0.5）")


if __name__ == "__main__":
    main()
