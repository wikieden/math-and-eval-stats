"""讲义：第 32 章 随机微积分（notes/lessons/32-stochastic-calculus.md）
运行：uv run python ch32_stochastic_calculus.py

例子：布朗运动的二次变差、∫W dW 的左端点和与中点和、几何布朗运动的 Euler-Maruyama 误差、带漂移的布朗运动
对照 Fokker-Planck 的解、一个一维 VE 型前向过程的反向 SDE 与概率流 ODE。
里程碑 ME-M21 的“OU 过程 + Fokker-Planck 平稳分布”留给自己做。
"""

import numpy as np


def quadratic_variation(n: int, t: float = 1.0, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """把 [0, t] 切成 n 段，Σ(ΔW)² 的均值和标准差：均值 = t，标准差 = t√(2/n) → 0。"""
    rng = np.random.default_rng(seed)
    dw = rng.normal(0, np.sqrt(t / n), (reps, n))
    qv = (dw ** 2).sum(axis=1)
    return float(qv.mean()), float(qv.std())


def ito_vs_strat(n: int = 1000, t: float = 1.0, reps: int = 4000, seed: int = 1) -> tuple[float, float, float]:
    """∫_0^T W dW：左端点和（Itô）= (W_T² − T)/2，期望 0；中点（W_k + W_{k+1})/2 的和（Stratonovich）= W_T²/2，期望 T/2。
    返回 (Itô 和的均值, Stratonovich 和的均值, Itô 和与 (W_T² − T)/2 的最大差)。"""
    rng = np.random.default_rng(seed)
    dw = rng.normal(0, np.sqrt(t / n), (reps, n))
    w = np.concatenate([np.zeros((reps, 1)), dw.cumsum(axis=1)], axis=1)
    ito = (w[:, :-1] * dw).sum(axis=1)
    strat = ((w[:, :-1] + w[:, 1:]) / 2 * dw).sum(axis=1)
    return float(ito.mean()), float(strat.mean()), float(np.abs(ito - (w[:, -1] ** 2 - t) / 2).max())


def gbm_strong_error(n: int, mu: float = 0.5, sigma: float = 0.8, t: float = 1.0, reps: int = 4000, seed: int = 2) -> float:
    """几何布朗运动 dX = μX dt + σX dW，X_0 = 1。Itô 引理给精确解 X_T = exp((μ − σ²/2)T + σW_T)。
    Euler-Maruyama 用同一条布朗路径，返回强误差 E|X_T^EM − X_T|（阶 1/2）。"""
    rng = np.random.default_rng(seed)
    h = t / n
    dw = rng.normal(0, np.sqrt(h), (reps, n))
    x = np.ones(reps)
    for k in range(n):
        x = x + mu * x * h + sigma * x * dw[:, k]
    exact = np.exp((mu - sigma ** 2 / 2) * t + sigma * dw.sum(axis=1))
    return float(np.abs(x - exact).mean())


def exp_w_mean(t: float = 1.0, n: int = 400000, seed: int = 3) -> float:
    """E[e^{W_t}] = e^{t/2}（不是 e^0 = 1）：Itô 引理多出来的那项 ½f″ dt。"""
    return float(np.exp(np.random.default_rng(seed).normal(0, np.sqrt(t), n)).mean())


def drift_bm(mu: float = 1.0, sigma: float = 0.5, t: float = 2.0, n: int = 200, reps: int = 20000, seed: int = 4):
    """dX = μ dt + σ dW，X_0 = 0。Fokker-Planck ∂p/∂t = −μ ∂p/∂x + ½σ² ∂²p/∂x² 的解是 N(μt, σ²t)。返回 EM 的 (均值, 方差)。"""
    rng = np.random.default_rng(seed)
    h = t / n
    x = np.zeros(reps)
    for _ in range(n):
        x = x + mu * h + sigma * rng.normal(0, np.sqrt(h), reps)
    return float(x.mean()), float(x.var())


S0, TT = 0.5, 1.0  # 数据 N(0, 0.5²)，前向 dX = dW 跑到 T = 1：p_t = N(0, s0² + t)


def reverse(kind: str, n: int = 500, reps: int = 20000, seed: int = 5) -> float:
    """从 x_T ~ N(0, s0² + T) 往回走到 t = 0，score ∇log p_t(x) = −x/(s0² + t) 用精确值。
    反向 SDE：dx = −∇log p_t dt（往回走所以符号翻转后写成 x += ∇log p·h）+ dW̄；
    概率流 ODE：dx/dt = −½∇log p_t，同样的边际分布但没有噪声。返回 t = 0 时的样本标准差（目标 0.5）。"""
    rng = np.random.default_rng(seed)
    h = TT / n
    x = rng.normal(0, np.sqrt(S0 ** 2 + TT), reps)
    for k in range(n, 0, -1):
        t = k * h
        sc = -x / (S0 ** 2 + t)
        if kind == "sde":
            x = x + sc * h + rng.normal(0, np.sqrt(h), reps)
        else:
            x = x + 0.5 * sc * h
    return float(x.std())


def main() -> None:
    print("\n第一步 布朗运动：(dW)² = dt")
    for n in (10, 100, 10000):
        m, s = quadratic_variation(n)
        print(f"  切 {n} 段：Σ(ΔW)² 均值 {m:.4f}、标准差 {s:.4f}")

    print("\n第二步 Itô 积分：取左端点")
    a, b, d = ito_vs_strat()
    print(f"  ∫W dW：左端点和的均值 {a:.4f}（Itô，理论 0），中点和的均值 {b:.4f}（Stratonovich，理论 0.5）")
    print(f"  左端点和与 (W_T² − T)/2 的最大差 {d:.4f}（只差离散化误差）")

    print("\n第三步 Itô 引理：df = f′dX + ½f″(dX)²")
    print(f"  E[e^(W_1)] ≈ {exp_w_mean():.4f}（理论 e^0.5 = {np.exp(0.5):.4f}，不是 1）")
    for n in (10, 100, 1000):
        print(f"  几何布朗运动 Euler-Maruyama {n} 步：强误差 {gbm_strong_error(n):.4f}")

    print("\n第四步 Fokker-Planck：密度怎么随时间变")
    m, v = drift_bm()
    print(f"  dX = 1·dt + 0.5·dW 跑到 t = 2：EM 均值 {m:.4f}、方差 {v:.4f}（Fokker-Planck 解 N(2, 0.5)）")

    print("\n第五步 反向 SDE 与概率流 ODE：用 score 把噪声变回数据")
    print(f"  数据 N(0, 0.5²)，前向 dX = dW 到 T = 1；从 N(0, 1.25) 往回走：")
    print(f"  反向 SDE 得到标准差 {reverse('sde'):.4f}，概率流 ODE 得到 {reverse('ode'):.4f}（目标 0.5）")


if __name__ == "__main__":
    main()
