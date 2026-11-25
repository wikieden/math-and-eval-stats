"""讲义：第 27 章 扩散模型推导（notes/lessons/27-diffusion-derivation.md）
运行：uv run python ch27_diffusion_derivation.py

例子：DDPM 的线性 β 调度（β 从 1e-4 到 0.02，T = 1000）。数据是一维高斯 N(2, 0.5²)，这时每个 x_t 的分布、score 和
E[x_0 | x_t] 都有闭式，可以拿精确的噪声预测去跑 DDIM（相当于“网络完美”）。
里程碑 ME-M21 的“蒙特卡洛模拟验证前向闭式和 Tweedie 公式”留给自己做；这里只用解析式对数。
"""

import numpy as np

T = 1000
BETA = np.linspace(1e-4, 0.02, T)
ALPHA = 1 - BETA
ABAR = np.cumprod(ALPHA)
M0, S0 = 2.0, 0.5  # 数据分布 N(2, 0.5²)


def abar(t: int) -> float:
    """t 从 1 到 T；ᾱ_0 = 1。"""
    return 1.0 if t == 0 else float(ABAR[t - 1])


def posterior_coef(t: int) -> tuple[float, float, float]:
    """q(x_{t−1} | x_t, x_0) = N(c0·x_0 + ct·x_t, β̃_t)。返回 (c0, ct, β̃_t)。"""
    b, a, ab, ab1 = BETA[t - 1], ALPHA[t - 1], abar(t), abar(t - 1)
    return np.sqrt(ab1) * b / (1 - ab), np.sqrt(a) * (1 - ab1) / (1 - ab), (1 - ab1) * b / (1 - ab)


def eps_weight(t: int) -> float:
    """ELBO 第 t 项 = ‖μ̃ − μ_θ‖²/(2σ_t²)，用 ε 参数化后 = w_t·‖ε − ε_θ‖²，σ_t² = β_t：w_t = β_t / (2 α_t (1 − ᾱ_t))。"""
    b, a, ab = BETA[t - 1], ALPHA[t - 1], abar(t)
    return b * b / (2 * b * a * (1 - ab))


def xt_stats(t: int) -> tuple[float, float]:
    """数据 N(m, s²) 时 x_t = √ᾱ x_0 + √(1−ᾱ) ε 的分布 N(√ᾱ m, ᾱ s² + 1 − ᾱ)。"""
    ab = abar(t)
    return np.sqrt(ab) * M0, ab * S0 ** 2 + 1 - ab


def score(x, t: int):
    m, v = xt_stats(t)
    return -(x - m) / v


def posterior_mean_x0(x, t: int):
    """高斯共轭直接算 E[x_0 | x_t]：x_t = √ᾱ x_0 + 噪声。"""
    ab = abar(t)
    prec = 1 / S0 ** 2 + ab / (1 - ab)
    return (M0 / S0 ** 2 + np.sqrt(ab) * x / (1 - ab)) / prec


def tweedie_x0(x, t: int):
    """Tweedie：E[x_0 | x_t] = (x_t + (1 − ᾱ_t)∇log p_t(x_t)) / √ᾱ_t。"""
    ab = abar(t)
    return (x + (1 - ab) * score(x, t)) / np.sqrt(ab)


def eps_hat(x, t: int):
    """完美的噪声预测：ε̂ = −√(1 − ᾱ_t)·∇log p_t(x_t)。"""
    return -np.sqrt(1 - abar(t)) * score(x, t)


def ddim(steps: int, n: int = 20000, seed: int = 0, return_samples: bool = False):
    """DDIM（η = 0）从 x_T ~ N(0, 1) 出发，在 steps 个均匀时间点上确定性地走回 t = 0。返回样本的 (均值, 标准差)。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    ts = np.linspace(T, 0, steps + 1).round().astype(int)
    for t, s in zip(ts[:-1], ts[1:]):
        e = eps_hat(x, t)
        x0 = (x - np.sqrt(1 - abar(t)) * e) / np.sqrt(abar(t))
        x = np.sqrt(abar(s)) * x0 + np.sqrt(1 - abar(s)) * e
    if return_samples:
        return x
    return float(x.mean()), float(x.std())


def main() -> None:
    print("\n第一步 前向闭式：x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε")
    for t in (1, 100, 500, 1000):
        ab = abar(t)
        print(f"  t = {t}：ᾱ = {ab:.6f}，信号系数 {np.sqrt(ab):.4f}，噪声系数 {np.sqrt(1 - ab):.4f}，SNR = {ab / (1 - ab):.3g}")

    print("\n第二步 后验 q(x_{t−1} | x_t, x_0) 是高斯，均值是 x_0、x_t 的加权")
    for t in (2, 500, 1000):
        c0, ct, bt = posterior_coef(t)
        print(f"  t = {t}：均值 = {c0:.4f}·x_0 + {ct:.4f}·x_t，方差 β̃ = {bt:.6f}（β_t = {BETA[t - 1]:.6f}）")

    print("\n第三步 ELBO 的每一项 = w_t·‖ε − ε_θ‖²；DDPM 把 w_t 全换成 1（简化损失）")
    for t in (1, 10, 100, 1000):
        print(f"  t = {t}：w_t = {eps_weight(t):.4f}")

    print("\n第四步 Tweedie：用 score 估 x_0")
    for t, x in ((100, 1.5), (500, 0.5)):
        print(f"  t = {t}、x_t = {x}：共轭公式 E[x_0|x_t] = {posterior_mean_x0(x, t):.4f}，Tweedie = {tweedie_x0(x, t):.4f}")

    print("\n第五步 DDIM：用 ε̂ 估 x_0，再按前向闭式跳到更早的时间点（不加新噪声）")
    for k in (1, 2, 10, 50):
        m, s = ddim(k)
        print(f"  {k} 步：样本均值 {m:.4f}、标准差 {s:.4f}（目标 2、0.5）")


if __name__ == "__main__":
    main()
