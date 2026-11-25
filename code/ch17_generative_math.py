"""讲义：第 17 章 生成模型的数学（notes/lessons/17-generative-math.md）
运行：uv run python ch17_generative_math.py

例子：一维潜变量模型 z ~ N(0, 1)、x | z ~ N(z, 0.5²)，观测 x = 1.2；∇_μ E[z²] 的两种梯度估计；
噪声 N(0, 1) 到数据 N(2, 0.5²) 的直线插值，验证连续性方程和"条件目标与边际目标只差常数"。
里程碑 ME-M13 的一维两峰 score 和 flow matching 训练留给自己做。
"""

import math

import numpy as np

SIG_X = 0.5   # 似然的标准差
X_OBS = 1.2
M1, S1 = 2.0, 0.5  # 数据分布 N(2, 0.5²)


def log_normal(x, m, s):
    return -0.5 * math.log(2 * math.pi * s * s) - 0.5 * ((x - m) / s) ** 2


def log_evidence(x: float = X_OBS) -> float:
    """log p(x)：z 积分掉以后 x ~ N(0, 1 + σ²)。"""
    return log_normal(x, 0.0, math.sqrt(1 + SIG_X**2))


def posterior(x: float = X_OBS) -> tuple[float, float]:
    """真后验 p(z | x) = N(x/(1 + σ²), σ²/(1 + σ²))，返回 (均值, 标准差)。"""
    v = SIG_X**2 / (1 + SIG_X**2)
    return x / (1 + SIG_X**2), math.sqrt(v)


def elbo(m: float, s: float, x: float = X_OBS) -> float:
    """q = N(m, s²) 时的 ELBO = E_q[log p(x|z)] − KL(q ‖ N(0, 1))，两项都有闭式。"""
    rec = -0.5 * math.log(2 * math.pi * SIG_X**2) - ((x - m) ** 2 + s * s) / (2 * SIG_X**2)
    kl = 0.5 * (m * m + s * s - 1 - math.log(s * s))
    return rec - kl


def kl_gauss(m1, s1, m2, s2) -> float:
    """KL(N(m1, s1²) ‖ N(m2, s2²))。"""
    return math.log(s2 / s1) + (s1 * s1 + (m1 - m2) ** 2) / (2 * s2 * s2) - 0.5


def grad_estimators(mu: float = 1.0, n: int = 100000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """估 ∇_μ E_{z~N(μ,1)}[z²]（真值 2μ）。返回 {名字: (均值, 单样本标准差)}。"""
    eps = np.random.default_rng(seed).standard_normal(n)
    z = mu + eps
    reparam = 2 * z                       # d/dμ (μ + ε)² = 2(μ + ε)
    score = z**2 * (z - mu)               # f(z) · ∇_μ log N(z; μ, 1)
    base = (z**2 - (mu**2 + 1)) * (z - mu)  # 减去基线 E[z²] = μ² + 1
    return {k: (float(v.mean()), float(v.std())) for k, v in (("重参数化", reparam), ("score function", score), ("score function + 基线", base))}


def path_stats(t: float) -> tuple[float, float]:
    """x_t = (1 − t)·x₀ + t·x₁，x₀ ~ N(0, 1)、x₁ ~ N(M1, S1²) 独立：x_t ~ N(t·M1, (1 − t)² + t²·S1²)。返回 (均值, 标准差)。"""
    return t * M1, math.sqrt((1 - t) ** 2 + t * t * S1 * S1)


def marginal_velocity(x, t: float):
    """高斯路径的边际速度 v = μ'_t + (σ'_t/σ_t)·(x − μ_t)。"""
    mu, sd = path_stats(t)
    dsd = (-(1 - t) + t * S1 * S1) / sd
    return M1 + dsd / sd * (x - mu)


def continuity_residual(x: float, t: float, h: float = 1e-4) -> tuple[float, float]:
    """有限差分核对连续性方程 ∂p/∂t + ∂(p·v)/∂x = 0。返回 (∂p/∂t, ∂(pv)/∂x)。"""
    p = lambda xx, tt: math.exp(log_normal(xx, *path_stats(tt)))
    dpdt = (p(x, t + h) - p(x, t - h)) / (2 * h)
    flux = lambda xx: p(xx, t) * marginal_velocity(xx, t)
    return dpdt, (flux(x + h) - flux(x - h)) / (2 * h)


def fm_vs_cfm(t: float = 0.5, cs=(0.0, 0.5, 1.0, 2.0), n: int = 400000, seed: int = 0):
    """对一族 v_c(x) = c·x：条件目标 E‖v_c − (x₁ − x₀)‖² 和边际目标 E‖v_c − v(x)‖²，差值和 c 无关。返回 [(c, CFM, FM, 差)]。"""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.normal(M1, S1, n)
    xt = (1 - t) * x0 + t * x1
    out = []
    for c in cs:
        cfm = float(((c * xt - (x1 - x0)) ** 2).mean())
        fm = float(((c * xt - marginal_velocity(xt, t)) ** 2).mean())
        out.append((c, cfm, fm, cfm - fm))
    return out


def main() -> None:
    print("\n第一步 ELBO：log p(x) = ELBO + KL(q ‖ 真后验)")
    pm, ps = posterior()
    print(f"  log p(x = 1.2) = {log_evidence():.4f}；真后验 N({pm:.2f}, {ps:.4f}²)")
    for m, s in ((pm, ps), (0.0, 1.0), (1.5, 0.3)):
        e = elbo(m, s)
        print(f"  q = N({m:.2f}, {s:.2f}²)：ELBO = {e:.4f}，差 {log_evidence() - e:.4f}，KL(q‖后验) = {kl_gauss(m, s, pm, ps):.4f}")

    print("\n第二步 两种梯度估计：∇_μ E[z²]，μ = 1，真值 2")
    for k, (mean, sd) in grad_estimators().items():
        print(f"  {k}：均值 {mean:.3f}，单样本标准差 {sd:.2f}")

    print("\n第三步 连续性方程：速度场推着密度走")
    for x, t in ((0.5, 0.3), (1.5, 0.7)):
        a, b = continuity_residual(x, t)
        print(f"  x = {x}、t = {t}：∂p/∂t = {a:+.5f}，∂(pv)/∂x = {b:+.5f}，相加 {a + b:+.1e}")

    print("\n第四步 flow matching：条件目标和边际目标只差一个常数")
    for c, cfm, fm, d in fm_vs_cfm():
        print(f"  v = {c}·x：条件目标 {cfm:.4f}，边际目标 {fm:.4f}，差 {d:.4f}")


if __name__ == "__main__":
    main()
