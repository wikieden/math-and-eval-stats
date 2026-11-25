"""讲义：第 34 章 信息几何（notes/lessons/34-information-geometry.md）
运行：uv run python ch34_information_geometry.py

例子：伯努利分布的 Fisher 信息和自然梯度（拟合硬币正面概率）；高斯策略 N(μ, σ²) 在 (μ, log σ) 坐标下的 Fisher 矩阵和
TRPO 的信任域步；softmax 策略的 Fisher 矩阵（奇异）。
"""

import numpy as np


def fisher_bernoulli(p: float, n: int = 400000, seed: int = 0) -> tuple[float, float]:
    """伯努利(p) 的 Fisher 信息：解析 1/(p(1−p))，和 score² 的样本均值比。score = x/p − (1−x)/(1−p)。"""
    x = (np.random.default_rng(seed).random(n) < p).astype(float)
    s = x / p - (1 - x) / (1 - p)
    return 1 / (p * (1 - p)), float((s ** 2).mean())


def kl_bern(p: float, q: float) -> float:
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_quadratic(p: float, d: float) -> tuple[float, float]:
    """KL(p ‖ p + d) 和二阶近似 ½ F d²。"""
    return kl_bern(p, p + d), 0.5 * d * d / (p * (1 - p))


def fit_coin(method: str, p_hat: float = 0.9, p0: float = 0.5, lr: float = 0.05, tol: float = 1e-6, max_it: int = 100000) -> int:
    """最大化平均对数似然 ℓ(p) = p̂ log p + (1 − p̂) log(1 − p)，看要几步到 |p − p̂| < tol。
    gd_p：在 p 上做梯度上升（∇ = (p̂ − p)/(p(1−p))，lr = 0.05）；gd_logit：在 logit η 上做（∇ = p̂ − p，lr = 0.05）；
    natural：自然梯度 F⁻¹∇，在 p 坐标里 = p̂ − p，步长 1。"""
    p = p0
    for it in range(1, max_it + 1):
        if method == "gd_p":
            p = np.clip(p + lr * (p_hat - p) / (p * (1 - p)), 1e-6, 1 - 1e-6)
        elif method == "gd_logit":
            eta = np.log(p / (1 - p)) + lr * (p_hat - p)
            p = 1 / (1 + np.exp(-eta))
        else:
            p = p + 1.0 * (p_hat - p)
        if abs(p - p_hat) < tol:
            return it
    return max_it


def kl_gauss(m1, s1, m2, s2) -> float:
    return float(np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5)


def trpo_step(g: np.ndarray, sigma: float = 0.5, eps: float = 0.01) -> tuple[np.ndarray, float]:
    """高斯策略参数 θ = (μ, log σ)，Fisher = diag(1/σ², 2)。信任域 max gᵀδ s.t. ½δᵀFδ ≤ ε 的解：
    δ = √(2ε / (gᵀF⁻¹g))·F⁻¹g。返回 (δ, 走完这一步后的真实 KL)。"""
    f = np.diag([1 / sigma ** 2, 2.0])
    nat = np.linalg.solve(f, g)
    d = np.sqrt(2 * eps / (g @ nat)) * nat
    return d, kl_gauss(0.0, sigma, d[0], sigma * np.exp(d[1]))


def softmax_fisher(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """softmax 策略对 logits 的 Fisher = diag(p) − ppᵀ；所有 logits 同加一个常数分布不变，所以有一个零特征值。"""
    f = np.diag(p) - np.outer(p, p)
    return f, np.linalg.eigvalsh(f)


def main() -> None:
    print("\n第一步 Fisher 信息：score 的方差 = 对数似然曲率的期望")
    for p in (0.5, 0.9):
        a, m = fisher_bernoulli(p)
        print(f"  伯努利 p = {p}：解析 1/(p(1−p)) = {a:.4f}，score² 的样本均值 {m:.4f}")

    print("\n第二步 KL 的局部二阶近似：KL(p ‖ p + d) ≈ ½ F d²")
    for p, d in ((0.5, 0.1), (0.5, 0.01), (0.9, 0.05)):
        e, q = kl_quadratic(p, d)
        print(f"  p = {p}、d = {d}：KL = {e:.6f}，½Fd² = {q:.6f}")

    print("\n第三步 自然梯度：按 KL 量步长，换参数化也不变")
    for m in ("gd_p", "gd_logit", "natural"):
        print(f"  {m}：从 0.5 拟合到 p̂ = 0.9 用了 {fit_coin(m)} 步")

    print("\n第四步 TRPO：在 KL 球里走最陡的方向")
    d, kl = trpo_step(np.array([1.0, 1.0]))
    print(f"  σ = 0.5，梯度 g = [1, 1]，ε = 0.01：δ = {np.round(d, 4).tolist()}，二阶近似 KL = 0.01，真实 KL = {kl:.5f}")
    plain = np.array([1.0, 1.0]) * np.sqrt(2 * 0.01 / (np.array([1.0, 1.0]) @ np.diag([4.0, 2.0]) @ np.array([1.0, 1.0])))
    print(f"  对比：沿普通梯度方向走到同样的 KL，δ = {np.round(plain, 4).tolist()}（μ 方向走得更少）")

    print("\n第五步 softmax 策略的 Fisher 是奇异的")
    f, ev = softmax_fisher(np.array([0.7, 0.2, 0.1]))
    print(f"  p = [0.7, 0.2, 0.1]：特征值 {(np.round(ev, 4) + 0.0).tolist()}（有一个 0：logits 整体平移不改变策略）")


if __name__ == "__main__":
    main()
