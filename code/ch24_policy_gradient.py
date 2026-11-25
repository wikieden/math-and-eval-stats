"""讲义：第 24 章 策略梯度推导（notes/lessons/24-policy-gradient.md）
运行：uv run python ch24_policy_gradient.py

玩具问题：两臂老虎机。动作 0 成功率 0.3，动作 1 成功率 0.8（成功奖励 1，失败 0）。
策略 π(a=1) = sigmoid(θ)。
"""

import math

import numpy as np

P_SUCCESS = np.array([0.3, 0.8])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def objective(theta: float) -> float:
    """第一步：J(θ) = Σ_a π(a)·E[r | a]。"""
    p1 = sigmoid(theta)
    return (1 - p1) * P_SUCCESS[0] + p1 * P_SUCCESS[1]


def true_gradient(theta: float) -> float:
    """第一步：直接求导 dJ/dθ = σ'(θ)·(0.8 - 0.3)，σ' = σ(1-σ)。"""
    p1 = sigmoid(theta)
    return p1 * (1 - p1) * (P_SUCCESS[1] - P_SUCCESS[0])


def score(theta: float, a: int) -> float:
    """第二步：∇θ log π(a)。a=1 时 1-σ(θ)，a=0 时 -σ(θ)。"""
    p1 = sigmoid(theta)
    return (1 - p1) if a == 1 else -p1


def reinforce_samples(theta: float, n: int, baseline: float = 0.0, seed: int = 0) -> np.ndarray:
    """第三、四步：每个样本的梯度估计 (r - b)·∇log π(a)。"""
    rng = np.random.default_rng(seed)
    p1 = sigmoid(theta)
    a = (rng.random(n) < p1).astype(int)
    r = (rng.random(n) < P_SUCCESS[a]).astype(float)
    s = np.where(a == 1, 1 - p1, -p1)
    return (r - baseline) * s


def ppo_clip_term(ratio: float, adv: float, eps: float = 0.2) -> float:
    """第五步：PPO 截断目标 min(r·A, clip(r, 1-ε, 1+ε)·A)。"""
    return min(ratio * adv, max(min(ratio, 1 + eps), 1 - eps) * adv)


def grpo_advantages(rewards) -> np.ndarray:
    """第五步：GRPO 组内归一化优势 (r - mean) / std。"""
    r = np.asarray(rewards, float)
    return (r - r.mean()) / (r.std() + 1e-8)


def main() -> None:
    theta = 0.0
    print("第一步 目标与真梯度：θ = 0，π(a=1) = 0.5")
    print(f"  J = {objective(theta):.3f}，dJ/dθ = {true_gradient(theta):.4f}")
    print("\n第二步 似然比：∇log π")
    print(f"  a=1：{score(theta, 1):+.2f}；a=0：{score(theta, 0):+.2f}")
    print(f"  精确期望 Σ π(a)·E[r|a]·∇log π(a) = {0.5 * 0.8 * 0.5 + 0.5 * 0.3 * (-0.5):.4f}  ← 和真梯度一样")
    print("\n第三、四步 REINFORCE 采样估计（每次 10 个样本，重复 20000 次）")
    for b in (0.0, objective(theta)):
        est = reinforce_samples(theta, 200_000, baseline=b).reshape(-1, 10).mean(axis=1)
        print(f"  基线 b = {b:.2f}：均值 {est.mean():.4f}，标准差 {est.std():.4f}")
    print("\n第五步 PPO 截断（ε = 0.2）")
    for ratio, adv in [(1.1, 1.0), (1.5, 1.0), (0.5, 1.0), (0.5, -1.0), (1.5, -1.0)]:
        print(f"  ratio {ratio}, A {adv:+.0f} → {ppo_clip_term(ratio, adv):+.2f}")
    print(f"  GRPO：一组 4 条回答奖励 [1, 0, 0, 1] → 优势 {np.round(grpo_advantages([1, 0, 0, 1]), 3).tolist()}")


if __name__ == "__main__":
    main()
