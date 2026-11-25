"""讲义：第 8 章 概率补课：期望方差、贝叶斯、MLE 与 MAP（notes/lessons/08-probability-mle-map.md）
运行：uv run python ch08_probability_mle_map.py
"""

import math

import numpy as np


def expectation(values, probs) -> float:
    """第一步：期望 E[X] = Σ x·P(x)。"""
    return float(sum(v * p for v, p in zip(values, probs)))


def variance(values, probs) -> float:
    """第二步：方差 Var(X) = E[(X - E[X])²] = E[X²] - E[X]²。"""
    m = expectation(values, probs)
    return float(sum(p * (v - m) ** 2 for v, p in zip(values, probs)))


def bayes_posterior(prior: float, sensitivity: float, specificity: float) -> float:
    """第三步：P(真成功 | 检测器说成功) = P(说成功 | 真成功)·P(真成功) / P(说成功)。"""
    says_yes = sensitivity * prior + (1 - specificity) * (1 - prior)
    return sensitivity * prior / says_yes


def bernoulli_log_likelihood(p: float, k: int, n: int) -> float:
    """第四步：伯努利对数似然 log L(p) = k·log p + (n-k)·log(1-p)。"""
    return k * math.log(p) + (n - k) * math.log(1 - p)


def mle_bernoulli(k: int, n: int) -> float:
    """第四步：令 d/dp log L = k/p - (n-k)/(1-p) = 0，得 p̂ = k/n。"""
    return k / n


def map_bernoulli(k: int, n: int, a: float, b: float) -> float:
    """第五步：Beta(a, b) 先验下的 MAP = (k + a - 1) / (n + a + b - 2)。"""
    return (k + a - 1) / (n + a + b - 2)


def bc_nll(probs_of_demo_actions) -> float:
    """第六步：行为克隆的损失 = 示范动作的平均负对数概率（离散动作）。"""
    return float(-np.mean(np.log(probs_of_demo_actions)))


def main() -> None:
    print("第一步 期望：一次抓取的奖励（成功 +1，掉落 -1，没抓到 0），概率 0.7 / 0.1 / 0.2")
    vals, ps = [1, -1, 0], [0.7, 0.1, 0.2]
    print(f"  E[R] = {expectation(vals, ps):.2f}")
    print("\n第二步 方差")
    print(f"  Var(R) = {variance(vals, ps):.2f}，标准差 = {math.sqrt(variance(vals, ps)):.3f}")
    print(f"  伯努利 p = 0.7：Var = p(1-p) = {variance([1, 0], [0.7, 0.3]):.2f}；30 次平均的方差 = 0.21/30 = {0.21 / 30:.4f}")
    print("\n第三步 贝叶斯：成功检测器 灵敏度 0.95、特异度 0.90，真成功率 0.70")
    print(f"  P(真成功 | 说成功) = {bayes_posterior(0.7, 0.95, 0.90):.3f}")
    print(f"  真成功率只有 0.20 时 = {bayes_posterior(0.2, 0.95, 0.90):.3f}")
    print("\n第四步 MLE：21/30 的对数似然")
    for p in [0.5, 0.6, 0.7, 0.8]:
        print(f"  log L({p}) = {bernoulli_log_likelihood(p, 21, 30):.3f}")
    print(f"  p̂_MLE = {mle_bernoulli(21, 30):.3f}")
    print("\n第五步 MAP：Beta(2, 2) 先验")
    print(f"  21/30 → {map_bernoulli(21, 30, 2, 2):.4f}；3/3 → MLE {mle_bernoulli(3, 3):.2f}，MAP {map_bernoulli(3, 3, 2, 2):.2f}")
    print("\n第六步 行为克隆 = MLE：示范动作在策略下的概率 0.9、0.6、0.3")
    print(f"  平均负对数似然 = {bc_nll([0.9, 0.6, 0.3]):.3f}")


if __name__ == "__main__":
    main()
