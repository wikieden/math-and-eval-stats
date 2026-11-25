"""讲义：第 10 章 信息论（notes/lessons/10-information-kl.md）
运行：uv run python ch10_information.py

例子都是编的：三类分布 P = [0.7, 0.2, 0.1]、Q = [0.4, 0.4, 0.2]；一个两峰分布拿单个高斯去拟合；一张 2×2 联合分布表。
"""

import math

import numpy as np

P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.4, 0.4, 0.2])
JOINT = np.array([[0.4, 0.1], [0.1, 0.4]])  # 编的：X、Y 各取 0/1 的联合概率


def entropy(p, base: float = math.e) -> float:
    """熵 H(P) = −Σ p log p（p = 0 的项算 0）。base = 2 时单位是比特。"""
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(base))


def cross_entropy(p, q) -> float:
    """交叉熵 H(P, Q) = −Σ p log q：用按 Q 设计的编码去编 P 的数据，平均要多少 nat。"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float(-(p[m] * np.log(q[m])).sum())


def kl(p, q) -> float:
    """KL(P‖Q) = Σ p log(p/q) = H(P, Q) − H(P) ≥ 0。"""
    return cross_entropy(p, q) - entropy(p)


def nll_of_labels(probs: np.ndarray, labels: np.ndarray) -> float:
    """分类器的平均负对数似然：−(1/n) Σ log q(y_i)，就是"标签的经验分布"和 q 的交叉熵。"""
    return float(-np.log(probs[np.arange(len(labels)), labels]).mean())


def mixture_pdf(x: np.ndarray) -> np.ndarray:
    """两峰：0.5·N(−2, 0.5²) + 0.5·N(2, 0.5²)。"""
    g = lambda m: np.exp(-0.5 * ((x - m) / 0.5) ** 2) / (0.5 * math.sqrt(2 * math.pi))
    return 0.5 * g(-2) + 0.5 * g(2)


def fit_gaussian(direction: str) -> tuple[float, float]:
    """用单个高斯 N(μ, s²) 拟合两峰：forward = 最小化 KL(P‖Q)，reverse = 最小化 KL(Q‖P)。网格搜索，返回 (μ, s)。"""
    x = np.linspace(-6, 6, 2401)
    dx = x[1] - x[0]
    p = mixture_pdf(x) + 1e-300
    best = (math.inf, 0.0, 0.0)
    for mu in np.linspace(-3, 3, 121):
        for s in np.linspace(0.2, 3.0, 141):
            q = np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi)) + 1e-300
            val = float(((p * np.log(p / q)) if direction == "forward" else (q * np.log(q / p))).sum() * dx)
            if val < best[0]:
                best = (val, float(mu), float(s))
    return best[1], best[2]


def mutual_information(joint: np.ndarray) -> float:
    """互信息 I(X; Y) = H(X) + H(Y) − H(X, Y)。"""
    return entropy(joint.sum(1)) + entropy(joint.sum(0)) - entropy(joint.ravel())


def main() -> None:
    print("\n第一步 熵：平均有多意外")
    print(f"  公平硬币 H = {entropy([0.5, 0.5], 2):.3f} 比特；9:1 的硬币 H = {entropy([0.9, 0.1], 2):.3f} 比特")
    print(f"  P = {P.tolist()}：H(P) = {entropy(P):.4f} nat")

    print("\n第二步 交叉熵 = 负对数似然")
    print(f"  H(P, Q) = {cross_entropy(P, Q):.4f} nat（比 H(P) 多出来的就是 KL）")
    probs = np.array([[0.7, 0.2, 0.1]] * 10)
    labels = np.array([0] * 7 + [1] * 2 + [2])  # 10 个样本，标签的经验分布正好是 P
    print(f"  10 个样本都预测 [0.7, 0.2, 0.1]，标签 7/2/1 个：平均 NLL = {nll_of_labels(probs, labels):.4f} = H(P)")

    print("\n第三步 KL 不对称")
    print(f"  KL(P‖Q) = {kl(P, Q):.4f}，KL(Q‖P) = {kl(Q, P):.4f}")

    print("\n第四步 正向 KL 盖住两峰，反向 KL 挑一个峰")
    for d in ("forward", "reverse"):
        mu, s = fit_gaussian(d)
        print(f"  {'KL(P‖Q) 正向' if d == 'forward' else 'KL(Q‖P) 反向'}：μ = {mu:+.2f}，s = {s:.2f}")

    print("\n第五步 互信息：知道 Y 以后，X 的不确定性少了多少")
    print(f"  联合表 {JOINT.tolist()}：I(X; Y) = {mutual_information(JOINT):.4f} nat = {mutual_information(JOINT) / math.log(2):.4f} 比特")


if __name__ == "__main__":
    main()
