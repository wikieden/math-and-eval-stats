"""讲义：第 61 章 多模态融合的数学（notes/lessons/61-multimodal-fusion.md）
运行：uv run python ch61_multimodal_fusion.py

例子：两个深度读数的逆方差加权（和一维 Kalman 更新对比）；乘积型标签上“晚融合”拟合不了、“早融合”可以；
Perceiver 交叉注意力的计算量账和一个 2 个潜变量 × 4 个输入的小例子；训练时做不做模态 dropout，测试时缺一个模态的误差。
里程碑 ME-M36 的“逆方差加权融合两个传感器（仿真一串带噪读数、验证融合后的方差）”留给自己做；这里只算一组数。
"""

import numpy as np


def inv_var(z1: float, v1: float, z2: float, v2: float) -> tuple[float, float, float]:
    """权重 w_i ∝ 1/σ_i²；返回 (w1, 融合值, 融合方差)。"""
    w1 = (1 / v1) / (1 / v1 + 1 / v2)
    return w1, w1 * z1 + (1 - w1) * z2, 1 / (1 / v1 + 1 / v2)


def kalman_1d(x: float, p: float, z: float, r: float) -> tuple[float, float, float]:
    """一维 Kalman 更新：先验 (x, p)、观测 (z, r)。返回 (K, 后验均值, 后验方差)。"""
    k = p / (p + r)
    return k, x + k * (z - x), (1 - k) * p


def fusion_fit(seed: int = 0, n: int = 400) -> dict[str, float]:
    """标签 y = a·b（两个模态的乘积）。晚融合 = f(a) + g(b)（各自的多项式特征相加），早融合 = 拼起来再加交叉项。"""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    y = a * b
    ones = np.ones(n)
    late = np.stack([ones, a, a**2, a**3, b, b**2, b**3], 1)
    early = np.concatenate([late, (a * b)[:, None]], 1)
    out = {}
    for name, x in (("晚融合", late), ("早融合", early)):
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[name] = float(1 - np.mean((x @ w - y) ** 2) / np.var(y))
    return out


def attn_cost(m: int = 224 * 224, n: int = 512) -> dict[str, int]:
    """注意力分数矩阵的元素个数：输入自注意力 M²、交叉注意力 M·N、潜变量自注意力 N²。"""
    return {"自注意力 M²": m * m, "交叉注意力 M·N": m * n, "潜变量自注意力 N²": n * n}


Q = np.array([[1.0, 0.0], [0.0, 1.0]])           # 2 个潜变量的 query
K = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]])  # 4 个输入的 key（两个图像、一个融合点、一个噪声）


def cross_attn(q=Q, k=K) -> np.ndarray:
    s = q @ k.T / np.sqrt(q.shape[1])
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def modality_dropout(seed: int = 0, n: int = 2000, p: float = 0.5) -> dict[str, float]:
    """两个相关模态 a、b = a + 噪声，y = a + b。训练时以概率 p 把 b 置 0（同时给一个“b 在不在”的标志位）。
    返回：各训练方式在“两个都在 / b 缺失”时的测试 MSE。"""
    rng = np.random.default_rng(seed)

    def data(m):
        a = rng.normal(size=m)
        b = a + 0.3 * rng.normal(size=m)
        return a, b, a + b

    def feats(a, b, present):
        return np.stack([a, b * present, present], 1)

    a, b, y = data(n)
    full = feats(a, b, np.ones(n))
    keep = (rng.uniform(size=n) > p).astype(float)
    drop = feats(a, b, keep)
    at, bt, yt = data(n)
    test_full, test_miss = feats(at, bt, np.ones(n)), feats(at, bt, np.zeros(n))
    out = {}
    for name, x in (("不做 dropout", full), ("模态 dropout", drop)):
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[name + "·都在"] = float(np.mean((test_full @ w - yt) ** 2))
        out[name + "·缺 b"] = float(np.mean((test_miss @ w - yt) ** 2))
    return out


def main() -> None:
    print("\n第一步 逆方差加权 = Kalman 更新")
    w1, x, v = inv_var(2.0, 0.04, 2.3, 0.16)
    print(f"  相机深度 2.0 m（σ² = 0.04）、雷达 2.3 m（σ² = 0.16）：w₁ = {w1:.2f}，融合 {x:.3f} m，方差 {v:.3f}")
    k, xk, pk = kalman_1d(2.0, 0.04, 2.3, 0.16)
    print(f"  把相机当先验、雷达当观测做 Kalman 更新：K = {k:.2f}，后验 {xk:.3f} m，方差 {pk:.3f}")
    w1b, xb, _ = inv_var(2.0, 0.04, 2.3, 0.01)
    print(f"  雷达谎报自己 σ² = 0.01：w₁ = {w1b:.2f}，融合 {xb:.3f} m（被拉向雷达）")

    print("\n第二步 早融合 vs 晚融合（标签 y = a·b）")
    for name, r2 in fusion_fit().items():
        print(f"  {name}：R² = {r2:.4f}")

    print("\n第三步 交叉注意力与 Perceiver")
    for name, c in attn_cost().items():
        print(f"  {name} = {c:,}")
    print(f"  2 个潜变量对 4 个输入的注意力权重：\n{cross_attn().round(3)}")

    print("\n第四步 模态缺失与模态 dropout")
    for name, mse in modality_dropout().items():
        print(f"  {name}：MSE = {mse:.3f}")


if __name__ == "__main__":
    main()
