"""讲义：第 39 章 归一化与信号传播（notes/lessons/39-normalization.md）
运行：uv run python ch39_normalization.py

例子：向量 x = [2, 4, 6, 8]；LayerNorm / RMSNorm 的几何和不变性；LayerNorm 的梯度方向；
Pre-LN 残差流的方差随层数增长、Post-LN 每层被拉回 1。
里程碑 ME-M24 的“比较 Pre-LN / Post-LN 12 层前向的梯度范数”留给自己做。
"""

import numpy as np

X = np.array([2.0, 4.0, 6.0, 8.0])


def layernorm(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + eps)


def rmsnorm(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def ln_grad(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """y = LN(x)（无仿射）：∂L/∂x = (g − mean(g) − y·mean(g·y)) / σ —— 和第 36 章 BatchNorm 同一个式子，只是沿特征维。"""
    y = layernorm(x)
    sigma = np.sqrt(((x - x.mean()) ** 2).mean())
    return (g - g.mean() - y * (g * y).mean()) / sigma


def residual_variance(layers: int, mode: str, d: int = 512, n: int = 2000, seed: int = 0) -> list[float]:
    """残差流 x（初始每个分量方差 1），每层子网络 F 是随机线性层（输出方差 ≈ 输入方差）。
    Pre-LN：x ← x + F(LN(x))；Post-LN：x ← LN(x + F(x))。返回每层后残差流的方差。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    out = []
    for _ in range(layers):
        w = rng.normal(0, 1 / np.sqrt(d), (d, d))
        x = x + layernorm(x) @ w if mode == "pre" else layernorm(x + x @ w)
        out.append(float(x.var()))
    return out


def main() -> None:
    print("\n第一步 LayerNorm 的几何：减均值 = 投影，除标准差 = 拉到半径 √d 的球面")
    y = layernorm(X)
    print(f"  x = {X.tolist()} → LN(x) = {np.round(y, 4).tolist()}；分量和 {y.sum():.1f}，长度 {np.linalg.norm(y):.4f}（√4 = 2）")
    r = rmsnorm(X)
    print(f"  RMSNorm(x) = {np.round(r, 4).tolist()}；不减均值，长度也是 {np.linalg.norm(r):.4f}")

    print("\n第二步 不变性")
    print(f"  LN(3x + 5) = {np.round(layernorm(3 * X + 5), 4).tolist()}（和 LN(x) 一样：平移、缩放都不变）")
    print(f"  RMSNorm(3x) = {np.round(rmsnorm(3 * X), 4).tolist()}（缩放不变）；RMSNorm(x + 5) = {np.round(rmsnorm(X + 5), 4).tolist()}（平移会变）")

    print("\n第三步 LayerNorm 的梯度：去掉“全 1”方向和 y 方向")
    g = np.array([1.0, -2.0, 0.5, 3.0])
    dx = ln_grad(X, g)
    print(f"  上游梯度 {g.tolist()} → ∂L/∂x = {np.round(dx, 4).tolist()}；Σ = {dx.sum() + 0:.1e}，和 y 的内积 {dx @ layernorm(X) + 0:.1e}")

    print("\n第四步 残差流的方差：Pre-LN 一路累加，Post-LN 每层拉回 1")
    pre, post = residual_variance(12, "pre"), residual_variance(12, "post")
    print(f"  Pre-LN 第 1、4、8、12 层后：{[round(pre[i], 2) for i in (0, 3, 7, 11)]}（约 1 + L）")
    print(f"  Post-LN 第 1、4、8、12 层后：{[round(post[i], 2) for i in (0, 3, 7, 11)]}")


if __name__ == "__main__":
    main()
