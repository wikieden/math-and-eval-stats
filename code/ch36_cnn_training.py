"""讲义：第 36 章 CNN 训练的数学（notes/lessons/36-cnn-training.md）
运行：uv run python ch36_cnn_training.py

例子：第 35 章的一维信号和核（反向传播 = 转置）；转置卷积的尺寸；单层的方差增益（Xavier / He）；
BatchNorm 的梯度性质；残差连接的梯度连乘。
里程碑 ME-M23 的“深层网络在 Xavier / He / 随机初始化下的激活方差图”留给自己做；这里只算一层。
"""

import numpy as np

from ch35_convolution import K, X, toeplitz_matrix


def conv_backward(g: np.ndarray, k: np.ndarray = K) -> np.ndarray:
    """前向 y = T x（valid 互相关），反向 ∂L/∂x = Tᵀ g = g 和核的 full 卷积。"""
    return np.convolve(g, k, "full")


def tconv_size(n: int, k: int, s: int = 1, p: int = 0, d: int = 1, op: int = 0) -> int:
    """转置卷积输出尺寸 (n − 1)s − 2p + d(k − 1) + output_padding + 1。"""
    return (n - 1) * s - 2 * p + d * (k - 1) + op + 1


def layer_gain(var_w: float, n_in: int, relu: bool = True) -> float:
    """一层 y = Wx：E[y²] = n_in·Var(w)·E[x²]；过 ReLU 后二阶矩再减半。返回 E[a_out²]/E[a_in²]。"""
    return n_in * var_w * (0.5 if relu else 1.0)


def one_layer_check(var_w: float, n_in: int = 512, n: int = 4000, seed: int = 0) -> float:
    """模拟一层：输入二阶矩 1，看输出 ReLU 后的二阶矩。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_in))
    w = rng.normal(0, np.sqrt(var_w), (n_in, n_in))
    a = np.maximum(x @ w, 0)
    return float((a ** 2).mean())


def batchnorm_grad(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """BatchNorm（γ = 1、β = 0）对输入的梯度：dx = (g − mean(g) − x̂·mean(g·x̂)) / σ。"""
    mu, var = x.mean(0), x.var(0)
    xh = (x - mu) / np.sqrt(var + eps)
    return (g - g.mean(0) - xh * (g * xh).mean(0)) / np.sqrt(var + eps)


def grad_products(depth: int = 50, seed: int = 0) -> tuple[float, float]:
    """普通堆叠：每层导数 w ~ U(0.5, 1.1)，连乘；残差：每层 1 + 0.1·u，u ~ U(−1, 1)。返回两个连乘。"""
    rng = np.random.default_rng(seed)
    plain = float(np.prod(rng.uniform(0.5, 1.1, depth)))
    res = float(np.prod(1 + 0.1 * rng.uniform(-1, 1, depth)))
    return plain, res


def main() -> None:
    print("\n第一步 卷积的反向传播 = 乘转置矩阵 = 和核做 full 卷积")
    t = toeplitz_matrix(len(X), K)
    g = np.array([1.0, 2.0, 0.0, -1.0])
    print(f"  上游梯度 g = {g}：Tᵀg = {t.T @ g}；np.convolve(g, k, 'full') = {conv_backward(g)}")

    print("\n第二步 转置卷积：把小图放大")
    for args in ((16, 4, 2, 1), (7, 3, 2, 1), (7, 3, 2, 1, 1, 1)):
        print(f"  输入 {args[0]}、核 {args[1]}、步长 {args[2]}、填充 {args[3]}{'、output_padding 1' if len(args) > 4 else ''}：输出 {tconv_size(*args)}")

    print("\n第三步 初始化：一层的方差增益")
    n_in = 512
    for name, vw in (("N(0, 0.01²)", 1e-4), ("Xavier 2/(n_in + n_out)", 2 / (2 * n_in)), ("He 2/n_in", 2 / n_in)):
        print(f"  {name}：理论增益 {layer_gain(vw, n_in):.4f}，模拟一层 ReLU 后二阶矩 {one_layer_check(vw):.4f}")

    print("\n第四步 BatchNorm 的梯度：减掉均值方向和 x̂ 方向")
    rng = np.random.default_rng(1)
    x = rng.normal(3, 2, (32, 4))
    gg = rng.standard_normal((32, 4))
    dx = batchnorm_grad(x, gg, eps=0.0)
    xh = (x - x.mean(0)) / x.std(0)
    print(f"  每个通道 Σ dx = {(np.round(dx.sum(0), 10) + 0.0).tolist()}；Σ dx·x̂ = {(np.round((dx * xh).sum(0), 10) + 0.0).tolist()}")

    print("\n第五步 残差：梯度连乘里多了一个 1")
    p, r = grad_products()
    print(f"  50 层，普通堆叠每层导数 U(0.5, 1.1)：连乘 {p:.2e}；残差每层 1 + 0.1·U(−1, 1)：连乘 {r:.4f}")


if __name__ == "__main__":
    main()
