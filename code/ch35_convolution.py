"""讲义：第 35 章 卷积的数学（notes/lessons/35-convolution.md）
运行：uv run python ch35_convolution.py

例子：一维信号 [3, 1, 4, 1, 5, 9] 和差分核 [1, 0, −1]（看相邻变化）；尺寸公式、感受野、im2col 的内存账。
里程碑 ME-M23 的“二维 conv2d = im2col × 矩阵乘法并和 PyTorch 对数”“用 FFT 验证卷积定理”留给自己做。
"""

import numpy as np

X = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
K = np.array([1.0, 0.0, -1.0])


def toeplitz_matrix(n: int, k: np.ndarray) -> np.ndarray:
    """valid 互相关写成矩阵：第 i 行在第 i..i+len(k)−1 列放核，其余是 0。"""
    m = n - len(k) + 1
    t = np.zeros((m, n))
    for i in range(m):
        t[i, i:i + len(k)] = k
    return t


def out_size(n: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:
    """输出尺寸 ⌊(n + 2p − d(k − 1) − 1)/s⌋ + 1（PyTorch Conv2d 文档的公式）。"""
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def receptive_field(layers: list[tuple[int, int, int]]) -> int:
    """layers 是 (k, s, d) 的列表。r ← r + (k − 1)·d·(前面所有层步长的乘积)。"""
    r, jump = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * jump
        jump *= s
    return r


def im2col_cost(c: int, h: int, w: int, k: int, s: int = 1, p: int = 1) -> tuple[tuple[int, int], int, int]:
    """im2col 矩阵的形状 (C·k·k, H_out·W_out)、它的元素数、原输入的元素数。"""
    ho, wo = out_size(h, k, s, p), out_size(w, k, s, p)
    shape = (c * k * k, ho * wo)
    return shape, shape[0] * shape[1], c * h * w


def main() -> None:
    print("\n第一步 卷积是一个稀疏、共享权重的线性映射")
    t = toeplitz_matrix(len(X), K)
    print(f"  Toeplitz 矩阵（4×6）：\n{t.astype(int)}")
    print(f"  矩阵乘 T·x = {t @ X}；np.correlate(x, k, 'valid') = {np.correlate(X, K, 'valid')}")
    print(f"  参数：全连接 4×6 = 24 个，卷积 3 个（共享）；非零元 {int((t != 0).sum())} / 24")

    print("\n第二步 深度学习里的“卷积”其实是互相关")
    print(f"  互相关 {np.correlate(X, K, 'valid')}；数学卷积（核翻转）{np.convolve(X, K, 'valid')}：差一个负号")

    print("\n第三步 尺寸公式")
    for args in ((32, 3, 1, 1, 1), (32, 3, 2, 1, 1), (32, 3, 1, 0, 2), (224, 7, 2, 3, 1)):
        print(f"  n = {args[0]}、k = {args[1]}、步长 {args[2]}、填充 {args[3]}、空洞 {args[4]}：输出 {out_size(*args)}")

    print("\n第四步 感受野")
    for name, layers in (("三层 3×3", [(3, 1, 1)] * 3), ("三层 3×3、前两层步长 2", [(3, 2, 1), (3, 2, 1), (3, 1, 1)]),
                         ("空洞 1、2、4", [(3, 1, 1), (3, 1, 2), (3, 1, 4)])):
        print(f"  {name}：感受野 {receptive_field(layers)}")

    print("\n第五步 卷积定理与 im2col 的代价")
    for n, k in ((1024, 3), (1024, 101), (65536, 1025)):
        print(f"  n = {n}、核长 {k}：直接算 {n * k:,} 次乘法；FFT 约 3·n·log₂n = {int(3 * n * np.log2(n)):,}")
    shape, cols, inp = im2col_cost(64, 56, 56, 3)
    print(f"  im2col：C = 64、56×56、3×3 核 → 矩阵 {shape}，{cols:,} 个元素，是输入（{inp:,}）的 {cols / inp:.0f} 倍")


if __name__ == "__main__":
    main()
