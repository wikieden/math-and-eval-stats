"""讲义：第 37 章 注意力的数学（notes/lessons/37-attention-math.md）
运行：uv run python ch37_attention_math.py

例子：softmax 在不同缩放下的饱和；一维“查表”：键是位置、值是 sin，看注意力 = 核平滑；多头的秩；因果 mask。
里程碑 ME-M24 的“模拟点积方差随 d 线性增长”留给自己做；这里只用公式。
"""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def dot_std(d: int, scaled: bool) -> float:
    """q、k 各分量独立、均值 0、方差 1：Var(q·k) = d；除以 √d 后方差 1。返回标准差。"""
    return 1.0 if scaled else float(np.sqrt(d))


def saturation(scale: float, z=(2.0, 1.0, 0.0, -1.0)) -> tuple[float, float, float]:
    """logits = scale·z：返回 (最大概率, 熵（nat）, softmax 雅可比的最大元素 p_i(1 − p_i))。"""
    p = softmax(scale * np.array(z))
    return float(p.max()), float(-(p * np.log(p + 1e-300)).sum()), float((p * (1 - p)).max())


def kernel_smoother(q: float, tau: float, n: int = 21) -> tuple[float, float]:
    """键 k_j = 位置 x_j（[0, 2π] 上 21 个点），值 v_j = sin x_j。注意力 logits = −(q − k)²/(2τ²)，
    即 softmax(q·k/τ² − k²/(2τ²) + 常数)：就是高斯核的 Nadaraya–Watson 平滑。返回 (输出, sin q)。"""
    x = np.linspace(0, 2 * np.pi, n)
    w = softmax(-((q - x) ** 2) / (2 * tau ** 2))
    return float(w @ np.sin(x)), float(np.sin(q))


def head_rank(d_model: int = 512, d_head: int = 64, seed: int = 0) -> int:
    """一个头的注意力打分矩阵 W_Q W_Kᵀ（d_model × d_model）的秩 ≤ d_head。"""
    rng = np.random.default_rng(seed)
    wq, wk = rng.standard_normal((d_model, d_head)), rng.standard_normal((d_model, d_head))
    return int(np.linalg.matrix_rank(wq @ wk.T))


def causal_weights(n: int = 4, seed: int = 0) -> np.ndarray:
    """上三角（未来）位置填 −∞ 再 softmax：每行只在 j ≤ i 上有权重。"""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, n))
    s[np.triu_indices(n, 1)] = -np.inf
    return softmax(s)


def main() -> None:
    print("\n第一步 为什么除以 √d：点积的方差是 d")
    for d in (16, 64, 256):
        print(f"  d = {d}：q·k 的标准差 {dot_std(d, False):.0f}，除以 √d 后 {dot_std(d, True):.0f}")

    print("\n第二步 softmax 的温度与饱和")
    for s in (0.25, 1, 4, 16):
        m, h, g = saturation(s)
        print(f"  logits = {s}×[2, 1, 0, −1]：最大概率 {m:.4f}，熵 {h:.4f} nat，最大局部梯度 p(1 − p) = {g:.2e}")

    print("\n第三步 注意力 = 核平滑（软查表）")
    for tau in (0.1, 0.5, 2.0):
        o, t = kernel_smoother(1.0, tau)
        print(f"  查询 q = 1、核宽 τ = {tau}：输出 {o:.4f}（真值 sin 1 = {t:.4f}）")

    print("\n第四步 多头 = 若干个低秩打分矩阵")
    print(f"  d_model = 512、d_head = 64：W_Q W_Kᵀ 是 512×512，但秩只有 {head_rank()}")
    print(f"  8 个头的 Q、K、V 参数 3 × 512 × 64 × 8 = {3 * 512 * 64 * 8:,}，和一个 512 维的头一样多")

    print("\n第五步 因果 mask：未来位置填 −∞")
    w = causal_weights()
    print(f"  4×4 权重（每行和 = {w.sum(1).round(6).tolist()}）：\n{np.round(w, 3)}")
    print(f"  n = 4096 时可见的 (i, j) 对：{4096 * 4097 // 2:,}，约占 n² 的一半")


if __name__ == "__main__":
    main()
