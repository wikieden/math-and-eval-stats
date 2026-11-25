"""讲义：第 42 章 参数化与初始化：标准参数化 vs μP（notes/lessons/42-mup.md）
运行：uv run python ch42_mup.py

例子：一个隐藏层 h = W x（宽度 n 从 64 到 4096），输入每个分量 O(1)。看一步 Adam 式更新（每个权重变化 ±lr）后，
h 的每个分量变化多大：标准参数化（SP）随宽度线性涨，μP 把隐藏层学习率缩放成 1/n 后不随宽度变。
"""

import numpy as np

WIDTHS = (64, 256, 1024, 4096)


def init_std(n: int, layer: str, mode: str) -> float:
    """SP：所有层 1/√fan_in。μP（Tensor Programs V 表 3 的 Adam 版本之一）：输入层、隐藏层 1/√fan_in，输出层 1/fan_in。"""
    if mode == "mup" and layer == "out":
        return 1.0 / n
    return 1.0 / np.sqrt(n)


def hidden_update_size(n: int, lr: float = 1e-3, mode: str = "sp", seed: int = 0) -> float:
    """Adam 的一步更新近似为 ΔW = −lr·sign(G)；梯度 G = g xᵀ（秩 1），所以 ΔW 的每一行和 x 的符号对齐。
    Δh = ΔW x 的每个分量 ≈ lr·Σ|x_j| ∝ lr·n。μP 把隐藏层学习率设为 lr·(n0/n)（n0 = 64）。返回 Δh 分量的均方根。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    g = rng.standard_normal(n)
    eta = lr if mode == "sp" else lr * 64 / n
    dw = -eta * np.sign(np.outer(g, x))
    return float(np.sqrt(((dw @ x) ** 2).mean()))


def output_logit_scale(n: int, mode: str, seed: int = 0) -> float:
    """初始化时输出 logit = w_outᵀ h（h 每个分量 O(1)）的大小：SP 是 O(1)，μP 输出层 1/n 初始化后是 O(1/√n)，宽了以后从 0 附近起步。"""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((2000, n))
    w = rng.normal(0, init_std(n, "out", mode), n)
    return float((h @ w).std())


def main() -> None:
    print("\n第一步 初始化只管前向：两种参数化都让隐藏层激活 O(1)")
    for n in WIDTHS:
        print(f"  宽度 {n}：隐藏层初始化标准差 1/√n = {init_std(n, 'hidden', 'sp'):.4f}")

    print("\n第二步 学习率才是问题：同样的 lr，宽度越大一步改得越多")
    for n in WIDTHS:
        print(f"  宽度 {n}：SP 一步后 Δh 分量 {hidden_update_size(n):.4f}；μP（lr × 64/n）{hidden_update_size(n, mode='mup'):.4f}")

    print("\n第三步 输出层：μP 用 1/n 初始化")
    for n in WIDTHS:
        print(f"  宽度 {n}：初始 logit 标准差 SP {output_logit_scale(n, 'sp'):.3f}，μP {output_logit_scale(n, 'mup'):.4f}")

    print("\n第四步 超参迁移：在宽 256 上调好 lr，μP 下直接用到宽 4096")
    print("  SP：最优 lr 随宽度大约按 1/n 变小，换宽度要重调；μP：隐藏层 lr 已经按 1/n 缩放，名义 lr 不变")


if __name__ == "__main__":
    main()
