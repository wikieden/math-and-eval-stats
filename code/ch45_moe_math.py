"""讲义：第 45 章 MoE 的数学（notes/lessons/45-moe-math.md）
运行：uv run python ch45_moe_math.py

例子：一个 token 对 4 个专家的门控 logits [2, 1, 0.5, −1]；Switch 负载均衡辅助损失在均匀 / 偏斜两种分布下的值；
容量因子与丢 token 的概率（二项分布）；DeepSeek-V3 无辅助损失的偏置调节一步；专家并行 all-to-all 的通信量。
里程碑 ME-M28 的“模拟 top-2 门控的负载分布、加辅助损失 / 偏置调节后对比”留给自己做。
"""

import numpy as np
from scipy.stats import binom


def topk_gate(logits: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """softmax 后取最大的 k 个，再在这 k 个里重新归一化（Mixtral 的做法）。返回 (专家编号, 权重)。"""
    p = np.exp(logits - logits.max())
    p /= p.sum()
    idx = np.argsort(-p)[:k]
    return idx, p[idx] / p[idx].sum()


def switch_aux(f: np.ndarray, p: np.ndarray) -> float:
    """Switch Transformer 的辅助损失 N·Σ f_i·P_i：f_i 是实际分到专家 i 的 token 比例，P_i 是路由概率的平均。均匀时 = 1。"""
    return float(len(f) * (f * p).sum())


def drop_rate(tokens: int, experts: int, cf: float, k: int = 1) -> float:
    """每个专家容量 C = cf·tokens·k/E。若路由完全随机均匀，一个专家收到的 token 数 ~ Binomial(tokens·k, 1/E)，
    超过 C 的部分被丢弃。返回期望丢弃比例。"""
    n, p = tokens * k, 1 / experts
    cap = int(cf * n / experts)
    x = np.arange(cap + 1, n + 1)
    return float(((x - cap) * binom.pmf(x, n, p)).sum() / (n * p))


def bias_update(load: np.ndarray, b: np.ndarray, gamma: float = 0.001) -> np.ndarray:
    """DeepSeek-V3：每个专家一个只用于选 top-k 的偏置 b_i；负载高于平均就减 γ，低于平均就加 γ（不进梯度、不改门控权重）。"""
    return b - gamma * np.sign(load - load.mean())


def a2a_bytes(tokens: int, k: int, hidden: int, bytes_per: int = 2) -> int:
    """专家并行一层的 all-to-all：派发（dispatch）和汇总（combine）各发一次，每个 token 发给 k 个专家。"""
    return 2 * tokens * k * hidden * bytes_per


def main() -> None:
    print("\n第一步 门控：softmax → top-k → 重新归一化")
    idx, w = topk_gate(np.array([2.0, 1.0, 0.5, -1.0]))
    print(f"  logits [2, 1, 0.5, −1]：选专家 {idx.tolist()}，权重 {np.round(w, 4).tolist()}；输出 = Σ 权重 × 专家输出")

    print("\n第二步 负载均衡辅助损失 N·Σ f_i P_i")
    for name, f, p in (("均匀", [0.25] * 4, [0.25] * 4), ("偏斜", [0.7, 0.1, 0.1, 0.1], [0.6, 0.15, 0.15, 0.1]),
                       ("全挤到一个", [1, 0, 0, 0], [0.9, 0.05, 0.03, 0.02])):
        print(f"  {name}：f = {f}、P = {p} → {switch_aux(np.array(f), np.array(p)):.4f}")

    print("\n第三步 容量因子：随机路由也会溢出")
    for cf in (1.0, 1.25, 2.0):
        print(f"  4096 个 token、64 个专家、top-1、容量因子 {cf}：期望丢弃 {drop_rate(4096, 64, cf):.2%}")

    print("\n第四步 无辅助损失的偏置调节（一步）")
    load = np.array([400.0, 100.0, 250.0, 250.0])
    print(f"  负载 {load.tolist()}（平均 250）：偏置 0 → {bias_update(load, np.zeros(4)).tolist()}")

    print("\n第五步 专家并行的通信量")
    print(f"  每卡 8192 个 token、top-8、hidden 7168、bf16：一层 all-to-all {a2a_bytes(8192, 8, 7168) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
