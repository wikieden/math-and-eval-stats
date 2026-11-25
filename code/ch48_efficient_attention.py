"""讲义：第 48 章 高效注意力与序列模型（notes/lessons/48-efficient-attention.md）
运行：uv run python ch48_efficient_attention.py

例子：online softmax 的 (m, l) 递推；FlashAttention 少读写的 HBM 字节；MLA 的“吸收”（矩阵结合律）；
线性注意力的递推和并行形式结果相同；一阶线性递推 h_t = a_t·h_{t−1} + b_t 用结合律做并行扫描。
里程碑 ME-M29 的“numpy 写分块注意力（online softmax 带 V）并和直接 softmax 对数”留给自己做。
"""

import numpy as np

Z = np.array([1.0, 3.0, 2.0, 5.0, 0.0, 4.0])


def online_softmax_stats(z: np.ndarray, block: int = 2) -> list[tuple[float, float]]:
    """分块扫一遍，维护当前最大值 m 和 l = Σ e^{z − m}；遇到更大的 m 就把旧的 l 乘 e^{m_old − m_new}。返回每块之后的 (m, l)。"""
    m, l, out = -np.inf, 0.0, []
    for i in range(0, len(z), block):
        b = z[i:i + block]
        mn = max(m, b.max())
        l = l * np.exp(m - mn) + np.exp(b - mn).sum()
        m = mn
        out.append((float(m), float(l)))
    return out


def hbm_bytes(n: int, d: int, sram: int = 100 * 1024, bytes_per: int = 2) -> tuple[float, float]:
    """标准实现：读写 N×N 的分数矩阵和概率矩阵（约 4N² 个元素）外加 Q、K、V、O；
    FlashAttention：Θ(N²d²/M) 次 HBM 访问（Dao 2022 定理 2），M 是 SRAM 能放的元素数。返回 (标准, Flash) 的字节数。"""
    m = sram / bytes_per
    std = (4 * n * n + 4 * n * d) * bytes_per
    flash = (n * n * d * d / m + 4 * n * d) * bytes_per
    return std, flash


def mla_absorb(d: int = 64, dc: int = 16, seed: int = 0) -> float:
    """MLA 只缓存低秩潜向量 c（dc 维），k = W_UK·c。打分 qᵀ(W_UK c) = (W_UKᵀ q)ᵀ c：把 W_UK 吸收进 q 的投影，
    推理时不必把 k 还原出来。返回两种算法的差。"""
    rng = np.random.default_rng(seed)
    q, c, w = rng.standard_normal(d), rng.standard_normal(dc), rng.standard_normal((d, dc))
    return float(abs(q @ (w @ c) - (w.T @ q) @ c))


def linear_attention(n: int = 6, d: int = 4, seed: int = 0) -> float:
    """因果线性注意力（特征映射 φ = elu + 1）：并行形式 Σ_{j≤i} (φ(q_i)·φ(k_j)) v_j / Σ φ(q_i)·φ(k_j)，
    递推形式 S_i = S_{i−1} + φ(k_i)v_iᵀ、z_i = z_{i−1} + φ(k_i)，o_i = S_iᵀφ(q_i)/(z_i·φ(q_i))。返回两者最大差。"""
    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((n, d)) for _ in range(3))
    phi = lambda x: np.where(x > 0, x + 1, np.exp(x))  # noqa: E731
    fq, fk = phi(q), phi(k)
    a = np.tril(fq @ fk.T)
    par = (a @ v) / a.sum(1, keepdims=True)
    s, z, rec = np.zeros((d, d)), np.zeros(d), []
    for i in range(n):
        s += np.outer(fk[i], v[i])
        z += fk[i]
        rec.append(s.T @ fq[i] / (z @ fq[i]))
    return float(np.abs(par - np.array(rec)).max())


def scan_combine(x, y):
    """线性递推 h = a·h + b 的“算子”(a, b) 的组合：先 x 后 y 等于 (a_y·a_x, a_y·b_x + b_y)，满足结合律。"""
    return (y[0] * x[0], y[0] * x[1] + y[1])


def parallel_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hillis–Steele 并行前缀扫描：log₂n 轮，每轮所有位置同时和左边距离 2^r 的位置组合。"""
    ops = list(zip(a, b))
    n, step = len(ops), 1
    while step < n:
        ops = [ops[i] if i < step else scan_combine(ops[i - step], ops[i]) for i in range(n)]
        step *= 2
    return np.array([o[1] for o in ops])


def main() -> None:
    print("\n第一步 online softmax：分块扫一遍就能得到最大值和分母")
    for i, (m, l) in enumerate(online_softmax_stats(Z), 1):
        print(f"  第 {i} 块后：m = {m}，l = {l:.4f}")
    print(f"  直接算：max = {Z.max()}，Σ e^(z − max) = {np.exp(Z - Z.max()).sum():.4f}")

    print("\n第二步 FlashAttention：不把 N×N 矩阵写回显存")
    for n in (2048, 8192):
        s, f = hbm_bytes(n, 128)
        print(f"  N = {n}、d = 128、SRAM 100 KB：标准 {s / 2**20:.0f} MiB，Flash 约 {f / 2**20:.0f} MiB（{s / f:.1f} 倍）")

    print("\n第三步 MQA / GQA / MLA：少存 KV")
    print("  32 个查询头：MHA 存 32 份 KV，GQA-8 存 8 份（1/4），MQA 存 1 份（1/32）")
    print(f"  MLA 只存潜向量 c；打分时把 W_UK 吸收进 q：两种算法相差 {mla_absorb():.1e}")

    print("\n第四步 线性注意力 = RNN")
    print(f"  并行形式和递推形式最大差 {linear_attention():.1e}；递推每步状态大小固定（d×d），和序列长度无关")

    print("\n第五步 SSM 的并行扫描")
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0.5, 1.0, 8), rng.standard_normal(8)
    h, seq = 0.0, []
    for t in range(8):
        h = a[t] * h + b[t]
        seq.append(h)
    print(f"  8 步线性递推：顺序算和并行扫描（3 轮）最大差 {np.abs(np.array(seq) - parallel_scan(a, b)).max():.1e}")


if __name__ == "__main__":
    main()
