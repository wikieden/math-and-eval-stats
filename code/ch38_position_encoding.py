"""讲义：第 38 章 位置编码的数学（notes/lessons/38-position-encoding.md）
运行：uv run python ch38_position_encoding.py

例子：d = 128、底数 10000 的 RoPE；把上下文从 4096 扩到 16384（s = 4）时 PI、NTK-aware、YaRN 各怎么改频率；
8 个头的 ALiBi 斜率。
里程碑 ME-M24 的“数值验证 RoPE 只依赖相对位置”留给自己做。
"""

import numpy as np

D, BASE, L_TRAIN = 128, 10000.0, 4096


def freqs(d: int = D, base: float = BASE) -> np.ndarray:
    """第 i 对维度的角频率 θ_i = base^{−2i/d}，i = 0 … d/2 − 1。"""
    return base ** (-np.arange(0, d, 2) / d)


def sinusoid_shift(p: int, k: int, i: int = 3, d: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """正弦编码第 i 对 (sin pω, cos pω)：PE(p + k) = R(kω)·PE(p)，R 是只和 k 有关的 2×2 矩阵。返回 (直接算, 用矩阵算)。"""
    w = 10000 ** (-2 * i / d)
    pe = lambda q: np.array([np.sin(q * w), np.cos(q * w)])  # noqa: E731
    r = np.array([[np.cos(k * w), np.sin(k * w)], [-np.sin(k * w), np.cos(k * w)]])
    return pe(p + k), r @ pe(p)


def wavelengths(th: np.ndarray) -> np.ndarray:
    return 2 * np.pi / th


def ntk_base(s: float, d: int = D, base: float = BASE) -> float:
    """NTK-aware：把底数换成 base·s^{d/(d−2)}，最低频正好被压缩 s 倍，最高频几乎不动（YaRN 附录 A.2）。"""
    return base * s ** (d / (d - 2))


def yarn_regimes(s: float = 4.0, alpha: float = 1.0, beta: float = 32.0) -> tuple[int, int, int, float]:
    """YaRN 的 NTK-by-parts：r = L/λ 是训练长度里转的圈数。r > β 不插值，r < α 完全按 PI 插值，中间线性过渡。
    返回 (不插值的维度对数, 过渡的, 完全插值的, 注意力温度 √(1/t) = 0.1 ln s + 1)。"""
    r = L_TRAIN / wavelengths(freqs())
    return int((r > beta).sum()), int(((r >= alpha) & (r <= beta)).sum()), int((r < alpha).sum()), 0.1 * np.log(s) + 1


def alibi_slopes(h: int = 8) -> list[float]:
    """ALiBi 第 k 个头的斜率 2^{−8k/h}，k = 1 … h；注意力分数加 −m·(i − j)。"""
    return [2 ** (-8 * k / h) for k in range(1, h + 1)]


def main() -> None:
    print("\n第一步 正弦编码：平移 k 是一个只依赖 k 的旋转")
    a, b = sinusoid_shift(10, 7)
    print(f"  第 3 对维度：PE(17) = {np.round(a, 4).tolist()}，R(7ω)·PE(10) = {np.round(b, 4).tolist()}")

    print("\n第二步 RoPE：每对维度按位置转一个角度")
    th = freqs()
    lam = wavelengths(th)
    print(f"  d = 128、底数 1e4：最快一对每个 token 转 {th[0]:.0f} 弧度（波长 {lam[0]:.2f}），最慢一对波长 {lam[-1]:,.0f}")
    print(f"  波长超过训练长度 4096 的维度对：{int((lam > L_TRAIN).sum())} / 64")

    print("\n第三步 扩长度（4096 → 16384，s = 4）")
    nb = ntk_base(4.0)
    th2 = freqs(base=nb)
    print(f"  位置插值 PI：所有频率除以 4（位置 m → m/4）")
    print(f"  NTK-aware：底数 1e4 → {nb:,.0f}；最高频 {th[0]:.4f} → {th2[0]:.4f}（几乎不变），最低频缩小 {th[-1] / th2[-1]:.4f} 倍")
    keep, ramp, interp, tmp = yarn_regimes()
    print(f"  YaRN（α = 1、β = 32）：不插值 {keep} 对、过渡 {ramp} 对、完全插值 {interp} 对；注意力温度 √(1/t) = {tmp:.4f}")

    print("\n第四步 ALiBi：不加位置向量，直接在分数上减距离")
    sl = alibi_slopes()
    print(f"  8 个头的斜率：{[f'1/{round(1 / m)}' for m in sl]}")
    print(f"  距离 100 时各头的偏置：{[round(-100 * m, 2) for m in sl]}")


if __name__ == "__main__":
    main()
