"""讲义：第 7 章 批判地读 loss 曲线和消融表（notes/lessons/07-read-curves-and-ablations.md）
运行：uv run python ch07_read_curves.py

例子全是编的：两个配置各 5 个种子的分数；一条带噪的 loss 曲线做 EMA 平滑；训练 / 验证 loss 的差距。
里程碑 ME-M6（审一张真表）留给自己做。
"""

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

SEEDS_A = np.array([71.2, 72.8, 70.1, 73.5, 71.9])  # 配置 A 五个种子的准确率（编的）
SEEDS_B = np.array([72.4, 71.0, 73.9, 72.2, 74.1])  # 配置 B（编的）


def seed_summary(x: np.ndarray) -> tuple[float, float]:
    """均值和种子间标准差（ddof = 1）。"""
    return float(x.mean()), float(x.std(ddof=1))


def welch(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t 检验（不假设方差相同）的双侧 p 值。"""
    return float(ttest_ind(b, a, equal_var=False).pvalue)


def prob_outperform(a: np.ndarray, b: np.ndarray) -> float:
    """Bouthillier 2021 式 (9)：P(B > A) = 所有 (b_i, a_j) 配对里 b 赢的比例（平局算一半）。"""
    wins = (b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum()
    return float(wins / (len(a) * len(b)))


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U 检验（只看排序）的双侧 p 值。"""
    return float(mannwhitneyu(b, a, alternative="two-sided").pvalue)


def ema(x: np.ndarray, beta: float) -> np.ndarray:
    """TensorBoard 式的指数滑动平均：s_t = β·s_{t−1} + (1 − β)·x_t，s_0 = x_0。"""
    s = np.empty_like(x)
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = beta * s[t - 1] + (1 - beta) * x[t]
    return s


def loss_curve(n: int = 600, drop_at: int = 200, seed: int = 0, noise: float = 0.15) -> np.ndarray:
    """编一条 loss 曲线：缓慢下降，在 drop_at 步突然降 0.5（比如学习率衰减），每步加噪声（noise = 0 就是干净曲线）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 2.0 * np.exp(-t / 300) + 0.5 - 0.5 * (t >= drop_at)
    return base + rng.normal(0, noise, n)


def lag_after_drop(beta: float, drop_at: int = 200, frac: float = 0.9) -> int:
    """在干净的阶跃上量：平滑后的曲线要过多少步，才走完这次 0.5 下降的 frac（只看阶跃，扣掉曲线本身的缓降）。"""
    x = loss_curve(noise=0.0, drop_at=drop_at)
    ref = loss_curve(noise=0.0, drop_at=10**9)[: len(x)]  # 同一条曲线但不突降
    gap = ema(ref, beta) - ema(x, beta)                     # 平滑后已经"感受到"的下降
    for k in range(drop_at, len(x)):
        if gap[k] >= frac * 0.5:
            return k - drop_at + 1
    return -1


def lag_formula(beta: float, frac: float = 0.9) -> int:
    """阶跃输入下 EMA 走完 frac 要的步数：1 − β^k ≥ frac → k = ⌈ln(1 − frac)/ln β⌉。"""
    return int(np.ceil(np.log(1 - frac) / np.log(beta))) if beta > 0 else 1


def seeds_needed(sd: float, delta: float, z: float = 2.8) -> int:
    """要分清相差 delta 的两个配置，每边要几个种子：n ≈ 2·z²·sd²/δ²（z = z_α/2 + z_β ≈ 1.96 + 0.84）。"""
    return int(np.ceil(2 * z**2 * sd**2 / delta**2))


def main() -> None:
    print("\n第一步 种子方差：同一个配置换种子，结果差多少")
    for name, x in (("A", SEEDS_A), ("B", SEEDS_B)):
        m, sd = seed_summary(x)
        print(f"  {name}：均值 {m:.2f}，种子间标准差 {sd:.2f}，最好的一个 {x.max():.1f}")
    print(f"  B − A = {SEEDS_B.mean() - SEEDS_A.mean():+.2f}；只挑最好的种子比：{SEEDS_B.max() - SEEDS_A.max():+.1f}")

    print("\n第二步 差异是不是噪声")
    print(f"  Welch t 检验 p = {welch(SEEDS_A, SEEDS_B):.3f}；Mann-Whitney p = {mann_whitney_p(SEEDS_A, SEEDS_B):.3f}")
    print(f"  P(B > A)（随便挑一个 B 种子赢随便一个 A 种子的概率）= {prob_outperform(SEEDS_A, SEEDS_B):.2f}")
    sd = np.sqrt((SEEDS_A.var(ddof=1) + SEEDS_B.var(ddof=1)) / 2)
    print(f"  合并标准差 {sd:.2f}：分清 1 个点的差要每边 {seeds_needed(sd, 1.0)} 个种子")

    print("\n第三步 平滑会拖后腿：loss 在第 200 步突降 0.5")
    noisy = loss_curve()
    print(f"  原始曲线每步噪声标准差 0.15；β = 0.9 平滑后残余抖动 {np.std(ema(noisy, 0.9)[300:] - loss_curve(noise=0)[300:]):.3f}")
    for beta in (0.0, 0.9, 0.99):
        print(f"  EMA β = {beta}：走完 90% 的下降要 {lag_after_drop(beta)} 步（公式 {lag_formula(beta)}）")

    print("\n第四步 过拟合：看训练和验证的差距，不看单条曲线")
    train = np.array([2.1, 1.4, 1.0, 0.7, 0.5, 0.35, 0.25])
    val = np.array([2.2, 1.6, 1.3, 1.2, 1.25, 1.35, 1.5])
    print(f"  验证 loss 最低在第 {int(val.argmin())} 个检查点；之后训练还在降，验证在升（差距 {val[-1] - train[-1]:.2f}）")


if __name__ == "__main__":
    main()
