"""讲义：第 44 章 训练稳定性（notes/lessons/44-training-stability.md）
运行：uv run python ch44_training_stability.py

例子：一组 logits 整体漂高时 softmax 不变、z-loss 会把它拉回来；QK-norm 给注意力 logit 一个上界；按全局范数裁剪梯度；
bf16 / fp16 / fp8 的数值范围；fp8 训练的 per-tensor 缩放因子。
里程碑 ME-M27 的“画 bf16 与 fp8 的可表示范围”留给自己做；这里只列数。
"""

import numpy as np


def logsumexp(z):
    m = np.max(z)
    return float(m + np.log(np.exp(z - m).sum()))


def ce_and_zloss(z: np.ndarray, target: int, coef: float = 1e-4) -> tuple[float, float]:
    """交叉熵 = logsumexp(z) − z_target；z-loss = coef·(log Z)²（ST-MoE 式 (5)），把 log Z 拉向 0。"""
    lz = logsumexp(z)
    return lz - float(z[target]), coef * lz * lz


def qk_logit_bound(d: int, g: float) -> float:
    """QK-norm：q、k 先各自 RMSNorm（长度 √d）再乘可学的 g，点积 / √d 的绝对值 ≤ g²·d·… 这里按长度 √d 的两个向量：
    |q·k|/√d ≤ √d·√d/√d·g² = g²√d。"""
    return g * g * np.sqrt(d)


def clip_global(grads: list[np.ndarray], max_norm: float) -> tuple[float, list[np.ndarray]]:
    """全局范数裁剪：所有参数的梯度拼成一个向量求范数，超过 max_norm 就整体乘 max_norm/范数（方向不变）。"""
    total = float(np.sqrt(sum((g ** 2).sum() for g in grads)))
    s = min(1.0, max_norm / (total + 1e-12))
    return total, [g * s for g in grads]


FORMATS = {   # 名字：(指数位, 尾数位, 最大值, 最小正规数)
    "fp32": (8, 23, 3.4028e38, 1.1755e-38),
    "bf16": (8, 7, 3.3895e38, 1.1755e-38),
    "fp16": (5, 10, 65504.0, 6.1035e-05),
    "fp8 E4M3": (4, 3, 448.0, 2 ** -6),
    "fp8 E5M2": (5, 2, 57344.0, 2 ** -14),
}


def fp8_scale(x: np.ndarray, fmt_max: float = 448.0) -> float:
    """per-tensor 缩放：scale = amax / fmt_max，存 x/scale（落进 E4M3 的范围），用时再乘回去。"""
    return float(np.abs(x).max() / fmt_max)


def round_e4m3(x: np.ndarray) -> np.ndarray:
    """模拟 E4M3：正规数保留 3 位尾数（就近），截到 ±448；|x| < 2^−6 时进入次正规数，间距固定为 2^−9（更小的直接舍成 0）。"""
    x = np.clip(x, -448, 448)
    e = np.floor(np.log2(np.maximum(np.abs(x), 1e-30)))
    q = 2.0 ** (np.maximum(e, -6) - 3)
    return np.round(x / q) * q


def main() -> None:
    print("\n第一步 logit 漂移：softmax 看不见，z-loss 看得见")
    z = np.array([2.0, 1.0, 0.0, -1.0])
    for shift in (0, 20, 50):
        ce, zl = ce_and_zloss(z + shift, 0)
        print(f"  logits + {shift}：交叉熵 {ce:.4f}（不变），log Z = {logsumexp(z + shift):.2f}，z-loss = {zl:.4f}")

    print("\n第二步 QK-norm：注意力 logit 有了上界")
    for g in (1.0, 2.0):
        print(f"  d = 128、增益 g = {g}：|q·k|/√d ≤ {qk_logit_bound(128, g):.2f}")

    print("\n第三步 梯度裁剪：按全局范数整体缩放")
    grads = [np.array([3.0, 4.0]), np.array([12.0])]
    tot, cl = clip_global(grads, 1.0)
    print(f"  全局范数 {tot:.1f} → 裁到 1：{[np.round(c, 4).tolist() for c in cl]}（方向不变）")

    print("\n第四步 低精度格式的范围")
    for k, (e, m, mx, mn) in FORMATS.items():
        print(f"  {k}：指数 {e} 位、尾数 {m} 位，最大 {mx:.4g}，最小正规数 {mn:.3g}，相对精度 2^-{m} = {2 ** -m:.3g}")

    print("\n第五步 fp8 训练：先按 amax 缩放再量化")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.001, 4096)
    x[0] = 0.05
    s = fp8_scale(x)
    direct = round_e4m3(x)
    scaled = round_e4m3(x / s) * s
    print(f"  梯度标准差 0.001、最大 0.05：scale = {s:.3e}")
    print(f"  不缩放直接转 E4M3：{(direct == 0).mean():.0%} 的值变成 0，相对误差 {np.abs(direct - x).mean() / np.abs(x).mean():.4f}，缩放后 {np.abs(scaled - x).mean() / np.abs(x).mean():.4f}")


if __name__ == "__main__":
    main()
