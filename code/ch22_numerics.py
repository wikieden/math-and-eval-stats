"""讲义：第 22 章 数值计算（notes/lessons/22-numerics.md）
运行：uv run python ch22_numerics.py

例子：ODE x' = cos(t)·x 从 t = 0 积到 1（精确解 e^{sin 1}）；softmax 的上溢；float16 / bfloat16 / float32 的累加；1 − cos x 的相消；Hilbert 矩阵的病态。
里程碑 ME-M18 的"同一个 ODE 画步数 → 误差图"留给自己做。
"""

import math

import numpy as np

EXACT = math.exp(math.sin(1.0))  # x' = cos(t)·x、x(0) = 1 在 t = 1 的精确值


def f(t: float, x: float) -> float:
    return math.cos(t) * x


def integrate(method: str, n: int) -> float:
    """从 t = 0 到 1 走 n 步：Euler（1 阶）、Heun（2 阶）、RK4（4 阶）。"""
    h, t, x = 1.0 / n, 0.0, 1.0
    for _ in range(n):
        if method == "euler":
            x = x + h * f(t, x)
        elif method == "heun":
            k1 = f(t, x)
            k2 = f(t + h, x + h * k1)
            x = x + h / 2 * (k1 + k2)
        else:
            k1 = f(t, x)
            k2 = f(t + h / 2, x + h / 2 * k1)
            k3 = f(t + h / 2, x + h / 2 * k2)
            k4 = f(t + h, x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return x


def softmax_naive(z: np.ndarray) -> np.ndarray:
    e = np.exp(z)
    return e / e.sum()


def softmax_stable(z: np.ndarray) -> np.ndarray:
    """先减最大值：e^{z − max} 不会上溢，结果不变。"""
    e = np.exp(z - z.max())
    return e / e.sum()


def logsumexp(z: np.ndarray) -> float:
    """log Σ e^z = max + log Σ e^{z − max}。"""
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


def accumulate(dtype, n: int = 10000, step: float = 1e-4) -> float:
    """在给定精度里把 step 加 n 次（真值 1.0）。"""
    s = np.array(0.0, dtype=dtype)
    d = np.array(step, dtype=dtype)
    for _ in range(n):
        s = (s + d).astype(dtype)
    return float(s)


def bf16(x: np.ndarray) -> np.ndarray:
    """用 float32 截掉低 16 位模拟 bfloat16（8 位指数、7 位尾数，就近舍入）。"""
    b = np.asarray(x, np.float32).view(np.uint32)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return b.view(np.float32)


def accumulate_bf16(n: int = 10000, step: float = 1e-4) -> float:
    s = np.float32(0.0)
    d = bf16(np.float32(step))
    for _ in range(n):
        s = bf16(np.float32(s + d))
    return float(s)


def cancel(x: float) -> tuple[float, float]:
    """1 − cos x 直接算 vs 换成 2 sin²(x/2)。"""
    return 1 - math.cos(x), 2 * math.sin(x / 2) ** 2


def hilbert_cond(n: int) -> float:
    h = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
    return float(np.linalg.cond(h))


def main() -> None:
    print("\n第一步 ODE 积分器：阶数决定步数翻倍时误差降多少")
    for n in (5, 10, 20):
        errs = [abs(integrate(m, n) - EXACT) for m in ("euler", "heun", "rk4")]
        print(f"  {n:2d} 步：Euler {errs[0]:.2e}，Heun {errs[1]:.2e}，RK4 {errs[2]:.2e}")
    print("  步数翻倍：Euler 误差约 ÷2，Heun 约 ÷4，RK4 约 ÷16")

    print("\n第二步 softmax 和 logsumexp：先减最大值")
    z = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        print(f"  直接算 softmax([1000, 1001, 1002]) = {softmax_naive(z).tolist()}")
    print(f"  先减最大值 = {np.round(softmax_stable(z), 4).tolist()}；logsumexp = {logsumexp(z):.4f}")

    print("\n第三步 浮点精度：把 0.0001 加一万次")
    print(f"  float32：{accumulate(np.float32):.6f}；float16：{accumulate(np.float16):.6f}；bfloat16：{accumulate_bf16():.6f}")
    print(f"  float16 最大 {np.finfo(np.float16).max:.0f}、最小正规数 {np.finfo(np.float16).tiny:.1e}；bfloat16 的范围和 float32 一样（约 3.4e38），但只有 3 位有效数字")

    print("\n第四步 相减相消：两个差不多的数相减，有效数字丢光")
    for x in (1e-3, 1e-8):
        a, b = cancel(x)
        print(f"  x = {x}：1 − cos x = {a:.6e}，2 sin²(x/2) = {b:.6e}")

    print("\n第五步 病态问题：Hilbert 矩阵")
    for n in (4, 8, 12):
        print(f"  {n}×{n}：条件数 {hilbert_cond(n):.1e}")


if __name__ == "__main__":
    main()
