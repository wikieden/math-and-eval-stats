"""讲义：第 30 章 泛函分析入门（notes/lessons/30-functional-analysis.md）
运行：uv run python ch30_functional_analysis.py

例子：函数 x^n 的两种范数、连续函数逼近阶跃、方波的 Fourier 展开、布朗运动协方差核的特征值、cos 的不动点、
高斯核岭回归（RKHS 的表示定理）。函数都在 [0, 1] 或 [−π, π] 上用细网格近似。
"""

import numpy as np


def xn_norms(n: int) -> tuple[float, float]:
    """f_n(x) = x^n 在 [0, 1] 上：sup 范数恒为 1，L2 范数 = 1/√(2n + 1) → 0。两种范数“看到”的收敛不一样。"""
    return 1.0, 1 / np.sqrt(2 * n + 1)


def step_gap(k: float, m: int = 200001) -> float:
    """连续函数 f_k(x) = 1/(1 + e^{−k(x − 0.5)}) 到阶跃函数的 L2 距离：k 越大越近，但极限不连续
    —— 连续函数空间在 L2 范数下不完备（不是 Banach 空间）。"""
    x = np.linspace(0, 1, m)
    f = 1 / (1 + np.exp(-k * (x - 0.5)))
    return float(np.sqrt(np.trapezoid((f - (x > 0.5)) ** 2, x)))


def square_wave_error(terms: int, m: int = 200001) -> tuple[float, float]:
    """方波 sign(x)（[−π, π]）的 Fourier 级数 (4/π) Σ sin((2k−1)x)/(2k−1)，取前 terms 项。
    返回 (L2 误差², 由 Parseval 算的剩余能量)。"""
    x = np.linspace(-np.pi, np.pi, m)
    s = sum(4 / np.pi * np.sin((2 * k - 1) * x) / (2 * k - 1) for k in range(1, terms + 1))
    err = float(np.trapezoid((np.sign(x) - s) ** 2, x))
    tail = 2 * np.pi - np.pi * sum((4 / np.pi / (2 * k - 1)) ** 2 for k in range(1, terms + 1))
    return err, float(tail)


def brownian_kernel_eigs(m: int = 1000, k: int = 3) -> tuple[list[float], list[float]]:
    """积分算子 (Kf)(x) = ∫_0^1 min(x, y) f(y) dy（布朗运动的协方差核）。离散成矩阵 min(x_i, x_j)/m 求特征值，
    和解析值 1/((j − 1/2)²π²) 比。"""
    x = (np.arange(m) + 0.5) / m
    ev = np.sort(np.linalg.eigvalsh(np.minimum.outer(x, x) / m))[::-1][:k]
    exact = [1 / ((j - 0.5) ** 2 * np.pi ** 2) for j in range(1, k + 1)]
    return ev.tolist(), exact


def cos_fixed_point(x0: float = 1.0, iters: int = 100) -> tuple[float, list[float]]:
    """x ← cos x：在 [cos 1, 1] 上 |cos′| = |sin x| ≤ sin 1 ≈ 0.84 < 1，是压缩映射，收敛到唯一不动点。"""
    x, gaps = x0, []
    for _ in range(iters):
        nx = np.cos(x)
        gaps.append(abs(nx - x))
        x = nx
    return float(x), gaps


def kernel_ridge(n: int = 20, lam: float = 1e-3, ell: float = 0.2, seed: int = 0) -> tuple[float, float]:
    """高斯核岭回归：min Σ(y_i − f(x_i))² + λ‖f‖²_H。表示定理：解是 f = Σ α_i k(x_i, ·)，α = (K + λI)⁻¹ y。
    数据 y = sin(2πx) + 0.1 噪声。返回 (训练点上的均方误差, 网格上和 sin(2πx) 的均方误差)。"""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(n)
    k = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * ell ** 2))  # noqa: E731
    alpha = np.linalg.solve(k(x, x) + lam * np.eye(n), y)
    g = np.linspace(0, 1, 501)
    fg = k(g, x) @ alpha
    return float(((k(x, x) @ alpha - y) ** 2).mean()), float(((fg - np.sin(2 * np.pi * g)) ** 2).mean())


def main() -> None:
    print("\n第一步 范数：同一个函数序列，用不同的尺子量，收敛与否不一样")
    for n in (1, 10, 100):
        s, l2 = xn_norms(n)
        print(f"  x^{n}：sup 范数 {s}，L2 范数 {l2:.4f}")

    print("\n第二步 完备性：连续函数在 L2 下能逼近一个不连续的极限")
    for k in (10, 100, 1000):
        print(f"  陡度 k = {k}：到阶跃函数的 L2 距离 {step_gap(k):.4f}")

    print("\n第三步 Hilbert 空间：正交展开与 Parseval")
    for t in (1, 5, 25):
        e, tail = square_wave_error(t)
        print(f"  方波取前 {t} 项：L2 误差² = {e:.4f}，Parseval 剩余能量 = {tail:.4f}")

    print("\n第四步 算子与谱：积分算子的特征值")
    ev, ex = brownian_kernel_eigs()
    print(f"  min(x, y) 核：离散特征值 {np.round(ev, 4).tolist()}，解析 {np.round(ex, 4).tolist()}")

    print("\n第五步 Banach 不动点定理")
    x, gaps = cos_fixed_point()
    print(f"  x ← cos x：不动点 {x:.6f}；相邻差 {[f'{g:.3g}' for g in gaps[:4]]} …，第 20 次 {gaps[19]:.2e}")
    print(f"  后期每次的比值 ≈ {gaps[30] / gaps[29]:.4f}（= |sin x*| = {abs(np.sin(x)):.4f}）")

    print("\n第六步 RKHS 与核方法：表示定理把无穷维问题变成 n 维")
    tr, te = kernel_ridge()
    print(f"  20 个点的高斯核岭回归：训练均方误差 {tr:.4f}，和真函数的均方误差 {te:.4f}")


if __name__ == "__main__":
    main()
