"""讲义：第 14 章 无约束优化（notes/lessons/14-unconstrained-optimization.md）
运行：uv run python ch14_optimization.py

例子：二次函数 f(x) = ½(x₁² + 10x₂²)；Adam / AdamW 的第一步；warmup + cosine；梯度裁剪；高斯-牛顿和 LM 拟合 y = a·e^{bx}。
里程碑 ME-M9 的 Rosenbrock 对比和与 torch.optim.AdamW 对齐留给自己做。
"""

import math

import numpy as np

H = np.diag([1.0, 10.0])  # f(x) = ½ xᵀ H x，最小点在 0
X0 = np.array([1.0, 1.0])
GX = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
GY = np.array([2.0, 3.3, 5.4, 9.0, 14.8])  # 编的：大约 y = 2·e^{x}


def gd(lr: float, steps: int, momentum: float = 0.0) -> tuple[np.ndarray, int]:
    """梯度下降（可选动量 v ← βv + g，x ← x − lr·v）。返回 (最后的 x, 第一次 ‖x‖ < 1e-3 的步数，没到就 −1)。"""
    x, v, hit = X0.copy(), np.zeros(2), -1
    for k in range(1, steps + 1):
        g = H @ x
        v = momentum * v + g
        x = x - lr * v
        if hit < 0 and np.linalg.norm(x) < 1e-3:
            hit = k
        if not np.isfinite(x).all() or np.linalg.norm(x) > 1e6:
            return x, -1
    return x, hit


def adam_step(x, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=True):
    """Adam Algorithm 1 的一步；wd > 0 时，decoupled=True 是 AdamW（直接衰减权重），False 是把 wd·x 加进梯度（L2）。"""
    if wd and not decoupled:
        g = g + wd * x
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1**t), v / (1 - b2**t)
    x = x - lr * mh / (np.sqrt(vh) + eps)
    if wd and decoupled:
        x = x - lr * wd * x
    return x, m, v


def warmup_cosine(step: int, total: int = 1000, warmup: int = 100, peak: float = 3e-4, floor: float = 3e-5) -> float:
    """线性 warmup 到 peak，再按余弦降到 floor。"""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))


def clip_by_norm(g: np.ndarray, max_norm: float) -> np.ndarray:
    """梯度范数超过 max_norm 就按比例缩小，方向不变。"""
    n = np.linalg.norm(g)
    return g * (max_norm / n) if n > max_norm else g


def gauss_newton(a: float = 1.0, b: float = 0.5, iters: int = 6) -> list[tuple[float, float, float]]:
    """拟合 y = a·e^{bx}：每步用残差的雅可比 J 解 (JᵀJ) Δ = −Jᵀr。返回每步的 (a, b, 残差平方和)。"""
    hist = []
    for _ in range(iters):
        e = np.exp(b * GX)
        r = a * e - GY
        hist.append((a, b, float(r @ r)))
        j = np.column_stack([e, a * GX * e])
        da, db = np.linalg.solve(j.T @ j, -j.T @ r)
        a, b = a + da, b + db
    r = a * np.exp(b * GX) - GY
    hist.append((a, b, float(r @ r)))
    return hist


def levenberg_marquardt(a: float = 1.0, b: float = 0.5, iters: int = 6, lam: float = 1.0) -> list[tuple[float, float, float]]:
    """LM：解 (JᵀJ + λI) Δ = −Jᵀr；这一步让残差变小就接受并把 λ 减半，否则拒绝并把 λ 乘 10。"""
    def sse(a, b):
        r = a * np.exp(b * GX) - GY
        return float(r @ r)
    hist = [(a, b, sse(a, b))]
    for _ in range(iters):
        while True:
            e = np.exp(b * GX)
            r = a * e - GY
            j = np.column_stack([e, a * GX * e])
            da, db = np.linalg.solve(j.T @ j + lam * np.eye(2), -j.T @ r)
            if sse(a + da, b + db) < sse(a, b):
                a, b, lam = a + da, b + db, lam / 2
                break
            lam *= 10
        hist.append((a, b, sse(a, b)))
    return hist


def main() -> None:
    print("\n第一步 梯度下降：步长不能超过 2/L（L = 最大曲率 10）")
    for lr in (0.05, 0.18, 0.21):
        x, hit = gd(lr, 500)
        print(f"  lr = {lr}：{'发散' if hit < 0 and np.linalg.norm(x) > 1 else f'{hit} 步收敛到 ‖x‖ < 1e-3'}")
    print(f"  条件数 κ = 10：沿平的方向（曲率 1）每步只缩 1 − lr·1")

    print("\n第二步 动量：在窄谷里少走之字")
    for beta in (0.0, 0.5, 0.8):
        _, hit = gd(0.05, 2000, beta)
        print(f"  lr = 0.05，β = {beta}：{hit} 步")

    print("\n第三步 Adam 的第一步：每个坐标都走约 lr，和梯度大小无关")
    g = np.array([0.1, 10.0])
    x1, _, _ = adam_step(np.zeros(2), g, np.zeros(2), np.zeros(2), 1)
    print(f"  梯度 {g.tolist()} → 更新量 {np.round(-x1, 6).tolist()}（lr = 1e-3）")
    w = np.array([1.0, 1.0])
    for dec in (True, False):
        xw, _, _ = adam_step(w, np.array([0.1, 10.0]), np.zeros(2), np.zeros(2), 1, wd=0.1, decoupled=dec)
        print(f"  {'AdamW（解耦）' if dec else 'Adam + L2'}，wd = 0.1：新权重 {np.round(xw, 6).tolist()}")

    print("\n第四步 学习率表与梯度裁剪")
    print("  warmup + cosine：", [f"{warmup_cosine(s):.2e}" for s in (0, 50, 99, 100, 550, 999)])
    print(f"  梯度 [3, 4]（范数 5）裁到 1：{np.round(clip_by_norm(np.array([3.0, 4.0]), 1.0), 4).tolist()}")

    print("\n第五步 高斯-牛顿：用一阶展开的最小二乘当二阶近似")
    for k, (a, b, sse) in enumerate(gauss_newton()):
        print(f"  第 {k} 步：a = {a:.4f}，b = {b:.4f}，残差平方和 {sse:.4f}")
    print("  LM（加阻尼 λ，变差就拒绝）：")
    for k, (a, b, sse) in enumerate(levenberg_marquardt()):
        print(f"  第 {k} 步：a = {a:.4f}，b = {b:.4f}，残差平方和 {sse:.4f}")


if __name__ == "__main__":
    main()
