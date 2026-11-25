"""讲义：第 13 章 多元微积分与自动微分（notes/lessons/13-calculus-autodiff.md）
运行：uv run python ch13_calculus_autodiff.py
"""

import math

import numpy as np


def grad_f(x: float, y: float) -> np.ndarray:
    """第一步：f(x, y) = x² + 3xy 的梯度 (2x + 3y, 3x)。"""
    return np.array([2 * x + 3 * y, 3 * x])


def fk_2link(t1: float, t2: float, l1: float = 1.0, l2: float = 1.0) -> np.ndarray:
    """第二步：两连杆平面臂的正运动学，末端 (x, y)。"""
    return np.array([l1 * math.cos(t1) + l2 * math.cos(t1 + t2), l1 * math.sin(t1) + l2 * math.sin(t1 + t2)])


def jacobian_2link(t1: float, t2: float, l1: float = 1.0, l2: float = 1.0) -> np.ndarray:
    """第二步：解析雅可比 J = ∂(x, y)/∂(θ1, θ2)。"""
    s1, c1, s12, c12 = math.sin(t1), math.cos(t1), math.sin(t1 + t2), math.cos(t1 + t2)
    return np.array([[-l1 * s1 - l2 * s12, -l2 * s12], [l1 * c1 + l2 * c12, l2 * c12]])


def numeric_jacobian(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """第五步：中心差分 (f(x+h e_i) - f(x-h e_i)) / 2h，用来核对解析 / 自动微分。"""
    cols = []
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        cols.append((f(*(x + e)) - f(*(x - e))) / (2 * h))
    return np.stack(cols, axis=1)


class Var:
    """第四步：50 行以内的标量反向模式自动微分（只支持 +、×、sin）。"""

    def __init__(self, value, parents=()):
        self.value, self.parents, self.grad = value, parents, 0.0

    def __add__(self, other):
        return Var(self.value + other.value, ((self, 1.0), (other, 1.0)))

    def __mul__(self, other):
        return Var(self.value * other.value, ((self, other.value), (other, self.value)))

    def sin(self):
        return Var(math.sin(self.value), ((self, math.cos(self.value)),))

    def backward(self):
        order, seen = [], set()

        def topo(v):  # 拓扑排序：保证一个节点的伴随量收齐了才往回传
            if id(v) not in seen:
                seen.add(id(v))
                for p, _ in v.parents:
                    topo(p)
                order.append(v)

        topo(self)
        self.grad = 1.0
        for v in reversed(order):
            for p, local in v.parents:
                p.grad += v.grad * local  # 伴随量 = 上游伴随 × 局部导数，多条路径相加


def main() -> None:
    print("第一步 梯度：f = x² + 3xy 在 (1, 2)")
    print(f"  ∇f = {grad_f(1, 2).tolist()}")
    t = (math.radians(30), math.radians(60))
    print("\n第二步 两连杆臂 θ = (30°, 60°)，l1 = l2 = 1")
    print(f"  末端 = {np.round(fk_2link(*t), 4).tolist()}")
    J = jacobian_2link(*t)
    print(f"  J = {np.round(J, 4).tolist()}，det J = {np.linalg.det(J):.4f}（= l1·l2·sin θ2）")
    print(f"  θ2 = 0（手臂伸直）时 det J = {np.linalg.det(jacobian_2link(t[0], 0.0)):.4f}  ← 奇异")
    print("\n第三步 链式法则：关节速度 (0.1, -0.2) rad/s 对应的末端速度 = J·θ̇")
    print(f"  v = {np.round(J @ np.array([0.1, -0.2]), 4).tolist()}")
    print("\n第四步 反向模式：f = x·y + sin x 在 x = 2, y = 3")
    x, y = Var(2.0), Var(3.0)
    f = x * y + x.sin()
    f.backward()
    print(f"  f = {f.value:.4f}，∂f/∂x = {x.grad:.4f}（= y + cos x），∂f/∂y = {y.grad:.4f}（= x）")
    print("\n第五步 数值核对雅可比")
    Jn = numeric_jacobian(fk_2link, np.array(t))
    print(f"  最大误差 = {np.abs(Jn - J).max():.2e}")


if __name__ == "__main__":
    main()
