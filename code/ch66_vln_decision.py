"""讲义：第 66 章 VLN 的序列决策（notes/lessons/66-vln-decision.md）
运行：uv run python ch66_vln_decision.py

例子：两个房间的信念更新（看到冰箱之后在厨房的概率）；走廊上的直方图滤波（移动一步、再观测一次）；
行为克隆和 DAgger 的误差上界（只代入公式）；FiLM 式的语言条件；两个观测动作的信息增益。
里程碑 ME-M36 的“模拟行为克隆的误差累积 vs DAgger”留给自己做；这里只代入 DAgger 论文里的上界。
"""

import math

import numpy as np


def bayes(prior, lik) -> np.ndarray:
    """b'(s) ∝ P(o | s) · b(s)。"""
    b = np.asarray(prior, float) * np.asarray(lik, float)
    return b / b.sum()


DOORS = np.array([1, 0, 0, 1, 0])        # 走廊 5 格，哪些格有门


def corridor_step(belief, move: int = 1, p_ok: float = 0.8, saw_door: bool = True, p_hit: float = 0.9, p_false: float = 0.2):
    """预测：以 0.8 的概率真的走了 move 格（走廊是环形），0.2 原地不动。更新：门格看到门的概率 0.9，非门格 0.2。"""
    b = np.asarray(belief, float)
    pred = p_ok * np.roll(b, move) + (1 - p_ok) * b
    lik = np.where(DOORS == 1, p_hit, p_false) if saw_door else np.where(DOORS == 1, 1 - p_hit, 1 - p_false)
    return pred, bayes(pred, lik)


def bc_bound(eps: float, T: int) -> float:
    """行为克隆（DAgger 论文 §2）：每步犯错率 ε，T 步的总代价比专家多至多 T²ε（再大也不超过 T）。"""
    return min(T * T * eps, T)


def dagger_bound(eps: float, T: int, u: float = 1.0) -> float:
    """DAgger（定理 3.2 的主项）：多出的代价 O(u·T·ε)，u 是一次犯错之后还要多付的代价。"""
    return min(u * T * eps, T)


def film(h, lang, Wg, Wb) -> np.ndarray:
    """FiLM：γ = W_γ·l、β = W_β·l，输出 γ ⊙ h + β。同一视觉特征 h，不同指令得到不同的调制。"""
    return (Wg @ lang) * h + Wb @ lang


H = np.array([1.0, 0.5, -0.5])
WG = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
WB = np.array([[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]])
LANG = {"左转": np.array([1.0, 0.0]), "右转": np.array([0.0, 1.0])}


def entropy(p) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(prior, lik_table) -> float:
    """IG = H(b) − Σ_o P(o) H(b | o)；lik_table[o][s] = P(o | s)。"""
    prior = np.asarray(prior, float)
    out = entropy(prior)
    for lik in lik_table:
        po = float(np.dot(lik, prior))
        if po > 0:
            out -= po * entropy(bayes(prior, lik))
    return out


PRIOR4 = [0.4, 0.3, 0.2, 0.1]                     # 目标在 4 个房间里的信念
LOOK_LEFT = [[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]]   # 看左边：能分清 {0, 1} 和 {2, 3}
LOOK_RIGHT = [[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]]  # 看右边：只能认出房间 0


def main() -> None:
    print("\n第一步 信念状态：两个房间")
    b = bayes([0.5, 0.5], [0.8, 0.1])
    print(f"  先验 [厨房 0.5, 卧室 0.5]，看到冰箱（P = 0.8 / 0.1）：后验 [{b[0]:.4f}, {b[1]:.4f}]")
    b2 = bayes(b, [0.8, 0.1])
    print(f"  再看到一次冰箱：[{b2[0]:.4f}, {b2[1]:.4f}]")

    print("\n第二步 走廊上的直方图滤波（门在格 0、3）")
    b0 = np.full(5, 0.2)
    _, b1 = corridor_step(b0, move=0, p_ok=1.0)
    print(f"  均匀先验，原地看到门：{b1.round(4).tolist()}")
    pred, b2 = corridor_step(b1, move=1)
    print(f"  往右走一格（预测）：{pred.round(4).tolist()}")
    _, b3 = corridor_step(b1, move=1, saw_door=False)
    print(f"  走完没看到门：{b3.round(4).tolist()}")

    print("\n第三步 行为克隆 vs DAgger 的上界（只代入公式）")
    for T in (10, 100, 1000):
        print(f"  ε = 0.001、T = {T}：BC ≤ {bc_bound(0.001, T):g}，DAgger ~ {dagger_bound(0.001, T):g}；一步都不错的概率 {(1 - 0.001) ** T:.4f}")

    print("\n第四步 语言条件策略：FiLM")
    for name, l in LANG.items():
        print(f"  同一视觉特征 {H.tolist()}、指令“{name}”：{film(H, l, WG, WB).round(3).tolist()}")

    print("\n第五步 探索的信息增益")
    print(f"  信念 {PRIOR4}：熵 {entropy(PRIOR4):.4f} bit")
    for name, t in (("看左边", LOOK_LEFT), ("看右边", LOOK_RIGHT)):
        print(f"  {name}：信息增益 {info_gain(PRIOR4, t):.4f} bit")


if __name__ == "__main__":
    main()
