"""讲义：第 51 章 后训练的数学（notes/lessons/51-post-training.md）
运行：uv run python ch51_post_training.py

例子：一个只有 3 个回答（A、B、C）的问题；参考策略 π_ref = [0.5, 0.3, 0.2]，奖励 r = [1, 2, 0]。
SFT 的交叉熵 = KL + 熵；Bradley-Terry 偏好概率；KL 正则目标的闭式最优策略；DPO 隐含奖励；GRPO 的组内归一化和 k3 KL 估计。
里程碑 ME-M31 的“5 选项玩具偏好数据上实现 DPO 和 GRPO 的一步更新”留给自己做。
"""

import numpy as np

PREF = np.array([0.5, 0.3, 0.2])
R = np.array([1.0, 2.0, 0.0])


def sft_decomposition(data: np.ndarray, pi: np.ndarray) -> tuple[float, float, float]:
    """交叉熵 H(data, π) = KL(data ‖ π) + H(data)：最小化交叉熵 = 最小化 KL = 最大似然。返回 (交叉熵, KL, 熵)。"""
    ce = -(data * np.log(pi)).sum()
    h = -(data * np.log(data)).sum()
    return float(ce), float(ce - h), float(h)


def bt_prob(ra: float, rb: float) -> float:
    """Bradley-Terry：P(a ≻ b) = σ(r_a − r_b)。"""
    return float(1 / (1 + np.exp(-(ra - rb))))


def optimal_policy(beta: float, pref: np.ndarray = PREF, r: np.ndarray = R) -> np.ndarray:
    """max E_π[r] − β·KL(π ‖ π_ref) 的解：π*(y) ∝ π_ref(y)·exp(r(y)/β)。"""
    w = pref * np.exp(r / beta)
    return w / w.sum()


def objective(pi: np.ndarray, beta: float) -> float:
    return float((pi * R).sum() - beta * (pi * np.log(pi / PREF)).sum())


def implied_reward(pi: np.ndarray, beta: float) -> np.ndarray:
    """DPO 的关键反解：r(y) = β log(π*(y)/π_ref(y)) + β log Z。log Z 对所有回答相同，比较两个回答时消掉。"""
    return beta * np.log(pi / PREF)


def dpo_loss(pi: np.ndarray, w: int, l: int, beta: float) -> float:
    """一对偏好 (y_w ≻ y_l) 的 DPO 损失 −log σ(β log(π(y_w)/π_ref(y_w)) − β log(π(y_l)/π_ref(y_l)))。"""
    d = beta * (np.log(pi[w] / PREF[w]) - np.log(pi[l] / PREF[l]))
    return float(np.log1p(np.exp(-d)))


def grpo_adv(rewards: np.ndarray) -> np.ndarray:
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def k3(ratio_ref_over_pi: np.ndarray) -> np.ndarray:
    """GRPO 用的 KL 估计 k3 = r − log r − 1，r = π_ref/π_θ（对 π_θ 采样时无偏、恒非负）。"""
    r = ratio_ref_over_pi
    return r - np.log(r) - 1


def main() -> None:
    print("\n第一步 SFT = 最大似然 = 最小化 KL")
    data = np.array([0.6, 0.3, 0.1])
    for name, pi in (("π_ref", PREF), ("和数据一样", data)):
        ce, kl, h = sft_decomposition(data, pi)
        print(f"  数据分布 [0.6, 0.3, 0.1]，模型 {name}：交叉熵 {ce:.4f} = KL {kl:.4f} + 熵 {h:.4f}")

    print("\n第二步 Bradley-Terry 奖励模型")
    for d in (0.0, 1.0, 3.0):
        print(f"  r_a − r_b = {d}：P(a ≻ b) = {bt_prob(d, 0):.4f}，奖励模型损失 −log σ = {-np.log(bt_prob(d, 0)):.4f}")

    print("\n第三步 KL 正则目标的闭式最优策略 π* ∝ π_ref·e^(r/β)")
    for beta in (0.1, 1.0, 10.0):
        ps = optimal_policy(beta)
        print(f"  β = {beta}：π* = {np.round(ps, 4).tolist()}，目标值 {objective(ps, beta):.4f}（π_ref 时 {objective(PREF, beta):.4f}）")

    print("\n第四步 DPO：奖励可以用策略表示")
    beta = 1.0
    ps = optimal_policy(beta)
    ir = implied_reward(ps, beta)
    print(f"  β = 1：β log(π*/π_ref) = {np.round(ir, 4).tolist()}；减去 r = {R.tolist()} 后都等于 {np.round(ir - R, 4).tolist()}（= −log Z，共同的常数）")
    print(f"  偏好对 B ≻ C：π = π_ref 时 DPO 损失 {dpo_loss(PREF, 1, 2, beta):.4f}，π = π* 时 {dpo_loss(ps, 1, 2, beta):.4f}")

    print("\n第五步 GRPO：组内归一化的优势和 k3 KL 估计")
    rw = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    print(f"  6 个回答的奖励 {rw.tolist()} → 优势 {np.round(grpo_adv(rw), 4).tolist()}")
    print(f"  k3(π_ref/π_θ) 在比率 0.5、1、2 时 = {np.round(k3(np.array([0.5, 1.0, 2.0])), 4).tolist()}（恒 ≥ 0）")


if __name__ == "__main__":
    main()
