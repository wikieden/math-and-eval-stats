"""讲义：第 23 章 MDP 与 Bellman 的完整推导（notes/lessons/23-mdp-bellman.md）
运行：uv run python ch23_mdp_bellman.py

例子：一个 3 个状态、2 个动作的 MDP（编的）：状态 0 → 1 → 2 是一条链；在 0、1 上"走"往前一格、扣 1 分，"停"原地不动、0 分；
到了 2 以后，不管做什么都留在 2、每步 +1（无限时域，V*(2) = 1/(1 − γ) = 10）。
里程碑 ME-M19 的 4×4 网格收敛曲线留给自己做。
"""

import numpy as np

GAMMA = 0.9
# P[a][s, s']、R[a][s]：a = 0 停、a = 1 走
P = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              [[0, 1, 0], [0, 0, 1], [0, 0, 1]]], float)
R = np.array([[0.0, 0.0, 1.0],
              [-1.0, -1.0, 1.0]])


def policy_value(pi: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Bellman 期望方程：V^π = r_π + γ P_π V^π → V^π = (I − γP_π)⁻¹ r_π。pi[s] 是在 s 选的动作。"""
    p = np.array([P[pi[s], s] for s in range(3)])
    r = np.array([R[pi[s], s] for s in range(3)])
    return np.linalg.solve(np.eye(3) - gamma * p, r)


def bellman_optimal(v: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Bellman 最优算子：(T V)(s) = max_a [R(a, s) + γ Σ P(a, s, s') V(s')]。"""
    return np.max(R + gamma * P @ v, axis=0)


def value_iteration(tol: float = 1e-8, gamma: float = GAMMA) -> tuple[np.ndarray, int, list[float]]:
    """反复套最优算子，返回 (V*, 迭代次数, 每次的 ‖V_{k+1} − V_k‖∞)。"""
    v, gaps = np.zeros(3), []
    while True:
        nv = bellman_optimal(v, gamma)
        gaps.append(float(np.abs(nv - v).max()))
        v = nv
        if gaps[-1] < tol:
            return v, len(gaps), gaps


def contraction_check(seed: int = 0, gamma: float = GAMMA) -> float:
    """随便取两个 V、U，看 ‖TV − TU‖∞ / ‖V − U‖∞ 是不是 ≤ γ。"""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(1000):
        v, u = rng.normal(0, 10, 3), rng.normal(0, 10, 3)
        worst = max(worst, np.abs(bellman_optimal(v, gamma) - bellman_optimal(u, gamma)).max() / np.abs(v - u).max())
    return worst


def iterations_needed(eps: float, gamma: float, v0_gap: float) -> int:
    """误差每次至少乘 γ：要从 v0_gap 降到 eps，至少 ⌈log(eps/v0_gap)/log γ⌉ 次。"""
    return int(np.ceil(np.log(eps / v0_gap) / np.log(gamma)))


def greedy(v: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """按 V 贪心取动作。"""
    return np.argmax(R + gamma * P @ v, axis=0)


def policy_iteration(gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray, int]:
    """策略评估（解线性方程）+ 贪心改进，直到策略不变。"""
    pi, k = np.zeros(3, int), 0
    while True:
        v = policy_value(pi, gamma)
        new = greedy(v, gamma)
        k += 1
        if np.array_equal(new, pi):
            return pi, v, k
        pi = new


def main() -> None:
    print("\n第一步 Bellman 期望方程：固定策略，值函数是一个线性方程组的解")
    for name, pi in (("一直停", np.array([0, 0, 0])), ("一直走", np.array([1, 1, 0]))):
        print(f"  {name}：V = {np.round(policy_value(pi), 4).tolist()}")

    print("\n第二步 Bellman 最优方程与值迭代")
    v, n, gaps = value_iteration()
    print(f"  V* = {np.round(v, 4).tolist()}，{n} 次收敛；最优策略 {greedy(v).tolist()}（0 停 / 1 走）")
    print(f"  前几次的 ‖V_(k+1) − V_k‖∞：{[round(g, 4) for g in gaps[:5]]}")

    print("\n第三步 压缩映射：‖TV − TU‖∞ ≤ γ‖V − U‖∞")
    print(f"  随机 1000 对 V、U，比值最大 {contraction_check():.4f}（γ = {GAMMA}）")

    print("\n第四步 要多少次迭代：误差每次至少乘 γ")
    for g in (0.9, 0.99):
        print(f"  γ = {g}：初始误差 10 降到 1e-6 至少要 {iterations_needed(1e-6, g, 10.0)} 次")

    print("\n第五步 策略迭代：评估 + 贪心改进，几步就停")
    pi, v, k = policy_iteration()
    print(f"  {k} 轮策略迭代得到 {pi.tolist()}，V = {np.round(v, 4).tolist()}")


if __name__ == "__main__":
    main()
