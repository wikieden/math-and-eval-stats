"""讲义：第 50 章 解码与采样（notes/lessons/50-decoding.md）
运行：uv run python ch50_decoding.py

例子：6 个候选 token 的 logits [2, 1.5, 1, 0, −1, −2]；温度、top-k、top-p；一个两步的小树上贪心不等于最优；
投机解码的接受率、每步期望 token 数，以及“接受—拒绝后重采”的输出分布恰好等于目标分布（解析计算）；约束解码。
里程碑 ME-M30 的“实现投机采样并用 10 万次采样验证输出分布不变”留给自己做。
"""

import numpy as np

LOGITS = np.array([2.0, 1.5, 1.0, 0.0, -1.0, -2.0])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def top_k(p: np.ndarray, k: int) -> np.ndarray:
    q = np.zeros_like(p)
    idx = np.argsort(-p)[:k]
    q[idx] = p[idx]
    return q / q.sum()


def top_p(p: np.ndarray, thr: float) -> np.ndarray:
    """核采样：按概率从大到小累加，保留累计概率刚好 ≥ thr 的最小集合。"""
    order = np.argsort(-p)
    keep = order[: int(np.searchsorted(np.cumsum(p[order]), thr) + 1)]
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()


TREE = {"A": (0.6, {"x": 0.4, "y": 0.3, "z": 0.3}), "B": (0.4, {"x": 0.9, "y": 0.1})}


def greedy_vs_best() -> tuple[tuple[str, float], tuple[str, float]]:
    """两步：第一步选 A（0.6）或 B（0.4），第二步按条件概率。贪心每步取最大；beam 宽 2 保留两个前缀再比整句概率。"""
    a = max(TREE, key=lambda k: TREE[k][0])
    g2 = max(TREE[a][1], key=TREE[a][1].get)
    greedy = (a + g2, TREE[a][0] * TREE[a][1][g2])
    cands = [(k1 + k2, p1 * p2) for k1, (p1, nxt) in TREE.items() for k2, p2 in nxt.items()]
    return greedy, max(cands, key=lambda c: c[1])


def speculative_output(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    """草稿模型按 q 提议 x，以 min(1, p(x)/q(x)) 接受；拒绝时从 norm(max(0, p − q)) 重采。
    输出分布 = q·min(1, p/q) + (1 − α)·norm(max(0, p − q))，α = Σ min(p, q)。返回 (输出分布, α)。"""
    acc = np.minimum(p, q)
    alpha = acc.sum()
    resid = np.maximum(p - q, 0)
    return acc + (1 - alpha) * resid / resid.sum(), float(alpha)


def expected_tokens(alpha: float, gamma: int) -> float:
    """一次提议 γ 个草稿 token、每个独立以概率 α 被接受：每次大模型前向期望产出 (1 − α^{γ+1})/(1 − α) 个 token（Leviathan 2022）。"""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def constrained(p: np.ndarray, allowed: list[int]) -> np.ndarray:
    """约束解码：不允许的 token 概率置 0（logit 填 −∞）再归一化。"""
    q = np.zeros_like(p)
    q[allowed] = p[allowed]
    return q / q.sum()


def main() -> None:
    p = softmax(LOGITS)
    print("\n第一步 温度")
    for t in (0.5, 1.0, 2.0):
        print(f"  T = {t}：{np.round(softmax(LOGITS / t), 3).tolist()}")

    print("\n第二步 top-k 与 top-p")
    print(f"  原分布 {np.round(p, 3).tolist()}")
    print(f"  top-k = 2：{np.round(top_k(p, 2), 3).tolist()}")
    print(f"  top-p = 0.8：{np.round(top_p(p, 0.8), 3).tolist()}")

    print("\n第三步 贪心不一定得到概率最大的整句")
    (gs, gp), (bs, bp) = greedy_vs_best()
    print(f"  贪心：{gs}，整句概率 {gp:.2f}；beam 宽 2 找到：{bs}，整句概率 {bp:.2f}")

    print("\n第四步 投机解码：输出分布不变")
    q = softmax(np.array([1.5, 2.0, 0.5, 0.0, -1.0, -1.0]))
    out, alpha = speculative_output(p, q)
    print(f"  目标 p {np.round(p, 4).tolist()}")
    print(f"  草稿 q {np.round(q, 4).tolist()}")
    print(f"  接受—重采后的分布 {np.round(out, 4).tolist()}（和 p 的最大差 {np.abs(out - p).max():.1e}），接受率 α = {alpha:.4f}")
    for g in (1, 4, 8):
        print(f"  一次提议 γ = {g} 个：每次大模型前向期望得到 {expected_tokens(alpha, g):.2f} 个 token")

    print("\n第五步 约束解码：只允许 token 2、3、4")
    print(f"  {np.round(constrained(p, [2, 3, 4]), 3).tolist()}")


if __name__ == "__main__":
    main()
