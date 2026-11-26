"""讲义：第 65 章 VLN 的几何与评测（notes/lessons/65-vln-metrics.md）
运行：uv run python ch65_vln_metrics.py

例子：一维序列上的 DTW 动态规划表；由 DTW 值换算 nDTW、SDTW；4 个回合的 SR 和 SPL；
小栅格上的前沿格子；VLMaps 式地把像素特征投到俯视栅格，再用文字嵌入查询。
里程碑 ME-M36 的“实现 DTW / nDTW / SPL 并在 R2R 风格的二维玩具轨迹上算”留给自己做；这里只用一维的数列。
"""

import math

import numpy as np


def dtw_table(r, q) -> np.ndarray:
    """D[i, j] = |r_i − q_j| + min(D[i−1, j], D[i, j−1], D[i−1, j−1])，D[0, 0] 从 0 起。"""
    n, m = len(r), len(q)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(r[i - 1] - q[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[1:, 1:]


R = [0, 1, 2, 3, 4]                    # 参考路径（沿走廊的位置，单位 m）
Q1 = [0, 1, 1, 2, 3, 4]                # 犹豫了一下，但一直沿着走
Q2 = [0, -1, -2, 0, 2, 4]              # 先走反了，再折回来；终点一样


def ndtw(dtw: float, n_ref: int, d_th: float = 3.0) -> float:
    """nDTW = exp(−DTW / (|R|·d_th))，R2R 里 d_th = 3 m。"""
    return math.exp(-dtw / (n_ref * d_th))


EPISODES = [(1, 10.0, 10.0), (1, 10.0, 15.0), (0, 8.0, 20.0), (1, 12.0, 24.0)]   # (成功, 最短路长 ℓ, 实际路长 p)


def spl(eps=EPISODES) -> tuple[float, float]:
    """SR = 成功率；SPL = (1/N) Σ S_i · ℓ_i / max(p_i, ℓ_i)。"""
    sr = sum(s for s, _, _ in eps) / len(eps)
    return sr, sum(s * l / max(p, l) for s, l, p in eps) / len(eps)


GRID = np.array([   # −1 未知，0 空闲，1 障碍
    [0, 0, 0, 1, -1, -1],
    [0, 0, 0, 1, -1, -1],
    [0, 0, 0, 0, -1, -1],
    [1, 1, 0, 0, 0, -1],
    [-1, -1, 0, 0, 0, -1],
])


def frontiers(g=GRID) -> list[tuple[int, int]]:
    """前沿 = 空闲格里，上下左右至少有一个未知格的那些。"""
    out = []
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i, j] != 0:
                continue
            nb = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if any(0 <= a < g.shape[0] and 0 <= b < g.shape[1] and g[a, b] == -1 for a, b in nb):
                out.append((i, j))
    return out


FX, CX = 100.0, 50.0                    # 一条图像行（宽 100 像素）的焦距和主点
PIXELS = [  # (列 u, 深度 m, 像素的视觉-语言特征)
    (30, 2.0, [0.9, 0.1, 0.0]),
    (35, 2.1, [0.8, 0.2, 0.1]),
    (70, 3.0, [0.1, 0.9, 0.2]),
    (80, 3.2, [0.2, 0.8, 0.1]),
]


def vlmap(pixels=PIXELS, cell: float = 1.0) -> dict[tuple[int, int], np.ndarray]:
    """反投影到相机坐标 x = (u − c_x)·z/f、前方 z，按 1 m 格子分桶，格内特征取平均。"""
    bins: dict[tuple[int, int], list] = {}
    for u, z, f in pixels:
        x = (u - CX) * z / FX
        bins.setdefault((int(math.floor(x / cell)), int(math.floor(z / cell))), []).append(f)
    return {k: np.mean(v, 0) for k, v in bins.items()}


def query(text_emb, m=None) -> tuple[tuple[int, int], dict]:
    m = vlmap() if m is None else m
    t = np.asarray(text_emb, float) / np.linalg.norm(text_emb)
    s = {k: float(v @ t / np.linalg.norm(v)) for k, v in m.items()}
    return max(s, key=s.get), s


def main() -> None:
    print("\n第一步 DTW 的动态规划表（R = [0, 1, 2, 3, 4]）")
    for name, q in (("Q1", Q1), ("Q2", Q2)):
        D = dtw_table(R, q)
        print(f"  {name} = {q}：DTW = {D[-1, -1]:.0f}\n{D.astype(int)}")

    print("\n第二步 nDTW 与 SDTW（d_th = 3 m）")
    for name, q in (("Q1", Q1), ("Q2", Q2)):
        d = dtw_table(R, q)[-1, -1]
        print(f"  {name}：nDTW = {ndtw(d, len(R)):.4f}；两条都到了终点，SR = 1，SDTW = nDTW")

    print("\n第三步 SR 与 SPL")
    sr, s = spl()
    print(f"  回合 {EPISODES}：SR = {sr:.2f}，SPL = {s:.4f}")

    print("\n第四步 前沿探索")
    fr = frontiers()
    print(f"  前沿格子 {fr}，共 {len(fr)} 个")

    print("\n第五步 VLMaps：像素特征投到俯视栅格")
    for k, v in vlmap().items():
        print(f"  格子 {k}：平均特征 {v.round(3).tolist()}")
    for name, t in (("“沙发”", [1, 0, 0]), ("“电视”", [0, 1, 0])):
        best, _ = query(t)
        print(f"  查询 {name}：最相似的格子 {best}")


if __name__ == "__main__":
    main()
