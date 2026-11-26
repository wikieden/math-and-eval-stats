"""讲义：第 57 章 地图表示（notes/lessons/57-maps.md）
运行：uv run python ch57_maps.py

例子：一个格子的占用概率在“命中、命中、未命中、命中”下的 log-odds 更新；一条射线上 3 次带噪声的深度观测融合成 TSDF；
一维 / 二维的欧氏距离场（ESDF）；稠密体素和八叉树存一个球面要多少格子。
里程碑 ME-M34 的“整张占用栅格的 log-odds 更新”留给自己做；这里只看一个格子。
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def occupancy_sequence(obs: list[int], p_hit: float = 0.7, p_miss: float = 0.4, clamp: float = 3.5) -> list[float]:
    """log-odds 贝叶斯更新：l ← l + log(p/(1−p)) − l0，l0 是先验（0.5 → 0）。命中加 0.847，未命中加 −0.405；夹在 ±clamp。返回每步后的概率。"""
    l, out = 0.0, []
    for o in obs:
        l += logit(p_hit) if o else logit(p_miss)
        l = float(np.clip(l, -clamp, clamp))
        out.append(float(1 - 1 / (1 + np.exp(l))))
    return out


def tsdf_fuse(depths: list[float], weights: list[float], xs: np.ndarray, trunc: float = 0.3) -> np.ndarray:
    """沿射线的每个位置 x：一次观测给出截断的有符号距离 d = clip(depth − x, −τ, τ)，多次观测按权重平均。"""
    num, den = np.zeros_like(xs), np.zeros_like(xs)
    for d, w in zip(depths, weights):
        sd = np.clip(d - xs, -trunc, trunc)
        num += w * sd
        den += w
    return num / den


def zero_crossing(xs: np.ndarray, f: np.ndarray) -> float:
    i = int(np.where(np.diff(np.sign(f)) != 0)[0][0])
    return float(xs[i] - f[i] * (xs[i + 1] - xs[i]) / (f[i + 1] - f[i]))


def esdf_2d() -> np.ndarray:
    """7×7 栅格中间一个 2×2 的障碍物，每个空闲格到最近障碍物的欧氏距离（格子数）。"""
    occ = np.zeros((7, 7), bool)
    occ[2:4, 3:5] = True
    return distance_transform_edt(~occ)


def octree_vs_dense(res: float, r: float = 1.0, box: float = 2.0) -> tuple[int, int]:
    """边长 2·box 的立方体里一个半径 r 的球面：稠密体素数 (2box/res)³；只存和球面相交的格子（近似八叉树叶子数）≈ 球面积 / res²。"""
    dense = int((2 * box / res) ** 3)
    surface = int(4 * np.pi * r * r / (res * res))
    return dense, surface


def main() -> None:
    print("\n第一步 占用栅格：log-odds 让贝叶斯更新变成加法")
    seq = [1, 1, 0, 1]
    ps = occupancy_sequence(seq)
    print(f"  命中 log-odds +{logit(0.7):.3f}、未命中 {logit(0.4):.3f}；观测 {seq}：概率 {[round(p, 4) for p in ps]}")
    long = occupancy_sequence([1] * 20 + [0] * 3)
    print(f"  连续命中 20 次再未命中 3 次（夹在 ±3.5）：{long[19]:.4f} → {long[-1]:.4f}；不夹的话要很久才翻得过来")

    print("\n第二步 TSDF：多次观测按权重平均，零交叉就是表面")
    xs = np.linspace(1.5, 2.5, 101)
    f = tsdf_fuse([2.02, 1.97, 2.04], [1, 1, 1], xs)
    print(f"  3 次深度观测 2.02、1.97、2.04 m：融合后零交叉 {zero_crossing(xs, f):.4f} m（平均 {np.mean([2.02, 1.97, 2.04]):.4f}）")

    print("\n第三步 ESDF：每个格子到最近障碍物的距离")
    d = esdf_2d()
    print(f"  7×7 栅格、中间 2×2 障碍物：\n{np.round(d, 2)}")

    print("\n第四步 八叉树：只细分有东西的地方")
    for res in (0.1, 0.05, 0.02):
        dn, sf = octree_vs_dense(res)
        print(f"  分辨率 {res} m：稠密体素 {dn:,} 个，球面附近的格子约 {sf:,} 个（{dn / sf:.0f} 倍）")


if __name__ == "__main__":
    main()
