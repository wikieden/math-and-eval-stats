"""讲义：第 60 章 感知的学习与评测（notes/lessons/60-perception-metrics.md）
运行：uv run python ch60_perception_metrics.py

例子：两个框的 IoU；3 个重叠的检测框做 NMS；8 个检测、5 个真值的 PR 曲线和 AP；3 个预测对 3 个真值的匈牙利匹配
（和贪心匹配比较）；三元组损失；InfoNCE；NetVLAD 的维数。
里程碑 ME-M35 的“IoU、NMS、mAP、匈牙利匹配的完整实现和测试”留给自己做；这里每个只算一个小例子。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b) -> float:
    """框 (x1, y1, x2, y2)：交集面积 / 并集面积。"""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


BOXES = [((10, 10, 50, 50), 0.9), ((12, 14, 52, 54), 0.8), ((60, 10, 100, 50), 0.7), ((15, 12, 48, 49), 0.6)]


def nms(boxes=BOXES, thr: float = 0.5) -> list[int]:
    """按分数从高到低，保留当前框，删掉和它 IoU > thr 的框。"""
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i][0], boxes[j][0]) <= thr]
    return keep


HITS = [1, 1, 0, 1, 0, 0, 1, 0]   # 按分数排好的 8 个检测是否命中（IoU ≥ 0.5 且没被更高分的检测占用）
N_GT = 5


def ap(hits=HITS, n_gt: int = N_GT) -> tuple[list[float], list[float], float]:
    """精确率 P_k = 命中数/k，召回率 R_k = 命中数/真值数；AP = 精确率包络（右侧最大值）对召回率的面积。"""
    tp = np.cumsum(hits)
    prec = tp / np.arange(1, len(hits) + 1)
    rec = tp / n_gt
    env = np.maximum.accumulate(prec[::-1])[::-1]
    r_prev, area = 0.0, 0.0
    for r, p in zip(rec, env):
        area += (r - r_prev) * p
        r_prev = r
    return prec.round(4).tolist(), rec.round(4).tolist(), float(area)


COST = np.array([[1.0, 2.0, 8.0], [1.5, 6.0, 9.0], [7.0, 3.0, 4.0]])


def hungarian(cost=COST) -> tuple[list[int], float]:
    r, c = linear_sum_assignment(cost)
    return c.tolist(), float(cost[r, c].sum())


def greedy(cost=COST) -> tuple[list[int], float]:
    """每次挑全局最小的一对，划掉这一行一列。"""
    m = cost.copy()
    assign, total = [-1] * len(m), 0.0
    for _ in range(len(m)):
        i, j = np.unravel_index(np.argmin(m), m.shape)
        assign[i] = int(j)
        total += cost[i, j]
        m[i, :], m[:, j] = np.inf, np.inf
    return assign, total


def triplet(da: float, dn: float, margin: float = 0.2) -> float:
    """三元组损失 max(0, d(a, p) − d(a, n) + margin)。"""
    return max(0.0, da - dn + margin)


def infonce(sim_pos: float, sim_negs: list[float], tau: float) -> float:
    """InfoNCE = −log( e^{s⁺/τ} / Σ_all e^{s/τ} )。"""
    z = np.array([sim_pos] + list(sim_negs)) / tau
    return float(-(z[0] - (z.max() + np.log(np.exp(z - z.max()).sum()))))


def main() -> None:
    print("\n第一步 IoU")
    print(f"  (10, 10, 50, 50) 和 (12, 14, 52, 54)：IoU = {iou(BOXES[0][0], BOXES[1][0]):.4f}")
    print(f"  (10, 10, 50, 50) 和 (60, 10, 100, 50)：IoU = {iou(BOXES[0][0], BOXES[2][0]):.4f}")

    print("\n第二步 NMS")
    print(f"  4 个框（分数 0.9、0.8、0.7、0.6），阈值 0.5：保留 {nms()}")

    print("\n第三步 PR 曲线与 AP")
    p, r, a = ap()
    print(f"  命中序列 {HITS}、真值 {N_GT} 个：精确率 {p}")
    print(f"  召回率 {r}；AP = {a:.4f}")

    print("\n第四步 匈牙利匹配 vs 贪心")
    ha, hc = hungarian()
    ga, gc = greedy()
    print(f"  代价矩阵 {COST.tolist()}：匈牙利 {ha}，总代价 {hc}；贪心 {ga}，总代价 {gc}")

    print("\n第五步 度量学习：三元组损失与 InfoNCE")
    for da, dn in ((0.3, 0.9), (0.5, 0.6), (0.8, 0.5)):
        print(f"  d(a, p) = {da}、d(a, n) = {dn}：三元组损失 {triplet(da, dn):.2f}")
    for tau in (1.0, 0.1, 0.05):
        print(f"  正样本相似度 0.8、负样本 [0.6, 0.3, 0.1, 0.0]，τ = {tau}：InfoNCE = {infonce(0.8, [0.6, 0.3, 0.1, 0.0], tau):.4f}")
    print(f"  NetVLAD：K = 64 个聚类中心、D = 512 维局部特征 → {64 * 512:,} 维全局描述子")


if __name__ == "__main__":
    main()
