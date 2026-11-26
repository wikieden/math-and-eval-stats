"""讲义：第 54 章 点云与配准（notes/lessons/54-point-cloud.md）
运行：uv run python ch54_point_cloud.py

例子：KD 树找最近邻（和暴力比较次数）；法向量 = 局部协方差最小特征值对应的特征向量；Kabsch / Procrustes 的反射修正；
点到点与点到面的残差；ICP 的一次“找对应 → 求解”。
里程碑 ME-M33 的“Kabsch 求旋转、完整的点到点 ICP 循环”留给自己做；这里只演示公式和一步。
"""

import numpy as np
from scipy.spatial import cKDTree


def nn_compare(n: int = 20000, q: int = 200, seed: int = 0) -> tuple[float, float]:
    """暴力最近邻要算 n 次距离；KD 树平均只访问 O(log n) 个节点。返回 (最大距离差, 暴力的距离计算次数)。"""
    rng = np.random.default_rng(seed)
    pts, qs = rng.uniform(0, 10, (n, 3)), rng.uniform(0, 10, (q, 3))
    d_tree, _ = cKDTree(pts).query(qs)
    d_brute = np.sqrt(((qs[:, None, :] - pts[None]) ** 2).sum(-1)).min(1)
    return float(np.abs(d_tree - d_brute).max()), n


def normal_from_neighbors(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """平面 z = 0.2x + 0.1y 上 50 个点加小噪声：协方差最小特征值对应的特征向量就是法向量。返回 (估计, 真值)。"""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (50, 2))
    z = 0.2 * xy[:, 0] + 0.1 * xy[:, 1] + 0.005 * rng.standard_normal(50)
    p = np.c_[xy, z]
    w, v = np.linalg.eigh(np.cov((p - p.mean(0)).T))
    n = v[:, 0] * np.sign(v[2, 0])
    true = np.array([-0.2, -0.1, 1.0])
    return n, true / np.linalg.norm(true)


def kabsch(p: np.ndarray, q: np.ndarray, fix_reflection: bool = True) -> np.ndarray:
    """找 R 使 Σ‖R p_i − q_i‖² 最小（点已去中心）：H = Σ p qᵀ = UΣVᵀ，R = V diag(1, 1, det(VUᵀ)) Uᵀ。"""
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T)) if fix_reflection else 1.0
    return vt.T @ np.diag([1, 1, d]) @ u.T


def residuals(p: np.ndarray, q: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """点到点 ‖p − q‖ 与点到面 |(p − q)·n|：沿表面滑动的误差点到面不罚。"""
    return float(np.linalg.norm(p - q)), float(abs((p - q) @ n))


def icp_one_step(seed: int = 0) -> tuple[float, float, float]:
    """目标是一个随机点云，源 = 目标转 15°、平移 0.1。最近邻找对应 → 去中心 → Kabsch → 更新。返回 (初始 RMS, 一步后 RMS, 一步后的残余转角)。"""
    rng = np.random.default_rng(seed)
    tgt = rng.uniform(-1, 1, (300, 3))
    th = np.radians(15)
    r = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    src = tgt @ r.T + 0.1
    _, idx = cKDTree(tgt).query(src)
    q = tgt[idx]
    ps, qs = src - src.mean(0), q - q.mean(0)
    rr = kabsch(ps, qs)
    new = (src - src.mean(0)) @ rr.T + q.mean(0)
    _, idx2 = cKDTree(tgt).query(new)
    rest = rr @ r
    ang = np.degrees(np.arccos(np.clip((np.trace(rest) - 1) / 2, -1, 1)))
    return float(np.sqrt(((src - tgt) ** 2).sum(1).mean())), float(np.sqrt(((new - tgt) ** 2).sum(1).mean())), float(ang)


def main() -> None:
    print("\n第一步 最近邻与 KD 树")
    d, n = nn_compare()
    print(f"  2 万个点、200 次查询：KD 树和暴力结果最大差 {d:.1e}；暴力每次要算 {n} 个距离，KD 树平均只看几十个节点")

    print("\n第二步 法向量 = 局部协方差最小特征值的特征向量")
    nn_, true = normal_from_neighbors()
    print(f"  平面 z = 0.2x + 0.1y：估计 {np.round(nn_, 4).tolist()}，真值 {np.round(true, 4).tolist()}，夹角 {np.degrees(np.arccos(nn_ @ true)):.2f}°")

    print("\n第三步 Kabsch：SVD 求最优旋转，注意反射")
    p = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]) * 0.5
    q = p * np.array([1, 1, -1])            # 镜像：最优“正交矩阵”是反射
    r0, r1 = kabsch(p, q, False), kabsch(p, q, True)
    print(f"  目标是源的镜像：不修正 det = {np.linalg.det(r0):.0f}（反射，不是旋转）；修正后 det = {np.linalg.det(r1):.0f}")

    print("\n第四步 点到点 vs 点到面")
    for name, pp in (("沿表面错开 0.1", np.array([0.1, 0, 0])), ("离开表面 0.1", np.array([0, 0, 0.1]))):
        a, b = residuals(pp, np.zeros(3), np.array([0, 0, 1.0]))
        print(f"  {name}：点到点 {a:.3f}，点到面 {b:.3f}")

    print("\n第五步 ICP 的一步")
    r0_, r1_, ang = icp_one_step()
    print(f"  源 = 目标转 15°、平移 0.1：初始 RMS {r0_:.4f}；一次“最近邻对应 + Kabsch”后 {r1_:.4f}，还差 {ang:.2f}°")


if __name__ == "__main__":
    main()
