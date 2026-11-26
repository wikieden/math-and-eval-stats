"""讲义：第 58 章 路径规划（notes/lessons/58-path-planning.md）
运行：uv run python ch58_path_planning.py

例子：一张 7 个节点的小路网（带坐标）；Dijkstra 与 A* 的扩展顺序；把启发函数放大 3 倍（不可采纳）会得到更差的路径；
构型空间里“机器人变成点、障碍物膨胀”；RRT* 的重连半径；势场法的局部极小。
里程碑 ME-M34 的“在栅格上跑 Dijkstra / A* 比较扩展节点数、再跑 RRT / RRT*”留给自己做。
"""

import heapq

import numpy as np

XY = {"S": (0, 0), "A": (2, 1), "B": (1, -2), "C": (4, 2), "D": (3, -1), "E": (5, -2), "G": (6, 0)}
EDGES = [("S", "A", 2.3), ("S", "B", 2.3), ("A", "C", 2.3), ("A", "D", 2.3), ("B", "D", 2.3), ("B", "E", 4.0),
         ("C", "G", 3.6), ("D", "G", 3.2), ("E", "G", 2.3), ("D", "E", 2.3)]


def graph():
    g = {k: [] for k in XY}
    for a, b, w in EDGES:
        g[a].append((b, w))
        g[b].append((a, w))
    return g


def search(h_scale: float = 0.0) -> tuple[list[str], float, list[str]]:
    """h_scale = 0：Dijkstra；1：A*（欧氏距离，可采纳）；> 1：加权 A*（不可采纳）。返回 (路径, 代价, 扩展顺序)。"""
    g = graph()
    h = {k: h_scale * float(np.hypot(XY[k][0] - XY["G"][0], XY[k][1] - XY["G"][1])) for k in XY}
    pq, best, parent, closed, order = [(h["S"], 0.0, "S")], {"S": 0.0}, {"S": None}, set(), []
    while pq:
        _, c, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        order.append(u)
        if u == "G":
            break
        for v, w in g[u]:
            if v not in closed and c + w < best.get(v, np.inf):
                best[v], parent[v] = c + w, u
                heapq.heappush(pq, (c + w + h[v], c + w, v))
    path, n = [], "G"
    while n:
        path.append(n)
        n = parent[n]
    return path[::-1], best["G"], order


def cspace_area(obstacle_r: float, robot_r: float) -> tuple[float, float]:
    """圆形障碍物、圆形机器人：在构型空间里机器人是点，障碍物半径膨胀成 r_obs + r_robot（Minkowski 和）。返回 (原面积, 膨胀后面积)。"""
    return float(np.pi * obstacle_r ** 2), float(np.pi * (obstacle_r + robot_r) ** 2)


def rrt_star_radius(n: int, d: int = 2, gamma: float = 6.0) -> float:
    """RRT* 的重连半径 r(n) = γ (log n / n)^{1/d}（Karaman & Frazzoli）：随节点数缓慢缩小，保证渐近最优又不至于每次连太多。"""
    return gamma * (np.log(n) / n) ** (1 / d)


def potential_1d() -> float:
    """一维势场：目标在 x = 10，吸引势 ½(x − 10)²；障碍物在 x = 5，斥力势 k/(x − 5)²（k = 20，只在 x < 5 一侧）。
    从左边出发沿负梯度走，停在吸引力和斥力平衡的地方。返回停下的位置。"""
    x = 0.0
    for _ in range(20000):
        grad = (x - 10) - 2 * 20 / (x - 5) ** 3 * (-1) * (-1)
        x -= 0.001 * grad
    return x


def main() -> None:
    print("\n第一步 Dijkstra：按已走代价从小到大扩展")
    p, c, o = search(0)
    print(f"  路径 {'→'.join(p)}，代价 {c:.1f}；扩展顺序 {o}（{len(o)} 个）")

    print("\n第二步 A*：加上到终点的直线距离（可采纳）")
    p, c, o = search(1)
    print(f"  路径 {'→'.join(p)}，代价 {c:.1f}；扩展顺序 {o}（{len(o)} 个）")
    p, c, o = search(3)
    print(f"  启发 × 3（不可采纳）：路径 {'→'.join(p)}，代价 {c:.1f}（更差）；扩展 {len(o)} 个")

    print("\n第三步 构型空间：机器人缩成点，障碍物膨胀")
    a0, a1 = cspace_area(0.5, 0.3)
    print(f"  半径 0.5 的障碍物、半径 0.3 的机器人：障碍物面积 {a0:.3f} → {a1:.3f}（{a1 / a0:.2f} 倍）")

    print("\n第四步 RRT* 的重连半径")
    for n in (100, 1000, 10000):
        print(f"  n = {n}：r = {rrt_star_radius(n):.3f}")

    print("\n第五步 势场法：会卡在局部极小")
    print(f"  目标 x = 10，障碍物 x = 5：从 x = 0 出发，停在 x = {potential_1d():.3f}（到不了目标）")


if __name__ == "__main__":
    main()
