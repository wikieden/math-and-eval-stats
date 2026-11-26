"""讲义：第 59 章 神经场景表示（notes/lessons/59-neural-rendering.md）
运行：uv run python ch59_neural_rendering.py

例子：一条射线穿过一块密度均匀的“雾”（σ = 2，厚 0.5，颜色 0.8），后面是颜色 0.2 的不透明墙：体渲染积分的闭式解和离散求和；
α 合成的等价写法；NeRF 的位置编码维数和“近的点编码后不再近”；3DGS 里一个各向同性高斯投影到图像的大小；球谐系数个数。
里程碑 ME-M35 的“通用的单射线离散体渲染”“任意 3D 高斯协方差的投影”留给自己做。
"""

import numpy as np

SIGMA, A, B, C_FOG, C_WALL = 2.0, 1.0, 1.5, 0.8, 0.2


def closed_form() -> float:
    """C = ∫ T(t) σ c dt：雾这段贡献 c_fog·(1 − e^{−σ(b−a)})，穿过雾的透射率 e^{−σ(b−a)} 乘墙的颜色。"""
    t = np.exp(-SIGMA * (B - A))
    return C_FOG * (1 - t) + t * C_WALL


def discrete(n: int) -> float:
    """把雾切成 n 段，每段 α_i = 1 − e^{−σδ}，T_i = Π_{j<i}(1 − α_j)，C = Σ T_i α_i c_i + T_end·c_wall。"""
    delta = (B - A) / n
    alpha = 1 - np.exp(-SIGMA * delta)
    t, c = 1.0, 0.0
    for _ in range(n):
        c += t * alpha * C_FOG
        t *= 1 - alpha
    return c + t * C_WALL


def naive_riemann(n: int) -> float:
    """如果直接用 σδ 当权重（不用 1 − e^{−σδ}），段数少时会算错。"""
    delta = (B - A) / n
    t, c = 1.0, 0.0
    for _ in range(n):
        c += t * SIGMA * delta * C_FOG
        t *= np.exp(-SIGMA * delta)
    return c + t * C_WALL


def posenc(p: float, levels: int = 10) -> np.ndarray:
    """NeRF 的位置编码 γ(p) = (sin 2⁰πp, cos 2⁰πp, …, sin 2^{L−1}πp, cos 2^{L−1}πp)。"""
    f = 2.0 ** np.arange(levels) * np.pi
    return np.r_[np.sin(f * p), np.cos(f * p)]


def gaussian_2d_std(sigma3d: float, z: float, f: float = 500.0) -> float:
    """光轴上深度 z 处、各向同性（标准差 σ）的 3D 高斯：透视投影的雅可比在光轴上是 diag(f/z, f/z)，所以 2D 标准差 = fσ/z 像素。"""
    return f * sigma3d / z


def sh_coeffs(degree: int) -> int:
    """球谐到 l 阶共 (l + 1)² 个基函数；每个颜色通道一组。"""
    return (degree + 1) ** 2


def main() -> None:
    print("\n第一步 体渲染积分：透射率 × 密度 × 颜色")
    print(f"  雾 σ = 2、厚 0.5、颜色 0.8，后面墙颜色 0.2：闭式解 C = {closed_form():.6f}；穿过雾的透射率 {np.exp(-SIGMA * (B - A)):.4f}")

    print("\n第二步 离散化：α = 1 − e^{−σδ} 才对")
    for n in (1, 4, 64):
        print(f"  切 {n} 段：α 合成 {discrete(n):.6f}；直接用 σδ 当权重 {naive_riemann(n):.6f}")

    print("\n第三步 位置编码：让网络能表示高频细节")
    a, b = posenc(0.50), posenc(0.51)
    print(f"  L = 10：编码后 {len(a)} 维（三个坐标共 {3 * len(a)} 维）")
    print(f"  0.50 和 0.51 相差 0.01；编码后相差 {np.linalg.norm(a - b):.3f}（最高频那对每 1/512 就转一圈）")

    print("\n第四步 3D 高斯投影到图像")
    for z in (2.0, 5.0, 10.0):
        print(f"  标准差 0.05 m 的高斯在 {z} m 处：图像上标准差 {gaussian_2d_std(0.05, z):.1f} 像素")

    print("\n第五步 球谐颜色")
    for l in (0, 1, 3):
        print(f"  {l} 阶：{sh_coeffs(l)} 个系数 / 通道，RGB 共 {3 * sh_coeffs(l)} 个")


if __name__ == "__main__":
    main()
