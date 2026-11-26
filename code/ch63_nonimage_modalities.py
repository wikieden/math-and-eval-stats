"""讲义：第 63 章 非图像模态的表示（notes/lessons/63-nonimage-modalities.md）
运行：uv run python ch63_nonimage_modalities.py

例子：PointNet 的对称函数（打乱顺序不变、关键点集合）；一个场景体素化后有多稀疏；
1 秒 16 kHz 音频的 STFT 帧数、频率分辨率和 Mel 刻度；触觉阵列的压力中心；本体感受的三种归一化遇到离群值。
"""

import math

import numpy as np


def pointnet_feature(pts: np.ndarray, seed: int = 0, d: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """共享的逐点映射 h（随机两层感知机）+ 逐维 max。返回 (全局特征, 每一维取到最大值的点的下标)。"""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=(pts.shape[1], 16)), rng.normal(size=(16, d))
    h = np.maximum(pts @ w1, 0) @ w2
    return h.max(0), h.argmax(0)


def cloud(seed: int = 1, n: int = 100) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (n, 2))


def critical_demo() -> dict[str, float]:
    pts = cloud()
    g, idx = pointnet_feature(pts)
    perm = np.random.default_rng(2).permutation(len(pts))
    g_perm, _ = pointnet_feature(pts[perm])
    crit = np.unique(idx)
    g_crit, _ = pointnet_feature(pts[crit])
    w = np.random.default_rng(0).normal(size=pts.size)  # 对照：把点按顺序拼成一个长向量再过线性层
    flat = lambda p: float(p.reshape(-1) @ w)  # noqa: E731
    return {
        "打乱后 max 特征变化": float(np.abs(g - g_perm).max()),
        "关键点个数": int(len(crit)),
        "只留关键点的特征变化": float(np.abs(g - g_crit).max()),
        "拼接写法打乱后变化": abs(flat(pts) - flat(pts[perm])) > 1e-6,
    }


def voxel_sparsity(seed: int = 0, voxel: float = 0.2) -> dict[str, float]:
    """100 m × 100 m × 8 m 的范围；地面上的 10 万个点 + 5 辆车（长方体表面）各 4000 点。"""
    rng = np.random.default_rng(seed)
    ground = np.c_[rng.uniform(-50, 50, (100_000, 2)), rng.normal(0, 0.03, 100_000)]
    cars = []
    for _ in range(5):
        c = rng.uniform(-40, 40, 2)
        p = rng.uniform([-2, -1, 0], [2, 1, 1.6], (4000, 3))
        k = rng.integers(0, 3, 4000)
        p[np.arange(4000), k] = np.array([[-2, -1, 0], [2, 1, 1.6]])[rng.integers(0, 2, 4000), k]
        cars.append(p + [c[0], c[1], 0])
    pts = np.vstack([ground, *cars])
    ijk = np.floor((pts - [-50, -50, -1]) / voxel).astype(int)
    occ = len(np.unique(ijk, axis=0))
    total = int(100 / voxel) ** 2 * int(8 / voxel)
    return {"点数": len(pts), "体素总数": total, "非空体素": occ, "占比": occ / total}


def stft_frames(sr: int = 16000, secs: float = 1.0, win: int = 400, hop: int = 160, center: bool = False) -> int:
    n = int(sr * secs)
    if center:
        return 1 + n // hop
    return 1 + (n - win) // hop


def hz_to_mel(f: float) -> float:
    return 2595 * math.log10(1 + f / 700)


def two_tone_peaks(sr: int = 16000, n_fft: int = 512) -> list[float]:
    """440 Hz + 1000 Hz 的一帧加汉宁窗做 FFT，取两个最大峰的频率。"""
    t = np.arange(n_fft) / sr
    x = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)
    mag = np.abs(np.fft.rfft(x * np.hanning(n_fft)))
    peaks = [i for i in range(1, len(mag) - 1) if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]]
    top = sorted(peaks, key=lambda i: -mag[i])[:2]
    return sorted(i * sr / n_fft for i in top)


TACTILE = np.zeros((4, 4))
TACTILE[1, 1], TACTILE[1, 2], TACTILE[2, 2] = 2.0, 1.0, 1.0   # 4×4 触觉阵列（每格的压力，单位 N）


def center_of_pressure(p=TACTILE, pitch: float = 0.005) -> tuple[float, float, float]:
    """总力 Σp、压力中心 (Σp·x/Σp, Σp·y/Σp)，格距 5 mm。"""
    ys, xs = np.mgrid[0 : p.shape[0], 0 : p.shape[1]] * pitch
    f = p.sum()
    return float(f), float((p * xs).sum() / f), float((p * ys).sum() / f)


JOINT = np.array([0.10, 0.12, 0.08, 0.11, 0.09, 0.10, 0.13, 0.07, 0.12, 2.90])  # 一个关节 10 帧的角度（rad），最后一帧是离群值


def normalize_all(x=JOINT) -> dict[str, list[float]]:
    z = (x - x.mean()) / x.std()
    mm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    lo, hi = np.quantile(x, [0.1, 0.9], method="lower")   # 数据多时用 1% / 99% 分位数；这里只有 10 帧
    q = np.clip(2 * (x - lo) / (hi - lo) - 1, -1, 1)
    return {"z-score": z.round(2).tolist(), "min-max": mm.round(2).tolist(), "分位数": q.round(2).tolist()}


def main() -> None:
    print("\n第一步 PointNet：对称函数")
    for k, v in critical_demo().items():
        print(f"  {k}：{v}")

    print("\n第二步 体素化有多稀疏（体素 0.2 m）")
    for k, v in voxel_sparsity().items():
        print(f"  {k}：{v:,.4g}" if isinstance(v, float) else f"  {k}：{v:,}")

    print("\n第三步 音频：STFT 与 Mel")
    print(f"  1 秒 16 kHz、窗 400（25 ms）、跳 160（10 ms）：{stft_frames()} 帧；center=True（torch.stft 默认）：{stft_frames(center=True)} 帧")
    print(f"  n_fft = 512：{512 // 2 + 1} 个频率格，每格 {16000 / 512} Hz")
    print(f"  Mel：1000 Hz → {hz_to_mel(1000):.1f}，2000 Hz → {hz_to_mel(2000):.1f}，4000 Hz → {hz_to_mel(4000):.1f}，8000 Hz → {hz_to_mel(8000):.1f}")
    print(f"  440 Hz + 1000 Hz 一帧的两个峰：{two_tone_peaks()} Hz")

    print("\n第四步 触觉阵列")
    f, cx, cy = center_of_pressure()
    print(f"  总力 {f} N，压力中心 ({cx * 1000:.2f} mm, {cy * 1000:.2f} mm)")

    print("\n第五步 本体感受的归一化（最后一帧是离群值 2.90）")
    for k, v in normalize_all().items():
        print(f"  {k}：{v}")


if __name__ == "__main__":
    main()
