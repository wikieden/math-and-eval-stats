"""讲义：第 40 章 张量的数学（notes/lessons/40-tensor-math.md）
运行：uv run python ch40_tensor_math.py

例子：一个形状 (2, 3, 4) 的 float32 张量；转置后的步长；广播规则；einsum 写注意力；矩阵乘和加偏置的 VJP。
里程碑 ME-M25 的“手算地址并和 as_strided 对数”“写一个 einsum 解释器”留给自己做。
"""

import numpy as np

T = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def elem_strides(a: np.ndarray) -> tuple[int, ...]:
    """NumPy 的 strides 是字节数，除以每个元素的字节数得到“元素步长”（PyTorch 的 stride() 就是这个）。"""
    return tuple(s // a.itemsize for s in a.strides)


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """行优先连续：最后一维步长 1，往前每一维乘上后面所有维的大小。"""
    out, acc = [], 1
    for n in reversed(shape):
        out.append(acc)
        acc *= n
    return tuple(reversed(out))


def offset(idx: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return sum(i * s for i, s in zip(idx, strides))


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...] | str:
    """右对齐；每一维要么相等，要么有一个是 1（或缺失）；否则报错。"""
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x != y and 1 not in (x, y):
            return f"不能广播：{x} 和 {y}"
        out.append(max(x, y))
    return tuple(out)


def attention_einsum(b: int = 2, h: int = 4, n: int = 8, d: int = 16, seed: int = 0) -> tuple[float, int]:
    """scores = einsum('bhqd,bhkd->bhqk')，out = einsum('bhqk,bhkd->bhqd')。返回 (和“转置 + matmul”写法的最大差, 两个 einsum 的乘加次数)。"""
    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((b, h, n, d)) for _ in range(3))
    s = np.einsum("bhqd,bhkd->bhqk", q, k)
    o = np.einsum("bhqk,bhkd->bhqd", s, v)
    ref = (q @ k.transpose(0, 1, 3, 2)) @ v
    return float(np.abs(o - ref).max()), 2 * b * h * n * n * d


def vjp_check(seed: int = 0) -> tuple[float, float, float]:
    """y = x W + b，L = Σ y·g（g 是上游梯度）。解析 VJP：∂L/∂x = g Wᵀ，∂L/∂W = xᵀ g，∂L/∂b = g 沿 batch 求和。
    和有限差分比，返回三个最大误差。"""
    rng = np.random.default_rng(seed)
    x, w, b, g = rng.standard_normal((5, 3)), rng.standard_normal((3, 2)), rng.standard_normal(2), rng.standard_normal((5, 2))
    loss = lambda x_, w_, b_: float(((x_ @ w_ + b_) * g).sum())  # noqa: E731

    def fd(f, a, h=1e-6):
        out = np.zeros_like(a)
        for i in np.ndindex(a.shape):
            e = np.zeros_like(a)
            e[i] = h
            out[i] = (f(a + e) - f(a - e)) / (2 * h)
        return out
    ex = np.abs(g @ w.T - fd(lambda a: loss(a, w, b), x)).max()
    ew = np.abs(x.T @ g - fd(lambda a: loss(x, a, b), w)).max()
    eb = np.abs(g.sum(0) - fd(lambda a: loss(x, w, a), b)).max()
    return float(ex), float(ew), float(eb)


def main() -> None:
    print("\n第一步 步长：多维下标 → 一维地址")
    print(f"  形状 {T.shape}：元素步长 {elem_strides(T)}（公式 {contiguous_strides(T.shape)}）")
    print(f"  T[1, 2, 3] 的偏移 = 1×12 + 2×4 + 3×1 = {offset((1, 2, 3), elem_strides(T))}，值 {T[1, 2, 3]:.0f}")

    print("\n第二步 视图与连续性：转置只改步长，不搬数据")
    tt = T.transpose(0, 2, 1)
    print(f"  T.transpose(0, 2, 1)：形状 {tt.shape}，步长 {elem_strides(tt)}，连续吗 {tt.flags['C_CONTIGUOUS']}，共享内存 {np.shares_memory(T, tt)}")
    r = tt.reshape(2, 12)
    print(f"  再 reshape(2, 12)：共享内存 {np.shares_memory(T, r)}（后两维步长 (1, 4) 没法合成一个步长，只能拷贝）")
    r2 = T.transpose(2, 0, 1).reshape(4, 6)
    print(f"  对比 T.transpose(2, 0, 1).reshape(4, 6)：共享内存 {np.shares_memory(T, r2)}（被合并的两维步长 (12, 4) 正好接得上）")

    print("\n第三步 广播：右对齐，1 可以拉伸（步长当 0）")
    for a, b in (((3, 1), (4,)), ((8, 1, 6, 1), (7, 1, 5)), ((2, 3), (3, 2))):
        print(f"  {a} 和 {b} → {broadcast_shape(a, b)}")
    e = np.broadcast_to(np.arange(3.0)[:, None], (3, 4))
    print(f"  把 (3, 1) 广播成 (3, 4)：步长 {elem_strides(e)}（第二维步长 0：同一个数被读 4 次）")

    print("\n第四步 einsum：用下标说清楚哪些维相乘、哪些维求和")
    err, flops = attention_einsum()
    print(f"  注意力两个 einsum 和 matmul 写法的最大差 {err:.1e}；乘加次数 2·B·H·N²·d = {flops:,}")

    print("\n第五步 计算图上的 VJP：y = xW + b")
    print(f"  解析 VJP 和有限差分的最大误差（x、W、b）：{[f'{v:.1e}' for v in vjp_check()]}")


if __name__ == "__main__":
    main()
