"""讲义：第 47 章 推理成本模型（notes/lessons/47-inference-cost.md）
运行：uv run python ch47_inference_cost.py

例子：一个 7B 级的模型（32 层、hidden 4096、32 个注意力头、每头 128 维，参数约 6.7B，bf16）跑在一张 H100 SXM 上
（bf16 稠密算力约 989 TFLOP/s、显存带宽约 3.35 TB/s）。
里程碑 ME-M29 的“KV cache 计算器（MHA / GQA / MLA）”和“用 roofline 判断任意配置受什么限制”留给自己做。
"""

PEAK, BW = 989e12, 3.35e12        # FLOP/s、byte/s
LAYERS, HIDDEN, HEADS, HDIM, PARAMS = 32, 4096, 32, 128, 6.7e9


def ridge() -> float:
    """roofline 的拐点：算术强度（FLOP/byte）超过它才是算力受限。"""
    return PEAK / BW


def matmul_intensity(m: int, k: int, n: int, bytes_per: int = 2) -> float:
    """(m×k)·(k×n)：2mkn 次浮点运算，至少读写 (mk + kn + mn) 个元素。"""
    return 2 * m * k * n / (bytes_per * (m * k + k * n + m * n))


def attainable(intensity: float) -> float:
    return min(PEAK, intensity * BW)


def kv_bytes_per_token(kv_heads: int, layers: int = LAYERS, hdim: int = HDIM, bytes_per: int = 2) -> int:
    """每个 token 的 KV cache：K 和 V 各一份，每层 kv_heads × head_dim 个数。"""
    return 2 * layers * kv_heads * hdim * bytes_per


def decode_floor_ms(batch: int, ctx: int, kv_heads: int = HEADS) -> float:
    """decode 一步的下限（带宽受限）：把权重读一遍 + 把这批请求的 KV cache 读一遍，除以带宽。"""
    return (2 * PARAMS + batch * ctx * kv_bytes_per_token(kv_heads)) / BW * 1e3


def main() -> None:
    print("\n第一步 roofline：算术强度决定受算力还是带宽限制")
    print(f"  H100 SXM 拐点 {ridge():.0f} FLOP/byte")
    for tok in (1, 16, 256, 2048):
        i = matmul_intensity(tok, HIDDEN, HIDDEN)
        print(f"  一次处理 {tok} 个 token 过 4096×4096 的线性层：强度 {i:.1f}，可达 {attainable(i) / 1e12:.0f} TFLOP/s"
              f"（{'算力' if i > ridge() else '带宽'}受限）")

    print("\n第二步 prefill vs decode")
    print(f"  prefill 2048 个 token：一次矩阵乘处理整段，强度约 {matmul_intensity(2048, HIDDEN, HIDDEN):.0f}")
    print(f"  decode：每步每个请求只有 1 个新 token，batch = 1 时强度约 {matmul_intensity(1, HIDDEN, HIDDEN):.2f}")

    print("\n第三步 KV cache 大小")
    for name, kvh in (("MHA（32 个 KV 头）", 32), ("GQA（8 个 KV 头）", 8)):
        per = kv_bytes_per_token(kvh)
        print(f"  {name}：每 token {per / 2**10:.0f} KiB；4096 token 一个请求 {per * 4096 / 2**30:.2f} GiB")

    print("\n第四步 decode 一步的时间下限：读权重 + 读 KV")
    for b in (1, 8, 64):
        print(f"  batch {b}、上下文 4096、MHA：{decode_floor_ms(b, 4096):.1f} ms/步，吞吐上限 {b / decode_floor_ms(b, 4096) * 1e3:.0f} token/s")
    print(f"  同样 batch 64 换成 GQA：{decode_floor_ms(64, 4096, 8):.1f} ms/步")


if __name__ == "__main__":
    main()
