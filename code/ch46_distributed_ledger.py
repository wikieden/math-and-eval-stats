"""讲义：第 46 章 分布式训练账本（notes/lessons/46-distributed-ledger.md）
运行：uv run python ch46_distributed_ledger.py

例子：混合精度 Adam 的“每个参数 16 字节”；ZeRO 论文图 1 的 7.5B 模型、64 卡；Transformer 一层的激活显存（Korthikanti 2022）；
ring all-reduce 的通信量；流水线气泡。
里程碑 ME-M28 的“显存与通信量计算器（参数、优化器状态、ZeRO 1/2/3、TP / PP），和 Playbook 表对数”留给自己做。
"""


def model_states(psi: float, nd: int, stage: int) -> float:
    """ZeRO 论文 §3：混合精度 Adam，参数 fp16 2Ψ + 梯度 fp16 2Ψ + 优化器状态 12Ψ（fp32 主权重、m、v）= 16Ψ 字节。
    stage 1 分片优化器状态，2 再分片梯度，3 再分片参数。返回每卡字节数。"""
    p, g, o = 2 * psi, 2 * psi, 12 * psi
    if stage >= 1:
        o /= nd
    if stage >= 2:
        g /= nd
    if stage >= 3:
        p /= nd
    return p + g + o


def activation_bytes(s: int, b: int, h: int, a: int, tp: int = 1) -> float:
    """一层 Transformer 的激活显存（不重算、不做序列并行）：s·b·h·(34 + 5·a·s/h) 字节（Korthikanti 2022 式 (2)）；
    张量并行 t 时大部分除以 t：s·b·h·(10 + 24/t + 5·a·s/(h·t))。"""
    if tp == 1:
        return s * b * h * (34 + 5 * a * s / h)
    return s * b * h * (10 + 24 / tp + 5 * a * s / (h * tp))


def ring_allreduce_bytes(m: float, n: int) -> float:
    """ring all-reduce：reduce-scatter + all-gather，每张卡发送 2(n − 1)/n·M 字节。"""
    return 2 * (n - 1) / n * m


def bubble(p: int, m: int) -> float:
    """GPipe / 1F1B 的流水线气泡比例 (p − 1)/(m + p − 1)，p 是流水线级数，m 是微批数。"""
    return (p - 1) / (m + p - 1)


def main() -> None:
    gb = 1e9
    print("\n第一步 模型状态：每个参数 16 字节")
    print(f"  7.5B 参数：{model_states(7.5e9, 1, 0) / gb:.0f} GB（还没算激活）")

    print("\n第二步 ZeRO 分片（64 卡）")
    for st, name in ((0, "数据并行"), (1, "ZeRO-1（优化器状态）"), (2, "ZeRO-2（+ 梯度）"), (3, "ZeRO-3（+ 参数）")):
        print(f"  {name}：每卡 {model_states(7.5e9, 64, st) / gb:.1f} GB")

    print("\n第三步 激活：随序列长度平方增长的那一项")
    for s in (2048, 8192):
        print(f"  s = {s}、b = 1、h = 4096、32 个头：一层 {activation_bytes(s, 1, 4096, 32) / gb:.2f} GB；"
              f"TP = 8 时 {activation_bytes(s, 1, 4096, 32, 8) / gb:.2f} GB")

    print("\n第四步 通信：ring all-reduce")
    for n in (2, 8, 64):
        print(f"  {n} 卡同步 14 GB 梯度（7B、bf16）：每卡发送 {ring_allreduce_bytes(14 * gb, n) / gb:.2f} GB")

    print("\n第五步 流水线气泡")
    for p, m in ((4, 4), (4, 16), (8, 64)):
        print(f"  {p} 级、{m} 个微批：气泡 {bubble(p, m):.1%}")


if __name__ == "__main__":
    main()
