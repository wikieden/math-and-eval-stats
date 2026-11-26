"""讲义：第 62 章 视觉-语言对齐的数学（notes/lessons/62-vl-alignment.md）
运行：uv run python ch62_vl_alignment.py

例子：温度 τ 决定 softmax 能有多“尖”（N 个候选时损失的下限）；InfoNCE 给出的互信息下界 log N − L 和它的上限 log N；
SigLIP 初始化 b = −10 的原因（每张图 N − 1 个负样本在初始时的损失）；零样本分类 = 余弦相似度的 argmax（含提示词集成）；
视觉 token 数账（ViT patch、Q-Former、Qwen2-VL 的 2×2 合并）。
里程碑 ME-M36 的“在小批量上实现 CLIP InfoNCE 与 SigLIP 损失，看温度的作用”留给自己做；这里不写整批的损失函数。
"""

import math

import numpy as np


def clip_floor(n: int, tau: float) -> float:
    """余弦相似度在 [−1, 1]：正样本 +1、其余 N − 1 个都是 −1 时，softmax 交叉熵最小也有 log(1 + (N−1)e^{−2/τ})。"""
    return math.log1p((n - 1) * math.exp(-2 / tau))


def mi_bound(n: int, loss: float) -> float:
    """InfoNCE 下界：I ≥ log N − L（单位 nats），所以最多只能证明 log N。"""
    return math.log(n) - loss


def siglip_init_loss(n: int, b: float, t: float = 10.0, cos: float = 0.0) -> tuple[float, float]:
    """一张图对一批 N 段文字：1 个正样本 −log σ(t·cos + b)，N − 1 个负样本 −log σ(−(t·cos + b))。
    初始化时嵌入随机，cos ≈ 0。返回 (正样本项, 负样本项之和)。"""
    z = t * cos + b
    pos = math.log1p(math.exp(-z))
    neg = (n - 1) * math.log1p(math.exp(z))
    return pos, neg


def normalize(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


IMG = normalize([0.9, 0.3, 0.1])                       # 一张猫的图
PROMPTS = {                                            # 每类两个提示词的文本嵌入
    "猫": normalize([[0.5, 0.8, 0.3], [0.95, 0.0, 0.3]]),
    "狗": normalize([[0.75, 0.6, 0.1], [0.2, 0.9, 0.3]]),
    "车": normalize([[0.1, 0.2, 0.9], [0.0, 0.4, 0.9]]),
}


def zero_shot(img=IMG, prompts=PROMPTS, ensemble: bool = True) -> dict[str, float]:
    """每类的文本嵌入（集成 = 先平均再归一化；不集成 = 只用第一个提示词）和图像嵌入的余弦相似度。"""
    out = {}
    for c, e in prompts.items():
        t = normalize(e.mean(0)) if ensemble else e[0]
        out[c] = float(img @ t)
    return out


def vit_tokens(side: int, patch: int) -> int:
    return (side // patch) ** 2


def qwen2vl_tokens(h: int, w: int) -> int:
    """patch 14、再 2×2 合并 → 每个 token 对应 28×28 像素。"""
    return (h // 28) * (w // 28)


def main() -> None:
    print("\n第一步 温度：softmax 能有多尖")
    for tau in (1.0, 0.07, 0.01):
        print(f"  N = 32,768、τ = {tau}：损失下限 {clip_floor(32768, tau):.3g}（logit 范围 ±{1 / tau:.1f}）")

    print("\n第二步 InfoNCE 与互信息下界")
    for n, loss in ((256, 2.0), (32768, 2.0), (32768, 0.0)):
        print(f"  N = {n:,}、损失 {loss}：I ≥ {mi_bound(n, loss):.2f} nats（上限 log N = {math.log(n):.2f}）")

    print("\n第三步 SigLIP 的偏置初始化（t = 10，初始 cos ≈ 0）")
    for b in (0.0, -10.0):
        pos, neg = siglip_init_loss(16384, b)
        print(f"  N = 16,384、b = {b}：正样本项 {pos:.3f}，负样本项合计 {neg:.3f}")

    print("\n第四步 零样本分类 = 余弦相似度")
    for ens in (False, True):
        s = zero_shot(ensemble=ens)
        print(f"  {'提示词集成' if ens else '只用一个提示词'}：" + "，".join(f"{c} {v:.4f}" for c, v in s.items()) + f" → {max(s, key=s.get)}")

    print("\n第五步 视觉 token 数账")
    print(f"  ViT-L/14 @ 224：{vit_tokens(224, 14)} 个；@ 336（LLaVA-1.5）：{vit_tokens(336, 14)} 个")
    print("  BLIP-2 Q-Former：固定 32 个 query")
    print(f"  Qwen2-VL 1344×896 的图：{qwen2vl_tokens(896, 1344)} 个；8 帧 448×448 的视频（两帧再合一）：{qwen2vl_tokens(448, 448) * 8 // 2} 个")


if __name__ == "__main__":
    main()
