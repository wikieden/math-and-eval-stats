# 数学与实验统计

**在线阅读：<https://wikieden.github.io/math-and-eval-stats/>** · [讲义目录](https://wikieden.github.io/math-and-eval-stats/lessons/index.html) · [全部动画](https://wikieden.github.io/math-and-eval-stats/animations/index.html) · [复习卡片](https://wikieden.github.io/math-and-eval-stats/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)

> 66 章（全部写完）· 66 个动画（GIF + 可交互页面）· 144 张静态图 · 215 张复习卡片 · 评测统计在前（第 1–7 章），Physical AI 的数学底座在后（第 8–22 章：概率与信息、线代与微积分、优化、采样与生成、几何估计与动力系统）；深入篇在第 23–34 章（RL / 控制 / 扩散推导与严格理论），应用篇在第 35–66 章（CNN / Transformer / PyTorch、现代 LLM 训练与推理、导航与感知、多模态与 VLN）
> 在线阅读：<https://wikieden.github.io/math-and-eval-stats/>（66 章全部写完）

## 这本书讲什么

从"30 次试验成功 21 次"这一个数字讲起：它有多准、两个这样的数怎么比、要测多少次才分得出高下、配对和多任务时怎么算、怎么定 A/B 判定规则，最后落到批判地读 loss 曲线和消融表。后 15 章从 Physical AI 的算法倒推数学：VLA 和动作头要的概率、高斯、KL 与生成模型数学，训练要的线代、自动微分与优化，运控要的 QP 与 Riccati，感知要的旋转、李群与 Kalman，扩散采样要的 ODE 积分器。再往下两篇：深入篇（第 23–34 章）把 RL、LQR / MPC、扩散与 flow 的推导写完整，补测度论、泛函、流形、随机微积分、最优传输、信息几何；应用篇（第 35–66 章）讲 CNN、Transformer、PyTorch 张量背后的数学，现代 LLM 的 scaling law、μP、Muon / SOAP、MoE、分布式账本、KV cache、FlashAttention、量化、投机解码、DPO / GRPO，以及导航感知（多视几何、SLAM、VIO、规划、NeRF / 3DGS）、多模态融合和 VLN。哪个算法用到哪块数学，见 Physical AI 数学需求地图。

- **适合谁**：要审实验、做评测、拍板"新模型能不能上"的人。只需要高中数学；缺的概率知识在第 1 章和第 8 章现场补。
- **读完能做什么**：当场回答 算法总监路线 · 达标自测 第 1 题（21/30 对 18/30，为什么下不了结论、要测多少次）；给团队的成功率表配上正确的区间；审一张消融表；做 学习计划 的里程碑 ME-M1–M36；读 π0 的 flow 动作头、PPO 的重要性比率、ESKF 的四元数扰动、凸 MPC 的 QP 时，知道每一步数学在干什么。
- **怎么读**：按章顺序读；只想学 VLA 或运动控制的，按 [章节索引](https://wikieden.github.io/math-and-eval-stats/lessons/appendix-reading-paths.html) 挑章读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍"讲。
- **能用图讲的都用图讲**：数量不设上限。关键机制做成可交互页面（▶ 播放、→ 单步、↺ 重来，有参数的配滑块），讲义里嵌 GIF；数字之间的关系画成图表，结构和流程画成静态图。
- **对照原文**：讲到原文的地方标本地文本行号（如 NIST §7.2.4.1 L22–L27），引用原句。
- **保留真实的学习过程**：练习与作答记录里是学习时的原话、对错和点评。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释，全书缩写见 [缩写与术语表](https://wikieden.github.io/math-and-eval-stats/lessons/appendix-glossary.html)。
- **每章都有代码对照**：讲义里的数字用 Python（NumPy、SciPy）算一遍，配测试核对；只算讲过的例子，不提前给练习答案。
- **每章收尾有口诀**：全部口诀汇总在本页末尾。

## 示例动画

| 动画 | 章 | 演什么 |
|---|---|---|
| ![置信区间覆盖率](https://wikieden.github.io/math-and-eval-stats/animations/01-ci-coverage.gif) | [第 1 章 一个成功率有多准](https://wikieden.github.io/math-and-eval-stats/lessons/01-success-rate-ci.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/01-ci-coverage.html) | 重复做"测 n 次"的实验，数区间盖住真值的比例：Wald 名义 95%、实际约 81%；Wilson 约 97% |
| ![零假设下的差值](https://wikieden.github.io/math-and-eval-stats/animations/02-null-distribution.gif) | [第 2 章 两个成功率比一比](https://wikieden.github.io/math-and-eval-stats/lessons/02-two-proportions.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/02-null-distribution.html) | 在"两者一样"的世界里重做 A/B 实验，红色尾巴的比例就是 p 值 |
| ![似然与后验](https://wikieden.github.io/math-and-eval-stats/animations/08-likelihood-posterior.gif) | [第 8 章 概率补课](https://wikieden.github.io/math-and-eval-stats/lessons/08-probability-mle-map.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/08-likelihood-posterior.html) | 逐个加试验结果，看 MLE 和 MAP 怎么移动、先验什么时候被压过 |
| ![二维高斯与条件分布](https://wikieden.github.io/math-and-eval-stats/animations/09-gaussian-conditional.gif) | [第 9 章 高斯分布](https://wikieden.github.io/math-and-eval-stats/lessons/09-gaussian.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/09-gaussian-conditional.html) | 协方差椭圆；已知 x₂ 后 x₁ 的分布变窄 |
| ![两连杆雅可比](https://wikieden.github.io/math-and-eval-stats/animations/13-jacobian-2link.gif) | [第 13 章 微积分与自动微分](https://wikieden.github.io/math-and-eval-stats/lessons/13-calculus-autodiff.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/13-jacobian-2link.html) | 雅可比两列与速度椭圆，手臂伸直时压成一条线 |
| ![欧拉角与万向锁](https://wikieden.github.io/math-and-eval-stats/animations/18-euler-gimbal-lock.gif) | [第 18 章 3D 旋转](https://wikieden.github.io/math-and-eval-stats/lessons/18-rotations-rigid-transforms.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/18-euler-gimbal-lock.html) | pitch 到 90° 后 yaw、roll 同时转，方块不动 |
| ![REINFORCE 与基线](https://wikieden.github.io/math-and-eval-stats/animations/24-reinforce-baseline.gif) | [第 24 章 策略梯度推导](https://wikieden.github.io/math-and-eval-stats/lessons/24-policy-gradient.html) · [▶ 交互](https://wikieden.github.io/math-and-eval-stats/animations/24-reinforce-baseline.html) | 同一串随机数，减基线后梯度估计更集中、学得更稳 |

## 目录

| 章 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|
| 1 | [一个成功率有多准](https://wikieden.github.io/math-and-eval-stats/lessons/01-success-rate-ci.html) | p̂ 是估计；SE = √(p̂(1−p̂)/n)；小样本靠边用 Wilson；半宽 ∝ 1/√n；95% 是方法的覆盖率 | [覆盖率](https://wikieden.github.io/math-and-eval-stats/animations/01-ci-coverage.gif) | ST01–ST05 | ✅ 讲解（练习待作答） |
| 2 | [两个成功率比一比](https://wikieden.github.io/math-and-eval-stats/lessons/02-two-proportions.html) | 差值 SE 是两份方差相加；p 值 = 零假设下这么极端的概率；小样本用 Fisher；区间重叠 ≠ 不显著 | [零假设下的差值](https://wikieden.github.io/math-and-eval-stats/animations/02-null-distribution.gif) | ST06–ST10 | ✅ 讲解（练习待作答） |
| 3 | [要测多少次](https://wikieden.github.io/math-and-eval-stats/lessons/03-power-sample-size.html) | 一类 / 二类错误、功效 80%、MDE；样本量公式；"10 个点要几百次" | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/03-power-sample-size.gif) | ST31–ST33 | ✅ 讲解（练习待作答） |
| 4 | [配对比较](https://wikieden.github.io/math-and-eval-stats/lessons/04-paired-comparison.html) | 同一初始条件 A/B 交替跑；McNemar 只看不一致的对；配对降方差 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/04-paired-comparison.gif) | ST34–ST35 | ✅ 讲解（练习待作答） |
| 5 | [Bootstrap 与 Beta 后验](https://wikieden.github.io/math-and-eval-stats/lessons/05-bootstrap-bayes.html) | 重抽样估区间；Beta(1+k, 1+n−k)；P(B > A) | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/05-bootstrap-bayes.gif) | ST36–ST38 | ✅ 讲解（练习待作答） |
| 6 | [A/B 判定规则与多重比较](https://wikieden.github.io/math-and-eval-stats/lessons/06-ab-rules-multiple-comparisons.html) | 先写判定规则再测；偷看提前停；Bonferroni / Holm；多任务多种子用 IQM + 分层 bootstrap | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/06-ab-rules-multiple-comparisons.gif) | ST39–ST42 | ✅ 讲解（练习待作答） |
| 7 | [批判地读 loss 曲线和消融表](https://wikieden.github.io/math-and-eval-stats/lessons/07-read-curves-and-ablations.html) | 种子方差、平滑、过拟合、评测漂移；审稿追问清单 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/07-read-curves-and-ablations.gif) | ST43–ST45 | ✅ 讲解（练习待作答） |
| 8 | [概率补课：期望方差、贝叶斯、MLE 与 MAP](https://wikieden.github.io/math-and-eval-stats/lessons/08-probability-mle-map.html) | 平均的方差除以 n；先验低时检测器误报多；伯努利 MLE = k/n；MAP 往中间拉；行为克隆 = NLL | [似然与后验](https://wikieden.github.io/math-and-eval-stats/animations/08-likelihood-posterior.gif) | ST11–ST14 | ✅ 讲解（练习待作答） |
| 9 | [高斯分布](https://wikieden.github.io/math-and-eval-stats/lessons/09-gaussian.html) | 协方差 = 椭圆（特征分解）；线性变换还是高斯；条件方差只减不增；高斯头 log 概率；单峰平均撞墙 | [二维高斯与条件分布](https://wikieden.github.io/math-and-eval-stats/animations/09-gaussian-conditional.gif) | ST15–ST18 | ✅ 讲解（练习待作答） |
| 10 | [信息论：熵、交叉熵、KL](https://wikieden.github.io/math-and-eval-stats/lessons/10-information-kl.html) | 交叉熵 = 负对数似然；KL ≥ 0 且不对称，正向 / 反向 KL 各偏向什么；互信息 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/10-information-kl.gif) | ST46–ST48 | ✅ 讲解（练习待作答） |
| 11 | [线代一：矩阵乘法、秩、特征分解](https://wikieden.github.io/math-and-eval-stats/lessons/11-linear-algebra-basics.html) | 矩阵乘法三种看法；秩；特征分解；正定与二次型（LQR 代价、Hessian） | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/11-linear-algebra-basics.gif) | ST49–ST51 | ✅ 讲解（练习待作答） |
| 12 | [线代二：SVD、低秩、最小二乘](https://wikieden.github.io/math-and-eval-stats/lessons/12-svd-least-squares.html) | SVD 与最优低秩近似、LoRA 参数量；最小二乘 = 正交投影；伪逆、条件数 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/12-svd-least-squares.gif) | ST52–ST55 | ✅ 讲解（练习待作答） |
| 13 | [多元微积分与自动微分](https://wikieden.github.io/math-and-eval-stats/lessons/13-calculus-autodiff.html) | 梯度、雅可比（det J = sin θ₂）；链式法则 = 雅可比相乘；反向模式算 VJP；中心差分核对 | [两连杆雅可比](https://wikieden.github.io/math-and-eval-stats/animations/13-jacobian-2link.gif) | ST19–ST22 | ✅ 讲解（练习待作答） |
| 14 | [无约束优化](https://wikieden.github.io/math-and-eval-stats/lessons/14-unconstrained-optimization.html) | 步长与收敛、动量、Adam / AdamW、warmup + cosine、梯度裁剪；牛顿、高斯-牛顿、LM | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/14-unconstrained-optimization.gif) | ST56–ST59 | ✅ 讲解（练习待作答） |
| 15 | [约束与凸优化](https://wikieden.github.io/math-and-eval-stats/lessons/15-constrained-convex-qp.html) | 拉格朗日、KKT、凸性判断、QP 标准形；MPC、全身控制、带限位 IK 写成 QP | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/15-constrained-convex-qp.gif) | ST60–ST62 | ✅ 讲解（练习待作答） |
| 16 | [蒙特卡洛与采样](https://wikieden.github.io/math-and-eval-stats/lessons/16-monte-carlo-sampling.html) | MC 误差 ∝ 1/√N；重要性采样（PPO 比率、off-policy）；基线降方差；CEM、MPPI 加权 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/16-monte-carlo-sampling.gif) | ST63–ST65 | ✅ 讲解（练习待作答） |
| 17 | [生成模型的数学](https://wikieden.github.io/math-and-eval-stats/lessons/17-generative-math.html) | 潜变量与 ELBO、重参数化、score、SDE / ODE 视角、连续性方程、flow matching 目标 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/17-generative-math.gif) | ST66–ST68 | ✅ 讲解（练习待作答） |
| 18 | [3D 旋转与刚体变换](https://wikieden.github.io/math-and-eval-stats/lessons/18-rotations-rigid-transforms.html) | 旋转矩阵四性质；万向锁；Rodrigues；四元数半角与双覆盖；网络输出用 6D；SE(3) 求逆 | [欧拉角与万向锁](https://wikieden.github.io/math-and-eval-stats/animations/18-euler-gimbal-lock.gif) | ST23–ST26 | ✅ 讲解（练习待作答） |
| 19 | [李群入门](https://wikieden.github.io/math-and-eval-stats/lessons/19-lie-groups.html) | SO(3) / SE(3) 的 exp / log、切空间扰动、左右雅可比、SLERP 与旋转平均 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/19-lie-groups.gif) | ST69–ST71 | ✅ 讲解（练习待作答） |
| 20 | [状态估计](https://wikieden.github.io/math-and-eval-stats/lessons/20-state-estimation.html) | 预测 + 更新两步；Kalman 增益；EKF / ESKF；UKF 与粒子滤波各适合什么；因子图 = 非线性最小二乘 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/20-state-estimation.gif) | ST72–ST75 | ✅ 讲解（练习待作答） |
| 21 | [动力系统与控制的数学](https://wikieden.github.io/math-and-eval-stats/lessons/21-dynamics-control.html) | 状态空间、线性化、ZOH 离散化、特征值判稳定、Riccati 与 LQR、Bellman 是不动点方程 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/21-dynamics-control.gif) | ST76–ST78 | ✅ 讲解（练习待作答） |
| 22 | [数值计算](https://wikieden.github.io/math-and-eval-stats/lessons/22-numerics.html) | Euler / Heun / RK4 与扩散采样步数；logsumexp；float16 / bfloat16；病态问题 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/22-numerics.gif) | ST79–ST82 | ✅ 讲解（练习待作答） |
| 23 | [MDP 与 Bellman 的完整推导](https://wikieden.github.io/math-and-eval-stats/lessons/23-mdp-bellman.html) | 值函数、Bellman 期望 / 最优方程、压缩映射与值迭代收敛 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/23-mdp-bellman.gif) | ST83–ST85 | ✅ 讲解（练习待作答） |
| 24 | [策略梯度推导](https://wikieden.github.io/math-and-eval-stats/lessons/24-policy-gradient.html) | 似然比搬梯度；环境项消失；基线不偏降方差；PPO 只截得利侧；GRPO 组内归一化 | [REINFORCE 与基线](https://wikieden.github.io/math-and-eval-stats/animations/24-reinforce-baseline.gif) | ST27–ST30 | ✅ 讲解（练习待作答） |
| 25 | [LQR 与 Riccati](https://wikieden.github.io/math-and-eval-stats/lessons/25-lqr-riccati.html) | 动态规划推 Riccati、有限 / 无限时域、可控性 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/25-lqr-riccati.gif) | ST86–ST88 | ✅ 讲解（练习待作答） |
| 26 | [轨迹优化与 MPC](https://wikieden.github.io/math-and-eval-stats/lessons/26-trajopt-mpc.html) | 线性化后写成 QP（稠密 / 稀疏形式）、iLQR 的二阶展开与回传、MPPI 的路径积分推导 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/26-trajopt-mpc.gif) | ST89–ST91 | ✅ 讲解（练习待作答） |
| 27 | [扩散模型推导](https://wikieden.github.io/math-and-eval-stats/lessons/27-diffusion-derivation.html) | 前向闭式、后验 $q(x_{t-1}\mid x_t,x_0)$、ELBO 到噪声预测损失、DDIM、Tweedie 公式与 score | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/27-diffusion-derivation.gif) | ST92–ST95 | ✅ 讲解（练习待作答） |
| 28 | [Flow matching 推导](https://wikieden.github.io/math-and-eval-stats/lessons/28-flow-matching.html) | 概率路径、条件与边际目标等价、rectified flow、π0 动作头的训练与采样 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/28-flow-matching.gif) | ST96–ST98 | ✅ 讲解（练习待作答） |
| 29 | [测度论概率入门](https://wikieden.github.io/math-and-eval-stats/lessons/29-measure-probability.html) | σ-代数、测度、Lebesgue 积分与期望、密度 = Radon-Nikodym 导数、条件期望、收敛方式与大数定律 / CLT 的严格陈述 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/29-measure-probability.gif) | ST99–ST101 | ✅ 讲解（练习待作答） |
| 30 | [泛函分析入门](https://wikieden.github.io/math-and-eval-stats/lessons/30-functional-analysis.html) | 赋范空间、Banach / Hilbert 空间、算子与谱、Banach 不动点定理（Bellman 收敛）、再生核 Hilbert 空间（RKHS）与核方法 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/30-functional-analysis.gif) | ST102–ST104 | ✅ 讲解（练习待作答） |
| 31 | [微分几何与流形优化](https://wikieden.github.io/math-and-eval-stats/lessons/31-manifold-optimization.html) | 流形、切空间、黎曼度量、测地线、黎曼梯度与收缩映射（retraction）、和第 19 章李群接上 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/31-manifold-optimization.gif) | ST105–ST107 | ✅ 讲解（练习待作答） |
| 32 | [随机微积分](https://wikieden.github.io/math-and-eval-stats/lessons/32-stochastic-calculus.html) | 布朗运动、Itô 积分与 Itô 引理、SDE 的数值解、Fokker-Planck 方程、反向时间 SDE、概率流 ODE | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/32-stochastic-calculus.gif) | ST108–ST111 | ✅ 讲解（练习待作答） |
| 33 | [最优传输](https://wikieden.github.io/math-and-eval-stats/lessons/33-optimal-transport.html) | Monge / Kantorovich、Wasserstein 距离、熵正则与 Sinkhorn、动态最优传输与 flow matching 的联系 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/33-optimal-transport.gif) | ST112–ST114 | ✅ 讲解（练习待作答） |
| 34 | [信息几何](https://wikieden.github.io/math-and-eval-stats/lessons/34-information-geometry.html) | Fisher 信息、自然梯度、KL 的局部二阶近似、TRPO / PPO 的信任域视角 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/34-information-geometry.gif) | ST115–ST117 | ✅ 讲解（练习待作答） |
| 35 | [卷积的数学](https://wikieden.github.io/math-and-eval-stats/lessons/35-convolution.html) | 卷积 = 稀疏共享权重的线性映射（Toeplitz）、互相关与卷积、尺寸公式（填充 / 步长 / 空洞）、感受野、卷积定理与 FFT、im2col = 矩阵乘法 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/35-convolution.gif) | ST118–ST120 | ✅ 讲解（练习待作答） |
| 36 | [CNN 训练的数学](https://wikieden.github.io/math-and-eval-stats/lessons/36-cnn-training.html) | 卷积的反向传播与转置卷积、Xavier / He 初始化的方差推导、BatchNorm 的统计与梯度、残差的梯度流 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/36-cnn-training.gif) | ST121–ST123 | ✅ 讲解（练习待作答） |
| 37 | [注意力的数学](https://wikieden.github.io/math-and-eval-stats/lessons/37-attention-math.html) | 缩放点积与 $\sqrt{d}$ 的方差推导、softmax 温度与饱和、注意力 = 核平滑 / 软检索、多头的低秩视角、因果 mask | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/37-attention-math.gif) | ST124–ST126 | ✅ 讲解（练习待作答） |
| 38 | [位置编码的数学](https://wikieden.github.io/math-and-eval-stats/lessons/38-position-encoding.html) | 正弦编码、RoPE = 复数旋转与相对位置、长度外推（位置插值、NTK、YaRN）、ALiBi | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/38-position-encoding.gif) | ST127–ST129 | ✅ 讲解（练习待作答） |
| 39 | [归一化与信号传播](https://wikieden.github.io/math-and-eval-stats/lessons/39-normalization.html) | LayerNorm / RMSNorm 的几何、Pre-LN vs Post-LN 的梯度、深层网络的方差传播 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/39-normalization.gif) | ST130–ST132 | ✅ 讲解（练习待作答） |
| 40 | [张量的数学（PyTorch）](https://wikieden.github.io/math-and-eval-stats/lessons/40-tensor-math.html) | stride 与索引映射、广播规则、einsum 记号、视图与连续性、计算图上的 VJP 组合 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/40-tensor-math.gif) | ST133–ST135 | ✅ 讲解（练习待作答） |
| 41 | [Scaling law](https://wikieden.github.io/math-and-eval-stats/lessons/41-scaling-law.html) | 幂律与 log-log 拟合、Kaplan vs Chinchilla、计算最优 N / D、6ND 计算账、外推误差 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/41-scaling-law.gif) | ST136–ST138 | ✅ 讲解（练习待作答） |
| 42 | [参数化与初始化](https://wikieden.github.io/math-and-eval-stats/lessons/42-mup.html) | 标准参数化 vs μP、超参从小模型迁移、宽度缩放下的学习率 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/42-mup.gif) | ST139–ST140 | ✅ 讲解（练习待作答） |
| 43 | [现代优化器](https://wikieden.github.io/math-and-eval-stats/lessons/43-modern-optimizers.html) | Adam 的预条件视角、Shampoo / SOAP（Kronecker 预条件）、Muon（Newton-Schulz 正交化）、Lion（符号更新）、WSD 调度、梯度噪声尺度与临界批量 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/43-modern-optimizers.gif) | ST141–ST144 | ✅ 讲解（练习待作答） |
| 44 | [训练稳定性](https://wikieden.github.io/math-and-eval-stats/lessons/44-training-stability.html) | loss spike、logit 增长与 z-loss、QK-norm、梯度裁剪、bf16 / fp8 的数值范围与缩放因子 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/44-training-stability.gif) | ST145–ST147 | ✅ 讲解（练习待作答） |
| 45 | [MoE 的数学](https://wikieden.github.io/math-and-eval-stats/lessons/45-moe-math.html) | 门控与 top-k、负载均衡辅助损失、无辅助损失的偏置调节、容量因子、专家并行的通信量 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/45-moe-math.gif) | ST148–ST150 | ✅ 讲解（练习待作答） |
| 46 | [分布式训练账本](https://wikieden.github.io/math-and-eval-stats/lessons/46-distributed-ledger.html) | 显存四项（参数 / 梯度 / 优化器状态 / 激活）、ZeRO 分片、DP / TP / PP 通信量公式、流水线气泡、all-reduce 成本模型 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/46-distributed-ledger.gif) | ST151–ST153 | ✅ 讲解（练习待作答） |
| 47 | [推理成本模型](https://wikieden.github.io/math-and-eval-stats/lessons/47-inference-cost.html) | roofline 与算术强度、prefill vs decode、KV cache 大小公式、批处理与延迟的权衡 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/47-inference-cost.gif) | ST154–ST156 | ✅ 讲解（练习待作答） |
| 48 | [高效注意力与序列模型](https://wikieden.github.io/math-and-eval-stats/lessons/48-efficient-attention.html) | online softmax 与 FlashAttention 分块、MQA / GQA、MLA 低秩 KV、稀疏注意力（NSA）、线性注意力与 SSM 的并行扫描（Mamba） | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/48-efficient-attention.gif) | ST157–ST160 | ✅ 讲解（练习待作答） |
| 49 | [量化的数学](https://wikieden.github.io/math-and-eval-stats/lessons/49-quantization.html) | 均匀量化的 scale / zero-point、舍入误差方差、per-channel / group、离群值、GPTQ 的二阶误差补偿、AWQ、SmoothQuant 的等价变换、FP8 / FP4 格式 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/49-quantization.gif) | ST161–ST163 | ✅ 讲解（练习待作答） |
| 50 | [解码与采样](https://wikieden.github.io/math-and-eval-stats/lessons/50-decoding.html) | 温度、top-k / top-p、beam search、投机解码（拒绝采样为什么不改分布）、Medusa / EAGLE、约束解码 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/50-decoding.gif) | ST164–ST166 | ✅ 讲解（练习待作答） |
| 51 | [后训练的数学](https://wikieden.github.io/math-and-eval-stats/lessons/51-post-training.html) | SFT = MLE；Bradley-Terry 奖励模型；KL 正则目标与闭式最优策略；DPO 推导；GRPO / 推理 RL 的优势归一化 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/51-post-training.gif) | ST167–ST170 | ✅ 讲解（练习待作答） |
| 52 | [相机几何](https://wikieden.github.io/math-and-eval-stats/lessons/52-camera-geometry.html) | 针孔模型、内参 / 外参、畸变模型、反投影、逆透视变换与 BEV（Lift-Splat-Shoot） | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/52-camera-geometry.gif) | ST171–ST173 | ✅ 讲解（练习待作答） |
| 53 | [多视几何](https://wikieden.github.io/math-and-eval-stats/lessons/53-multiview-geometry.html) | 对极约束、本质 / 基础矩阵、单应、三角化、PnP、RANSAC 迭代次数公式；DUSt3R 怎么绕开这条流水线 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/53-multiview-geometry.gif) | ST174–ST176 | ✅ 讲解（练习待作答） |
| 54 | [点云与配准](https://wikieden.github.io/math-and-eval-stats/lessons/54-point-cloud.html) | 最近邻与 KD 树、Procrustes / Kabsch（SVD 求最优旋转）、ICP（点到点 / 点到面）、法向量 = 局部协方差的最小特征向量 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/54-point-cloud.gif) | ST177–ST179 | ✅ 讲解（练习待作答） |
| 55 | [SLAM 的优化](https://wikieden.github.io/math-and-eval-stats/lessons/55-slam-optimization.html) | 重投影误差与 BA、稀疏 Hessian 与 Schur 补、位姿图、鲁棒核（Huber / Cauchy）、边缘化 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/55-slam-optimization.gif) | ST180–ST182 | ✅ 讲解（练习待作答） |
| 56 | [VIO](https://wikieden.github.io/math-and-eval-stats/lessons/56-vio.html) | IMU 运动学与噪声 / 偏置模型、流形上的预积分、紧耦合滑窗、可观测性 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/56-vio.gif) | ST183–ST185 | ✅ 讲解（练习待作答） |
| 57 | [地图表示](https://wikieden.github.io/math-and-eval-stats/lessons/57-maps.html) | 占用栅格的 log-odds 贝叶斯更新、TSDF / ESDF、体素与八叉树、语义地图 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/57-maps.gif) | ST186–ST188 | ✅ 讲解（练习待作答） |
| 58 | [路径规划](https://wikieden.github.io/math-and-eval-stats/lessons/58-path-planning.html) | 图搜索（Dijkstra、A* 与可采纳启发）、构型空间、采样规划（PRM、RRT、RRT* 渐近最优）、代价地图与势场 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/58-path-planning.gif) | ST189–ST191 | ✅ 讲解（练习待作答） |
| 59 | [神经场景表示](https://wikieden.github.io/math-and-eval-stats/lessons/59-neural-rendering.html) | 体渲染积分与 α 合成、NeRF 的位置编码、3DGS 的协方差投影雅可比与球谐颜色 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/59-neural-rendering.gif) | ST192–ST194 | ✅ 讲解（练习待作答） |
| 60 | [感知的学习与评测](https://wikieden.github.io/math-and-eval-stats/lessons/60-perception-metrics.html) | IoU、NMS、PR 曲线与 mAP、匈牙利匹配（DETR）、度量学习与地点识别（三元组损失、NetVLAD、InfoNCE） | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/60-perception-metrics.gif) | ST195–ST197 | ✅ 讲解（练习待作答） |
| 61 | [多模态融合的数学](https://wikieden.github.io/math-and-eval-stats/lessons/61-multimodal-fusion.html) | 逆方差加权（Kalman 的特例）、早 / 中 / 晚融合、交叉注意力与 Perceiver 潜变量、模态缺失与模态 dropout | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/61-multimodal-fusion.gif) | ST198–ST200 | ✅ 讲解（练习待作答） |
| 62 | [视觉-语言对齐](https://wikieden.github.io/math-and-eval-stats/lessons/62-vl-alignment.html) | CLIP 对比损失与温度、SigLIP 的 sigmoid 损失、InfoNCE 与互信息下界、零样本分类 = 余弦相似度、视觉 token 化（patch、Q-Former、投影器）与 token 数账、ImageBind | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/62-vl-alignment.gif) | ST201–ST203 | ✅ 讲解（练习待作答） |
| 63 | [非图像模态的表示](https://wikieden.github.io/math-and-eval-stats/lessons/63-nonimage-modalities.html) | 点云的置换不变（PointNet 对称函数）、体素化与稀疏卷积、音频的 STFT / Mel 频谱、力 / 触觉阵列、本体感受的归一化 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/63-nonimage-modalities.gif) | ST204–ST206 | ✅ 讲解（练习待作答） |
| 64 | [多传感器标定与对齐](https://wikieden.github.io/math-and-eval-stats/lessons/64-calibration.html) | 外参标定（手眼 AX = XB）、时间同步与插值（含 SO(3) 上的插值）、相机-激光雷达投影、BEVFusion | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/64-calibration.gif) | ST207–ST209 | ✅ 讲解（练习待作答） |
| 65 | [VLN 的几何与评测](https://wikieden.github.io/math-and-eval-stats/lessons/65-vln-metrics.html) | 指令-轨迹对齐（DTW、nDTW、SDTW）、SPL、拓扑图与图搜索、前沿探索、语言特征投到俯视地图（VLMaps） | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/65-vln-metrics.gif) | ST210–ST212 | ✅ 讲解（练习待作答） |
| 66 | [VLN 的序列决策](https://wikieden.github.io/math-and-eval-stats/lessons/66-vln-decision.html) | POMDP 与信念状态、行为克隆误差 O(T²) 累积与 DAgger 的 O(T)、语言条件策略、探索的信息增益 | [GIF](https://wikieden.github.io/math-and-eval-stats/animations/66-vln-decision.gif) | ST213–ST215 | ✅ 讲解（练习待作答） |

## 联系与合作

欢迎来信：<wikieden@gmail.com>。可以合作的方向：

- 讲义勘误与建议
- 课程 / 企业内训
- 内容合作与转载授权
- 赞助本系列持续更新

赞助请发邮件联系。

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
