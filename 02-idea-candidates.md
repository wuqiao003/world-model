# World Model 论文选题候选(2026.08)

选题依据来自 [`01-related-work.md`](./01-related-work.md)。筛选原则:

1. **瓶颈是 idea 而不是数据中心。** 不做需要 1M+ 小时视频预训练、14B 训练run、或真机集群的题。
   Odyssey-2 Max 用了"几百张 B200",Cosmos 用 2 亿精选片段 —— 在 fidelity 上正面竞争必输。
2. **每条都有至少一篇 2025–2026 一手文献明确说这是 open problem**,而不是我们自己想象的缺口。
3. **优先"在冻结的开源 backbone 上做机制研究"和"测量别人不测的东西"。**
   开权重可用的:Cosmos-Predict2.5-2B、Matrix-Game 2.0(1.3B)/3.0、Yume-1.5(5B)、
   Infinite-World、V-JEPA 2.1(80M/300M/1B/2B)、Newt(200+ checkpoint)、LeWM(15M,单卡几小时)。
4. **避开已经拥挤的赛道。** 2026 年 2–6 月出了至少四个通用交互 benchmark
   (WorldMark / iWorldBench / MIND / Omni-WorldBench),再做一个"通用 benchmark"很难中。

---

## 快速索引

| # | 一句话 | 算力 | 风险 | 天花板 | 推荐 |
|---|---|---|---|---|---|
| **1** | 用可辨识性理论造一个**能预测下游效用**的反事实动作保真度指标 | 8 GPU | 中(需差异化) | 高 | ★★★ |
| **2** | 画出**记忆 ↔ 可控性**的 Pareto 前沿,并用 action-conditioned gating 推动它 | 8 GPU | 低 | 中高 | ★★★ |
| **3** | 记忆该**写什么**:事件驱动的选择性写入 | 8 GPU | 低 | 中 | ★★★ |
| **4** | 为世界模型**故意收集更差的数据**(excitation-aware 数据策划) | 8 GPU | 中高 | 很高 | ★★☆ |
| **5** | **画外世界仍在演化**:off-screen state evolution | 16 GPU + 造数据 | 中 | 高 | ★★☆ |
| **6** | 把**相机位姿与化身位姿解耦**(第三人称控制) | 16 GPU + 造数据 | 中 | 高 | ★★☆ |
| **7** | 用**便宜的几何 verifier 门控昂贵的视频生成器** | 8–16 GPU | 中 | 中高 | ★★ |
| **8** | 算力归一化的物理评测:**知识天花板 vs 采样缺口** | 4–8 GPU | 很低 | 中 | ★★ |
| **9** | 实测 **safe horizon**,并澄清 "search 有害" vs "imagination 有效" 的矛盾 | 16 GPU | 低 | 中 | ★★ |

---

# Tier A — 首选

## Idea 1:反事实动作保真度(CAF)—— 从可辨识性理论到能预测下游效用的指标

**一句话.** 现有所有 world model 指标都在测"前向 rollout 好不好看";我们要测的是
"**如果我换一个动作,预测的未来是否恰好按该动作应有的方式改变、且不多不少**",
并且证明这个指标比 FVD / 物理分数更能预测下游 planning 与 policy-evaluation 的成败。

**为什么现在.** 三条线在 2026 年撞到了一起,但没人把它们接起来:

- WMBench / GigaWorld-1(2607.02642,32.4 万 rollout)的结论是
  "**评估器质量由长视界、动作保真的 rollout 一致性主导,而非短期视觉真实度**" ——
  但他们**没有给出一个可计算的动作保真度指标**。
- What-If World(2605.27589)发现了 "**contrastive bottleneck**":模型对"轻踩刹车"和"重踩刹车"
  生成**几乎相同的视频**,同时通过每一项单视频检查。但它是 prompt 级、依赖 VLM/人评的。
- 可辨识性理论(2607.22430)**证明了**这件事为什么必然会发生:在 conditional action excitation
  不足时,"准确的 on-policy 预测"与"大得多的 counterfactual 误差"可以共存。
  配套的 2605.26379 给了一个可复现的最小例子:**Reacher 在 OU 噪声采样下可辨识,
  在 goal-directed policy 下不可辨识**。

也就是说:**现象被发现了、理论解释了它、下游研究说它才是要紧的 —— 但缺一个能自动算、
能被采纳的指标,以及缺"这个指标真的预测下游效用"的证据。**这就是这篇论文。

**核心贡献设计(三段式,第三段才是真贡献):**

1. **指标.** 定义一族无需人评、无需新标注的成对干预量:
   - `action-response gain` = ∂(预测的末端/自车位移) / ∂(指令位移),理想为 1,塌缩时趋 0;
   - `action-insensitivity rate` = 在成对干预下预测轨迹差异低于噪声阈值的比例(直接量化 contrastive bottleneck);
   - `branch-divergence spectrum` = 从同一状态在 a 与 a′ 下 rollout,分解差异为
     "动作应致的分量" vs "动作无关的漂移分量"(用光流/点跟踪 + 刚体分解),
     后者就是"改了动作顺手把无关的东西也改了"这一被完全忽视的失败。
2. **数据构造(零人评).** 在已有 (obs, action, next-obs) 数据上用**动作扰动族**造配对:
   镜像、缩放、时间反转、夹爪反转、正交位移。DROID / Bridge V2 / Open-X / nuScenes(NAVSIM)/
   Minecraft VPT contractor 全都自带真动作,所以扰动后的**期望效应方向是已知的**,不需要新真值视频。
3. **杀手实验(论文的重心).** 证明 **predictive validity**:
   把 CAF、FVD、PSNR、VideoPhy 分、WorldScore 一起算在同一批开源 world model 上,
   然后用 WorldGym(2506.00613)/ WMBench 协议做 policy-evaluation rank fidelity,
   以及在 RoboTwin/ManiSkill 里做真实规划成功率,报**相关系数表**。
   预期结论:CAF 与下游效用的相关显著高于所有像素指标 —— 这正是 Lambert et al.(2002.04523)
   的 objective mismatch 在视频世界模型上的第一次定量落地。

**评测对象(全部开权重):** Cosmos-Predict2.5-2B、Ctrl-World、iVideoGPT、Genie Envisioner GE-Base、
RoboScape、DreamDojo、Matrix-Game 2.0、Vista(驾驶)。

**算力.** 主体是推理 + 一个小 IDM/点跟踪器。8 GPU 充裕。

**风险与缓解.**
- *最大风险:被当成"又一个 benchmark"。* → **不要把 benchmark 当卖点**。
  标题和 abstract 都压在 predictive validity 上:"哪些指标预测控制效用,哪些不预测"。
  Benchmark 是副产品,理论 grounding + 相关性证据是主产品。
- *与 What-If World / ACT-Bench / Omni-WorldBench 重叠。* → 差异点写清楚三条:
  (a) 自动、无人评、无新标注(它们都依赖 VLM judge 或人评,而 PhyGround 测出
  Gemini-3.1-Pro 在物理打分上有 16.6% 聚合相对偏差);
  (b) 有可辨识性理论的 grounding 与可测前置条件;
  (c) 有下游 predictive validity 证据 —— 上述三者都没有。
- *judge 漂移.* → 全程不用闭源 judge。这本身是一个卖点。

**相关工作必引.** 2607.02642 · 2605.27589 · 2607.22430 · 2605.26379 · 2412.05337(ACT-Bench)·
2603.22212 · 2506.00613 · 2410.18072(WorldSimBench 的 implicit manipulative eval)· 2002.04523 ·
2310.06253 · 2404.12391(FVD 内容偏差)· 2505.13934(RLVR-World)。

**投稿.** NeurIPS D&B / ICLR / ICML。ICLR 尤其合适(理论+实证+可用工件)。

---

## Idea 2:记忆 ↔ 可控性 的 Pareto 前沿,与 action-conditioned memory gating

**一句话.** 领域里已经有五个互相独立的散点在说同一件事 —— 记忆变强会让动作响应变弱 ——
但没有人把这条曲线画出来,也没有人给出一个能支配整条曲线的方法。

**为什么现在.** 证据已经攒够了,而且是彼此不知情地攒出来的:

- 综述 2606.01164 §6.1 原话意译:历史观测越多长时一致性越好,"**但同时削弱了对动作的响应**"。
- Astra(ICLR 2026)的做法是**给 conditioning frame 加噪**,刻意模糊历史影响以逼出动作响应度 ——
  这等于承认二者冲突,而且它只是曲线上的**一个点**。
- MIND(2602.08025)发现在 action-space shift 下,**带记忆的 video-to-world 反而不如无记忆的
  image-to-world** —— 记忆会主动倒忙。
- WorldMem(NeurIPS 2025)自己的 ablation:记忆帧 1→8 全面变好,**16 帧全面变差**。
- Rolling Forcing 指出加噪这一族的代价:"缓解漂移…但**剥夺了模型的干净参考,损害时间一致性**"。

**核心贡献设计.**

1. **前沿测绘.** 在 2–3 个冻结的开源 backbone(Matrix-Game 2.0 1.3B / Yume-1.5 5B / Infinite-World)
   上取四个可连续调的旋钮:context length、conditioning noise σ(Astra 式)、retrieval top-K、
   memory-attention 增益。扫出 consistency(WorldMark/MIND 的一致性维度)
   vs controllability(平移/旋转误差)的二维前沿。**这张图本身就会被反复引用**,
   因为现在每篇记忆论文都只能报单点。
2. **机制解释.** 为什么冲突?假设是**注意力质量守恒**:softmax 预算被历史 token 抢走,
   动作 token 的有效增益下降。用 attention mass 分配 + 对动作 token 做因果消融来验证。
   若成立,这给出了一个比"经验上要加噪"更强的解释。
3. **方法.** `action-conditioned memory gating`:让记忆读取强度由当前动作决定 ——
   **转身/回看(指向已探索区域)时重记忆,推进入新区域时重动作**。
   实现上是一个小 gate,输入为动作、当前位姿相对已探索区域的覆盖度、以及检索命中的几何重叠;
   输出为 memory attention 的温度/权重。冻结 backbone 只训 gate。
4. **主张.** 单个 gate 支配整条 baseline 前沿(在同等一致性下可控性更好,反之亦然)。

**算力.** 冻结 backbone + 小 gate。8 GPU。**这是本清单里投入产出比最高的一条。**

**风险与缓解.**
- *只是一张工程曲线,不够"新"。* → 必须有第 2 步的机制解释和第 3 步的方法。
  只有前沿图的话是 workshop paper;加上 gating 就是正会。
- *不同 backbone 前沿形状不一致,难下统一结论。* → 那本身是结论(前沿形状由记忆架构族决定),
  按 implicit-latent / explicit-geometry / hybrid 三类分组报告。

**相关工作必引.** 2606.01164 §6.1 · Astra(ICLR 2026)· 2602.08025 · 2504.12369 · 2509.25161 ·
2602.06028 · 2510.03198 · 2604.21686 · 2606.31672(中段崩塌被首尾指标掩盖)。

**投稿.** ICLR / CVPR / NeurIPS。

---

## Idea 3:记忆该**写什么** —— 事件驱动的选择性写入

**一句话.** 每个记忆系统都在研究怎么**读**,几乎没人研究怎么**写**;
而探索轨迹里绝大多数帧是冗余的,少数帧承载了几乎全部 loop-closure 信息。

**为什么现在.**
- RELIC(压缩长视界 cache)、Infinite-World(递归蒸馏到固定预算)、Context Forcing
  (固定间隔 consolidation)—— **三个 2026 年的记忆压缩方案全都是均匀压缩**。
  Context Forcing 的 "surprisal-based context consolidation" 是最接近的,但仍是粗粒度的
  2-chunk-interval 启发式。
- WorldMem 的非单调性(8 好 16 坏)说明瓶颈是**检索精度而非容量** ——
  少写、写对,应当同时改善质量和成本。
- **工程红利很实在**:Matrix-Game 3.0 的 ablation 显示去掉 GPU 端 memory retrieval 省 33 fps,
  即蒸馏之后 **retrieval 已是吞吐瓶颈**。缩小记忆库直接换 fps —— 这条论文有"又好又快"的双卖点,
  在这个领域非常吃香。

**核心贡献设计.**
1. 学一个 **write-gate**:在固定预算下决定是否把当前帧 commit 为 keyframe。触发信号候选:
   surprisal(预测误差)、几何覆盖增益(新观察到的 surfel/体素占比)、
   语义事件(放下物体、开门、首次看到 landmark —— 可用冻结 VLM 弱标注)。
2. **训练信号是关键设计点。** 不要用重构损失训 gate(那会退化成均匀采样)。
   用 **downstream 检索命中后的生成损失下降** 做 credit assignment:
   一帧的价值 = 它在后续 revisit 时被检索到并降低了多少生成损失。
   可以用 REINFORCE 或直接用离线贪心 oracle 蒸馏。
3. **主张(两条,都要):** (a) 在 1/4 记忆预算下超过均匀压缩;
   (b) 给出"记忆预算 vs 长视界一致性"的 scaling 曲线 —— 现在没人有这条曲线。

**算力.** 冻结 backbone,只训 gate。8 GPU。

**风险与缓解.**
- *gate 学成"每 k 帧写一次"。* → 这是最可能的失败模式。必须报 gate 触发时刻与
  ground-truth 事件(引擎里可取)的对齐度,并把"退化成均匀"当作显式 baseline 对照。
- *Context Forcing 已经做了 surprisal。* → 差异化在于:它是 consolidation 的调度启发式,
  我们是**带预算的可学习写入决策 + 用下游生成收益做 credit assignment**,并且要正面比较。

**相关工作必引.** 2512.04040 · 2602.02393 · 2602.06028 · 2604.08995 · 2504.12369 ·
2603.17117(MosaicMem)· 2603.03482(PERSIST)。

**投稿.** ICLR / CVPR / NeurIPS。

---

# Tier B — 更有野心,风险中等

## Idea 4:为世界模型**故意收集更差的数据**(excitation-aware 数据策划)

**一句话.** 如果可辨识性理论对,那么 **demonstration 可能是训练世界模型最糟糕的数据** ——
因为目标导向策略让 latent 边缘分布塌缩到低熵区,使转移不可辨识。
这条如果成立,是本清单里冲击力最大的结论。

**为什么现在.** 两篇 2026 年理论论文给了具体的可操作抓手,而**没人在真实数据上测过**:
- 2607.22430:**conditional action excitation** 不足会让 counterfactual 误差与 on-policy 误差解耦。
- 2605.26379 的最小例子:**Reacher 在 OU 噪声采样下可辨识,在 goal-directed policy 下不可辨识**,
  因为后者的边缘分布塌到低熵 latent 区域。
- 而机器人综述 2605.00080 §8.1 从建模侧说的是同一件事:
  "技术瓶颈是**弱动作条件化**:许多预测式世界模型目标主要从观测历史与任务意图训练,
  所以它们的未来可以合理却并不因果地系于将要执行的动作。"

**核心贡献设计.**
1. **测量.** 在 DROID / Open-X / D4RL / VPT contractor 上定义并计算 conditional action excitation
   的可用代理量(理论条件是线性高斯,真实数据需要设计代理 —— **这个代理量的设计本身就是贡献**)。
   候选:动作条件下 latent 增量的条件协方差最小特征值、动作序列的谱平坦度、
   给定观测后动作的条件熵。
2. **相关性.** 证明该量预测下游 planning / policy-eval 成功率(与 Idea 1 的 CAF 天然互补,
   可共用评测栈)。
3. **干预.** 给出 excitation 最大化的三种可落地手段并比较:
   (a) 数据混合权重重加权;(b) action relabeling / 抖动增强;
   (c) 主动采集(在仿真里可做:注入 OU 噪声的探索策略 vs 纯 demo,固定数据量对比)。
4. **头条结论目标.** 固定数据预算下,**混入或替换为"更差的"高 excitation 数据的世界模型,
   下游规划更好** —— 且这与它在 on-policy 预测损失上更差并不矛盾。

**算力.** 8 GPU(小模型 + 大量统计)。可先在 DMC/Meta-World 做干净版本,再迁 DROID。

**风险与缓解.**
- *理论条件(可逆观测、线性高斯 latent、精确表征约束)离真实数据很远。* →
  分两层报告:仿真里满足条件的干净验证 + 真实数据上的代理量相关性。**不要声称理论保证迁移了。**
- *结论可能是"没关系"。* → 那也是可发表的负面结果,但要提前想好:
  若相关性弱,fallback 是把它做成 Idea 1 的一个诊断维度。

**相关工作必引.** 2607.22430 · 2605.26379 · 2605.00080 §8.1 · 2606.31232(Delta-JEPA)·
2506.01622 · 2002.04523。

**投稿.** ICML / NeurIPS(偏理论-实证结合的会更合适)。

---

## Idea 5:画外世界仍在演化(off-screen state evolution)

**一句话.** 除了 LiveWorld 和 HM-World,所有世界模型都把视锥外的世界**冻结**,
所以回头看到的是过期状态。MBench 的典型失败:镜头穿过爆炸烟雾,回来街道完好无损。

**为什么现在.**
- MBench(2606.00793)把这个现象提炼成了领域判词:
  "**causal memory 是核心未解难题**" —— 很多 action-conditioned 模型保住了**外观**,
  但**世界不演化**。它们的聚合分只在 43–47 区间。
- 只有两篇 2026.03 的论文(LiveWorld 2603.07145、HM-World 2603.25716)正面碰这个问题 ——
  **赛道几乎是空的**,而问题定义清晰。
- WorldMem 已经在记忆里存了 timestamp,但只用于静态几何检索 —— **机制已经在那里,没被用起来**。

**核心贡献设计.**
1. **数据与 metric(这是主要工作量,也是护城河).** 游戏引擎(Minecraft / Unreal / Habitat)
   免费提供画外真值。构造"离开 T 秒后回看"的评测集,定义 `revisit-state-error`:
   回看时的状态与引擎真值状态之差,并按"该演化的东西(火在烧、水在流、NPC 在走)"
   与"不该变的东西(墙、地形)"分开报,以区分**演化**与**漂移**。
2. **方法.** 时间戳 + 事件条件的记忆更新:记忆条目不再是"某时刻的快照",
   而是"快照 + 一个可外推的演化算子"。最简单可行版本:
   在检索到的记忆帧上按 Δt 施加一个学到的 latent 演化步,再送入 generator。
3. **与 Idea 3 天然组合**:写入决定存什么,演化算子决定存的东西如何随时间前进。两者可以是同一篇的两节。

**算力.** 16 GPU + 明确的数据工程投入。

**风险与缓解.**
- *造数据成本高。* → 先在 Minecraft 上做(Dreamer 4 / Oasis / Matrix-Game 生态都在这里,
  baseline 现成,而且 VPT contractor 数据可用),Unreal 只做泛化验证。
- *可能被 LiveWorld / HM-World 抢先。* → 这两篇都是方法论文;
  把重心放在**评测协议 + 演化/漂移解耦**上,方法作为示范,更抗抢。

**相关工作必引.** 2606.00793(MBench)· 2603.07145 · 2603.25716 · 2504.12369(timestamp)·
2602.02393(revisit-dense 数据的重要性)· 2606.31672。

---

## Idea 6:把相机位姿与化身位姿解耦(第三人称控制)

**一句话.** 文献里被**量化过的最大单项能力缺口**:Matrix-Game 2.0 从第一人称切到第三人称,
旋转误差从 1.32° 涨到 27.6°(约 20×)。原因是所有现有动作空间把相机和化身纠缠在一起。

**为什么现在.**
- WorldMark(2604.21686)在 500 个可控 case 上量出了这个 20× 崩塌,
  并且发现 **domain-specific 训练完全不迁移**(Minecraft 上训的 Open-Oasis
  在真实与风格化场景上全指标失败)。
- Genie 3 的**第一条** limitation 就是:"promptable world events 允许广泛的环境干预,
  但它们**不一定由 agent 自己执行**;agent 能直接做的动作范围目前受限。"
- MIND(2602.08025)刻意做了第一/第三人称共享动作空间 —— 说明社区已经意识到,
  但没人给出解法。

**核心贡献设计.**
1. **因子化动作空间**:`camera extrinsics × agent root motion × agent articulation`。
   三者分别注入(相机走连续位姿 / root motion 走连续位移 / articulation 走离散或低维连续),
   并在训练中做**交叉独立性正则** —— 只改相机时化身不该动,只改化身时背景不该动。
   这正好是 Idea 1 里 `branch-divergence spectrum` 要测的东西,可以互相引用支撑。
2. **数据.** 第三人称游戏录像(带引擎位姿最好)+ 合成 Unreal/Unity 数据(免费拿到相机与骨骼真值)。
3. **metric.** 分别报相机误差与化身误差,并报**交叉泄漏率**(改 A 引起 B 的变化量)。

**算力.** 16 GPU + 数据获取是主要成本。

**风险与缓解.**
- *数据获取是硬成本。* → 合成数据先行;这一路的好处是真值完全免费且可控。
- *可能被大厂顺手解决。* → 概率存在,但因子化 + 交叉泄漏 metric 是学术味的贡献,
  即使大模型能力上来了,这套分析框架仍然有价值。

**相关工作必引.** 2604.21686 · Genie 3 blog(limitations)· 2602.08025 · 2506.18701 · 2506.17201 ·
2603.16871(WorldCam,把相机位姿当统一几何表示)。

---

## Idea 7:用便宜的几何 verifier 门控昂贵的视频生成器

**一句话.** WorldLens 发现**纹理强的模型违反物理、几何稳的模型行为不真实**;
WMReward 证明物理失败**很大一部分是采样问题而非知识问题**。
把这两条接起来的显然做法 —— 用便宜的几何/占据模型当 critic 去搜索昂贵视频模型的输出 ——
我没找到有人做。

**为什么现在.**
- WorldLens(2512.10958,CVPR 2026 Oral):"**几乎所有现有世界模型都会触发碰撞或偏离道路**",
  且"没有一个世界模型全面胜出:纹理强的常违反物理,几何稳的缺行为保真度。"
- WMReward(2601.10553,CVPR 2026):用 V-JEPA-2 的 surprise 当推理时 reward 做 Best-of-N,
  **PhysicsIQ Challenge 第一,+7.42pp** —— 证明 critic-guided 搜索这条路很有效。
- occupancy 世界模型(OccWorld/OccSora 系)比多视角视频**便宜约一个数量级**,
  且在 nuScenes/OpenOccupancy 上跑得动。

**核心贡献设计.** 用 occupancy / 点云世界模型作为几何 verifier,对视频世界模型做
Best-of-N 重排序 + 去噪梯度引导;在 nuScenes + NAVSIM v2 上证明比纯视频 reward
更能降低碰撞率与偏离道路率。附带一个有意思的消融:verifier 需要多便宜才划算
(报 verifier FLOPs vs 闭环收益曲线)。

**算力.** 8–16 GPU。

**风险.** 工程量不小(要接两套栈);且驾驶闭环评测协议本身有坑
(NAVSIM↔Bench2Drive 相关性 ρ=0.90 但**非单调、有排序反转**,且那个研究只有 n=8)。
好处是 World Engine(2606.19836)在 nuPlan 上完全开源,可以直接站在上面。

**相关工作必引.** 2512.10958 · 2601.10553 · 2606.03159(OmniDreams 的退化优雅性论证)·
2606.19836 · 2405.20337 · 2605.00066(NAVSIM↔Bench2Drive)。

---

# Tier C — 低风险、快出、天花板中等

## Idea 8:算力归一化的物理评测 —— 知识天花板 vs 采样缺口

**一句话.** Physics-IQ 排行榜把单样本条目和 Best-of-N 条目**混在一起**,而 BoN 值 +7~25 分。
所以"模型 A 比模型 B 更懂物理"和"模型 A 被搜索得更狠"目前**无法区分**。
测 pass@1…pass@64,把"模型不会"和"采样没采到"拆开。

**为什么这条值得做.** 它便宜(4–8 GPU,纯推理),而且**直接改写领域叙事**:
如果大部分物理失败是采样缺口,那么"scaling 修不好物理"(PhyGenBench、PhyWorld)
和"BoN 拿到 SOTA"(WMReward)就统一了 —— 知识在流形上,采样器找不到它。
顺带可以提议排行榜必须报 compute-normalized 分数,这种"方法论卫生"类贡献引用量往往很好。

**要做的.** 在 Physics-IQ **Verified 协议**(2606.18943,必须用这个,原协议已被证明
仅改协议就把排名打乱到 Kendall τ=0.46)和 PhyGround 上,对开源模型
(Wan 2.2、Hunyuan Video 1.5、CogVideoX、Magi-1、Cosmos-Predict、LTX-Video)测 pass@k 曲线,
带/不带 V-JEPA-2 重排序,按物理类别拟合曲线,给出**每类的知识天花板与采样缺口估计**。
必须用开源 judge(PhyJudge-9B 偏差 3.3% vs Gemini-3.1-Pro 16.6%)。

**可低成本加挂的第二节(强烈建议加,能把它从"测量报告"提升为正会论文):**
**judge 审计** —— 交换呈现顺序、改写 rubric、塞入"好看但物理错"和"物理对但画质差(如仿真渲染)"的样本、
测 judge 对自家模型族的自偏好、测闭源 judge 的版本漂移。
交付一个 judge 鲁棒性套件 + 一个"不改物理就能刷榜"的攻击。

**相关工作必引.** 2501.09038 · **2606.18943** · 2601.10553 · 2605.10806 · 2411.02385 ·
2410.05363 · 2404.12391。

---

## Idea 9:实测 safe horizon,并澄清 "search 有害" vs "imagination 有效"

**一句话.** 2026 年 MBRL 文献里有一个公开的、没人解释的矛盾;同时有一个刚被证明的理论量
(safe horizon)从没被测过。两件事可以合成一篇很干净的澄清型论文。

**矛盾是什么.**
- **MRS.Q**(ICML 2026, 2601.21306):"传统看法认为长程预测与误差累积是 MBRL 的主要障碍。
  我们挑战这一观点… 令人惊讶的是,**即使模型高度准确,search 也可能损害性能**。
  缓解分布偏移比提升模型或价值函数精度更重要。"
- **Dreamer 4**(2509.24527):imagination RL 恰恰是拿到钻石的原因 ——
  仅 WM+BC 是 **0.0%**,加了在(自认不完美的)模型里做 rollout 之后是 **0.7%**,
  iron pickaxe 从 16.9% 到 29.0%。
- **TD-M(PC)²**(L4DC 2026):罪魁是 planner/actor 策略错配,把 h1hand-run 上
  **2159% 的价值高估打到 ≈0%**。
- 三者逻辑上兼容,但**没人画出这张地图**。

**要做的.**
1. 在匹配的模型质量与匹配的 value-ensemble 处理下,隔离
   **(a) rollout 用于 policy gradient** 与 **(b) rollout 用于测试时 search** 两种用法。
   DMC + Atari 100k 足够。
2. 实测 **safe horizon**:2605.15960 证明"在大策略集上利用基本不可避免"并导出了
   ε-exploitability 的 safe horizon,但**是纯理论**。在 DreamerV3 / TD-MPC2 / DIAMOND 上
   测经验 safe horizon vs 理论界,回答"这个界松到没用吗"。
3. 顺手给一个 **自适应 horizon 调度器**(随实测模型可靠度增长扩展 imagination horizon,
   而不是按惯例固定 H=15)。注意 MRS.Q 把 MPPI 固定到 3 步反而拿到 SOTA ——
   这是"领域默认视界早已超过 safe horizon"的弱证据,正好是这条论文的引子。

**算力.** 16 GPU。**注意** MRS.Q 用了 10 个 Q 函数的 ensemble,他们自己承认代价"substantial" ——
预算要留够。

**相关工作必引.** 2601.21306 · 2509.24527 · TD-M(PC)² · 2605.15960 · 2002.04523 · 2310.06253 ·
Stealing That Free Lunch(ICML 2025,Dyna 怀疑论)。

---

# 明确不建议做的

1. **在 fidelity / 分辨率 / 时长上正面竞争。** Odyssey-2 Max 用了几百张 B200,Cosmos 用 2 亿精选片段,
   Genie 3 不可复现。这是稳输的仗。
2. **再做一个通用 benchmark。** 2026 年 2–6 月已出至少四个(WorldMark / iWorldBench / MIND /
   Omni-WorldBench),还有 MBench / WorldArena / WorldRoamBench。要做评测就必须**测别人不测的维度**
   且**有 predictive validity 证据**(Idea 1、8 就是按这个原则设计的)。
3. **把 LIBERO 当 headline 数字。** 已饱和:六个架构完全不同的方法全在 97.2–98.5 的噪声里;
   而且 RynnVLA-002 的离散变体在 LIBERO 好看却在真实 SO-100 上**完全失败**。
4. **把 FVD / PSNR / LPIPS 当主要优化目标。** 至少四个独立来源(2404.12391 的 FVD 内容偏差、
   2510.02283 发现 VBench 偏好过曝退化帧、2510.18135、2607.02642)说这些不预测控制效用。
5. **需要 1M+ 小时视频预训练或百万级机器人轨迹的题。** WMPO 的贵处不是 RL 而是那个世界模型。
   要用 scale 就去用别人放出来的 checkpoint(V-JEPA 2.1 的 80M/300M/1B/2B、Newt 的 200+、
   Cosmos-Predict2.5-2B、Matrix-Game 2.0/3.0)。

---

# 组合建议

这些不是九个孤立选项,有三条自然的成篇路线:

**路线 α(评测/诊断,最稳):Idea 1 → Idea 8 → Idea 4。**
先立一个能预测下游效用的动作保真度指标(1),用算力归一化把物理评测的叙事修正(8),
再用可辨识性把它推到数据侧的因果结论(4)。三篇共用一套评测栈,叙事完整:
"我们在测错的东西 → 正确的东西该怎么测 → 测对之后数据该怎么收"。**8 GPU 全程够用。**

**路线 β(记忆机制,最快出成果):Idea 2 → Idea 3 → Idea 5。**
先画出记忆-可控性前沿并给 gating(2),再解决"写什么"(3),再解决"存的东西如何随时间演化"(5)。
共用同一个冻结 backbone 和同一套长视界评测,是一个非常完整的"世界模型记忆"系列。
Idea 2 和 3 甚至可以合成一篇强论文。

**路线 γ(能力缺口,风险最高但最亮眼):Idea 6 + Idea 5。**
第三人称/因子化动作空间 + 画外演化,都是"文献里被量化过但赛道空着"的硬缺口。
需要愿意投入数据工程。

**如果只选一条:Idea 1。** 理由:它同时满足"有一手文献明确说这是瓶颈"
(WMBench 直接说了评估器质量由动作保真度主导却没给指标)、"零人评零新标注"、
"8 GPU 够"、"结论对整个领域都有用"、以及"有理论 grounding 可以抵御 incremental 的批评"。
风险只有一个 —— 别把它包装成 benchmark 论文。

---

# 立刻可做的三件事

1. **核实关键 ID。** 见 `01-related-work.md` 的验证注意事项。最要紧的三个:
   2607.22430(Idea 1/4 的理论基石)、2607.02642(Idea 1 的动机来源)、
   2605.27589(Idea 1 最强的竞品)。这三篇不成立,路线 α 要重新设计。
2. **拉 baseline。** Matrix-Game 2.0(1.3B,能在少量卡上跑)、Cosmos-Predict2.5-2B、
   V-JEPA 2.1-300M、LeWM(15M,单卡几小时,迭代速度极快)。
   先把"人拿键盘进去玩"跑通 —— Dreamer 4 用的就是这个作为保真度检查,不要小看。
3. **扫 ICLR 2026 World Models workshop 的 94 篇接收列表。** 这是目前信息密度最高的地方,
   也是判断"我这个 idea 是不是已经有人在做"的最快途径。
