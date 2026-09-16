# World Model 近期相关工作(2024 – 2026.08)

> 说明:本清单由 2026-08-04 的网络检索汇总而成,**不是逐篇通读原文的结果**。
> 2026 年的 arXiv ID(26xx.xxxxx)有相当一部分来自综述的参考文献表或二手页面,已在文末
> [验证注意事项](#验证注意事项) 中逐条标注。**投稿引用前请逐一核对 ID、标题与会议归属。**

---

## 0. 一句话总览

到 2026 年中,这个领域发生了三件结构性的事:

1. **"world model" 的定义收紧了。** 从"能生成好看视频"变成"能被闭环使用的 action-conditioned
   predictor"。2026 年的两篇机器人综述(2605.00080、2606.00113)都把这条当作组织全文的主轴,
   并且都得出同一个结论:**视觉真实度是控制效用的弱代理**。
2. **交互式生成的工程问题基本被解决了**,在 1–5B / 20–40 fps 这个量级。配方收敛为
   DiT → causalization → per-frame action injection → forcing 系训练 → few-step 蒸馏 → 外部记忆。
3. **概念前沿转移到了小实验室能碰的地方。** 大厂垄断的是 artifact(Genie 3、Cosmos、GWM-1、
   OmniDreams 都不可复现),但真正没解决的是 objective / identifiability / evaluation /
   exploitation 这几类问题 —— 这些用 8–32 GPU 就能做。

---

## A. 交互式视频世界模型:主干、因果化、动作注入、蒸馏

| 名称 | 出处 + 时间 | 贡献 |
|---|---|---|
| Genie | ICML 2024 | 奠基:无监督 Latent Action Model + video tokenizer + dynamics,11B,~2s |
| Genie 2 | DeepMind blog, 2024.12 | 单图起始的自回归 latent diffusion,键鼠控制,720p,10–20s |
| **Genie 3** | DeepMind blog, **2025.08.05**;Project Genie 2026.01;Street View grounding 2026.05 | 首个实时通用世界模型,720p@24fps,分钟级一致性,~1min 视觉记忆,promptable world events。**无技术报告**,一致性被官方描述为 emergent(无显式 3D) |
| GameNGen | ICLR 2025 | DOOM 以 ~20fps 被扩散模型模拟,"neural game engine" 成为严肃主张 |
| Oasis | Decart/Etched, 2024 | 首个大规模可玩实时扩散游戏模型(Minecraft),500M DiT,640×360@20fps |
| Matrix-Game | arXiv 2506.18701, 2025.06 | 17B;**离散键盘走 cross-attn、连续鼠标/pitch 走 concat + temporal attn**(此后成为约定);3700h 动作标注语料 |
| Matrix-Game 2.0 | arXiv 2508.13009, 2025.08 | 开源实时流式,1.3B,25fps,~1min;把 Self-Forcing 带进交互式世界模型 |
| **Matrix-Game 3.0** | arXiv 2604.08995, 2026.04 | 720p@**40fps**/5B:camera-aware memory retrieval、残差误差重注入自纠、multi-segment DMD、INT8 + VAE 剪枝 |
| Hunyuan-GameCraft | arXiv 2506.17201, 2025.06 | 13B,720p@24fps;**把键盘+鼠标统一到共享连续相机空间**(另一条主流约定);8-step consistency 蒸馏 |
| WorldPlay / HY-World 1.5 | arXiv 2512.14614, 2025.12 | Context Forcing + memory-augmented self-rollout;短时 temporal + 长时 spatial 混合记忆;Temporal Reframing 修 RoPE 外推 |
| Yume / Yume-1.5 | 2507.17744 / 2512.22096;CVPR 2026 | 键盘驱动探索**真实场景**(Sekai, 400h);1.5 版加文本控制 + 记忆,704×1280@16fps 仅 5B |
| Dreamer 4 | arXiv 2509.24527, 2025.09 | **Shortcut forcing**(diffusion forcing + shortcut model)→ 每帧 4 次前向,单卡实时;极少动作标签 + 大量无标注视频;首个纯离线拿到 Minecraft 钻石 |
| Cosmos-Predict2.5 / Transfer2.5 | arXiv 2511.00062, 2025.10–11 | flow-based WFM 统一 T2W/I2W/V2W;Cosmos-Reason1 当 text encoder;2B/14B **权重开放**;可后训成 action-conditioned policy-eval 模型 |
| Runway GWM-1 | Runway, 2025.12.11 | 首个商用自回归通用世界模型族;24fps HD,**37ms/帧**;条件含相机位姿、机器人指令、音频 |
| World Labs RTFM | 2025.10 | 实时**学习式渲染器**:每帧带 3D pose,posed frames 本身就是空间记忆;"context juggling" 按区域检索邻近帧 → 单张 H100 上无界持久 |
| World Labs Marble | GA 2025.11.12 | 相反的赌注:生成**可下载的持久 3D 世界**(splat/mesh/video),用生成灵活性换零漂移 |
| Odyssey-2 / Pro / Max | 2025.10 / 2026.01 / **2026.04.21** | AR DiT + continuous flow matching + few-step 蒸馏;720p ~20–22fps,40–50ms/帧,120s+;Max 报 VBench-2 physics 58.52(vs 49.67) |
| Vid2World | ICLR 2026 | 把预训练**双向**视频扩散模型改造成因果 action-conditioned 世界模型的系统配方 |
| Self Forcing → Self-Forcing++ | 2506.08009 → 2510.02283 | SF:在自生成 rollout 上训练以闭合 train/test gap。SF++:从自生成长 rollout 采窗 + 反向加噪 + 短视界 teacher 做 DMD → **4min15s**,50×;并指出 **VBench 偏好过曝退化帧** |
| Rolling Forcing | 2509.25161, 2025.09 | 滚动窗口**联合**去噪(放松严格因果,让误差在定稿前被纠正)+ 初始帧 attention sink + 冻结相对 RoPE |
| Context Forcing | 2602.06028, 2026.02 | 修"长上下文带记忆的 student 从短上下文无记忆的 teacher 蒸馏"这一错配;surprisal-based context consolidation 把可用上下文推过 20s |
| Geometry Forcing | 2507.07982, 2025.07 | 把内部状态对齐到几何基础模型特征 → 无显式 3D 表示的 3D-aware latent |
| PAN | 2511.09057, 2025.11 | LLM 语义层 + 长视界可交互世界仿真 |
| WorldGen (Meta) | 2511.16825;CVPR 2026 | 文本 → **可行走** 3D 世界:LLM 布局推理 + 程序化生成 + 实例分解 + navmesh |
| Lyra 2.0 (NVIDIA) | 2604.13036, 2026.04 | 可探索生成式 3D 世界(3DGS + 点云),可部署进仿真引擎 |
| 综述:Towards Interactive Video World Modeling | 2606.01164, 2026.06 | 该方向参考综述,按 controllability / memory / responsiveness 分类;维护 Awesome-Interactive-World-Model |

**收敛后的技术共识(可直接当 method section 的 background):**
DiT 已完全胜出,且多数团队不再从零训练,而是后训一个开源视频基座(Wan 2.2 最常见)再因果化。
双向注意力与交互天然冲突(未来帧要依赖用户还没做的动作),统一解法是在交错的
(observation, action) block 上做 block-causal attention + KV cache。动作注入四种范式:
concat(主流)/ AdaLN 调制 / 相机控制渲染或检索 / 全景矩阵变换。实时性配方:few-step 蒸馏
(DMD/DMD2 或 consistency model)到 4–8 步 + 滚动窗 KV cache + INT8 + VAE 解码器剪枝/蒸馏。

**两个值得记住的工程事实:**
- Matrix-Game 3.0 的 ablation:从 ~40fps 起,去掉 GPU 端 memory retrieval 省 33fps、VAE 剪枝 14、
  INT8 12 —— 蒸馏之后 **retrieval 而非去噪成了吞吐瓶颈**。
- DiT 优化到位后,**VAE 解码成为流式瓶颈**,所以剪枝/蒸馏解码器突然到处都是。

---

## B. 记忆与长视界一致性

| 名称 | 出处 | 贡献 |
|---|---|---|
| **WorldMem** | 2504.12369;**NeurIPS 2025** | 记忆库设计的参考实现:存 memory frame + state(Plücker 位姿嵌入、时间戳),state-aware memory attention 检索;时间戳让它能建模世界**演化** |
| VMem | ICCV 2025 | **Surfel-indexed view memory**:按观察到的 surfel 索引历史视图,取与目标相机 3D 重叠最大的 K 个 |
| Video WM with Long-Term Spatial Memory (Spmem) | NeurIPS 2025 | 拆成 **spatial**(增量更新静态 point map)+ **episodic**(稀疏参考帧保动态前景身份) |
| RELIC | 2512.04040, 2025.12 | 未压缩 KV cache(近期滑窗)+ 压缩长视界 spatial cache(latent + action + 绝对相机位姿);Teacher Forcing 与 Diffusion Forcing 混合作为因果蒸馏初始化 |
| Memory Forcing | 2510.03198, 2025.10 | 训练时决定**何时**依赖记忆 vs 近期上下文 |
| Infinite-World | 2602.02393;**ICML 2026** | **Pose-free** 分层记忆压缩器,递归蒸馏历史到固定预算(次线性、显存平坦)+ 三态 uncertainty-aware 动作标注 + **30 分钟 revisit-dense 微调集**即可在 1000+ 帧激活 loop closure |
| MosaicMem | 2603.17117, 2026.03 | 显式几何 + 隐式 latent 混合记忆,论证任一单独都不够 |
| PERSIST | 2603.03482, 2026.03 | 用显式**持久 3D 状态**当记忆,而非像素历史 |
| LiveWorld | 2603.07145, 2026.03 | 少数正面处理**画外区域仍在演化**的工作之一 |
| HM-World | 2603.25716, 2026.03 | 混合记忆 + 同样关注 off-screen 演化 |
| Long-Context State-Space Video World Models | ICCV 2025 | SSM + diffusion 混合,次二次代价的长上下文 |

**载重性实证事实(做记忆方向必须知道):**
- **记忆不是越多越好。** WorldMem 自己的 ablation:1→8 memory frame 上 PSNR/LPIPS/rFID 全面改善,
  但 **16 帧全面变差**("过多记忆帧可能引入噪声或降低检索精度")。瓶颈是**检索精度而非容量**。
- WorldMem 自陈三个 limitation,其中之一是 **记忆占用线性增长**。
- Infinite-World 指出真实视频难训的两个原因:位姿估计有噪声、以及 **视角重访极其稀缺** ——
  互联网视频几乎从不回到同一视角,所以 loop closure 没有天然监督信号。
- **记忆与可控性冲突。** 综述 2606.01164 §6.1:历史观测越多长时一致性越好,
  "但同时**削弱了对动作的响应**"。Astra(ICLR 2026)干脆给 conditioning frame 加噪来模糊历史影响。
- Rolling Forcing 对这一族的评价:给历史帧加噪"缓解漂移…但**剥夺了模型的干净参考,损害时间一致性**"。

---

## C. Latent action 与 action-free 预训练

| 名称 | 出处 | 贡献 |
|---|---|---|
| LAPO | ICLR 2024 | 原始配方:联合 latent IDM + latent world model,从纯视频恢复动作空间结构 |
| LAPA | 2410.11758;ICLR 2025 | VQ-VAE latent action → VLM 预训练 → 少量真动作微调;**只用人类视频**预训练也能超 OpenVLA |
| Moto | 2412.04445;ICCV 2025 | Latent Motion Token 作为桥接语言;next-motion-token 预训练后共同微调到真动作 |
| AdaWorld | ICML 2025 | 把 latent-action-aware 变成**预训练目标**,使动作可跨上下文迁移 |
| GameFactory | ICCV 2025 | 解耦游戏风格**动作控制**与场景生成,使控制迁移到开放域 |
| **Olaf-World** | 2602.10104;**ICML 2026** | 当前对"latent action 不迁移"的最好回答:**Seq∆-REPA** 把积分后的 latent action 对齐到冻结自监督视频编码器的时序特征差 → 跨上下文**共享坐标系**;用 **~1 分钟** 标签达到 AdaWorld 用 2 小时标签的水平 |

**开放问题(综述 2606.01164 §8.1 原话意译):** 发现出的动作"往往只**弱**地锚定在真实控制语义上,
并可能与场景外观、相机运动或数据集偏差**纠缠**"。

---

## D. 决策 / MBRL 世界模型

| 名称 | 出处 | 贡献 |
|---|---|---|
| DreamerV3 | **Nature** 640:647–653, 2025.04.02 | 单一超参配置在 150+ 任务上超专用方法;首个无人类数据从零挖到 Minecraft 钻石 |
| Dreamer 4 | 2509.24527, 2025.09 | 见上;**纯离线**拿钻石,imagination RL 是关键(仅 WM+BC 为 0.0%) |
| TD-MPC2 | ICLR 2024 | decoder-free 隐式世界模型 + MPPI;104 任务单配置,扩到 317M |
| Newt + MMBench | **ICLR 2026** (2511.19584) | 200 任务在线 RL benchmark;语言条件 TD-MPC2 后继;**开放 200+ checkpoint、4000+ demo**,5M/20M/80M 变体 |
| TD-M(PC)² | L4DC 2026 | 对 TD-MPC2 加 ~10 行的 soft KL actor 约束,把 h1hand-run 上 **2159% 的价值高估打到 ≈0%** |
| **MRS.Q**:The Surprising Difficulty of Search in MBRL | **ICML 2026** (2601.21306) | 反共识:**即使模型几乎完美,search 也可能损害性能**;瓶颈是 planner/critic 策略错配导致的高估偏差,而非 compounding error |
| DIAMOND | NeurIPS 2024 Spotlight | 像素空间扩散世界模型,Atari100k mean HNS 1.46 |
| Δ-IRIS | ICML 2024 | tokenize 帧间**差分**,比 IRIS 快 10× |
| EMERALD | ICML 2025 (2507.04075) | MaskGIT prior over 空间 latent;**首个在 Crafter 上超人类专家**(58.1% vs 50.5%),30M 参数 |
| Simulus | 2502.11537 | "world model 的 Rainbow";首个 planning-free 在 Atari100k median 与 IQM 双达人类水平 |
| DyMoDreamer | NeurIPS 2025 (2509.24804) | 帧间差分掩码把运动线索注入 RSSM;Atari100k 1.566 |
| EDELINE | 2502.00466 | Mamba SSM 替换扩散世界模型的固定上下文窗;Atari100k 1.87 mean / 0.94 IQM |
| JEDI | 2605.13013, 2026.05 | 首个端到端 **latent** 扩散世界模型(latent 由去噪损失在 JEPA 框架下学出);比 DIAMOND 省 43% 显存、采样快 3× |
| Horizon Imagination | ICLR 2026 (2602.08032) | 并行去噪多个未来观测,半步预算下保持控制性能 |
| Mixture-of-World Models | ICLR 2026 (2602.01270) | 任务条件 MoE 动力学 + 梯度式任务聚类;**单**智能体 26 游戏 110.4% mean HNS,参数减半 |
| RLVR-World | **NeurIPS 2025** (2505.13934) | 用可验证奖励在**解码后的预测指标**上后训世界模型,直接打 MLE/任务目标错配 |
| WMPO | **ICLR 2026** (2511.09515) | 在**像素空间**世界模型里对 VLA 做 on-policy GRPO(刻意不用 latent 以匹配 VLA 预训练);真机 53%→70% |

**SOTA 数字(注意各文 game 数、seed 数、mean/median/IQM 口径不一,慎比):**
- Atari 100k mean HNS:IRIS 1.046 → DreamerV3 ~1.10 → Δ-IRIS 1.39 → DIAMOND 1.46 →
  DyMoDreamer 1.566 → EDELINE 1.87 → EASimulus 1.818(声称首个 MBRL 超人 IQM,**未核实**)。
  但 **model-free 的 BBF 的 IQM 2.25 至今未被世界模型方法在 IQM 上可信超越**。Atari100k 作为研究信号已接近饱和。
- Crafter:human expert 50.5% / DreamerV3-XL 39.6% / Δ-IRIS 42.5% / **EMERALD 58.1%**。
- Minecraft(Dreamer 4 Table 7,1000 episode,60min,原始像素+低级键鼠+真 GUI 合成):
  iron pickaxe VLA 11.2 / WM+BC 16.9 / **Dreamer 4 29.0**;**diamond 0.0 / 0.0 / 0.7**。
  成功回合平均 20.7 分钟拿到钻石。
- 世界模型保真度(Dreamer 4 Table 1,人在模型里玩 16 个脚本交互任务):
  Lucid-v1 0/16、Oasis(small) 0/16、Oasis(large) 5/16、**Dreamer 4 14/16**,且上下文 9.6s(前作 6×)。
  **注意:2025 年最好的前沿世界模型保真度指标,是"一个人拿键盘进去玩"。**

---

## E. JEPA / 隐预测世界模型

| 名称 | 出处 | 贡献 |
|---|---|---|
| V-JEPA 2 / 2-AC | 2506.09985, 2025.06 | 1M+ 小时视频 action-free 预训练;300M action-conditioned predictor 仅用 **<62h 无标注 DROID** 后训 → CEM/MPC 零样本 Franka 抓放,两个陌生实验室 |
| V-JEPA 2.1 | 2603.14482, 2026.03 | dense predictive loss + 深度自监督 + 多模态 tokenizer;**抓取成功率 +20%**,导航规划快 10×;80M/300M/1B/2B 全部放出 |
| DINO-WM | ICML 2025 (2411.04983) | 在冻结 DINOv2 特征上学 latent dynamics;task-agnostic、reward-free、零样本 goal-reaching |
| Intuitive physics emerges from SSL on natural videos | 2502.11831, 2025.02 | **V-JEPA 在 IntPhys 上零样本 98%、InfLevel 62%;像素预测模型与 MLLM 接近随机。** "在表征空间预测"这一立场最强的实证支持 |
| LeJEPA | 2511.08544, 2025.11 (Balestriero & LeCun) | 证明各向同性高斯是最优嵌入分布;SIGReg 用 ~50 行替掉 stop-grad/EMA/teacher-student,只剩一个超参;**训练损失与下游性能相关 85–99% → 可无标签模型选择** |
| LeWorldModel (LeWM) | 2603.19312, 2026.03 | 首个从像素端到端稳定训练的 JEPA,仅 2 个损失项;**15M 参数、单 GPU 几小时**,规划比基础模型式世界模型快 48× |
| TD-JEPA | **ICLR 2026 Oral** (2510.00739) | 在 latent 上做 TD → 恢复长期策略动力学的低秩分解 → zero-shot reward optimization,65 任务 |
| When Does LeJEPA Learn a World Model? | 2605.26379, 2026.05 | 证明 LeJEPA 达到 latent 的**线性可辨识性**,且高斯是唯一使之成立的分布 |
| Delta-JEPA | 2606.31232, 2026.06 | 从 latent **位移**而非拼接端点解码执行的动作 → 无需 VICReg/SIGReg 也抗塌缩且 action-sensitive |
| Critiques of World Models (Xing et al., GLP/PAN) | 2507.05169 | decoder-free JEPA 的主要反方:有损不可逆 encoder 不能保证**充分统计量**;decoder 是对"静默无状态化"的诊断工具 |

**LeCun 动向(仅媒体报道,无技术出版物):** 2025.11.19 宣布离开 Meta,在巴黎创立 AMI Labs;
2026.03.10 宣布 $1.03B 种子轮 / $3.5B pre-money。明确是押 JEPA / world model 反 LLM scaling。
他 2026.01 对 MIT Tech Review 说:"JEPA 不是生成式 AI…关键是学到世界的抽象表征并在那个抽象空间里预测,
忽略你无法预测的细节",并且说 **"目前最令人兴奋的工作来自学术界,而非困在 LLM 世界里的大型工业实验室"** ——
这句话对下面的小实验室定位很有用。

---

## F. 具身 / VLA + 世界模型(World-Action Model, WAM)

| 名称 | 出处 | 贡献 |
|---|---|---|
| UniPi / UniSim / RoboDreamer | NeurIPS 2023 / ICLR 2024 Outstanding / ICML 2024 | 奠基三篇:video-as-policy、可交互真实世界模拟器、组合式世界建模 |
| iVideoGPT | 2405.15223, 2024.05 | 压缩式 tokenization + 在 (obs, action, reward) token 上自回归 |
| EnerVerse | 2501.01895;NeurIPS 2025 | chunk-wise 自回归视频扩散 + 稀疏记忆 + Free Anchor Views;EnerVerse-D 配 4DGS 形成 sim2real 飞轮;单 4090 上 8 步 chunk ~280ms |
| GR00T N1 | 2503.14734, 2025.03 | 开源双系统人形 VLA(VLM@10Hz + DiT flow-matching head@120Hz) |
| **DreamGen** | 2505.12705, 2025.05 | 经典"梦数据"流水线:视频 WM 微调到目标本体 → 生成 → IDM/latent action 打标 → 训策略 |
| EnerVerse-AC | 2505.09723, 2025.05 | action-conditional 多视角 WM,同时做数据引擎与策略评估器;**刻意用失败轨迹训练** |
| WorldVLA | 2506.21539, 2025.06 | 单一自回归模型统一 action + 图像理解/生成;action attention masking 阻断 chunk 内误差传播 |
| RoboScape | 2506.23135;**NeurIPS 2025 spotlight** | 物理信息化:联合 RGB + 时序深度 + 关键点动力学;5 万 AgiBot 片段,32×A800 ~24h |
| Genie Envisioner | 2508.05635;**ICLR 2026** | 一个视频扩散 latent 空间同时当策略主干、神经仿真器和 benchmark(EWMBench) |
| WoW | 2509.22642, 2025.09 | 14B,**200 万真实机器人轨迹**(5275 任务 / 12 机器人);SOPHIA VLM critic 约束幻觉;共训 IDM |
| **Ctrl-World** | 2510.10125;**ICLR 2026** | DROID 上多视角(含腕部)action-conditioned WM + 位姿条件记忆检索;把 π₀.₅-DROID 新指令成功率 **38.7% → 83.4%** |
| World-in-World | 2510.18135, 2025.10 | 首个按**任务成功率**排名异质世界模型的闭环平台 |
| GigaBrain-0 / GigaWorld-0 | 2510.19430 / 2511.19861 | WM-as-data-engine 全栈:视频分支 + 3D 分支(3DGS + 可微系统辨识) |
| RynnVLA-002 | 2511.17502, 2025.11 | 共享词表 over image/text/state/action + 连续 Action Transformer head;LIBERO 97.4 **无需预训练**;真机 SO-100 因 WM 提升 50 点 |
| **Cosmos Policy** | 2601.16163, 2026.01 | "latent frame injection":把 action、future state、**value** 当额外 latent frame 塞进 Cosmos-Predict2,**不改架构**。LIBERO 98.5,真机 ALOHA 93.6 |
| DreamDojo | 2602.06949;**ICML 2026** | 在 **44k 小时第一人称人类视频**上用连续 latent action 当代理标签预训练;蒸馏到 ~10.8 FPS |
| **DreamZero** | 2602.15922, 2026.02 | 14B Wan-2.1 主干联合去噪 video + action;**7Hz 闭环**,泛化 >2× SOTA VLA;10–20 分钟**纯视频**跨本体演示带来 +42% 相对提升 |
| GR00T N2 | NVIDIA GTC 2026 | 基于 DreamZero 的 WAM。**无论文,厂商声明** |
| Fast-WAM | 2603.16666(经综述) | 唱反调:收益可能主要来自**训练时的 video co-training**,而非测试时 imagination —— 后者可在推理时丢掉 |
| Aether | **ICCV 2025** (2503.18945) | 几何锚定地统一 4D 重建 + action-conditioned 预测 + goal-conditioned 规划,**用相机轨迹当动作空间**;纯合成训练零样本到真实 |
| TesserAct | 2504.20995, 2025.04 | 4D WM 联合预测 RGB + depth + normal |
| WorldGrow | 2510.21682;**AAAI 2026 Oral** | 无界**显式** 3D 场景生成(block-wise 3D inpainting + 粗到细) |

**真机上被验证有效的(vs demo-only):**
- **生成数据确实能训真策略,效果大但不神奇。** DreamGen 最干净:9 个真实任务、每任务仅 10–13 条真实
  轨迹,GR-1 37%→46.4%、Franka 23%→37%、SO-100 21%→45.5%。真正惊人的是从 0 到 1 那些:
  只训 pick-and-place 的 GR00T N1 在多数新动词上 **0%**,DreamGen 训后新行为(已见环境)43.2%、
  新行为+未见环境 28.5%。RoboCasa 里把神经轨迹放大 333× 得到对数线性改善。
- **在 WM 生成的成功轨迹上微调有效,但只对指令跟随。** Ctrl-World 38.7%→83.4%,作者自己划界:
  "我们预期我们的模型不够准确,无法改善其他方面,比如已见指令上的底层成功率。"
- **在世界模型里做 RL 在真机难任务上有效。** WMPO,Mobile ALOHA 5mm 插入:53% base → 60% offline DPO → **70% WMPO**(n=30,小样本但是真闭环)。
- **latent planning 零样本可行(简单任务)。** V-JEPA 2-AC:reach 100%、grasp cup 60%、pick-place cup 80%,
  对照 Octo 10%/0%/10%。最强的"像素生成非必需"证据,但任务简单且 goal 必须是图像。
- **一个被低估的负面结果:** RynnVLA-002 的**离散**动作变体 —— 也就是 LIBERO 上分数好看的那个 ——
  **在真实 SO-100 臂上完全失败**(过拟合 + 轨迹不连续)。
- **LIBERO 已饱和,不应用来排名世界模型:** Cosmos Policy 98.5、LingBot-VA 98.5、Say-Dream-ACT 98.1、
  Motus 97.7、RynnVLA-002 97.4、VLA-JEPA 97.2 —— 架构完全不同却全在噪声内;
  综述 2605.00080 还指出跨 benchmark 迁移很差(RoboTwin 强不预示 CALVIN 或 SIMPLER 强)。

---

## G. 自动驾驶世界模型

| 名称 | 出处 | 贡献 |
|---|---|---|
| GAIA-1 / GAIA-2 (Wayve) | 2309.17080 / 2503.20523 | 驾驶 WM 作为多模态 next-token;GAIA-2 为 latent diffusion 环视,5 相机 448×960,结构化条件(自车动力学/他车/天气/道路语义),英美德三国 |
| Copilot4D (Waabi) | ICLR 2024 | VQ-VAE LiDAR tokenization + 离散扩散做 4D 点云预测;1s Chamfer 降 >65% |
| Vista | **NeurIPS 2024** (2405.17398) | 可泛化 10Hz/576×1024 驾驶 WM,统一控制接口;**首个把 WM 自身当 reward** 评估动作而不需真动作标签 |
| OccSora / OccWorld | 2405.20337 等 | 4D occupancy 世界模型,DiT over 4D scene tokenizer,16s 轨迹条件占据预测 |
| Doe-1 | 2412.09627, 2024.12 | 感知+预测+规划统一为 observation/description/action token 的 next-token 问题 |
| DriveDreamer-2 | **AAAI 2025** | LLM 驱动的用户指定场景生成;FID 11.2 / FVD 55.7;改善 3D 检测与跟踪 |
| DrivingSphere | **CVPR 2025** | 闭环仿真:4D occupancy 世界(OccDreamer)渲染到多视角视频(VideoDreamer)—— occupancy 当基底、video 当传感器模型 |
| Cosmos-Drive-Dreams | 2506.09042, 2025.06 | 开放流水线 + 权重 + **81,802 片段**;**即使叠加在大规模真实数据之上仍有可测增益** |
| **DriveVLA-W0** | 2510.12796;**ICLR 2026** | 未来图像预测作为稠密自监督**放大数据 scaling law**:7000 万帧时相比纯动作监督 ADE +28.8%、碰撞 −15.9% |
| WorldLens | 2512.10958;**CVPR 2026 Oral** | 五轴 24 维 benchmark + 26K 人工标注 + 蒸馏 VLM critic。**"几乎所有现有世界模型都会触发碰撞或偏离道路"** |
| Waymo World Model | Waymo, 2026.02 | Genie 3 为驾驶后训,联合输出时序一致的 **camera + lidar**,把行车记录仪视频变成多模态仿真 log。仅公告 |
| DriveDreamer-Policy | 2604.01765, 2026 | 几何锚定驾驶 WAM(depth + future video + planning);**NAVSIM v1 PDMS 89.2、v2 EPDMS 88.7** |
| **NVIDIA OmniDreams** | 2606.03159, 2026.06 | 文档最完整的闭环驾驶 WM:2.1 万小时训练,**单 GB300 上单相机 720p 68FPS / 16 GPU 四视角每相机 105FPS**,接入 AlpaSim + Alpamayo 策略 |
| World Engine (OpenDriveLab) | 2606.19836, 2026.06 | 失败挖掘 → 3DGS 重建 → 行为增强 → RL 后训。生产栈上**碰撞 −45.5%**、200km 无接管。**nuPlan 上完全开源** |
| Tesla neural world simulator | 讲座(ICCV 2025, ScaledML 2026) | 8 相机 36fps 5MP >1min;同网络泛化到 Optimus 室内。**无论文无数字** |

**最有用的两个结论:**
- **OmniDreams 保住了策略排序,且随策略偏离记录轨迹的退化远比 3DGS 重建优雅** ——
  NuRec 的 FVD 一离开采集路径就快速上升,OmniDreams 保持稳定。这是"生成式仿真优于重建式仿真"
  这一整套论证的核心,而且这里是**被测量出来的而非断言的**。
- **驾驶里最不争议的胜利是数据引擎**,以及一个 sleeper:DriveVLA-W0 发现未来图像预测**放大**数据
  scaling law(收益随数据加速,而纯动作监督饱和)—— 这比任何仿真器结果都重要,因为它正好
  作用在工业界花钱的地方。

**评测基础设施:** NAVSIM / v2(NeurIPS 2024 D&B;v2 2025.02.28,PDMS→EPDMS)、
Bench2Drive(CARLA 闭环,220 route)、以及 **NAVSIM↔Bench2Drive 相关性研究**(2605.00066):
ρ=0.90 但**非单调、有明显排序反转**;Ego Progress 是最强单一预测因子(ρ=0.83),
碰撞指标弱得多(ρ=0.45);**n 只有 8 对方法**。

---

## H. LLM / code 作为世界模型

| 名称 | 出处 | 贡献 |
|---|---|---|
| WebDreamer | **TMLR 2025** (2411.06559) | LLM 用自然语言模拟"点这个会怎样";VWA +34.1%、Online-Mind2Web +42.3%;比 tree search 高效 4–5× |
| PoE-World | **NeurIPS 2025** | LLM 合成的世界模型 = 一堆小程序的指数加权**乘积**;少量观测学随机动力学,泛化到未见 Montezuma 关卡 |
| Code World Models for General Game Playing | **ICLR 2026** | 把自然语言规则编译成可执行 Python(OpenSpiel)供 MCTS 使用 |
| From Word to World | **ACL 2026** long (2512.18832) | 辩论中最强的**正方**:dynamics-aligned SFT 下 LLM WM 可预测地 scaling;WebShop 上动作验证给 GPT-4o +5.5%,SciWorld 上 warm-start RL +15% |
| Code2World | 2602.09856, 2026.02 | 通过生成 HTML 再**渲染**来预测下一帧 GUI 截图;AndroidWorld 导航 +9.5% |
| CUWM (Microsoft) | 2602.17365, 2026.02 | 两阶段 UI 动力学(文本 Δ描述 → 图像编辑),冻结 agent 上做测试时动作搜索 |
| Bridging the Agent-World Gap(综述) | 2606.09032, 2026.06 | 目前最好的 LLM-as-WM vs code-as-WM 分类 |
| Do LLMs Build Spatial World Models? | ICLR 2026 workshop (2604.10690) | 互补的**负面**结果:Gemini-2.5-Flash 在邻接表表示的 5×5–7×7 迷宫上 80–86%,**视觉网格上只有 16–34%** —— 同一任务 2–5× 摆动;且"尽管推理轨迹语义覆盖率 96–99%,模型仍无法把这种理解用于一致的空间计算,说明它把每个问题独立处理,而非建立累积的空间知识" |

---

## I. 评测、benchmark、物理合理性

| 名称 | 出处 | 测什么 / 头条数字 |
|---|---|---|
| **Physics-IQ** | 2501.09038;**WACV 2026** | 对拍摄真值的真实世界视频续写,66 场景×3 视角×2 次 = 396 视频;real-vs-real 方差归一为 100%。发布时最好 VideoPoet(multiframe)**29.5**,Sora(i2v)10.0。**视觉真实度与物理不相关:r = −0.46, p = .249** |
| **Physics-IQ Verified** | **2606.18943, 2026.06** | 对上者的审计:修正 **57.6% 的样本、34.8% 的 prompt**;**29.8% 的真值视频有 artifact**。仅协议改动就把排名打乱到 **Kendall τ = 0.46 / Spearman ρ = 0.65**。并记录 **Sora 2 在 2026.04 的分数低于 2025.10**(静默模型漂移) |
| VideoPhy / VideoPhy-2 | ICLR 2025 (2406.03520) / **ICLR 2026** (2503.06800) | 688 人工核验 caption;后者 197 动作 → 3940 prompt。Dream Machine 语义 61.9% vs 物理 21.8%;模型在**质量与动量守恒**上最差;闭源模型**并未**优于最好的开源模型 |
| PhyGenBench / PhyGenEval | **ICML 2025** | 160 prompt / 27 物理定律。**scaling 与 prompt engineering 都修不好动态物理现象** |
| PhysBench | **ICLR 2025 Oral** | 测**理解**而非生成:10,002 图文视频交错题,75 个 VLM 全部与常识推理有大差距 |
| IPV-Bench(Impossible Videos) | **ICML 2025** | 反向任务:能否生成/理解物理上**不可能**的视频。最好生成器 Mochi 1 **37.3%**。失败模式:artifact,或者悄悄"纠正"回正常场景 |
| Morpheus | **ICML 2026** (2504.02918) | 用 PINN + SAM2 跟踪按**守恒律**评分而非像素匹配。模型生成审美好但非保守的视频 |
| **PhyWorld** | **ICML 2025** (2411.02385) | 受控 2D 力学:**分布内完美泛化、组合泛化随 scaling 改善、OOD 彻底失败**。机制是 **case-based** 泛化,参考优先级 **color > size > velocity > shape** |
| PhyGround + PhyJudge-9B | 2605.10806, 2026.05 | 至今质量控制最好的人评:**459 标注者、5796 次标注、37.4K 标签**,split-half ρ>0.90。**PhyJudge-9B 聚合相对偏差 3.3%,Gemini-3.1-Pro 16.6%** |
| WorldModelBench | **NeurIPS 2025 D&B** | 350 prompt、7 领域;**67K 人工标签**,14 个前沿模型;2B 微调 judger 超 GPT-4o |
| **WorldScore** | **ICCV 2025** (2504.00983) | 统一 3D/4D/I2V/T2V 为带显式相机轨迹的序贯下一场景生成。**视频模型不会相机控制**:最好的 CogVideoX-T2V 只有 **40.22**,低于每一个 3D 方法(WonderWorld 72.69) |
| WorldSimBench | **ICML 2025** | 双框架:显式感知评测 + **隐式操控评测**(生成的视频能否被转成正确控制信号) |
| EWMBench | **BMVC 2025** | AgiBot World 上的具身 WM:场景一致性、运动正确性、语义对齐;对称 Hausdorff + 归一化 DTW,voxel 化以容许多条合法轨迹 |
| **ACT-Bench + Terra** | ICLR 2025 WM workshop (2412.05337) | 驾驶 WM 的开源动作保真度。**SOTA 的 Vista 在 FID/FVD 上赢,但指令遵循比 Terra 差**;Terra v2 把指令执行一致性翻倍以上 |
| **WorldPrediction** | 2506.04363, 2025.06 | 高层语义世界建模(POSMDP 基础,用 action equivalents 阻断低层连续性捷径)。**前沿模型 57.0% (WM) / 38.1% (PP),人类全对;视频扩散世界模型 ~30%(接近随机)** |
| MME-CoF | 2510.26802, 2025.10 | 对"视频模型是零样本推理者"的直接实证反驳:短程空间连贯尚可,**长程因果一致性、几何约束、抽象逻辑失败**。判词:"pattern-driven, not principle-driven" |
| **What-If World** | **2605.27589, 2026.05** | **因果/干预**:319 对比 prompt **对**(nuScenes + DROID 真实帧),APEO rubric 单视频与成对两种模式。最好 Grok Imagine **51.7% 成对 APEO**。命名 **"contrastive bottleneck"**:模型通过每一项单视频检查,却对相反干预产出**几乎相同的**轨迹 |
| Omni-WorldBench | 2603.22212, 2026.03 | 首个交互中心 benchmark,1068 prompt / 3 交互层级。最好 Wan2.2 AgenticScore 75.92%。**时序闪烁与运动平滑度对几乎所有模型已饱和(>95%)—— 视觉指标不再有区分力**,交互保真度才有 |
| MIND | 2602.08025, 2026.02 | 闭环**重访轨迹** benchmark(记忆一致性 + 动作控制),250 视频 1080p/24fps,第一/第三人称共享动作空间。**记忆在 action-space shift 下有害**;记录到 Matrix-Game-2.0 被要求左移时完全不动 |
| WorldRoamBench / WorldOdysseyBench | 2606.31672, 2026.06 | 长视界稳定性四轴,**controllability-gated** 物理评分。**10+ 开闭源模型中没有一个满足全部四维**;并论证轨迹级动作指标与首尾漂移对比会掩盖中段崩塌 |
| MBench | 2606.00793, 2026.06 | 把记忆当作世界模型的定义性能力,实体/环境/因果一致性 → 12 子维。续写类模型聚合分**只在 43–47 区间**。结论:**"causal memory 是核心未解难题"** —— 很多 action-conditioned 模型保住了外观,但世界不演化 |
| WorldArena | 2602.08971, 2026.02 | 感知**与功能**双评:16 视频指标 + WM 当(a)数据引擎(b)策略评估器(c)闭环规划器。头条:可测量的 **perception–functionality gap** |
| WorldGym | 2506.00613, 2025.06 | WM **当环境**做策略评估:模型内成功率与真实成功率高度相关,**跨版本/规模/checkpoint 保持相对排序** |
| **WMBench / GigaWorld-1** | 2607.02642, 2026.07 | 什么让 WM 成为好的**策略评估器**:7 个 WM、4 种动作编码、**32.4 万+ 仿真 rollout**。核心结论:**"评估器质量由长视界、动作保真的 rollout 一致性主导,而非短期视觉真实度"** |
| WorldMark | 2604.21686, 2026.04 | 首个可对齐比较的交互 benchmark:**统一 WASD+L/R 动作词表** + 每模型适配器,500 case,20–60s 分档 |
| WMReward | **CVPR 2026** (2601.10553) | 把 V-JEPA-2 的 "surprise" 当推理时物理 reward 做 Best-of-N + 梯度引导。**62.64%,ICCV 2025 PhysicsIQ Challenge 第一,+7.42pp**。含义重大:**物理失败很大一部分是采样/推理问题,不是知识问题** |
| On the Content Bias in FVD | **CVPR 2024** (2404.12391) | **FVD 对时间质量基本不敏感**:更严重时序退化的集合得 310.52,更轻的得 317.10;对无运动生成视频重采样可**把 FVD 砍半**。原因是 I3D/K400 特征偏外观;VideoMAE-v2 自监督特征基本能修 |

**当前模型可测量地失败在什么地方(带数字):**
- Physics-IQ 原协议 v2v 最好从 VideoPoet 29.5(2025.02)升到 Magi-1 24B + BoN 64.5(2026.06),
  但**未增强单样本**条目低得多。在更严格的 **Verified 协议**下整个领域被压缩:
  Cosmos3-Super-I2V 39.5±0.8、Grok Imagine 34.8±0.6、Sora 2 26.5±0.8 ——
  相对 100% 的真实视频方差上限,**最好的公开 i2v 模型只复现了不到 40% 的真实物理行为**。
  且 BoN 重排序值 **+7~+25 分**,意味着相当一部分"物理理解"其实是在固定生成流形上做搜索。
- **视觉真实度与物理正确性解耦,已被三种独立方式证明**:Physics-IQ 的 r=−0.46(p=.249);
  VideoPhy 的 Dream Machine 语义 61.9% vs 物理 21.8%;Omni-WorldBench 的时序平滑度 >95% 饱和
  而交互保真度仍差几十分。WorldArena 把同一现象结构化命名为 perception–functionality gap。
- **可控性是比合理性更严重的独立失败**:WorldScore 上最好的视频模型相机控制 40.22,
  而每一个 3D 场景生成方法都更高(数个 >84);ACT-Bench 上 Vista FID/FVD 领先但指令遵循更差;
  MIND 记录到模型被要求左移时干脆不动。
- **因果/干预敏感性接近失效**:What-If World 最好 51.7% 成对 APEO,而 contrastive bottleneck
  意味着"轻踩刹车"和"重踩刹车"生成的是**同一段视频**。因为此前所有 benchmark 都是单视频打分,
  这个失败在 2026 年之前是不可见的。
- **机制是 case-based 检索而非定律抽象**:PhyWorld 的 OOD 彻底失败 + 参考优先级
  color > size > velocity > shape;Vafa et al. 的模型在轨道数据上训练后每个数据切片给出**不同且荒谬的**引力定律。

---

## J. 理论与批评

| 名称 | 出处 | 贡献 |
|---|---|---|
| **General agents need world models** | **ICML 2025**, PMLR 267:51659–51687 (2506.01622) | 任何在深度 N 目标上有界 regret 的 agent **必然**学到了一个 transition model,且可**仅从其策略**恢复,误差 ≤ (2σ/((n−1)(1−δ)))^½ |
| **Evaluating the World Model Implicit in a Generative Model** | **NeurIPS 2024 Spotlight** (Vafa et al.) | 用 Myhill-Nerode 启发的两个指标(sequence compression / distinction)形式化世界模型恢复。Transformer 产出合法 next-turn 近 **100%**,状态表征看似编码了位置,但**重建出的街道图与曼哈顿几乎无关**(含不存在的街道);绕路时性能崩塌 |
| **What Has a Foundation Model Found?** | **ICML 2025** (2507.06952) | Inductive bias probe(R-IB / D-IB)。轨道模型微调去预测力向量时给出**每个数据切片都不同的荒谬引力定律**;Othello/lattice 模型的归纳偏置指向**合法下一 token 集合而非棋盘状态** |
| Objective Mismatch in MBRL | L4DC 2020 (2002.04523) | 奠基性陈述:**验证损失与 episode reward 不可靠相关**;全局准确的模型不必在要紧的地方局部准确 |
| A Unified View on Solving Objective Mismatch | 2310.06253 | decision-aware MBRL 的四分类法(distribution correction / control-as-inference / value equivalence / differentiable planning) |
| **Imperfect World Models are Exploitable** | 2605.15960, 2026.05 | 首个世界模型的**序**式安全概念;证明在大策略集上**利用基本不可避免**,并导出 "safe horizon" |
| **On the Identifiability of Controlled World Models** | 2607.22430, 2026.07 | action-conditioned latent 预测在 spectral separation + **conditional action excitation** 下联合辨识表征(至正交变换)与转移;**state-only 预测则不能**。关键推论:**弱 conditional action excitation 会让"准确的 on-policy 预测"与"大得多的 counterfactual 误差"共存** |
| When Does LeJEPA Learn a World Model? | 2605.26379, 2026.05 | 见 §E。**Reacher 上同一物理系统在 OU 噪声采样下可辨识、在 goal-directed policy 下不可辨识** |
| Critiques of World Models | 2507.05169 | 见 §E。反 decoder-free JEPA |
| Beyond World Models: Rethinking Understanding in AI | **AAAI 2026** (2511.12239) | 科学哲学式批评:world-model 框架不足以刻画"理解"。**注意:不是 Kambhampati 的文章** |

**三极立场(不是两极):**
- **LeCun / JEPA**:生成式像素预测是错的目标,预测必须发生在学到的抽象空间。最强实证支持是
  V-JEPA 的 IntPhys 零样本 98% vs 像素模型接近随机。
- **Xing et al. / GLP-PAN**:反过来 —— 纯 latent 预测缺乏 grounding,应保留 decoder 作为监督与诊断。
- **Sutton / OaK**:与前两者正交 —— 业界把 "world model" 塌缩成了"低层物理仿真器",
  而 agent 需要的是由大量关于 option 的预测(GVF)组成的**高层**转移模型,从第一人称经验持续学习。
  **在这一立场下,上表几乎所有 benchmark 都在测错的东西。**

**而实证文献已经悄悄裁决了一个版本的争论:** 序列预测确实**没有**诱导出所假设的世界模型
(Vafa ×2),scaling **修不好** OOD 物理(PhyWorld),而当前最好的物理分数来自
**用一个 latent world model 当 critic 去搜索生成器的流形**(WMReward),而非更好的生成式预训练。

---

## 验证注意事项

**高置信(已由主源确认):** Nature DOI(DreamerV3)、PMLR 页(Richens、Vafa)、
ACL Anthology(From Word to World)、以及 2509.24527、2506.09985、2603.14482、2511.09515、
2411.06559、2507.06952、2506.01622、2502.11537、2602.09856、2602.17365、2512.18832、
2511.19584、2510.00739、2605.13013、2605.26379、2607.22430、2605.15960、2511.08544、
2603.19312、2602.01270、2606.31232、2606.09032、2601.21306、2501.09038、2606.18943、
2504.12369、2506.18701、2508.13009、2604.08995、2506.17201、2512.14614、2507.17744、
2512.22096、2506.08009、2510.02283、2509.25161、2507.07982、2602.06028、2602.02393、
2602.10104、2512.04040、2510.03198。

**需要核实后再引用:**
- Genie 1 的 arXiv ID(2402.15391)未在检索结果中直接看到 —— 保险起见按 "Bruce et al., ICML 2024" 引。
- **Genie 3 无任何架构细节公开。** 网上流传的 spatiotemporal VQ-VAE / 8 维 latent action token /
  decoder-only transformer 说法来自第三方访谈转述;11B 参数量来自二手博客。**全部按未确认处理。**
- 来自综述 2605.00080 参考文献表而非原文的 ID:Fast-WAM (2603.16666)、WorldArena (2602.08971)、
  LeWorldModel 的部分元信息、GigaWorld-Policy (2603.17240)、Motus (2512.13030)、
  V-JEPA 2.1 的部分数字、VLA-JEPA (2602.10098)、JEPA-VLA (2602.11832)。
- 未独立核实存在性:Learning Latent Action World Models In The Wild (2601.05230, 声称 ICML 2026)、
  Causal-JEPA (2602.11389, 声称 ICML 2026)、EASimulus (2601.19336,声称 Atari100k 1.818 与
  "首个 MBRL 超人 IQM" —— 这是本清单里最强的单项 Atari 声明,**务必先核实**)。
- **STORM 的 Atari100k 数字有争议**:原文 126.7% mean HNS,MoW 复现报 114.2%。
  引用 126.7% 时必须写 "as reported by the authors"。
- **Richens et al. 的标题与作者数跨源不一致**:PMLR 列 3 作者,arXiv/paperswithcode 列 4;
  且一次抓取中页面标题是 "General agents **contain** world models" 而摘要里是 "need"。
  疑为后续修订改题,引用前查当前版本。
- **同一 arXiv ID 两个标题**:2606.31672 的 v2 是 "WorldOdysseyBench",DOI 落地页写 "WorldRoamBench"。引 ID 并注明改名。
- **同一 benchmark 跨版本数字差异巨大**:VideoPhy-2 的 arXiv v1(2025.03)hard subset 最好 22%,
  camera-ready 是 Wan2.2-27B-A14B 55.4% full / 47.7% hard。这是**不同模型池 + 一年进展**,
  不是勘误,**不可互换引用**。类似地 Morpheus 的样本数在 v1/ICML/HF 三处是 80/130/124,
  WISA 数据集从 32K 变 80K,WorldModelBench 的 judger 对比在 arXiv 与 camera-ready 是
  "+8.6% accuracy" vs "−9.9% error" 两种表述。
- **Dreamer 4 无同行评审 venue**;截至 2026.08.04 未发现 Dreamer 5 或其直接后继。
- 未验证的厂商声明:GR00T N2(无论文,且 GTC 日期在二手报道中有 2026.03 与 2026.06 两说)、
  Tesla 与 Waymo 的世界模型(仅讲座/博客,无数字无协议)、Waabi 的闭环安全验证(媒体报道多于论文)。
- AMI Labs 的融资数字来自三家媒体一致报道,非技术出版物;AMI 至今无论文。
- **ICLR 2026 World Models workshop 接收了 94 篇**,大部分未逐篇查阅 —— 若你的子问题很窄,
  那份列表是下一步信息密度最高的地方。
