# Mem–Ctrl Pareto：冻结 backbone 实验设计（草稿）

> 目标：把合成 Mem v3 的结论迁到**真实开源交互视频 WM**上，画出 consistency ↔ controllability 前沿。  
> 前置：CFS 暂无候选权重；本设计在权重到位后即可开跑。

## 1. 候选 backbone（按可得性格）

| 优先级 | 模型 | 规模 | 备注 |
|--------|------|------|------|
| A | Matrix-Game 2.0 | ~1.3B | 开源、动作条件清晰 |
| B | Yume-1.5 / Infinite-World | ~5B | 记忆相关论文常用 |
| C | Cosmos-Predict2.5-2B | 2B | 若权重可得 |
| 对照 | Wan2.2-TI2V-5B | 5B | **无 action**；只作「非条件视频」负对照，不进 Pareto 主图 |

## 2. 可调旋钮（连续）

1. **context length** \(L\)：条件帧数 / 记忆 token 数  
2. **conditioning noise** \(\sigma\)：Astra 式，对历史帧加噪  
3. **retrieval top-K**（若模型有显式记忆库）  
4. **memory-attention gain** \(g\)：缩放记忆支路（需改 forward 或 LoRA 钩子）

扫描网格示例：\(L\in\{1,2,4,8\}\)，\(\sigma\in\{0,0.1,0.3,0.5\}\)，每点 2 seed。

## 3. 指标（与合成对齐）

**Controllability（主）**
- frozen-memory 协议不可用时的近似：成对动作干预（镜像 / 缩放 / 正交）下的  
  `action-response gain`、`self_cf_gap`、相对 oracle/IDM 的 PV（有真动作数据时）

**Consistency**
- WorldMark / 短视界 PSNR·LPIPS 于 on-policy rollout  
- 长视界漂移：固定动作重复下的帧间一致性

**Pareto 图**
- x = controllability（越高越好）  
- y = consistency（越高越好）  
- 每个 \((L,\sigma)\) 一个点；学到的 gate / 推荐配置标星

## 4. 数据

- 驾驶：nuScenes / NAVSIM 短片段（真动作）  
- 机器人：Bridge / DROID 子集  
- 游戏：若用 Matrix-Game，跟其官方评测动作空间

## 5. 成功标准（论文可讲）

1. 前沿非退化：增大 \(L\) 时 consistency↑ 且 controllability↓（复现综述叙事）  
2. 至少一个 gating / \(\sigma\) 配方 **支配** 朴素加长 context 的曲线段  
3. CAF 与下游 proxy（短视界动作误差）相关高于 FVD

## 6. 算力

- 推理为主：单卡 A800 可扫小网格；8 卡并行 seed×旋钮  
- 不训练 14B；最多轻量 LoRA 挂钩子 \(g\)

## 7. 阻塞与下一步

- **阻塞：** 集群无 A/B/C 权重  
- **下一步：** 外取 Matrix-Game 1.3B 或 Cosmos-2B 到 `/mnt/group/jxdong/wm_exp/ckpts/`；失败则本设计保持为投稿计划正文的 Methods 骨架
