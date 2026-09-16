# 选题与实验总汇报（截至 2026-08-24）

> 综合：`02-idea-candidates.md`、`01-related-work.md`、`10-synth-findings.md`、集群实验。  
> **引用注意：** 部分 2026 arXiv ID 来自二手汇总，投稿前请逐一核对。

---

## 0. 总览

| Idea | 一句话 | 实验水位 | 建议优先级 |
|------|--------|----------|------------|
| **1 CAF** | 反事实动作保真度指标（可预测控制效用） | 合成 ✅；真 WM ❌ | ★★★ 主线 |
| **2 Mem–Ctrl** | 记忆↔可控性 Pareto + 正确 CF 协议 | 合成 ✅（主信号稳） | ★★★ 主线 |
| **4 Excitation** | 高激励数据提升反事实可辨识性 | 合成 v4 ✅ | ★★☆ 配套 |
| **3 Sel-Mem** | 事件驱动选择性写入 | 合成暂停 ❌ | 降级 / future |
| **7 Geom verifier** | 几何检查门控生成 | TRELLIS smoke + verifier ✅ | 旁路底座 |
| 5/6/8/9 | off-screen / 第三人称 / 物理算力 / safe horizon | 未开实验 | 暂缓 |

**叙事建议：** 一篇主文 = CAF + Mem–Ctrl + Excitation（见 `14-paper-narrative.md` 选项 A）；权重不到可拆 B1/B2。

---

## Idea 1 — CAF（Counterfactual Action Fidelity）

### 问题
现有指标多测「前向好看不好看」；控制真正需要的是：换动作后，预测是否**按该动作应有方式**改变，且改变忠实于世界动力学。

### 对照论文（动机 / 缺口）

| 文献 | 角色 | 我们相对它做什么 |
|------|------|------------------|
| WMBench / GigaWorld-1（2607.02642） | 结论：评估器质量由**长视界、动作保真**主导，非短视界视觉 | 他们没给出可自动算的动作保真指标 → 我们给 CAF |
| What-If World（2605.27589） | 发现 contrastive bottleneck（轻重刹车视频几乎一样） | 他们偏 prompt + VLM/人评 → 我们要自动、成对干预 |
| 可辨识性（2607.22430）+ Reacher 例（2605.26379） | 证明：excitation 不足时 on-policy 准 ≠ CF 准 | 理论 grounding；CAF 直接测 CF 侧 |
| ACT-Bench（2412.05337）、Omni-WorldBench 等 | 动作/物理评测 | 差异：无人评、有理论前置、强调 **predictive validity** |
| Lambert et al.（2002.04523） | objective mismatch | CAF 与下游效用相关 = 该论述在视频 WM 上的落地 |
| FVD（及 2404.12391 内容偏差） | 像素/分布指标 | 负对照：不应当控制效用主指标 |

### 参考方法（我们采用 / 计划采用）

- **成对动作干预：** 镜像、取反、缩放、正交扰动（在有真动作的数据上，期望效应方向可知）
- **指标族：** `self_cf_gap`、`pred_validity`（相对 oracle/IDM）、`CAF = gap / (pv+ε)`、response gain、insensitivity
- **杀手实验（未做）：** 多开源 WM 上 CAF vs FVD/PSNR 与 WorldGym/WMBench / 规划成功率的相关表

### 已有实验结果

| 实验 | 结果 | 判决 |
|------|------|------|
| CAF v2（FiLM vs ignore） | film gain≈0.20，ignore insens=1 | ✅ 可分 |
| CAF v3 / joint GPU（3 seeds） | film CAF≈0.38，ignore=0 | ✅ 主信号 |
| 单看 self_gap | demo OOD 可虚高 | ❌ 不作唯一主指标 |

**缺口：** 真实 action-conditioned WM（Cosmos / Matrix-Game / LeWM）权重未挂上；Wan TI2V 无 action，仅视频栈冒烟。

---

## Idea 2 — Memory ↔ Controllability Pareto

### 问题
记忆变强常提升一致性，但削弱即时动作响应；文献多是散点，缺统一前沿与正确反事实协议。

### 对照论文

| 文献 | 角色 | 我们相对它做什么 |
|------|------|------------------|
| 综述 2606.01164 §6.1 | 明确写出：历史↑ → 一致性↑，**动作响应↓** | 把冲突画成 **Pareto 曲线** |
| Astra（ICLR 2026） | conditioning frame 加噪逼出动作响应 | 只是前沿上一个点 → 我们系统扫 `L, σ, K, g` |
| MIND（2602.08025） | action-space shift 下有记忆的 V2W 反而不如 I2W | 记忆会倒忙的实证 |
| WorldMem（NeurIPS 2025, 2504.12369） | 记忆库参考实现；1→8 好、**16 帧变差** | 记忆≠越多越好；我们用可控旋钮测绘 |
| Rolling Forcing（2509.25161） | 加噪缓解漂移但伤干净参考/一致性 | 解释 σ 旋钮的代价 |
| Infinite-World / RELIC / Memory Forcing 等 | 各类记忆工程 | 扫 frontier 的 backbone / 旋钮来源 |

### 参考方法

- **CF 协议（关键）：** **frozen-memory + 只翻转当前动作**（否决全序列 negate）
- **合成：** 记忆/动作通道干净隔离的 `force_w` 扫描
- **计划（真 backbone）：** Matrix-Game 1.3B / Yume / Cosmos；扫 context、Astra 式 σ、retrieval K、memory gain（见 `11-mem-ctrl-backbone-plan.md`）
- **指标：** instant_cf_gap（可控性）、consistency（on-policy / 长视界漂移）

### 已有实验结果

| 实验 | 结果 | 判决 |
|------|------|------|
| Mem v3 seed1 | gaps 严格单调，`w=1→0` | ✅ |
| multiseed 0–4 | **w1_zero = 5/5**；严格单调 3/5 | ✅ 主信号；中段可有小反弹 |
| 全序列 negate（clean） | 不单调 | ❌ 否决该协议 |

**论文表述：** 主打端点效应 +「记忆占满则消即时控制」；严格单调作补充。

---

## Idea 4 — Excitation-aware 数据

### 问题
demo / 目标导向策略动作激励不足 → 模型可辨识性差；应用故意提高 excitation 的数据，改善 CF。

### 对照论文

| 文献 | 角色 | 我们相对它做什么 |
|------|------|------------------|
| 2607.22430（excitation / identifiability） | 理论：激励不足 → CF 误差可很大 | 实验验证「换数据分布」 |
| 2605.26379（Reacher：OU vs goal-directed） | 最小例子：采样策略决定可辨识 | 合成上对比 demo vs high_exc |
| WMBench 等 | 强调动作保真评估 | Excitation 解释 CAF 为何在 demo 上塌 |

### 参考方法

- 构造 **demo（近零动作）vs high_exc（大幅值 + 扫描轨迹）** 训练分布
- 在 **同一 high_exc 评测集** 上比 oracle PV 与 fact_err
- **禁止**只报 self_cf_gap（demo 会 OOD「假敏感」）

### 已有实验结果

| 实验 | 结果 | 判决 |
|------|------|------|
| Excitation v1（delta aux） | 过简，cf_gap≈0 | ❌ |
| v3 | oracle PV 方向对 | ✅ 弱 |
| joint 短训 / 严阈值 ×0.9 | 易假阴 | ⚠️ |
| **v4（强动力学，3 seeds）** | PV∧fact 全胜；OOD gap 膨胀 3/3 | ✅ **定稿** |

例 seed1：PV 0.103→0.067，fact 0.070→0.029，self_gap 0.161→0.017。

---

## Idea 3 — Selective memory writes（暂停）

### 问题
记忆该写什么：稀有高价值事件才写入，避免噪声覆盖。

### 对照 / 参考
- Memory Forcing（何时依赖记忆）、WorldMem / Infinite-World（检索与压缩）
- 事件驱动记忆、surprise / utility 门控（RL / 认知架构常见思路）

### 实验结果
| 版本 | 结果 |
|------|------|
| v1–v3 | always 不差于/优于 oracle；selective 无事件分辨 |
| **判决** | **暂停合成线**；非 idea 必死，需更强 POMDP 或真 backbone |

---

## Idea 7 — Geometry verifier（旁路）

### 对照 / 参考
- TRELLIS（微软图生 3D）、Marble（显式 3D 世界）、Geometry Forcing（几何特征对齐）

### 实验结果
- TRELLIS smoke → `sample.ply` ✅  
- geom_verifier：点数/包围盒/质心检查 ✅  
- **未**接到「门控视频生成器」闭环

---

## 工程底座（非选题，但支撑实验）

| 项 | 状态 |
|----|------|
| 集群主战场 | `25mhn`（少频 SSH） |
| Wan2.2-TI2V-5B | 推理通；**无 action**，不作 CAF 主实验 |
| LeWM ckpt | 目录空；HF 外取曾失败 |
| 文档 | Methods：`12`/`13`；叙事：`14`；图：`reports/figures/` |

---

## 对照关系简图

```text
可辨识性理论 ──► Excitation(数据) ──► 解释 CAF 何时可靠
       │                                    │
       └──────────► CAF(指标) ◄─────────────┘
                         │
WMBench「要测动作保真」───┘
What-If「bottleneck」──────┘

综述/Astra/WorldMem「记忆伤控制」──► Mem–Ctrl Pareto(协议+前沿)
```

---

## 还缺什么（按影响排序）

1. **真实 action WM 上的 CAF + Mem 扫图**（权重）  
2. CAF 与下游规划/policy-eval 的 **相关系数表**（杀手实验）  
3. Mem gating / σ 配方证明能 **支配** 朴素加长 context  
4. Sel-Mem 仅在更强设定下重启  

---

## 一句话结论

- **已立住（合成）：** CAF 可分听/不听动作；Mem 正确 CF 下记忆↑则即时控制↓（w=1 消控制）；Excitation 改善 PV∧fact，且揭露 self_gap 虚高。  
- **对照锚点：** WMBench / What-If / 可辨识性理论；记忆综述 + Astra + WorldMem。  
- **主阻塞：** 无真实 action-conditioned 开源权重 → 杀手实验与 Pareto 真图未开。



主线三 idea（建议拼一篇）
Idea 1 — CAF（反事实动作保真）

主张：测「换动作后预测是否按该动作该变的方式变」，且用 CAF ≈ self_gap / pred_validity。
实验：合成 film vs ignore 稳定可分（joint GPU：film CAF≈0.38，ignore=0）；禁止单看 self_gap。
对照：WMBench（要测动作保真但无自动指标）、What-If World（contrastive bottleneck）、可辨识性理论 2607.22430 / Reacher 例、Lambert objective mismatch；FVD 作负对照。
缺口：真 action WM 未挂上。
Idea 2 — Memory ↔ Controllability

主张：记忆↑ → 即时控制↓；CF 必须 frozen-memory + 只翻当前动作。
实验：w=1 → gap=0 为 5/5；严格单调 3/5（中段可有小反弹）→ 主打端点效应。
对照：综述 2606.01164、Astra 加噪、MIND、WorldMem（16 帧变差）、Rolling Forcing。
方法：force_w 扫描；计划在 Matrix-Game / Yume / Cosmos 上扫 L,σ,K,g。
Idea 4 — Excitation 数据

主张：高激励训练提升 high_exc 上的 PV∧fact；demo 的 self_gap 会 OOD 虚高。
实验：v4 定稿，3/3 seed 全过（例 seed1：PV 0.103→0.067，fact 0.070→0.029）。
对照：同一套可辨识性 / Reacher 激励理论；服务 CAF「何时可信」。
降级 / 旁路
Idea	状态	对照/方法要点
3 Sel-Mem
合成暂停（gate 无事件分辨）
Memory Forcing / WorldMem 系
7 Geom verifier
TRELLIS ply + 几何检查 ✅，未闭环门控视频
TRELLIS / Marble / Geometry Forcing
5/6/8/9
未开实验
暂缓
一句话
合成上 CAF + Mem 协议 + Excitation 已立住；对照锚在「评估/可辨识性/记忆–控制冲突」三条文献线；真正缺的是 带动作的开源 WM 权重 上的杀手实验与 Pareto 真图。