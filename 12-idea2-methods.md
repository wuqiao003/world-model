# Idea 2 Methods 扩写：Memory ↔ Controllability Pareto

> 配套：[`11-mem-ctrl-backbone-plan.md`](./11-mem-ctrl-backbone-plan.md)、[`10-synth-findings.md`](./10-synth-findings.md)、图 `reports/figures/mem_v3_pareto_seed1.svg`

## 1. Problem statement

交互式视频世界模型里，加长历史 / 加强记忆通常提升时序一致性，但会削弱对当前动作的响应。文献中已有多个独立散点（WorldMem ablation、Astra 加噪、MIND 分布偏移等），缺少：

1. **同一 backbone 上可复现的二维前沿**（consistency vs controllability）  
2. **正确的反事实协议**（避免「全序列 negate」污染记忆状态）  
3. 能 **支配** 朴素加长 context 的 gating / 加噪配方

## 2. Synthetic evidence (done)

协议：**frozen-memory + 仅翻转当前动作编码**。

在可控玩具动力学上扫描 `force_w ∈ {0,0.25,0.5,0.75,1}`：

- seed1：`instant_cf_gap` **严格单调递减**，`force_w=1 → 0`  
- 说明：记忆通道占满时，即时动作被完全闸掉 —— Pareto 张力可被定量测到

见图：`reports/figures/mem_v3_pareto_seed1.svg`。

**否决的错误协议：** 对整段动作序列做 negate（记忆 GRU 也吃到反事实），曲线不单调、不可解释。

## 3. Real-backbone protocol (planned)

### 3.1 Models
优先 Matrix-Game 1.3B / Cosmos-Predict2.5-2B / Yume；Wan-TI2V **仅负对照**（无 action）。

### 3.2 Knobs
`L`（context）、`σ`（history noise）、retrieval `K`、memory gain `g`。

### 3.3 Metrics
- **Controllability：** 成对动作干预下的 response gain / CAF（有真动作时对齐 IDM/oracle 方向）  
- **Consistency：** on-policy 短视界失真 + 长视界漂移  

### 3.4 Claim
存在非退化前沿；至少一个 `(σ, g)` 或 action-conditioned gate **支配**「只加长 L」的曲线段。

## 4. Blocking

集群 CFS 无候选权重；HF 经代理拉取 LeWM 失败（2026-08-22）。Methods 与合成图可先写入论文草稿；真实扫前沿待权重落盘。

## 5. Minimal next experiment when weights arrive

1. 冻结 backbone，只扫 `L × σ` 小网格（2 seed）  
2. 画出 Pareto；标出 Astra 式 `σ>0` 是否外推支配  
3. 同点计算 CAF，报与短视界动作误差的相关
