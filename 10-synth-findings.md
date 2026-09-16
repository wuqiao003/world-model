# CAF / Mem–Ctrl 合成实验结论（可引用笔记）

> 更新至 2026-08-24。集群玩具实验，非论文终稿数字。

## CAF（Idea 1）

**主张：** 应用「动作是否改变预测」+「反事实是否对齐 oracle」联合衡量；单独 self-gap 会把 OOD 爆炸误判为可控。

**合成证据：**
- film vs ignore：`self_cf_gap` 可分；ignore ≈ 0
- joint GPU（3 seeds）：mean CAF film≈0.38，ignore=0
- 正式方向：`caf ≈ self_gap / pred_validity_vs_oracle`

**否决：** 弱注入 CAF v1；仅用相对 resp_ratio 而无绝对 gap 的过松准则。

**缺口：** 真实 **action-conditioned** WM：**LeWM 已挂上**（K1 smoke ✅）；缺真实帧 / 下游 proxy 与第 2 个 WM。

### Predictive validity（K2，主会关键）

| 轮次 | 下游 | 结论 |
|------|------|------|
| K2v1 | on-policy plan | fact_err 相关虚高；raw CAF 失败 |
| K2v2 | CF plan（大/负动作） | 效用排序 film≻ignore≻weak；**raw CAF 反序**（weak OOD gap）→ **否决 raw CAF** |
| K2v3 | 同上 + gated | `caf_gated` 稳把 film 排第一；本 3 模共线时仍难稳胜 fact |
| K2v4 | +`pos_only` 非对称；CF=负 ang | **2/3 seeds pass**；seed0：`caf_gated` Spearman **1.0** vs `neg_fact` **0.4** |

**主张改写：** 主卖 **gated CAF**（fact 门控）/ 双轴，不卖 raw `gap/(pv+ε)`。

> ### 2026-08-31 重要更新：上述合成结论**未能迁移到真实模型**
>
> 3 个真实 LeWM PushT-FR3 ckpt × 5 动作管线 × 3 seed 上：
> 最佳可控性指标 `caca` ρ=+0.796 vs 最佳保真度指标 `fact_align` ρ=+0.754，
> **配对 bootstrap 的 Δ 置信区间跨 0**（−0.25…+0.35，P(不优)=0.40）；
> 在 `−gt_rank` 效用轴上保真度明显更好（0.961 vs 0.886）。
>
> 另外两条被真模型否掉的合成设计：
> - **保真度门控**会误杀 v4b（一步保真度最差但官方 cost-surface CV 最好的规划器）。
> - **复合式** `tanh(gap)·(1+caca)/2` 比 `caca` 单独差近一半（0.379 vs 0.796），
>   因为 `gap_ratio` 自身与效用无关（ρ=−0.04）。
>
> 详见 `reports/2026-08-31.md` §7 与 `16-main-track-roadmap.md` §6.4。
> **本文件的合成结果从此只能作为「动机」，不能作为证据。**

## Mem ↔ Controllability（Idea 2）

**主张：** 记忆权重 ↑ → 即时动作响应 ↓；CF 协议必须 **frozen-memory + 只翻当前动作**。

**合成证据：**
- Mem v3 seed1：严格单调，`w=1 → gap=0`
- **multiseed 0–4：** `frac_w1_zero=1.0`；严格单调 3/5（中段可有小反弹）
- **论文主打：** 端点效应 + w=1 消即时控制；严格单调作补充

**否决：** 全序列 negate（记忆也被反事实污染）。

## Excitation（Idea 4）— v4 定稿

**主张：** 高激励训练提升高激励评测上的 **oracle PV** 与 **fact_err**；demo 的 `self_cf_gap` 会 OOD 虚高。

**v4（3/3 seeds，强动力学）：**  
`frac_pv_better=1`，`frac_fact_better=1`，`frac_ood_gap_inflation=1`，**pass=true**。  
seed1：PV 0.103→0.067，fact 0.070→0.029，self_gap 0.161→0.017。

**否决：** self_cf_gap 作主指标；过严「PV×0.9」单阈值（小效应量易假阴）。

## Selective memory（Idea 3）

**状态：** 合成 v1–v3 **暂停**（learned gate 无法分辨事件；oracle 收益不稳）。

## 工程底座

- TRELLIS image→3D + geom verifier ✅  
- Wan2.2-TI2V-5B smoke mp4 ✅（无 action）  
- 主战场：`25mhn`
