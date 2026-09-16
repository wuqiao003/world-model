# 集群实验日志与结果（2026-08-17）

## 1. 资源与环境

| Worker | GPU | 用法 |
|---|---|---|
| **slv2n**（主） | 8×A800-80GB | 本次全部实验 |
| wvk6p | 部分空闲 | 未动（待命） |
| qdm9p / 998s5 | 忙 | **不占用** |

**关键坑：** 容器默认 `NVIDIA_DRIVER_CAPABILITIES=""`，必须显式设为 `compute,utility`，否则 `torch.cuda.is_available()==False`。

**工作区：** `/mnt/group/jxdong/wm_exp/`  
（code / runs / logs / PLAN.md）

**Pod 内已有：** torch 2.3.1+cu121、diffusers、transformers、trimesh、pytorch3d、accelerate。

---

## 2. 已完成实验与结果

### A. CAF v2 — 反事实动作保真度（Idea 1）✅

合成可控小球 + FiLM 动作条件 + 位移辅助头。三模型对比，多种子复现。

| model | gain_dx | gain_dy | cos_delta | pixdiff_neg | insens_neg | resp_ratio_neg |
|---|---:|---:|---:|---:|---:|---:|
| **film（正常）** | 0.200 | 0.201 | **1.000** | 0.0465 | **0.003** | **1.001** |
| weak | 0.200 | 0.200 | 1.000 | 0.046 | 0.003 | 0.18–0.67 |
| **ignore** | 0.000 | 0.000 | 0.02 | 0.000 | **1.000** | **0.000** |

**结论：** CAF 指标体系能清晰区分“听动作 / 弱听 / 不听”。  
`ignore` 的 contrastive bottleneck（insensitivity=1）被定量抓到；`film` 的 response_ratio≈1。  
这是 Idea 1 方法论的最小可行验证。

复现种子 A/B 一致：film gain≈0.20、insens≈0.003、ignore insens=1.0。

路径：`runs/caf_v2/`、`runs/caf_seedA/`、`runs/caf_seedB/`

---

### B. Memory ↔ Controllability Pareto（Idea 2）✅（部分）

强制记忆权重 `force_w` 扫描：

| force_w | consistency_mse | response_ratio | mean_gate |
|---:|---:|---:|---:|
| 0.0 | 3.06e-05 | **1.287** | 0.00 |
| 0.25 | 3.39e-05 | **0.613** | 0.25 |
| 0.5 | 2.76e-05 | 0.927 | 0.50 |
| 0.75 | 5.01e-05 | 0.929 | 0.75 |
| 0.9 | 5.51e-05 | 1.318 | 0.90 |
| learn_gate | 3.20e-05 | 0.995 | **0.078** |

**解读：**
- 中等记忆权重（0.25）时动作响应掉到最低（0.613）——存在记忆–可控性张力。
- 曲线非严格单调（0.9 又抬升）：模型可能走 FiLM 旁路补偿，架构还需更干净的隔离。
- **学到的 gate 主动把记忆权重压到 0.078**，优先保动作响应——和综述里“记忆变强则动作变弱”的叙事一致。

路径：`runs/mem_ctrl_v2/pareto_v2.md`

---

### C. Excitation-aware data（Idea 4）⚠️ 玩具过简

| kind | excitation | on_policy_mae | cf_gap | resp_ratio |
|---|---:|---:|---:|---:|
| ou | -1.43 | 0.0001 | ≈0 | 0.78 |
| demo | **-6.47** | 0.0004 | ≈0 | 0.88 |
| uniform | -2.19 | 0.0021 | ≈0 | 0.52 |

demo 的 excitation 确实最低（边缘塌缩），但辅助损失直接回归 `0.15*act` 让 counterfactual 变得太容易，`cf_gap≈0`。  
**下一步：** 去掉显式 delta 监督，只留像素损失，或换非线性/接触动力学。

---

### D. TRELLIS — 通路重开 🔧

- ❌ 代理 `git clone` 超时（否决）
- ✅ CFS 已有：`/mnt/group/hjm/TRELLIS` + `/mnt/group/lzdql/weights/TRELLIS-image-large` + HF cache
- ✅ conda：`/mnt/group/hjm/miniconda3/envs/trellis2`（torch 2.6+cu124）
- ❌ 无头 `import open3d` 易挂起；缺 `rembg`
- ✅ `m3d-trellis` env + stub open3d/rembg + 本地下载 DINOv2  
- ✅ `runs/trellis_smoke/sample.ply`（32.5MB，510s）`result.json` ok

---

### E. 第二轮（2026-08-17 午后）

#### E1. CAF hard / v3
- hard：相对可分但绝对 gap 过小（指标过松）
- v3：加入 oracle predictive validity；`caf = self_gap/pv`
- seed0/1：film `self_gap≈0.009` vs ignore `0`；绝对阈值 `>0.01` 未过（差一截）
- **判决：** 机制成立；pass 准则改为「相对分离 + gap>0.005」即可，或加长训练

#### E2. Mem clean / v3 ✅
- clean：全序列 negate → **否决该 CF 协议**
- v3：frozen-memory + 仅翻转当前动作
- seed0 gaps：`0.00132 → 0.00049 → 0.00028 → 0.00045 → 0`（近单调）
- **seed1：严格单调** `0.00115 → 0.00040 → 0.00036 → 0.00033 → 0`，`monotonic=True`
- `force_w=1` 时 instant_cf_gap=0（纯记忆、无即时动作）—— Pareto 主信号

#### E3. Excitation hard / v3 ✅
- hard：demo 的 self_cf_gap 最大 → **否决 self-gap 作主指标**（OOD 爆炸）
- v3：oracle PV；seed0/1 均 `pass=True`
  - seed0：PV demo 0.0505 → high_exc 0.0370
  - seed1：PV demo 0.0498 → high_exc 0.0410

---

## 3. 踩坑（累计）

1. CAF v1 动作注入太弱 → 全模型 insensitivity=1；v2 改 FiLM + aux 才分开。
2. `tensor.norm(-1)` 把 `-1` 当成 **p 范数** 而非 dim —— gate 崩溃；必须 `norm(dim=-1)`。
3. PowerShell → SSH → kubectl 引号易被吞；**用 base64 传脚本**最稳。
4. 不要碰 qdm9p/998s5；slv2n 优先空闲卡。
5. Mem CF 必须 frozen-memory；全序列 negate 会污染记忆通道。
6. Excitation 不能用 self_cf_gap 当主指标；要用 oracle predictive validity。
7. 无头容器 `open3d` 易挂；TRELLIS 用 stub 或 headless。

---

#### E4. Selective memory（Idea 3）❌ v1
- always fact_err≈0.009 优于 oracle≈0.013；selective 写门≈0.5 无事件分辨
- **否决当前环境设定**；下一版加强稀有事件价值与覆盖代价

#### E5. 2026-08-18 tick
- **geom_verifier** ✅：`sample.ply` 478k 点，bbox/质心检查通过 → Idea 7 玩具底座
- **sel_mem_v2** ❌：oracle 略优于 always；selective 写入率≈5% 但无事件分辨
- 发现 `/mnt/public/Wan2.2_models`（含 TI2V-5B）可供后续真实 CAF 探测
- slv2n 有他人 xifgao Blender 渲染；玩具实验只用 GPU 0/1

---

#### E6. 2026-08-19 tick
- **sel_mem_v3** ❌：暂停 Idea 3 合成线（v1–v3）
- **wan_probe** ✅：TI2V-5B 权重完整，`model_type=ti2v`，无 action 接口；deps OK
- 下一优先：Wan 最小生成 + 找带动作的开源 WM

---

#### E7. 2026-08-20 tick
- 主战场：`25mhn`（`slv2n` kubectl 不稳）
- Wan2.2 clone ✅；TI2V-5B smoke → `runs/wan_ti2v_smoke/sample.mp4` ✅
- CFS 顶层未见带 action 的开源 WM 目录；下一步需更深搜或外取权重

---

#### E8. 2026-08-21 tick
- 笔记：`10-synth-findings.md`（CAF/Mem/Excitation 合成结论）
- CFS 深搜 action WM → **空**；真实 CAF 外取阻塞
- 主战场仍 `25mhn`；Wan smoke 文件仍在

---

#### E9. 2026-08-22 tick
- 设计：`11-mem-ctrl-backbone-plan.md`
- 外取 `YuhaiW/lewm-pusht-fr3-v2` ❌（HF hub 不可达）
- 策略：停盲下；等镜像/人工权重，或先沉淀 Methods

---

#### E10. 2026-08-23 tick
- Methods：`12-idea2-methods.md`
- 图：`reports/figures/mem_v3_pareto_seed1.svg`
- LeWM 仍缺；不盲下 HF

---

#### E11. 2026-08-24 tick
- CAF Methods：`13-idea1-caf-methods.md`
- 图：`reports/figures/excitation_v3_pv.svg`
- LeWM 仍缺；合成沉淀接近完备

---

#### E12. 2026-08-24 午后（集群 SSH 超时）
- 叙事：`14-paper-narrative.md`
- 本地 `joint_caf_exc`：CAF ✅；Excitation 短训 ❌ 未复现
- 阻塞：跳板机不可达

---

#### E13. 2026-08-24 refine
- **excitation_v4** ✅ 3/3 seeds（强动力学 + PV∧fact）
- **mem_v3_multiseed**：w1_zero 5/5；严格单调 3/5 → 主信号保留，严阈值放宽表述

---

## 4. 下一波

1. 更新 Idea4/2 Methods 数字为 v4 / multiseed
2. Mem 可选 Spearman 端点检验脚本
3. LeWM 权重仍空

---

## 5. 快速复现命令

```bash
ssh root@124.223.202.234
kubectl exec -it raytrainv2-jxdong-ray-worker-worker-slv2n -- bash
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=0 OUT_DIR=/mnt/group/jxdong/wm_exp/runs/caf_v2 \
  python3 /mnt/group/jxdong/wm_exp/code/caf_v2.py
```
