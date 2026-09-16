# 汇报 2026-08-24（午后续跑）

> 触发：用户要求继续实验并汇报  
> 集群：`124.223.202.234` **SSH banner 超时**（连续失败）→ 改本地 CPU 合成实验

## 1. 进度

### 集群
- 不可达，GPU 实验暂停  
- 既有 CFS 结果（Mem v3 / Excitation v3 / Wan smoke 等）以本地笔记与缓存为准

### 联合实验（本地 CPU）

**短训 STEPS=350，seed0/1**
| 判定 | 结果 |
|------|------|
| CAF film vs ignore | ✅ ignore=0，film≈0.40 |
| Excitation PV | ❌ 方向反了（demo PV 更好） |

**全量 STEPS=700，seed0**（`runs/joint_caf_exc_full/`）
| 判定 | 结果 |
|------|------|
| CAF | ✅ 仍稳 |
| Excitation PV | 方向恢复：high_exc 0.0327 < demo 0.0359，但未过「×0.9」严阈值 → `pass_excitation_pv=false` |

结论：CAF 主信号稳；Excitation 需足够步数，效应量在本玩具上偏小。集群恢复后用多 seed + GPU 再标定阈值。

demo 的 `self_cf_gap`/`CAF` 仍偏高（OOD 敏感），on-policy `fact_err` 却更差 → **继续坚持 PV，不单看 CAF。**

### 既有水位（不变）
| 线 | 状态 |
|----|------|
| CAF / Mem–Ctrl / Excitation | 合成 + Methods + 图齐 |
| Wan TI2V smoke | 曾成功（集群） |
| 真实 action WM | LeWM 外取失败；集群现又断 |

## 2. 否决 / 阻塞
- 集群网络不通 → 不空等 HF/kubectl  
- 不把 Wan 无动作生成当 CAF 主实验

## 3. 下一步（集群恢复后）
1. 回 `25mhn` 重跑 `joint_caf_exc` 全量 seed（CUDA）  
2. 再探 LeWM / Matrix-Game 权重  
3. 按 `14-paper-narrative.md` 选项 A 扩写 intro

## 5. 集群恢复后续（少频 SSH）

- 主战场 **25mhn** 可用；`lewm_pusht/` 目录空
- **GPU joint 已完成**（~37s，seeds 0/1/2，STEPS=700）

| 判定 | 结果 |
|------|------|
| CAF film vs ignore | ✅ 全 seed；mean CAF film≈0.38，ignore=0 |
| Excitation PV | 均值略好（0.033 vs 0.037）但未过 ×0.9；仅 seed1 单独过关 |

摘要已落本地：`runs/joint_caf_exc_gpu_summary.json`。之后 SSH 仅在需要新动作时再连。
