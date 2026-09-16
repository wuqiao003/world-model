# Idea 1 Methods 短文：Counterfactual Action Fidelity (CAF)

> 配套：[`10-synth-findings.md`](./10-synth-findings.md)

## 1. Claim

前向视觉指标（FVD / PSNR）不能可靠预测控制效用。我们要测的是：

> 若替换动作 \(a\to a'\)，预测未来是否**按该动作应有的方式**改变，且改变量与 oracle / 可辨识方向对齐。

## 2. Metric family

对同一条件状态，比较 factual 预测 \(\hat y(a)\) 与 counterfactual \(\hat y(a')\)：

| 量 | 含义 |
|----|------|
| `self_cf_gap` | \(\|\hat y(a)-\hat y(a')\|\) — 是否「听动作」 |
| `pred_validity` | \(\|\hat y(a')-y^\star(a')\|\) — 反事实是否对 |
| **CAF** | `self_cf_gap / (pred_validity + \epsilon)` — 既敏感又忠实 |

辅助：action-response gain、insensitivity rate（contrastive bottleneck）。

## 3. Synthetic evidence

- film vs ignore：ignore 的 `self_cf_gap≈0`，film 可分  
- **否决**仅用 `self_cf_gap`：demo 策略训练的模型在 OOD 高激励上会「爆炸式敏感」，PV 更差（见 Idea 4）

## 4. Real evaluation (blocked)

需 **action-conditioned** 开源 WM + 真动作数据做成对干预，并报 CAF 与下游 planning / policy-eval 相关。

当前：Wan TI2V 无 action；LeWM 外取失败；CFS 无 Cosmos/Matrix-Game。

## 5. Positioning

不是又一个通用 video benchmark；卖点是 **predictive validity**（指标能否预测控制成败）+ 可辨识性理论 grounding。
