# 论文叙事大纲：拆合决策

> 基于合成证据 + Methods（Idea 1/2/4）。真实 backbone 仍阻塞。

## 选项 A — 一篇主文（推荐，若 4–6 周内能拿到 action WM）

**标题方向：** *What to Measure When World Models Must Control: Counterfactual Action Fidelity and the Memory–Controllability Frontier*

**故事线：**
1. 问题：视觉 fidelity ≠ 控制效用；记忆变强会伤动作响应  
2. CAF：定义 + 合成可分 +（计划）与下游相关  
3. Mem–Ctrl Pareto：正确 CF 协议 + 合成单调前沿 +（计划）真 backbone 扫图  
4. Excitation：解释 CAF 为何在 demo 数据上塌缩；PV 才是对的读数  
5. 讨论：Sel-Mem 合成失败说明「写什么」比「写不写」更难，留作 future work

**贡献清单（可投 ICLR/NeurIPS）：** 指标 + 协议 + 前沿图 + 数据激励结论；开源评测脚本。

## 选项 B — 拆成两篇（权重继续不到时）

| 篇 | 焦点 | 靠什么站住 |
|----|------|------------|
| B1 | CAF + Excitation | 理论 grounding + 合成 + 开源评测代码；弱依赖大模型权重 |
| B2 | Mem–Ctrl Pareto | 协议正确性 + 合成前沿；真 backbone 作为「一旦权重可得」的扩展 |

## 选项 C — 短平快 workshop / D&B

只投 **CAF 指标 + 合成验证 + 评测工具**，Mem Pareto 作附录。风险：被看成「又一个 metric」。

## 当前建议

- **短期（本周）：** 按 A 写 intro/related 骨架；实验以「协议与指标」为主表，真 WM 标为 pending  
- **权重一到：** 优先 Matrix-Game / LeWM 上复现 CAF + 小网格 Mem Pareto（见 `11-mem-ctrl-backbone-plan.md`）  
- **不要：** 无增量合成空转；HF 盲下；把 Wan-TI2V 写成 CAF 主实验
