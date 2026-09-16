# 3D AI 全方向相关工作清单（主流方向 · 2023 – 2026.08）

> 配套导读：[`08-3d-primer.md`](./08-3d-primer.md)（通俗全景介绍）  
> 更细的分方向原始调研：
> - [`04-survey-3d-reconstruction-neural-rendering.md`](./04-survey-3d-reconstruction-neural-rendering.md)
> - [`04-survey-3d-spatial.md`](./04-survey-3d-spatial.md)
> - [`04-survey-3d-asset-generation.md`](./04-survey-3d-asset-generation.md)
> - [`05-survey-3d4d-generative.md`](./05-survey-3d4d-generative.md)
> - [`05-survey-3d-geometry-foundation.md`](./05-survey-3d-geometry-foundation.md)
> - [`06-survey-radiance-fields-3dgs.md`](./06-survey-radiance-fields-3dgs.md)
>
> **引用前请核对 arXiv ID / 会议归属。** 2026 年部分条目来自会议页面或二手列表，文末有验证注意事项。

---

## 0. 一张总地图：3D AI 的九大主流方向

```text
真实世界 / 图像 / 文本 / 传感器
            │
            ▼
    ┌───【1. 表征 Representation】───┐
    │ 点云·体素·网格·SDF·NeRF·3DGS  │
    └──────────────┬─────────────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
【2. 获取/重建】 【3. 渲染】  【4. 生成】
SfM·DUSt3R·VGGT  光栅·NeRF·GS  物体·场景·4D
     │             │             │
     └──────┬──────┴──────┬──────┘
            ▼             ▼
      【5. 理解】   【6. 时间/4D】
   检测·占用·VLM    动态GS·视频→4D
            │             │
            └──────┬──────┘
                   ▼
         【7. SLAM / 在线地图】
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
【8. 驾驶/机器人 3D】  【9. 显式世界模型】
BEV·Occ·抓取·位姿     Marble·WorldGrow·Lyra
```

**到 2026 年中的四条结构性结论：**

1. **外观默认表示是 3DGS；几何前端是 feed-forward pointmap（DUSt3R→VGGT→DA3）。**
2. **物体级 3D 生成配方已收敛：** 3D-native VAE（VecSet 或 sparse latent）+ flow-matching DiT + 独立 PBR 贴图。SDS 时代结束。
3. **场景级生成与交互式视频世界模型分裂成两条赌注：** 显式持久 3D（Marble / WorldGrow）vs 隐式帧级想象（Genie 3）。
4. **瓶颈从“能不能出 3D”转移到“能不能当资产 / 能不能闭环用”。** 拓扑、UV、PBR、碰撞、动作保真度、评测诚实性，比 PSNR 更重要。

---

## 1. 表征（Representation）—— 所有方向的“货币”

| 表征 | 像什么 | 擅长 | 短板 | 2026 地位 |
|---|---|---|---|---|
| 点云 Point Cloud | 空中沙粒 | 传感器原生、稀疏几何 | 无表面、难渲染 | LiDAR / 深度相机母语 |
| 体素 / Occupancy | 乐高格子 | 占用、规划、驾驶 | 分辨率↑内存炸 | 驾驶与机器人规划主流 |
| 网格 Mesh | 三角皮肤 | 游戏/CAD/打印/碰撞 | 拓扑难、细节贵 | **产业交换格式** |
| SDF / Occupancy Field | “到表面多远” | 光滑表面、布尔 | 薄结构/开曲面难 | 生成与表面重建仍常用 |
| NeRF 辐射场 | “任意点看过去什么色” | 新视角、复杂外观 | 慢、几何不显式 | 研究 niche / 特殊材质 |
| **3D Gaussian** | 百万彩色小椭圆 | **实时 + 可微** | 几何噪声、存储大 | **NVS / 交互默认表示** |
| Triplane / Sparse Latent | 压缩的 3D 特征 | 生成友好 | 解码管线复杂 | TRELLIS / Hunyuan 系 |

**关键地标：** DeepSDF → NeRF (2020) → Instant-NGP → **3DGS (SIGGRAPH 2023)** → 2DGS / SuGaR（表面化）→ TRELLIS SLAT / O-Voxel（可多格式解码的生成 latent）。

**一句话：** 选表征先问三件事——要实时吗？要导出给引擎吗？要可微优化吗？

---

## 2. 获取与重建（Geometry Foundation）

### 2.1 经典管线仍活着

| 名称 | 出处 | 贡献 |
|---|---|---|
| COLMAP | 经典 | 多视角 SfM/MVS 金标准 |
| **GLOMAP** | ECCV 2024 | 全局 SfM，精度可比 COLMAP、快几个数量级；已并入 **COLMAP 4.0** |
| GLUEMAP | CVPR 2026 | 系统比较经典 vs feed-forward：经典在 ETH3D 仍更准，feed-forward 在野生照片更强 |

### 2.2 单目深度基础模型

| 名称 | 出处 | 贡献 |
|---|---|---|
| Depth Anything / V2 | CVPR 2024 / NeurIPS 2024 | 相对深度 foundation；V2 更锐 |
| Metric3D / v2 | ICCV 2023 / TPAMI 2024 | 零样本**度量**深度 |
| UniDepth / V2 | CVPR 2024 / 2025 | 无需测试时相机参数的度量深度 |
| MoGe / MoGe-2 | CVPR 2025 | 开放域单目 pointmap（→度量+法线） |
| **Depth Anything 3** | arXiv 2511.10647; ICLR 2026 | 极简 transformer + depth-ray；相对 VGGT 大增益（作者基准） |

### 2.3 Feed-forward 三维几何（2024–2026 主航道）

```text
DUSt3R (CVPR'24)
  → MASt3R (ECCV'24) → MASt3R-SfM / MASt3R-SLAM
  → Spann3R / CUT3R / Fast3R / MUSt3R (2025)
  → VGGT (CVPR'25 Best Paper)
  → π³ (ICLR'26) · StreamVGGT · VGGT-Long · MapAnything
  → DA3 · AMB3R · ZipMap · VGG-T³ · VGGT-Ω · D4RT (CVPR'26)
```

| 名称 | 出处 | 一句话 |
|---|---|---|
| **DUSt3R** | CVPR 2024 | 无标定图像对 → **pointmap**；统一单目/多目 |
| MASt3R | ECCV 2024 | + 稠密匹配；度量 pointmap |
| Spann3R / CUT3R | 3DV/CVPR 2025 | 空间记忆 / 循环状态 → 在线全局点图 |
| Fast3R | CVPR 2025 | 1000+ 视角一次前向 |
| **VGGT** | **CVPR 2025 Best Paper** | 一次前向出相机+深度+点图+轨迹；<1s |
| π³ | ICLR 2026 | 置换等变、去参考视角偏置；比 VGGT 更准更快 |
| MapAnything | 3DV 2026 | 因子化几何（射线/深度/位姿/尺度），多任务统一 |
| **VGGT-Ω** | CVPR 2026 Finalist | 干净 power-law：参数/数据 scaling 曲线 |
| **D4RT** | **CVPR 2026 Best Paper** | 视频编码一次，查询任意像素在任意时刻/相机下的 3D |
| ZipMap / VGG-T³ | CVPR 2026 | test-time training 固定状态 → **线性时间**长序列重建 |
| AMB3R | CVPR 2026 | 稀疏体素 backend 挂在冻结 VGGT 上；feed-forward VO 超优化 SLAM |

**收敛共识：** Pointmap / depth-ray 是统一输出；位姿可选；多视角 transformer 是默认骨干；**经典 BA 与 feed-forward 是互补而非替代**。

---

## 3. 神经渲染与 3DGS

### 3.1 NeRF → 3DGS

| 名称 | 出处 | 贡献 |
|---|---|---|
| NeRF | ECCV 2020 | 可微体渲染奠基 |
| Instant-NGP | SIGGRAPH 2022 | 哈希编码，秒级训练 |
| Mip-NeRF 360 / Zip-NeRF | CVPR 2022 / ICCV 2023 | 无界抗锯齿质量基线 |
| **3D Gaussian Splatting** | **SIGGRAPH 2023** | 各向异性高斯 + tile 光栅；实时 1080p NVS |
| Mip-Splatting | CVPR 2024 Best Student | 多尺度抗锯齿 |
| Scaffold-GS / Octree-GS | CVPR 2024 / TPAMI 2025 | 结构化锚点 + LOD |
| Spec-Gaussian | NeurIPS 2024 | 高光/各向异性外观 |
| AbsGS / GaussianPro | ACM MM / ICML 2024 | densification 修复 |
| VastGaussian / Hierarchical 3DGS / CityGaussian | CVPR/SIGGRAPH/ECCV 2024 | 城市场景：分区 + 合并 + LOD |

### 3.2 从“好看”到“有表面”

| 名称 | 出处 | 贡献 |
|---|---|---|
| NeuS / VolSDF / Neuralangelo | NeurIPS'21 / ICLR'21 / CVPR'23 | 神经 SDF 表面线 |
| **2DGS** | SIGGRAPH 2024 | 扁平高斯盘 → 强表面 + TSDF mesh |
| SuGaR / Gaussian Surfels / Frosting | CVPR/SIGGRAPH/ECCV 2024 | 表面对齐高斯 → 可编辑 mesh+GS |
| FlexiCubes / DiffMC | SIGGRAPH 2023 等 | 可微等值面抽取 |

### 3.3 动态 / 4D 重建（渲染侧）

| 名称 | 出处 | 贡献 |
|---|---|---|
| 4D-GS / Deformable 3DGS | CVPR 2024 | 规范高斯 + 变形场；实时动态 NVS |
| Ex4DGS | NeurIPS 2024 | 全显式 4DGS |
| MoSca | CVPR 2025 | 随意视频 + 运动脚手架 |
| Shape of Motion | ICCV 2025 | 单目 4D：SE(3) 运动基 + 跟踪先验 |

**两个体制：** 多相机工作室（基本可用）vs 随意单目视频（仍高度依赖先验、易失败）。

---

## 4. 3D / 4D 生成

### 4.1 物体级：三代演进，配方收敛

| 世代 | 代表 | 为何胜/败 |
|---|---|---|
| Gen-1 SDS | DreamFusion, Magic3D, ProlificDreamer | 慢、Janus、过饱和；有 Objaverse 后退出前沿 |
| Gen-2 MV+LRM | Zero-1-to-3, InstantMesh, LGM, TripoSR, CRM | 快；几何欠定 → 退居贴图前端 |
| **Gen-3 Native 3D** | **TRELLIS / TRELLIS.2, Hunyuan3D 2.x, CLAY, TripoSG** | 秒级、可扩展；**VecSet vs sparse SLAT/O-Voxel** |

| 名称 | 出处 | 贡献 | 权重 |
|---|---|---|---|---|
| 3DShape2VecSet | SIGGRAPH 2023 | 发明 **VecSet** latent | Open |
| TRELLIS | CVPR 2025 | **SLAT** + rectified flow；一解 mesh/GS/NeRF | Open |
| Hunyuan3D 2.0/2.1 | 2025.01/06 | Shape DiT + Paint；2.1 全开源 + **PBR** | Open |
| TRELLIS.2 | 2025.12 | **O-Voxel**；开曲面 + PBR；开源 SOTA | Open |
| CLAY / Rodin | SIGGRAPH 2024 | 1.5B VecSet；产品 Hyper3D Rodin | API |
| Meshtron / MeshMosaic | 2024–2026 | 自回归艺术家级 mesh（64K–100K+ 面） | 混合 |
| Seed3D 1.0 | 2025.10 | 面向仿真就绪资产 | API |

**已 converging 的约定：** 图像条件 ≫ 原生文本条件；几何与 PBR 两阶段；flow matching 优于 DDPM；稀疏注意力使 1024³ 可在 ~8 GPU 训练。

### 4.2 场景 / 世界资产生成

| 名称 | 出处 | 路线 |
|---|---|---|
| LucidDreamer / WonderWorld | 2023–24 | 单图递归扩展 3DGS |
| BlockFusion / InfiniCube | 2024 | 块/地图条件的可扩展世界 |
| SceneCraft | ICML 2024 | LLM → 场景图 → Blender 代码 |
| HunyuanWorld 1.0 | 2025.07 | 全景代理 → 分层 mesh；可导出 |
| **WorldGrow** | AAAI 2026 Oral | 块级 3D inpainting；无界可走世界 |
| Lyra / Lyra 2.0 | 2025–26 | 视频扩散知识蒸馏成前馈 3DGS |
| **WorldGen (Meta)** | CVPR 2026 | LLM 布局 + 程序化 + navmesh；引擎就绪 |
| **World Labs Marble** | GA 2025.11 | 产品化持久世界；splat/mesh/collider 导出 |

**成功配方：** 语义/几何脚手架（LLM 布局 / HD map / 全景层 / 块网格）+ lift（image-to-3D / 视频先验 / 程序化），**不是**一个巨型端到端 latent。

### 4.3 4D 生成

| 名称 | 出处 | 贡献 |
|---|---|---|
| Align Your Gaussians / DreamGaussian4D / 4D-fy | 2023–24 | SDS/优化式 text/video→4D |
| Diffusion4D / L4GM / SV4D | 2024 | 前馈 / 视频扩散驱动的动态资产 |
| DynamiCrafter | ECCV 2024 | 常被复用为运动驱动先验 |

**现状：** 单物体动画基本可用；**场景级、长时程、物理合理、多物体 4D 仍未解决。**

### 4.4 可控 / 贴图 / 资产生成化

| 方向 | 代表 |
|---|---|
| UV/PBR 贴图 | TEXTure, Paint3D, FlashTex, MaterialMVP |
| 部件级 | PartCrafter, OmniPart, Nano3D, Hunyuan3D-Omni |
| 重拓扑/序列 mesh | MeshGPT → BPT → Meshtron → MeshMosaic |
| 生产就绪 | Hunyuan3D Studio, SF3D, Seed3D；瓶颈=assetization |

---

## 5. 3D 理解（Understanding）

### 5.1 驾驶：BEV → Occupancy → 规划

| 名称 | 出处 | 贡献 |
|---|---|---|
| BEVFormer / PETR / Sparse4D | 2022–23 | 相机→BEV/稀疏查询检测 |
| TPVFormer / OccFormer / SurroundOcc | CVPR/ICCV 2023 | 视觉语义占用 |
| UniAD / VAD | CVPR/ICCV 2023 | 规划导向端到端 / 向量化场景 |
| OccWorld / OccSora | ECCV 2024 / 2024 | 占用空间世界模型（AR / 扩散） |
| GaussianOcc / EmbodiedOcc | ICCV 2025 | 高斯占用；室内在线记忆 |
| **NAVSIM** | NeurIPS 2024 D&B | 数据驱动规划评测（PDMS→EPDMS） |

### 5.2 点云基础模型与开放词汇

| 名称 | 出处 | 贡献 |
|---|---|---|
| Point-BERT / Point-MAE | CVPR/ECCV 2022 | 点云掩码预训练奠基 |
| **PTv3** | CVPR 2024 Oral | 序列化替代 KNN；默认骨干 |
| Pointcept | 代码库 | 该方向 Detectron 级基础设施 |
| Sonata | CVPR 2025 | SSL PTv3；线性探测终于像样 |
| Uni3D / OpenScene | ICLR 2024 / CVPR 2023 | CLIP 对齐的开放词汇 3D |

### 5.3 空间 VLM / 3D-LLM

| 名称 | 出处 | 贡献 |
|---|---|---|
| 3D-LLM / LEO / LL3DA / Chat-Scene | 2023–24 | 把 3D 特征注入 LLM |
| SpatialVLM / SpatialRGPT / SpatialBot | CVPR/NeurIPS 2024 / ICRA 2025 | **度量**空间问答与区域 grounding |
| LLaVA-3D | ICCV 2025 | 2D VLM + 3D 位置嵌入 |
| VGGT → SpatialStack 等 | 2025–26 | **几何 transformer 当 VLM 骨干** |

**关键判断：** 多数 VLM 仍是 2D 预训练；“会说左右”≠“真度量可靠”。VGGT 类几何骨干是 2025–26 的重要补丁。

---

## 6. SLAM 与在线建图

| 名称 | 出处 | 贡献 |
|---|---|---|
| ORB-SLAM / VINS 等 | 经典 | 几何 SLAM 基线 |
| GS-SLAM / SplaTAM / MonoGS / Photo-SLAM | CVPR 2024 | 第一波 3DGS 稠密 SLAM |
| WildGS-SLAM | CVPR 2025 | 动态环境下的单目 GS-SLAM |
| MASt3R-SLAM | CVPR 2025 | 双目重建先验上的实时 SLAM |
| VGGT-SLAM / 2.0 | NeurIPS 2025 / RSS 2026 | SL(4) 流形对齐；真机 Jetson |

**现状：** 静态室内 photoreal 地图可行；**动态、大尺度漂移、与 feed-forward 几何的在线耦合**仍是系统问题。

---

## 7. 机器人与具身 3D

| 方向 | 代表 | 要点 |
|---|---|---|
| 抓取 / 6D 位姿 | AnyGrasp, FoundationPose, BundleSDF | 从实例到类别级 |
| 高斯作机器人记忆 | EmbodiedOcc, 操作向 3DGS | 可编辑、可导出、可抓 |
| 几何接地世界模型 | Aether, TesserAct, WorldGrow | 相机轨迹当动作；RGB+深度+法线 |
| 视频/WAM 交叉 | 见 `01-related-work.md` / `03-survey-embodied-driving.md` | 显式 3D vs 视频想象 |

**室内数据：** ScanNet, ARKitScenes, Habitat, ProcTHOR, EmbodiedScan, SceneFun3D。

---

## 8. 与 World Model 的交界：显式 vs 隐式

| | **显式 3D（Marble 系）** | **隐式视频（Genie 系）** |
|---|---|---|
| 状态 | splat / mesh / occupancy | 像素 / latent token |
| 优点 | 可导出、可碰撞、可编辑、长期几何一致 | 交互强、动态丰富、易吃互联网视频 |
| 缺点 | 动态与开放交互弱、生成难 | 漂移、不可验证几何、难进物理引擎 |
| 代表 | Marble, WorldGrow, Lyra, OccWorld | Genie 3, Matrix-Game, Cosmos, Odyssey |
| 何时选 | 仿真、工具链、安全闭环、资产管线 | 开放交互、想象训练、快速原型 |

**World Labs（2026.06）功能三分法（有影响力，是否 canonical 仍早）：**
Renderer（出像素）/ Simulator（出可计算物理状态）/ Planner（出动作）。

**杂交正在发生：** Geometry Forcing、Lyra（视频→3DGS）、MosaicMem / PERSIST（显式记忆进视频 WM）、DrivingSphere（占用世界→视频传感器模型）。

---

## 9. 评测与数据（常被低估）

| 类型 | 代表 | 提醒 |
|---|---|---|
| NVS 质量 | Mip-NeRF 360 场景, PSNR/SSIM/LPIPS | **≠ 几何准、≠ 可规划** |
| 几何 | ETH3D, DTU, Tanks&Temples, AUC@θ | 紧阈值才测精度 |
| 物体生成 | Objaverse；3D Arena 人评 | 人偏爱贴图/渲染格式，与生产需求相反 |
| 驾驶 | nuScenes, Occ3D, **NAVSIM**, Bench2Drive | open-loop ≠ closed-loop |
| 空间语言 | SpatialVLM/SpatialBench 类 | 需区分定性与度量 |
| World Model | WorldScore, WorldMark, WorldLens… | 见 `01-related-work.md` |

---

## 10. 按方向的“已解决 / 未解决”

| 方向 | 相对已可用 | 明显未解决 |
|---|---|---|
| 静态 NVS | 3DGS 高保真实时 | 城市级可更新孪生 |
| 几何前端 | VGGT/DA3 粗糙到可用 | 与 COLMAP 精度对齐；动态长序列 |
| 物体生成 | 秒级 image→3D 草稿 | 生产级拓扑/UV/PBR/绑定 |
| 场景生成 | 可漫游 demo | 布局可控 + 物理交互 + 无限扩展 |
| 4D | 工作室多视角动态 | 单目→可仿真 4D |
| 驾驶占用 | 封闭集占用/检测 | 开放世界可靠性；闭环安全 |
| 空间 VLM | 能聊空间 | **真度量**与跨场景稳定 |
| SLAM+GS | 静态室内好看地图 | 动态 + 大尺度 + 边缘端 |
| World Model | 分钟级交互 / 可导出世界 | 小时级一致；动作因果；评测诚实 |

---

## 11. 小实验室（8–32 GPU）仍肥沃的缝

1. **Feed-forward 几何 × 动态/驾驶**（Amb3R/D4RT 之后的应用层仍稀）
2. **生成资产 → 可仿真**（碰撞、质量、关节；Seed3D 提出问题但未开放）
3. **开放词汇占用 × 具身导航**
4. **显式世界 × 可学习规划器**（renderer/simulator → planner）
5. **压缩 / 流式 4DGS**
6. **评测：视觉质量 ≠ 控制/资产效用**（与 world model 路线 α 同构）
7. **记忆写入 / 画外演化 / 相机-化身解耦**（见 `02-idea-candidates.md`）

**不建议：** 再刷一个静态 3DGS 小改进；再做一个无下游的通用 3D benchmark；在 fidelity 上硬刚 Marble/Genie。

---

## 12. 验证注意事项

- **高置信：** 3DGS 原论文、DUSt3R/MASt3R/VGGT、TRELLIS/Hunyuan3D 2.1、PTv3、OccWorld、NAVSIM、BEVFormer 等有 CVF/arXiv 主页支撑的条目。
- **需再核：** 部分 CVPR 2026 条目（ZipMap 别名、部分 efficiency 论文 ID）、Genie 3 / Marble 产品细节（无技术报告）、商业 API 排行（季度洗牌）。
- **GLUEMAP 的数字**表明：feed-forward **不是**全面取代 COLMAP；引用时务必说明场景类型。
- 与 world model 文档交叉的 2026 ID 规则同 `01-related-work.md`。

---

## 13. 推荐入口（先读这些）

| 目的 | 资源 |
|---|---|
| 表征总览 | arXiv 2410.06475；2606.04871 |
| 3DGS | Kerbl outlook 2510.26694；综述 2401.03890 |
| Feed-forward 几何 | 综述 2507.08448；VGGT 2503.11651 |
| 物体生成 | 生产向综述 2604.23629；TRELLIS；Hunyuan3D 2.1 |
| 场景/世界 | worldbench.github.io/survey；Marble / WorldGrow |
| 驾驶 3D | UniAD；OccWorld；NAVSIM |
| 空间语言 | SpatialVLM；VGGT→VLM 线 |
| 与 WM 交叉 | 本仓库 `01`/`02`/`03`；World Labs taxonomy 博文 |
