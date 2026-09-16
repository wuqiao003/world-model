# 3D Scene Understanding, Spatial AI, and Driving/Robotics 3D
### Literature survey, 2023 – August 2026
*Compiled 5 August 2026. Claims traced to primary sources where possible; uncertain items flagged with ⚠. No invented arXiv IDs.*

---

## 0. How to read this report

Three shifts organize 3D / spatial AI between early 2023 and August 2026:

1. **From BEV boxes to occupancy and then to generative 4D occupancy.** Camera-only pipelines matured (BEVFormer → PETR/Sparse4D → OccFormer/SurroundOcc/TPVFormer), then occupancy became both a perception target *and* a world-model substrate (OccWorld, OccSora). NAVSIM (NeurIPS 2024) reset how end-to-end planners are ranked.
2. **From task-specific 3D nets to point-cloud foundation models and CLIP-aligned open vocab.** PTv3 (CVPR 2024 Oral) + Pointcept became the default backbone stack; Sonata (CVPR 2025) asked whether linear probing finally “works” for points; Uni3D / OpenScene pulled language into 3D.
3. **From injecting point clouds into LLMs to geometry transformers as VLM backbones.** 3D-LLM / LEO / Chat-Scene / LL3DA / LLaVA-3D established the 3D-LLM class; SpatialVLM / SpatialRGPT / SpatialBot attacked *metric* spatial VQA; VGGT (CVPR 2025 Best Paper) made feed-forward multi-view geometry a reusable backbone for spatial VLMs (SpatialStack, VG-LLM line). Meanwhile World Labs’ Marble vs DeepMind’s Genie 3 crystallized the **explicit-3D simulator vs interactive video renderer** debate (taxonomy essay, June 2026).

**Anchor surveys / maps:**
- *3D and 4D World Modeling: A Survey* — arXiv [2509.07996](https://arxiv.org/abs/2509.07996); project <https://worldbench.github.io/survey> (VideoGen / OccGen / LiDARGen taxonomy).
- Companion WM survey in this folder: `03-survey-embodied-driving.md` (WAMs, Genie 3, OmniDreams, World Engine).

---

## 1. Taxonomy at a glance

| Axis | Dominant 2023 form | Dominant 2025–2026 form | Why it matters |
|---|---|---|---|
| Driving scene state | BEV features + 3D boxes | Semantic occupancy / Gaussian occupancy + vectorized maps | Occupancy expresses free space without fixed class lists; Gaussians speed rendering |
| Temporal model | Multi-frame BEV fusion | Occupancy / video / LiDAR *world models* | Forecasting moves from box tracks to full scene evolution |
| Point backbone | SparseUNet / PTv2 | **PTv3** (+ Sonata SSL) in Pointcept | Scale + serialization beat careful neighbor search |
| Open vocab 3D | PointCLIP-style adapters | OpenScene / Uni3D / language-aligned Gaussians | CLIP space becomes query interface for scenes |
| Spatial language | Caption / QA on ScanNet | Metric VQA + region grounding + VGGT-injected VLMs | Distance/size/layout become first-class |
| Robot memory | NeRF / TSDF | **3DGS** as editable scene + grasp / pose memory | Explicit primitives support edit, grasp, and export |
| World-model substrate | Video pixels | Split: **renderer** (Genie 3) vs **simulator** (Marble / Occ / 3DGS) | Explicit 3D wins when collision, editability, or physics export matter |

---

## 2. Driving 3D: detection, occupancy, planning, NAVSIM era

### 2A. Core methods (verified)

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 1 | **BEVFormer** | arXiv 2203.17270 (31 Mar 2022); ECCV 2022; later TPAMI | Spatiotemporal transformers over grid BEV queries; became the default camera→BEV encoder for UniAD-class stacks | [arXiv](https://arxiv.org/abs/2203.17270) · [code](https://github.com/fundamentalvision/BEVFormer) |
| 2 | **PETR** | arXiv 2203.05625 (10 Mar 2022); ECCV 2022 | 3D position embedding transformation — sparse DETR-style queries without explicit BEV | [arXiv](https://arxiv.org/abs/2203.05625) |
| 3 | **PETRv2** | arXiv 2206.01256 (2 Jun 2022); ICCV 2023 | Temporal PETR + multi-task queries (det / BEV seg / 3D lane) | [arXiv](https://arxiv.org/abs/2206.01256) |
| 4 | **Sparse4D** | arXiv 2211.10581 (19 Nov 2022) | Sparse 4D keypoint sampling over view/scale/time; edge-friendly alternative to dense BEV | [arXiv](https://arxiv.org/abs/2211.10581) · [code](https://github.com/linxuewu/Sparse4D) |
| 5 | **Sparse4D v2** | arXiv 2305.14018 (23 May 2023) | Recurrent temporal fusion: O(T)→O(1) cost; long-horizon sparse memory | [arXiv](https://arxiv.org/abs/2305.14018) |
| 6 | **TPVFormer** | arXiv 2302.07817 (15 Feb 2023); **CVPR 2023** | Tri-perspective view (three orthogonal planes) for vision-based semantic occupancy | [arXiv](https://arxiv.org/abs/2302.07817) |
| 7 | **OccFormer** | arXiv 2304.05316 (11 Apr 2023); **ICCV 2023** | Dual-path transformer on 3D voxels; Mask2Former-style occupancy decoder | [arXiv](https://arxiv.org/abs/2304.05316) · [code](https://github.com/zhangyp15/OccFormer) |
| 8 | **SurroundOcc** | arXiv 2303.09551 (16 Mar 2023); **ICCV 2023** | Multi-cam → 3D volume + pipeline to densify LiDAR into occupancy GT | [arXiv](https://arxiv.org/abs/2303.09551) · [code](https://github.com/weiyithu/SurroundOcc) |
| 9 | **UniAD** | arXiv 2212.10156 (20 Dec 2022); **CVPR 2023 Best Paper** | Planning-oriented full stack: track / map / motion / occ / plan with unified queries | [arXiv](https://arxiv.org/abs/2212.10156) |
| 10 | **VAD** | arXiv 2303.12077 (21 Mar 2023); **ICCV 2023** | Fully vectorized scene (agents + map) for efficient E2E planning; faster/safer than UniAD on nuScenes open-loop | [arXiv](https://arxiv.org/abs/2303.12077) |
| 11 | **OccWorld** | arXiv 2311.16038 (27 Nov 2023); **ECCV 2024** | GPT-like world model in *occupancy* space: joint future occ + ego trajectory | [arXiv](https://arxiv.org/abs/2311.16038) · [code](https://github.com/wzzheng/OccWorld) |
| 12 | **OccSora** | arXiv 2405.20337 (30 May 2024) | Diffusion transformer over 4D scene tokens; trajectory-conditioned ~16 s occupancy generation | [arXiv](https://arxiv.org/abs/2405.20337) |
| 13 | **GaussianOcc** | arXiv 2408.11447 (21 Aug 2024); **ICCV 2025** | Fully self-supervised occ via Gaussian splatting (no GT pose); ~2.7× faster train / 5× faster render vs volume rendering | [arXiv](https://arxiv.org/abs/2408.11447) · [project](https://ganwanshui.github.io/GaussianOcc/) |
| 14 | **EmbodiedOcc** | arXiv 2412.04380 (5 Dec 2024); **ICCV 2025** | Online *indoor* embodied occupancy with Gaussian memory updated during exploration; EmbodiedOcc-ScanNet | [arXiv](https://arxiv.org/abs/2412.04380) · [code](https://github.com/YkiWu/EmbodiedOcc) |
| 15 | **NAVSIM** | arXiv 2406.15349 (21 Jun 2024); **NeurIPS 2024** D&B | Non-reactive, data-driven planning benchmark (PDMS); CVPR 2024 challenge (143 teams). v2 / pseudo-simulation → CoRL 2025 / AGC 2025 | [arXiv](https://arxiv.org/abs/2406.15349) · [code](https://github.com/autonomousvision/navsim) |

### 2B. Key findings (driving)

- **BEV was necessary but not sufficient.** BEVFormer/PETR made camera-only 3D detection practical; occupancy (TPVFormer, OccFormer, SurroundOcc) added free-space geometry that boxes cannot express (construction zones, irregular obstacles).
- **Sparse queries won the deployment argument.** Sparse4D(v2/v3) and PETR-style methods avoid dense BEV cost; industry edge stacks often prefer this family even when BEV wins open-loop NDS.
- **Planning-oriented joint training (UniAD) then vectorized efficiency (VAD)** defined 2023 E2E driving. NAVSIM then showed that on hard curated scenarios, simple TransFuser-class models can match heavy UniAD-style stacks — *benchmark design dominates architecture narrative*.
- **Occupancy world models are the cheap 3D counterpart to video WMs.** OccWorld (AR tokens) and OccSora (diffusion) forecast scene evolution without rendering RGB; later DrivingSphere-style systems (see WM survey) render multi-view video *from* occupancy. Geometry-stable, appearance-weak.
- **Gaussians entered occupancy (2024–2025).** GaussianOcc (self-sup outdoor) and EmbodiedOcc (online indoor memory) treat Gaussians as both representation and renderable memory — bridging perception and reconstruction.

### 2C. Open problems (driving 3D)

- Closed-loop correlation of NAVSIM PDMS/EPDMS with reactive sim (Bench2Drive, nuPlan, AlpaSim) still rests on small paired samples in public analyses ⚠.
- Occupancy label pipelines (Poisson densification, Occ3D, OpenOccupancy) disagree on dynamic objects and long-tail classes.
- Long-horizon 4D occupancy (> a few seconds) still drifts; OccSora-style generation is controllable but not yet a trusted planner world model.
- Camera-only metric scale without pose/LiDAR remains brittle despite GaussianOcc-style self-supervision.

---

## 3. Point-cloud foundation models and open-vocabulary 3D

### 3A. Methods

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 16 | **Point-BERT** | arXiv 2111.14819 (29 Nov 2021); **CVPR 2022** | Masked point modeling with a dVAE tokenizer — founding BERT-style 3D pretrain | [arXiv](https://arxiv.org/abs/2111.14819) |
| 17 | **Point-MAE** | arXiv 2203.06604 (13 Mar 2022); ECCV 2022 | Masked autoencoding directly on point patches (no dVAE) | [arXiv](https://arxiv.org/abs/2203.06604) |
| 18 | **Point Transformer V3 (PTv3)** | arXiv 2312.10035 (15 Dec 2023); **CVPR 2024 Oral** | Serialization replaces KNN; receptive field 16→1024; 3× speed / 10× memory vs PTv2; SOTA across 20+ indoor/outdoor tasks | [arXiv](https://arxiv.org/abs/2312.10035) · [repo](https://github.com/Pointcept/PointTransformerV3) |
| 19 | **Pointcept** | GitHub 2023– | De-facto research codebase for PTv2/PTv3, PPT multi-dataset training, outdoor/indoor configs | [GitHub](https://github.com/Pointcept/Pointcept) |
| 20 | **Sonata** | arXiv 2503.16429 (20 Mar 2025); **CVPR 2025** | Self-supervised PTv3 (encoder-only, self-distillation) on ~140k scenes / 108M params; reliable *linear probing* | [arXiv](https://arxiv.org/abs/2503.16429) · [code](https://github.com/facebookresearch/sonata) |
| 21 | **Uni3D** | arXiv 2310.06773 (10 Oct 2023); **ICLR 2024 Spotlight** | Scale 3D ViT (Point-BERT tokenizer) aligned to CLIP image–text; explores ~1B-param regime | [arXiv](https://arxiv.org/abs/2310.06773) |
| 22 | **OpenScene** | arXiv 2211.15654 (v2 6 Apr 2023); **CVPR 2023** | Distill CLIP pixel features onto 3D points (2D fusion + 3D SparseUNet); zero-shot open-vocab queries | [arXiv](https://arxiv.org/abs/2211.15654) · [project](https://pengsongyou.github.io/openscene) |

### 3B. Key findings

- **Scaling + simplicity beat inductive-bias gymnastics.** PTv3’s thesis (“performance after scale cares more about efficiency than exact neighbors”) held across ScanNet / nuScenes / Waymo-class tasks.
- **SSL for points finally looks like SSL for images — with caveats.** Sonata’s linear-probe story is the 2025 milestone; outdoor LiDAR transfer still needs care (domain gap, beam patterns).
- **Open-vocab 3D is mostly “2D foundation → 3D lift.”** OpenScene (multi-view CLIP fusion) and Uni3D (CLIP-space alignment of point transformers) both borrow 2D language supervision rather than native 3D captions (which remain scarce).
- **Codebases matter.** Pointcept concentrated follow-on work the way Detectron/mmdet3d did for detection; academic novelty often appears first as a Pointcept config.

### 3C. Open problems

- No public “ImageNet moment” for raw LiDAR at internet scale; Objaverse helps shapes more than driving scenes.
- Open-vocab metrics still lean on closed ScanNet/nuScenes label sets; true free-text 3D retrieval lacks a stable leaderboard.
- Temporal / 4D point foundation models (streaming city-scale) lag image/video foundations by years.

---

## 4. Spatial VLMs and 3D LLMs

### 4A. Methods

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 23 | **3D-LLM** | arXiv 2307.12981 (24 Jul 2023) | Inject 3D (features / location tokens) into LLMs for captioning, QA, grounding, navigation | [arXiv](https://arxiv.org/abs/2307.12981) |
| 24 | **LEO** | arXiv 2311.12871 (18 Nov 2023); **ICML 2024** | Embodied generalist: 3D VL alignment → VLA instruction tuning (nav + manipulation) | [arXiv](https://arxiv.org/abs/2311.12871) · [project](https://embodied-generalist.github.io/) |
| 25 | **LL3DA** | arXiv 2311.18651 (30 Nov 2023) | Point-cloud scene encoder + visual prompts + frozen LLM; dense caption / 3D QA | [arXiv](https://arxiv.org/abs/2311.18651) · [project](https://ll3da.github.io/) |
| 26 | **Chat-Scene** | arXiv 2312.08168 (Dec 2023) | Object-identifier referencing for grounded 3D dialogue (bridges scene ↔ LLM) | [arXiv](https://arxiv.org/abs/2312.08168) |
| 27 | **Scene-LLM** | arXiv 2403.11401 (18 Mar 2024) | Hybrid scene-level + egocentric 3D features for interactive indoor planning | [arXiv](https://arxiv.org/abs/2403.11401) |
| 28 | **SpatialVLM** | arXiv 2401.12168 (22 Jan 2024); **CVPR 2024** | Internet-scale metric spatial VQA synthesis (~2B pairs / 10M images); qualitative + quantitative spatial QA | [arXiv](https://arxiv.org/abs/2401.12168) · [project](https://spatial-vlm.github.io/) |
| 29 | **SpatialRGPT** | arXiv 2406.01584 (3 Jun 2024); **NeurIPS 2024** | Region-level grounded spatial reasoning; depth plugin into VLM encoder | [arXiv](https://arxiv.org/abs/2406.01584) |
| 30 | **SpatialBot** | arXiv 2406.13642 (19 Jun 2024); **ICRA 2025** | RGB-D VLM + Depth API; SpatialQA / SpatialQA-E / SpatialBench; embodiment pick-place | [arXiv](https://arxiv.org/abs/2406.13642) · [code](https://github.com/BAAI-DCAI/SpatialBot) |
| 31 | **LLaVA-3D** | arXiv 2409.18125 (26 Sep 2024); **ICCV 2025** | 3D position embeddings on LLaVA patches; direct 3D box decode; keeps 2D ability | [arXiv](https://arxiv.org/abs/2409.18125) |
| 32 | **VGGT** | arXiv 2503.11651 (14 Mar 2025); **CVPR 2025 Best Paper** | Feed-forward transformer: cameras, depth, point maps, tracks from 1–hundreds of views in seconds | [arXiv](https://arxiv.org/abs/2503.11651) · [code](https://github.com/facebookresearch/vggt) |
| 33 | **SpatialStack** | **CVPR 2026** (arXiv 2603.27437 ⚠ verify version before cite) | Layered geometry→LLM fusion using VGGT-class encoders; inject geometry into LLM layers | [CVPR OA](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_SpatialStack_Layered_Geometry-Language_Fusion_for_3D_VLM_Spatial_Reasoning_CVPR_2026_paper.pdf) · [arXiv](https://arxiv.org/abs/2603.27437) |

**Cosine / Cosine3D:** No mainstream peer-reviewed method under this name was found in 2023–Aug 2026 searches (arXiv / CVF / major venues). ⚠ Treat as **non-existent or non-standard naming** unless a primary source is supplied. Do not confuse with unrelated “cosine similarity” losses or vendor product nicknames.

### 4B. Gemini / GPT spatial claims (vendor + academic stress tests)

| Claim source | What is actually supported | Caution |
|---|---|---|
| **GPT-4V / GPT-4o** product demos | Strong qualitative spatial language; coarse layout / counting | Academic benches (OpenEQA comparisons in LLaVA-3D; object-centric spatial benches e.g. arXiv [2509.21922](https://arxiv.org/abs/2509.21922)) show **fine-grained metric / relational gaps** vs specialized models |
| **Gemini / Gemini-Pro** | Competitive multimodal QA; localization features in later Gemini apps ⚠ product surface changes fast | LLaVA-3D reports surpassing Gemini-Pro on OpenEQA-style splits with far fewer params — *as of that paper’s eval*, not a permanent ranking |
| **SpatialVLM / SpatialRGPT / SpatialBot** | Explicitly close the metric-distance hole left by GPT/Gemini-style VLMs | Still template-sensitive; robotics needs real depth (SpatialBot’s thesis) |

### 4C. Key findings

- **Two lineages diverged then recombined:** (i) *explicit 3D tokens* into LLMs (3D-LLM, LEO, LL3DA, Chat-Scene, Scene-LLM, LLaVA-3D); (ii) *2D VLMs + synthetic spatial supervision* (SpatialVLM) or *depth/region plugins* (SpatialRGPT, SpatialBot).
- **VGGT is the 2025 backbone shock.** After DUSt3R/MASt3R, VGGT made multi-view geometry a single forward pass; 2025–2026 spatial VLMs increasingly treat VGGT (or cousins) as the vision tower instead of frozen CLIP-only towers.
- **Grounding remains the differentiator.** Chat-Scene’s object IDs and SpatialRGPT’s regions matter more for agents than open-ended 3D caption BLEU.
- **Embodiment needs sensors.** SpatialBot’s RGB-D / Depth-API design is the practical robotics stance; pure RGB metric VQA remains approximate.

### 4D. Open problems

- Unified benchmark spanning *metric distance*, *region grounding*, *dynamic 4D*, and *egocentric robotics* (VSI-Bench / 3DSRBench / SpatialBench / OpenEQA fragment the story).
- Hallucinated geometry when VGGT fails (specular, motion blur, textureless walls) propagates into LLM answers with high confidence.
- Proprietary GPT/Gemini spatial rankings are moving targets; cite dated eval protocols, not blog claims.

---

## 5. Robotics spatial: grasp, pose, Gaussians, camera-as-action WMs

### 5A. Methods

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 34 | **AnyGrasp** | arXiv 2212.08333 (16 Dec 2022); **IEEE T-RO 2023** | Dense 7-DoF grasp perception + temporal association; 93.3% bin-clearing on 300+ unseen objects (paper claim) | [arXiv](https://arxiv.org/abs/2212.08333) · [site](https://graspnet.net/anygrasp.html) |
| 35 | **BundleSDF** | arXiv 2303.14158 (24 Mar 2023); **CVPR 2023** | Near real-time 6-DoF track + neural SDF reconstruction of unknown objects from RGB-D | [arXiv](https://arxiv.org/abs/2303.14158) · [project](https://bundlesdf.github.io/) |
| 36 | **FoundationPose** | arXiv 2312.08344 (13 Dec 2023); **CVPR 2024 Highlight** | Unified novel-object 6D pose *and* tracking; CAD **or** few-shot references; large-scale synthetic + LLM-aided assets | [arXiv](https://arxiv.org/abs/2312.08344) · [project](https://nvlabs.github.io/FoundationPose/) |
| 37 | **ManiGaussian** | arXiv 2403.08321 (13 Mar 2024) | Dynamic Gaussian Splatting world model for language-conditioned multi-task manipulation (RLBench) | [arXiv](https://arxiv.org/abs/2403.08321) |
| 38 | **GraspSplats** | arXiv 2409.02084 (3 Sep 2024) | Feature-enhanced 3DGS in <30 s; part-level language grasps; editable under rigid motion | [arXiv](https://arxiv.org/abs/2409.02084) |
| 39 | **GaussianVLM** | arXiv 2507.00886 (Jul 2025) | Scene-centric VLM over language-aligned Gaussian primitives (detector-free) | [arXiv](https://arxiv.org/abs/2507.00886) |
| 40 | **Aether** | arXiv 2503.18945 (24 Mar 2025); **ICCV 2025** | Geometry-aware unified WM: 4D reconstruction + action-conditioned prediction + goal planning; **camera / raymap as action**; synthetic→real zero-shot | [arXiv](https://arxiv.org/abs/2503.18945) · [code](https://github.com/InternRobotics/Aether) |
| 41 | **TesserAct** | arXiv 2504.20995 (29 Apr 2025) | 4D embodied WM predicting RGB + depth + normals jointly for spatial consistency / IDM | [arXiv](https://arxiv.org/abs/2504.20995) |
| 42 | **WorldGrow** | arXiv 2510.21682 (24 Oct 2025); **AAAI 2026 Oral** | Unbounded explicit 3D world generation via block-wise 3D inpainting + refinement (persistent walkable scenes) | [arXiv](https://arxiv.org/abs/2510.21682) · [code](https://github.com/world-grow/WorldGrow) |

### 5B. Key findings

- **Pose/grasp foundations matured earlier than scene WMs.** AnyGrasp (grasp), BundleSDF (object field), FoundationPose (unified novel-object pose) are now default building blocks in many manipulation stacks.
- **3DGS replaced NeRF as the robotics scene memory** for speed and editability: ManiGaussian (dynamics + action), GraspSplats (part grasp), EmbodiedOcc / GaussianOcc (occupancy), Marble export (sim), GaussianVLM (language on splats).
- **Camera-as-action is a real design pattern.** Aether encodes camera trajectories (raymaps) as the controllable action interface — aligning geometric reconstruction with generative video prediction and visual planning. This is the cleanest academic articulation of “navigate the latent world by moving a camera.”
- **RGB-D-Normal (TesserAct) and explicit growable 3D (WorldGrow)** attack video WM failure modes: spatial inconsistency and lack of persistent geometry.

### 5C. Open problems

- Dynamic non-rigid / deformable / fluid scenes still break Gaussian robot memory.
- Pose foundations assume segmentation / references; cluttered egocentric video remains hard.
- Few public results close the loop from Gaussian memory → contact-rich control under domain shift (sim results ≠ kitchen).

---

## 6. Indoor scene understanding: datasets & simulators

| Resource | Era / venue | Scale & role | Link |
|---|---|---|---|
| **ScanNet** | 2017 (still standard) | ~1.5k RGB-D reconstructions; semantic/instance backbone of indoor 3D | [project](http://www.scan-net.org/) |
| **ARKitScenes** | 2021+ | ~5k real scans with LiDAR/RGB; boxes; entered EmbodiedScan v2 | [GitHub](https://github.com/apple/ARKitScenes) |
| **Habitat** | Facebook / Meta platform | Photoreal nav sim on Matterport/Gibson/HM3D; standard embodied nav | [site](https://aihabitat.org/) |
| **ProcTHOR** | NeurIPS 2022 | Procedural interactive houses at scale; pretrain → zero-shot Habitat gains | [project](https://procthor.allenai.org/) |
| **EmbodiedScan** | **CVPR 2024** | Ego-centric multi-modal suite: ~5k scans, ~1M RGB-D views, ~160k 9-DoF boxes (760+ cats), occ, language; baseline Embodied Perceptron | [project](https://tai-wang.github.io/embodiedscan/) · [code](https://github.com/InternRobotics/EmbodiedScan) |
| **SceneFun3D** | **CVPR 2024 Oral** | 710 high-res scenes, >14.8k functional interaction annotations (handles/knobs/…), motion params, language tasks; functionality seg / affordance / 3D motion | [project](https://scenefun3d.github.io/) |
| **EmbodiedOcc-ScanNet** | ICCV 2025 (with EmbodiedOcc) | Reorganized ScanNet local annotations for *online embodied occupancy* | [arXiv](https://arxiv.org/abs/2412.04380) |

**Finding:** 2024 shifted indoor 3D from “segment the mesh” to **ego-centric, language-grounded, affordance-aware** suites (EmbodiedScan, SceneFun3D). Simulators (Habitat, ProcTHOR) remain the only way to get interaction volume; real scans remain the fidelity anchor.

**Gap:** Motion / articulation labels (SceneFun3D) and dense occupancy (EmbodiedScan / EmbodiedOcc) are still poorly linked to learned world models that *act*.

---

## 7. Connection to world models: when explicit 3D beats video

### 7A. Marble vs Genie 3 (the public debate)

| | **Genie 3** (Google DeepMind) | **Marble** (World Labs) |
|---|---|---|
| Announced | Blog **5 Aug 2025** | Preview mid-2025; **general availability** blog (multimodal Marble); commercial product surface |
| Primary artifact | Interactive **video** frames @ ~720p / 24 fps, minutes of consistency | **Explicit 3D** worlds (Gaussian splats + meshes), editable/exportable |
| World Labs taxonomy (3 Jun 2026) | Classified as a **renderer** (pixels; no explicit 3D contract) | Positioned toward **simulator** (state you can collide / export) |
| Strength | Real-time playability, promptable events, agent research access | Persistence, editing, Omniverse/Isaac Sim export paths |
| Weakness (as argued by World Labs) | Looks right; may not be *structurally* right under novel camera/paths | Interactivity / long-horizon dynamics still “future chapter” |

Sources:
- Genie 3: <https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/>
- Marble multimodal blog: <https://www.worldlabs.ai/blog/marble-world-model>
- Taxonomy essay: <https://www.worldlabs.ai/blog/taxonomy-of-world-models> (3 Jun 2026) — **Renderer / Simulator / Planner**
- NVIDIA Isaac Sim + Marble workflow: <https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/>

⚠ Product pricing, access tiers, and capability lists change; treat blogs as dated snapshots. Genie 3 has **no peer-reviewed paper** as of this compilation.

### 7B. Decision rule: when is explicit 3D better?

Prefer **explicit 3D / occupancy / 3DGS / meshes** when:

1. **Collision / free-space** is the objective (planning cost maps, NAVSIM-style progress+TTC, robot motion planning).
2. **Editability & composition** matter (insert objects, grow rooms — WorldGrow, Marble edit/expand).
3. **Export to a physics engine** is required (Isaac Sim colliders, not just pretty rollouts).
4. **Metric spatial QA / manipulation** needs calibrated geometry (SpatialBot, FoundationPose, grasp sampling on Gaussians).
5. **Long-term persistence** beyond a video memory window (~1 min Genie-class visual memory vs stored splat/mesh).

Prefer **video world models / renderers** when:

1. **Appearance diversity & internet-scale priors** dominate (human video pretraining, DreamDojo-class).
2. You need **real-time interactive pixels** for human-in-the-loop or agent play (Genie 3).
3. Downstream policy is a **VLA that already lives in RGB** and only needs imagination rollouts (WAM line — see `03-survey-embodied-driving.md`).
4. Explicit 3D assets are unavailable and lifting would dominate the budget.

**Hybrid pattern gaining evidence (2025–2026):** occupancy or depth as *verifier / co-trainer* on top of video imagination (DriveDreamer-Policy-style depth+video; DrivingSphere occ→video; Aether geometry+generation). World Labs’ claim that the **simulator is the linchpin** is a strategic bet, not a settled theorem — but academic OccWorld/OccSora/WorldGrow/Aether results rhyme with it for control-facing metrics.

---

## 8. Academic-lab gaps (8–32 GPU, public data)

Directions that are idea-bound more than cluster-bound:

1. **Occupancy (or depth) as a verifier for video WMs.** Run OccWorld/OccSora-class predictors as reward/filter on open video WM rollouts; measure collision/off-road vs FVD. Explicitly motivated by WorldLens-type findings that pretty ≠ safe (see WM survey).
2. **VGGT features as frozen spatial backbone — systematic ablation.** Swap CLIP vs VGGT vs RGB-D towers inside one VLM (Qwen2.5-VL / LLaVA-class) on SpatialBench + OpenEQA + a robot sim; report *metric* error, not only VQA accuracy.
3. **Gaussian memory interfaces for VLAs.** ManiGaussian/GraspSplats show representation value; missing is a clean API: language → splat query → AnyGrasp/FoundationPose → action, with public RLBench/ManiSkill ablations.
4. **EmbodiedScan × SceneFun3D joint baseline.** Detection/occupancy stacks ignore affordance motion; affordance papers ignore ego online memory (EmbodiedOcc). A single ego-centric baseline predicting boxes + functional parts + short-horizon motion is still under-served.
5. **NAVSIM v2 pseudo-simulation stress tests for sparse vs dense perception.** Re-evaluate Sparse4D-backed vs BEVFormer-backed planners under identity/occlusion perturbations; architecture papers rarely report this.
6. **Sonata-style probing for *driving* LiDAR.** Sonata’s headline is indoor-centric linear probes; a rigorous outdoor recipe (nuScenes/Waymo) with frozen encoders would be widely cited.
7. **Counterfactual spatial VQA.** SpatialVLM generates metric questions from estimated depth — build a set where *geometry is systematically wrong* and measure whether models trust language priors over pixels (diagnostic, not a new foundation model).
8. **Avoid:** training another billion-param Uni3D; chasing GPT/Gemini leaderboard screenshots; inventing “Cosine3D”-branded wrappers without a real method.

---

## 9. Uncertainty register

| Item | Status |
|---|---|
| **Cosine / Cosine3D** | **Not found** as a mainstream 2023–2026 paper/product in this survey pass |
| **Sparse4D v3** details | Exists in community / follow-on literature; primary arXiv not re-pulled this session — cite v1 `2211.10581` / v2 `2305.14018` unless verified |
| **SpatialStack arXiv 2603.27437** | Seen via CVPR 2026 OA + ar5iv HTML; confirm latest version before camera-ready citation |
| **GaussianFormer** (Huang et al., ECCV-era occupancy-as-Gaussians) | Cited by EmbodiedOcc / GaussianOcc; treat as related prior — ID not independently re-fetched here |
| **Genie 3 / Marble** quantitative robotics claims | Blog / vendor / secondary press only; no peer-reviewed head-to-head |
| **GPT-4o / Gemini spatial rankings** | Eval-date dependent; LLaVA-3D and 2509.21922 give *snapshots*, not eternal orderings |
| **AnyGrasp success-rate figures** | From authors’ T-RO paper; hardware/setup sensitive |
| **World Labs taxonomy** | Strategic essay (3 Jun 2026), not an empirical meta-analysis — use as framing, not as proof |

---

## 10. Suggested citation spine (minimal reading list)

1. BEVFormer (`2203.17270`) → UniAD (`2212.10156`) → VAD (`2303.12077`) → NAVSIM (`2406.15349`)
2. TPVFormer (`2302.07817`) / OccFormer (`2304.05316`) / SurroundOcc (`2303.09551`) → OccWorld (`2311.16038`) → OccSora (`2405.20337`) → GaussianOcc (`2408.11447`)
3. Point-BERT (`2111.14819`) → PTv3 (`2312.10035`) → Sonata (`2503.16429`); side path OpenScene (`2211.15654`) / Uni3D (`2310.06773`)
4. 3D-LLM (`2307.12981`) → LEO (`2311.12871`) → SpatialVLM (`2401.12168`) → SpatialRGPT (`2406.01584`) → VGGT (`2503.11651`) → LLaVA-3D (`2409.18125`)
5. BundleSDF (`2303.14158`) → FoundationPose (`2312.08344`) → ManiGaussian (`2403.08321`) → Aether (`2503.18945`) / TesserAct (`2504.20995`) / WorldGrow (`2510.21682`)
6. EmbodiedScan (CVPR 2024) + SceneFun3D (CVPR 2024) + EmbodiedOcc (`2412.04380`)
7. Genie 3 blog + Marble blog + World Labs taxonomy (Jun 2026) + `03-survey-embodied-driving.md`

---

*End of survey. Cross-check any ⚠ row against arXiv abs pages before submitting camera-ready text.*
