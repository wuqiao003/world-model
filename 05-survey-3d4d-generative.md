# 3D / 4D Generative Models
### Literature survey, 2023 – August 2026 (emphasis 2025 – mid-2026)
*Compiled 5 August 2026. Every arXiv ID below was checked against a primary arXiv/HTML/project page in this session unless flagged ⚠. Companion deep-dive on object assetization (mesh topology, CAD, rigging, PBR pipelines): [`04-survey-3d-asset-generation.md`](04-survey-3d-asset-generation.md).*

---

## 0. How to read this report

Three structural shifts organize 3D/4D generation as of August 2026:

1. **Object geometry left SDS and multi-view-then-reconstruct.** The winning recipe is a 3D-native VAE (VecSet *or* sparse structured latents) + flow-matching DiT conditioned on one image, then a separate PBR painter. DreamFusion-style optimization is legacy for geometry; multi-view diffusion survives mainly as the *texture* stage.
2. **World generation split into two bets.** (A) *Persistent explicit 3D* — Marble, WorldGen, HunyuanWorld, WorldGrow, Lyra — exportable splats/meshes you can walk forever. (B) *Interactive video world models* — Genie 3, Matrix-Game, Cosmos — real-time but drift. Mid-2026 systems increasingly hybridize: video priors → explicit 3DGS (Lyra), or LLM layout + procedural + image-to-3D (WorldGen).
3. **4D is still the undercooked layer.** Feed-forward L4GM / SV4D beat hours-long SDS (AYG, 4D-fy), but long-horizon, physically plausible, multi-object 4D at scene scale remains open. Most “4D” papers are still single animated objects.

**Anchor surveys** (maps, not oracles):
- *From Visual Synthesis to Interactive Worlds: Toward Production-Ready 3D Asset Generation* — [2604.23629](https://arxiv.org/abs/2604.23629) (⚠ contains some wrong bibliography IDs; see §13 of the companion asset survey).
- *Recent Advances in 3D Object and Scene Generation* — [2504.11734](https://arxiv.org/abs/2504.11734).
- Microsoft TRELLIS / TRELLIS.2 project pages: [TRELLIS](https://microsoft.github.io/TRELLIS/) · [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

---

## 1. Taxonomy (~32 papers / systems)

Legend **Status**: **Open** = public weights · **Code** = code without full SOTA weights · **Closed** = API/product only · **Research** = paper/demo, limited access.

### 1A. Text / Image → 3D (object)

| # | Name | Date | One-line | Link | Status |
|---|---|---|---|---|---|
| 1 | **DreamFusion** | arXiv 2209.14988, Sep 2022; ICLR 2023 ⚠ | SDS: optimize NeRF so 2D diffusion likes its renders. Founded text-to-3D. | [arXiv](https://arxiv.org/abs/2209.14988) | Closed backbone |
| 2 | **Magic3D** | arXiv 2211.10440, Nov 2022; CVPR 2023 ⚠ | Coarse NeRF → high-res DMTet mesh via SDS; 2× faster / higher-res than DreamFusion. | [arXiv](https://arxiv.org/abs/2211.10440) | Closed |
| 3 | **Zero-1-to-3** | arXiv 2303.11328, Mar 2023; ICCV 2023 | Pose-conditioned Stable Diffusion: one image + relative camera → novel view. Hinge into Gen-2. | [arXiv](https://arxiv.org/abs/2303.11328) | Open |
| 4 | **SyncDreamer** | arXiv 2309.03453, Sep 2023; ICLR 2024 | Synchronized multi-view diffusion via shared 3D-aware feature volume. | [arXiv](https://arxiv.org/abs/2309.03453) | Open |
| 5 | **Instant3D** | arXiv 2311.06214, Nov 2023; ICLR 2024 | Text → 4 sparse views → LRM-style reconstructor; ~20s feed-forward text-to-3D. | [arXiv](https://arxiv.org/abs/2311.06214) | Closed |
| 6 | **LGM** | arXiv 2402.05054, Feb 2024 | Asymmetric U-Net → multi-view Gaussian features; high-res 3DGS in ~5s. | [arXiv](https://arxiv.org/abs/2402.05054) | Open |
| 7 | **TripoSR** | arXiv 2403.02151, Mar 2024 | Open LRM reimplementation; sub-second image→mesh on one GPU. | [arXiv](https://arxiv.org/abs/2403.02151) | Open (MIT) |
| 8 | **CRM** | arXiv 2403.05034, Mar 2024; ECCV 2024 ⚠ | Conv U-Net from 6 orthographic views + CCM → FlexiCubes mesh (~10s). | [arXiv](https://arxiv.org/abs/2403.05034) | Open |
| 9 | **InstantMesh** | arXiv 2404.07191, Apr 2024 | Sparse-view LRM + FlexiCubes iso-surface head; strong open 2024 baseline. | [arXiv](https://arxiv.org/abs/2404.07191) | Open |
| 10 | **Era3D** | arXiv 2405.11616, May 2024 | Camera-aware MV diffusion + row-wise attention → 512² multi-views efficiently. | [arXiv](https://arxiv.org/abs/2405.11616) | Open |
| 11 | **Unique3D** | arXiv 2405.20343, May 2024 | High-res normals + ISOMER reconstruction; detailed meshes in <30s on 4090. | [arXiv](https://arxiv.org/abs/2405.20343) | Open |
| 12 | **Meshtron** | arXiv 2412.09548, Dec 2024 | Autoregressive artist meshes: hourglass transformer, **64K faces** @ 1024 coord res. | [arXiv](https://arxiv.org/abs/2412.09548) | Closed ⚠ |
| 13 | **TRELLIS** | arXiv 2412.01506, Dec 2024; CVPR 2025 | Structured Latents (SLAT) + rectified-flow DiT; decode to mesh / 3DGS / NeRF. | [arXiv](https://arxiv.org/abs/2412.01506) · [page](https://microsoft.github.io/TRELLIS/) | Open (MIT) |
| 14 | **Hunyuan3D 2.0 / 2.1** | 2501.12202 (Jan 2025) · 2506.15442 (Jun 2025) | Shape DiT (flow) + Paint; 2.1 adds full training code + **PBR**. Open reference pipeline. | [2.0](https://arxiv.org/abs/2501.12202) · [2.1](https://arxiv.org/abs/2506.15442) | Open |
| 15 | **TRELLIS.2** | arXiv 2512.14692, Dec 2025 | **O-Voxel** field-free sparse latents + 4B flow DiT; open surfaces, PBR, up to 1536³. | [arXiv](https://arxiv.org/abs/2512.14692) · [HF](https://huggingface.co/microsoft/TRELLIS.2-4B) | Open (MIT) ⚠ license nuance |
| 16 | **CLAY / Rodin** | arXiv 2406.13897; SIGGRAPH 2024 | 1.5B VecSet DiT + multi-res VAE; bbox/voxel/point/image control. Product: Hyper3D Rodin. | [arXiv](https://arxiv.org/abs/2406.13897) | Closed (API) |
| 17 | **CraftsMan3D** | arXiv 2405.14979, May 2024 | Native 3D DiT coarse mesh + interactive normal-based refiner. | [arXiv](https://arxiv.org/abs/2405.14979) | Open |
| 18 | **TripoSG / Meshy / Luma Genie** | TripoSG 2502.06608 (Feb 2025); products ongoing | Commercial image/text→3D APIs; TripoSG = open rectified-flow shape backbone from Tripo/VAST. | [TripoSG](https://arxiv.org/abs/2502.06608) | Mixed |

### 1B. Native 3D diffusion / flow / structured latents

| # | Name | Date | One-line | Link | Status |
|---|---|---|---|---|---|
| 19 | **3DShape2VecSet** | arXiv 2301.11445, Jan 2023; SIGGRAPH 2023 | Invented **VecSet**: unordered latent tokens → occupancy via cross-attn. Industry latent #1. | [arXiv](https://arxiv.org/abs/2301.11445) | Open |
| 20 | **Michelangelo** | arXiv 2306.17115, Jun 2023; NeurIPS 2023 ⚠ | Shape–image–text aligned latent before diffusion. | [arXiv](https://arxiv.org/abs/2306.17115) | Open |
| 21 | **Direct3D** | arXiv 2405.14832, May 2024; NeurIPS 2024 | Native triplane VAE + DiT; no MV diffusion / SDS for image→3D. | [arXiv](https://arxiv.org/abs/2405.14832) | Open |
| 22 | **GaussianCube** | arXiv 2403.19655, Mar 2024 | OT-arrange fixed-count Gaussians onto a voxel grid → standard 3D U-Net diffusion. | [arXiv](https://arxiv.org/abs/2403.19655) | Open |
| 23 | **DiffSplat** | arXiv 2501.16764, Jan 2025 | Repurpose 2D image diffusion to emit multi-view Gaussian grids + 3D render loss. | [arXiv](https://arxiv.org/abs/2501.16764) | Open |
| 24 | **TRELLIS SLAT / TRELLIS.2 O-Voxel** | 2412.01506 · 2512.14692 | Latent family #2: sparse spatially-indexed codes; enables regional edit / multi-format decode. | see rows 13–15 | Open |

### 1C. Generative world / scene assets

| # | Name | Date | One-line | Link | Status |
|---|---|---|---|---|---|
| 25 | **LucidDreamer** | arXiv 2311.13384, Nov 2023; TVCG 2025 ⚠ | Domain-free scene 3DGS via recursive “dreaming” (inpaint) + alignment from one image. | [arXiv](https://arxiv.org/abs/2311.13384) · [page](https://luciddreamer-cvlab.github.io/) | Open |
| 26 | **WorldDreamer** | arXiv 2401.09985, Jan 2024 | Masked-token world model for *video* (STPT); text/image/action prompts — video WM, not mesh world. | [arXiv](https://arxiv.org/abs/2401.09985) | Research |
| 27 | **BlockFusion** | arXiv 2401.17053, Jan 2024 | Latent triplane diffusion of scene *blocks* + extrapolation for expandable scenes. | [arXiv](https://arxiv.org/abs/2401.17053) | Open |
| 28 | **SceneCraft** | arXiv 2403.01248, Mar 2024; ICML 2024 | LLM agent: text → scene graph → Blender Python; GPT-V refine + library learning. | [arXiv](https://arxiv.org/abs/2403.01248) | Code |
| 29 | **WonderWorld** | arXiv 2406.09394, Jun 2024 | *Interactive* single-image scene extrapolation (<10s/step) with connected geometry. | [arXiv](https://arxiv.org/abs/2406.09394) | Open |
| 30 | **InfiniCube** | arXiv 2412.03934, Dec 2024; ICCV 2025 | Map/bbox/text-conditioned unbounded *driving* voxel worlds → video → dynamic 3DGS (~300×400 m). | [arXiv](https://arxiv.org/abs/2412.03934) · [page](https://research.nvidia.com/labs/toronto-ai/infinicube/) | Research |
| 31 | **HunyuanWorld 1.0** | arXiv 2507.21809, Jul 2025 | Panorama proxy → semantic layered meshes; explorable/interactive, engine-exportable. | [arXiv](https://arxiv.org/abs/2507.21809) · [code](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) | Open |
| 32 | **Lyra / Lyra 2.0** | 2509.19296 (Sep 2025) · 2604.13036 (Apr 2026) | Distill video-diffusion 3D knowledge into feed-forward 3DGS; 2.0 = long-horizon explorable worlds. | [Lyra](https://arxiv.org/abs/2509.19296) · [2.0](https://arxiv.org/abs/2604.13036) | Open (NVIDIA HF) |
| 33 | **WorldGrow** | arXiv 2510.21682, Oct 2025; AAAI 2026 Oral | Unbounded explicit 3D via block-wise 3D inpainting + coarse-to-fine; walkable persistent worlds. | [arXiv](https://arxiv.org/abs/2510.21682) · [code](https://github.com/world-grow/WorldGrow) | Open |
| 34 | **WorldGen** (Meta) | arXiv 2511.16825, Nov 2025; CVPR 2026 ⚠ | LLM layout + procedural + navmesh-conditioned recon + object decompose; ~50×50 m engine-ready. | [arXiv](https://arxiv.org/abs/2511.16825) · [Meta](https://www.meta.com/blog/worldgen-3d-world-generation-reality-labs-generative-ai-research/) | Research (not public) |
| 35 | **World Labs Marble** | Beta Sep 2025; GA 12 Nov 2025; World API Jan 2026; Marble 1.1 Apr 2026 | Multimodal persistent 3D worlds (text/image/video/layout) → splat/mesh/video export. | [blog](https://www.worldlabs.ai/blog/marble-world-model) · [API](https://www.worldlabs.ai/blog/announcing-the-world-api) | Closed (product) |

### 1D. 4D generation

| # | Name | Date | One-line | Link | Status |
|---|---|---|---|---|---|
| 36 | **4D-fy** | arXiv 2311.17984, Nov 2023 | Hybrid SDS (image + 3D-aware + video) for text-to-4D NeRF scenes. | [arXiv](https://arxiv.org/abs/2311.17984) | Open |
| 37 | **Align Your Gaussians** | arXiv 2312.13763, Dec 2023; CVPR 2024 | Dynamic 3DGS + deformation + composed diffusion SDS; composable 4D assets. | [arXiv](https://arxiv.org/abs/2312.13763) | Open |
| 38 | **DreamGaussian4D** | arXiv 2312.17142, Dec 2023 | Static GS → HexPlane deformation driven by video; minutes not hours. | [arXiv](https://arxiv.org/abs/2312.17142) | Open |
| 39 | **Diffusion4D** | arXiv 2405.16645, May 2024; NeurIPS 2024 | Single 4D-aware video diffusion for orbital dynamic views + GS reconstruction. | [arXiv](https://arxiv.org/abs/2405.16645) | Open |
| 40 | **L4GM** | arXiv 2406.10324, Jun 2024 | First feed-forward 4D LRM: monocular video → dynamic Gaussian sequence (Objaverse anim). | [arXiv](https://arxiv.org/abs/2406.10324) | Open |
| 41 | **SV4D** | arXiv 2407.17470, Jul 2024 | Unified latent video DM: mono video → consistent novel-view videos → dynamic NeRF (no SDS). | [arXiv](https://arxiv.org/abs/2407.17470) | Open |
| 42 | **DynamiCrafter** (→4D pipelines) | arXiv 2310.12190, Oct 2023; ECCV 2024 Oral | Image-to-video prior widely reused as the *motion* driver for later image/video→4D stacks. | [arXiv](https://arxiv.org/abs/2310.12190) | Open |

### 1E. Controllable 3D / texture / PBR (representative)

| # | Name | Date | One-line | Link | Status |
|---|---|---|---|---|---|
| 43 | **TEXTure** | arXiv 2302.01721, Feb 2023; SIGGRAPH 2023 ⚠ | Iterative depth-to-image painting with trimap keep/refine/generate. | [arXiv](https://arxiv.org/abs/2302.01721) | Open |
| 44 | **Paint3D** | arXiv 2312.13913, Dec 2023; CVPR 2024 | Coarse MV bake + UV diffusion for **lighting-less** 2K textures. | [arXiv](https://arxiv.org/abs/2312.13913) | Open |
| 45 | **FlashTex** | arXiv 2402.13251, Feb 2024; ECCV 2024 | LightControlNet disentangles lighting → relightable PBR texturing (~10× faster SDS). | [arXiv](https://arxiv.org/abs/2402.13251) | Open |
| 46 | **MaterialMVP** | arXiv 2503.10289, Mar 2025; ICCV 2025 | Illumination-invariant multi-view PBR (albedo + MR); engine behind Hunyuan3D-Paint 2.1. | [arXiv](https://arxiv.org/abs/2503.10289) | Open |
| 47 | **Hunyuan3D-Omni** | arXiv 2509.21245, Sep 2025 | Point / voxel / bbox / **skeletal pose** control on Hunyuan3D 2.1 backbone. | [arXiv](https://arxiv.org/abs/2509.21245) | Open |
| 48 | **PartCrafter / OmniPart / Nano3D** | 2506.05573 · 2507.06165 · 2510.15019 | Part-native generation, TRELLIS part synthesis, training-free part edit + Edit-100k. | [PC](https://arxiv.org/abs/2506.05573) · [OP](https://arxiv.org/abs/2507.06165) · [N3](https://arxiv.org/abs/2510.15019) | Open |

---

## 2. Technical trends (2023 → Aug 2026)

### 2.1 Object generation: three generations, one settled recipe

| Era | Dominant method | Why it lost / won |
|---|---|---|
| Gen-1 (2022–23) | SDS optimization (DreamFusion → Magic3D → ProlificDreamer) | Slow, Janus, baked lighting; died once Objaverse-scale 3D data existed |
| Gen-2 (2023–24) | MV diffusion + LRM / Gaussian reconstructor (Zero-1-to-3 → InstantMesh / LGM / CRM) | Fast; geometry underdetermined by views → demoted to texturing front-end |
| Gen-3 (2024–26) | Native 3D latent + **flow / rectified flow** DiT | Seconds, scalable; **VecSet** (Hunyuan/TripoSG/CLAY) vs **structured sparse** (TRELLIS → TRELLIS.2) |

**Settled conventions (high confidence):**
- Image conditioning (DINOv2/v3) ≫ native text conditioning; “text-to-3D” ≈ T2I then I2I3D.
- Two-stage **geometry then PBR** (even TRELLIS.2 keeps a material stage).
- Flow matching over DDPM for 3D latents.
- Sparse voxels / sparse attention make 1024³ trainable on ~8 GPUs (Direct3D-S2 [2505.17412](https://arxiv.org/abs/2505.17412)).

### 2.2 World assets: layout + lift, not one giant latent

Successful large-world systems compose:
1. **Semantic / geometric scaffold** — LLM scene graph (SceneCraft, WorldGen), HD map (InfiniCube), panorama layers (HunyuanWorld), or block grid (BlockFusion, WorldGrow).
2. **Appearance fill** — image/video diffusion or image-to-3D priors.
3. **Export substrate** — mesh + navmesh (WorldGen, HunyuanWorld) or 3DGS (Marble, Lyra, LucidDreamer).

WorldGen’s own comparison (arXiv 2511.16825): Marble-quality near the conditioned view but fidelity drops within ~3–5 m; WorldGen targets ~50×50 m engine-importable scenes in ~5 minutes — research-only as of Meta’s post.

### 2.3 4D: from hybrid SDS → feed-forward video-conditioned reconstruction

- **2023–early 2024:** hours of hybrid SDS (4D-fy, AYG) or video-driven HexPlane (DreamGaussian4D).
- **Mid–late 2024:** amortize with video/MV diffusion (Diffusion4D, SV4D) or 4D LRM (L4GM).
- **2025–26 gap:** scene-scale multi-agent 4D, physical contact, and long temporal memory are still mostly video-WM territory (Genie 3 / Cosmos), not native 4D asset models.

### 2.4 Controllability moved into the latent

CLAY established rich 3D controls (bbox, voxel, point). TRELLIS’s spatial SLAT made *regional* edit natural (OmniPart, Nano3D). Texture arc: sequential inpaint (TEXTure) → sync MV (Paint3D lineage) → PBR-first (FlashTex, MaterialMVP, Hunyuan3D 2.1).

---

## 3. Commercial APIs & open models (status as of mid-2026)

*Product rankings below are from secondary market writeups (2025–2026 review blogs), not controlled benchmarks — use for market shape only.*

| System | Type | Mid-2026 status | Notes |
|---|---|---|---|
| **Microsoft TRELLIS / TRELLIS.2-4B** | Open weights | Strongest open object I2I3D | MIT on GitHub/HF; project page warns against commercial exploitation — resolve before shipping |
| **Tencent Hunyuan3D 2.1** | Open weights + train code | Best fully open *pipeline* (shape+PBR) | 2.5 / 3.0 / 3.1 moved API-only ([2604.23629](https://arxiv.org/abs/2604.23629) §8.4 flags this migration pattern) |
| **TripoSG** | Open shape model | Open research backbone | Product **Tripo** is closed API (speed / game topology niche) |
| **Step1X-3D** | Open | Reproducible two-stage system | [2505.07747](https://arxiv.org/abs/2505.07747) |
| **Rodin / Hyper3D (CLAY lineage)** | Closed API | Hero-asset / high-fidelity niche | Sketch-to-3D, PBR, LoRA-style style lock reported in product reviews |
| **Meshy** | Closed API | Full pipeline (gen→texture→rig→animate) | Strong indie default in 2026 roundups |
| **Luma Genie / Dream Machine 3D** | Closed (ecosystem) | Concept + capture combo | Genie for generation; Luma scanning/splats for real-world capture |
| **World Labs Marble** | Closed product + **World API** (Jan 2026) | Persistent multimodal worlds | GA Nov 2025; Marble 1.1 / 1.1 Plus Apr 2026 ([release notes](https://docs.worldlabs.ai/marble/release-notes)) |
| **Meta WorldGen** | Research | Not available to developers | Engine-ready meshes claimed; research phase (Meta Quest blog) |
| **NVIDIA Lyra 1.0 / 2.0** | Open HF weights | Explorable 3DGS worlds from video priors | [github.com/nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) |
| **HunyuanWorld 1.0** | Open | Panorama→layered mesh worlds | Released Jul 2025 with Hunyuan3D 2.1 ecosystem |

**Practical split:** object assets → TRELLIS.2 or Hunyuan3D 2.1 locally, or Tripo/Meshy/Rodin via API; persistent worlds → Marble API or open HunyuanWorld / Lyra / WorldGrow; do not expect Genie-3-class real-time interactive WM weights.

---

## 4. Open problems (with attribution)

**Assetization gap (object).**
> “Even when a method produces a visually compelling shape, the gap between its output and a deployable asset encompasses retopology, UV unwrapping, PBR separation, rigging, LOD creation, and collision fitting…”
> — *Production-Ready 3D Asset Generation*, [2604.23629](https://arxiv.org/abs/2604.23629) §8.1

**No complete production dataset.**
> “To our knowledge, no existing dataset simultaneously provides manifold topology, artist-quality UV layouts, complete PBR channels, skeletal rigs, and natural-language descriptions.”
> — same, §8.1

**Evaluation rewards the wrong thing.**
> Human votes favor splats over meshes (+16.6 ELO) and textured over untextured (+144.1 ELO); same model, different render format → 78 ELO gap.
> — *3D Arena*, [2506.18787](https://arxiv.org/abs/2506.18787) §4.3

**World extent vs fidelity.**
WorldGen authors: viewpoint-conditioned generators (incl. Marble-class) degrade within meters of the seed view; compositional/layout methods scale spatially but need procedural+LLM scaffolding ([2511.16825](https://arxiv.org/abs/2511.16825), qualitative discussion).

**O-Voxel resolution floor.**
> Features smaller than a voxel alias; dual-vertex QEF averages nearby surfaces → blurred materials.
> — TRELLIS.2, [2512.14692](https://arxiv.org/abs/2512.14692) Appendix F

**Part-level data starvation.**
> PartCrafter trains on ~50K part assets vs millions for whole objects.
> — PartCrafter, [2506.05573](https://arxiv.org/abs/2506.05573) Limitations

**Physics absent from generation.**
> Methods optimize visual plausibility; mass/friction/elasticity/articulation limits largely missing.
> — [2604.23629](https://arxiv.org/abs/2604.23629) §8.2

**4D still object-centric.** L4GM/SV4D papers themselves scope to Objaverse-style animated *objects*; multi-object contact-rich scene 4D is not claimed solved in those works.

**Editing remains local / backbone-bound.**
> Nano3D: “supports only localized edits… constrained by TRELLIS’s generative capacity.”
> — [2510.15019](https://arxiv.org/abs/2510.15019)

---

## 5. Under-explored gaps for 8–32 GPU labs

Do **not** train a new 4–10B image-to-3D foundation model. TRELLIS.2-4B and Hunyuan3D 2.1 already occupy that niche. High-leverage gaps:

### 5.1 Production-readiness benchmark (compute ≈ 0–2 GPUs)
Measure topology validity, UV distortion/seams, PBR relight consistency, engine import success, collision validity over open checkpoints (TRELLIS.2, Hunyuan3D 2.1, TripoSG, Step1X-3D). Predictable result: ranking ≠ 3D Arena ELO ([2604.23629](https://arxiv.org/abs/2604.23629) §3.4, §7.2; [2506.18787](https://arxiv.org/abs/2506.18787)).

### 5.2 Part / edit / UV paired data (CPU + inference budget)
Bootstrap with training-free pipelines (Nano3D pattern) for part splits, retopo pairs, software-UV→artist-UV pairs. Supervision-starved stages, not scale-starved.

### 5.3 World *consistency metrics* beyond FID/FVD
Camera-trajectory WorldScore-style eval already shows video models fail camera control; extend to **navmesh validity, traversable free-space, object editability after export** for Marble/WorldGrow/HunyuanWorld/Lyra outputs. Small models + careful protocol beat another generative demo.

### 5.4 Joint geometry–material ablation on an open backbone
Everyone assumes two-stage. Run controlled same-budget joint vs cascaded on TRELLIS.2 / Hunyuan3D 2.1 training code. Architectural question the field has assumed ([2604.23629](https://arxiv.org/abs/2604.23629); UniLat3D / Native3D dissent).

### 5.5 4D that is not a single Objaverse toy
Feed-forward multi-object 4D with contact / occlusion, or distill DynamiCrafter/SV4D motions into Lyra-style explicit 4DGS *scenes*. L4GM proved amortization for objects; scene 4D is still wide open and fits 8–32 GPUs if you freeze large video backbones and train only the 3D/4D head (Lyra recipe).

### 5.6 Physics / sim-ready attributes
Predict mass, friction, joint limits, collision proxies; validate in MuJoCo/Isaac. Seed3D and URDF-Anything+ point at the demand; almost nobody evaluates “does it rest / slide / articulate correctly.”

### 5.7 Layout-conditioned open world generation
WorldGen’s LLM+procedural+navmesh recipe is powerful but closed. Reproduce a *small* open variant: LLM scene graph → place open Hunyuan3D/TRELLIS assets → navmesh check → VLM critic loop (SceneCraft already did the Blender-code half). Compositionality > training another scene NeRF.

### 5.8 Things to avoid
- New foundation I2I3D from scratch.
- Headline Chamfer on GSO/Toys4K only.
- Optimizing purely for human A/B (rewards texture/splats over usable topology).
- Claiming “solved 4D worlds” from single-object orbital videos.

---

## 6. Compact genealogy (one screen)

```text
SDS era                MV+LRM era              Native 3D latent           Worlds / 4D
DreamFusion ──┐        Zero-1-to-3 ──┐         VecSet: 3DShape2VecSet
Magic3D       │        SyncDreamer   │                ├─ Michelangelo
              ├──────► Instant3D     ├──────►         ├─ CLAY/Rodin
              │        LGM/CRM       │                ├─ Hunyuan3D 2.x
              │        InstantMesh   │                └─ TripoSG
              │        Era3D/Unique3D│
              └─ legacy for geom.    │         Sparse: TRELLIS(SLAT) → TRELLIS.2(O-Voxel)
                                     │                └─ Direct3D / DiffSplat / GaussianCube
                                     ▼
                              Texture stage: TEXTure→Paint3D→FlashTex→MaterialMVP
                              Control: CLAY controls → Hunyuan3D-Omni → Part*/Nano3D

Worlds: LucidDreamer/WonderWorld → BlockFusion → InfiniCube
      → HunyuanWorld / WorldGrow / Lyra → WorldGen (Meta) / Marble (World Labs)

4D: 4D-fy/AYG/DG4D (SDS) → Diffusion4D/SV4D/L4GM (amortized)  … scene-scale still open
```

---

## 7. Verification caveats

- **Venues marked ⚠** were not re-checked against proceedings listings this session (taken from arXiv comments, project pages, or common citation).
- **TRELLIS.2 venue:** arXiv + authors call it a tech report; do not cite as CVPR 2026 without CVF confirmation.
- **TRELLIS.2 license:** MIT on GitHub/HF vs project-page commercial disclaimer — check before commercial use.
- **Hunyuan3D 3.0/3.1:** product announcements; no technical report found this session — omitted from main claims.
- **WorldDreamer** is a *video* world model (masked tokens), not a 3D mesh world generator — listed because the name appears in world-model taxonomies; do not conflate with Marble/WorldGen.
- **Commercial quality ordering** (Tripo vs Meshy vs Rodin) is from secondary blogs only.
- Companion file [`04-survey-3d-asset-generation.md`](04-survey-3d-asset-generation.md) has deeper tables on mesh sequence models, CAD, auto-rigging, and evaluation pathology (3D Arena, CADENA, Hi3DEval).

---

*End of survey. For embodied/driving world models (Genie 3, Cosmos, OmniDreams, WAMs), see [`03-survey-embodied-driving.md`](03-survey-embodied-driving.md).*
