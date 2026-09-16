# 3D Asset Generation
### Literature survey, 2024 – August 2026 (emphasis Aug 2025 – Aug 2026)
*Compiled 5 August 2026. Every arXiv ID in this document was checked against the arXiv API during this session unless explicitly flagged with ⚠. Venues taken from CVF Open Access, ACM DL, NeurIPS/OpenReview proceedings, or official repos where possible; unverified venues are flagged.*

---

## 0. How to read this report

Three things define the state of the field in August 2026, and they organize everything below.

**1. The architecture question is settled, and it is not the one people were arguing about in 2023.** The dominant recipe for object-level 3D generation is now: *a 3D-native VAE that compresses a mesh into a compact latent (either an unordered "VecSet" of tokens or a sparse-voxel grid of latents), plus a large flow-matching Diffusion Transformer that generates that latent conditioned on a single image, plus a second model that paints PBR materials onto the resulting surface.* Score-distillation optimization is dead as a frontier method, and multi-view-then-reconstruct survives mainly as the texturing stage, not the geometry stage. Every major 2025–2026 release — Hunyuan3D 2.x, TripoSG, Direct3D-S2, Step1X-3D, Sparc3D, Seed3D 1.0, TRELLIS.2 — is a variation on this recipe.

**2. The bottleneck moved from "can it make a plausible shape?" to "is the shape a usable asset?"** The 2026 survey that best captures this calls the gap the **assetization bottleneck**: a generated shape still needs retopology, UV unwrapping, PBR separation, rigging, LOD, and collision geometry before an engine will take it. Nearly all of the interesting Aug 2025 – Aug 2026 work is in those downstream stages — mesh topology, UV seams, part decomposition, auto-rigging, articulation — rather than in shape quality.

**3. Evaluation is measurably broken, and for once we have data proving it.** 3D Arena's 123,243 human votes show that people reward *rendering format* and *texture presence* over geometric quality (Gaussian splats beat meshes by 16.6 ELO with the same underlying model; textured beats untextured by 144.1 ELO), which is exactly backwards from what professional pipelines need. Meanwhile CADENA shows that on the standard CAD benchmark every method scores within noise of every other, and switching to real mechanical parts halves every method's score.

**Two anchor surveys**, both useful as maps:
- *From Visual Synthesis to Interactive Worlds: Toward Production-Ready 3D Asset Generation* — arXiv [2604.23629](https://arxiv.org/abs/2604.23629) (HKUST + Tencent). Organizes the literature by production pipeline stage rather than algorithm family. This is the single most useful entry point for this topic. **Caveat: it contains at least three incorrect arXiv IDs** (see §13).
- *Recent Advances in 3D Object and Scene Generation: A Survey* — arXiv [2504.11734](https://arxiv.org/abs/2504.11734), ACM Computing Surveys. Better on the algorithmic genealogy of SDS variants.

---

## 1. Table of key papers and systems

Legend for **Weights**: **Open** = downloadable checkpoints under a permissive/community license · **Restricted** = weights released but license limits commercial or regional use · **Code only** = code without full checkpoints · **Closed** = API/product only · **?** = not confirmed this session.

### 1A. Generation 1 & 2 — SDS optimization, multi-view diffusion, large reconstruction models

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 1 | **DreamFusion** (Poole et al.) | arXiv 2209.14988, 29 Sep 2022; ICLR 2023 ⚠venue not re-verified | Invented Score Distillation Sampling (SDS): optimize a NeRF so its renders look right to a frozen 2D text-to-image diffusion model. Founded text-to-3D. | [arXiv](https://arxiv.org/abs/2209.14988) | Closed (Imagen backbone) |
| 2 | **Magic3D** (Lin et al., NVIDIA) | arXiv 2211.10440, 18 Nov 2022; CVPR 2023 ⚠venue not re-verified | Coarse-to-fine SDS: low-res hash-grid NeRF, then a DMTet mesh refined at high resolution. 2× faster, 8× higher resolution than DreamFusion. | [arXiv](https://arxiv.org/abs/2211.10440) | Closed |
| 3 | **Fantasia3D** (Chen et al.) | arXiv 2303.13873; ICCV 2023 | Disentangled geometry/appearance SDS on DMTet with spatially-varying BRDF — first serious attempt at *relightable* SDS output. | [arXiv](https://arxiv.org/abs/2303.13873) | Open |
| 4 | **ProlificDreamer** (Wang et al.) | arXiv 2305.16213, 25 May 2023; NeurIPS 2023 ⚠venue not re-verified | Variational Score Distillation (VSD): treat the 3D scene as a *distribution* optimized by particle variational inference, fixing SDS's over-saturation and mode collapse. The high-water mark of the SDS era. | [arXiv](https://arxiv.org/abs/2305.16213) | Open |
| 5 | **Zero-1-to-3** (Liu et al.) | arXiv 2303.11328, 20 Mar 2023; ICCV 2023 | Fine-tuned Stable Diffusion to be *camera-pose-conditioned*: given one image and a relative viewpoint, synthesize that view. The hinge between generation 1 and 2. | [arXiv](https://arxiv.org/abs/2303.11328) | Open |
| 6 | **MVDream** (Shi et al.) | arXiv 2308.16512, 31 Aug 2023; ICLR 2024 | Generate four mutually consistent views *jointly* via 3D self-attention across views — the fix for the Janus (multi-face) problem. | [arXiv](https://arxiv.org/abs/2308.16512) | Open |
| 7 | **SyncDreamer** (Liu et al.) | arXiv 2309.03453, 7 Sep 2023; ICLR 2024 | Synchronized multi-view diffusion via a shared 3D-aware feature volume that correlates the noise across views at every denoising step. | [arXiv](https://arxiv.org/abs/2309.03453) | Open |
| 8 | **Wonder3D** (Long et al.) | arXiv 2310.15008, 23 Oct 2023; CVPR 2024 | Cross-domain diffusion producing color **and normal** maps together, then normal-guided surface fusion. Made MV→mesh geometry usable. | [arXiv](https://arxiv.org/abs/2310.15008) | Open |
| 9 | **Zero123++** | arXiv 2310.15110, 23 Oct 2023 | Practical engineering of the MV generator: 6 fixed views tiled into one image, reference attention, fixed elevation. Became the default MV front-end. | [arXiv](https://arxiv.org/abs/2310.15110) | Open |
| 10 | **LRM** (Hong et al.) | arXiv 2311.04400, 8 Nov 2023; ICLR 2024 | Large Reconstruction Model: a 500M transformer that regresses a tri-plane NeRF from one image in ~5s. Killed per-scene optimization for reconstruction. | [arXiv](https://arxiv.org/abs/2311.04400) | Closed (Adobe) |
| 11 | **Instant3D** | arXiv 2311.06214, 10 Nov 2023; ICLR 2024 | Text → 4 sparse views → LRM-style transformer → 3D in ~20s. Established the "MV generator + feed-forward reconstructor" two-stage template. | [arXiv](https://arxiv.org/abs/2311.06214) | Closed |
| 12 | **TripoSR** | arXiv 2403.02151, 4 Mar 2024 | Open LRM reimplementation with data/architecture fixes; sub-second single-image 3D on one GPU. The most-used open baseline of 2024. | [arXiv](https://arxiv.org/abs/2403.02151) | Open (MIT) |
| 13 | **InstantMesh** | arXiv 2404.07191, 10 Apr 2024 | Sparse-view LRM with a **FlexiCubes** differentiable iso-surface head, so it trains against mesh geometry directly rather than a radiance field. | [arXiv](https://arxiv.org/abs/2404.07191) | Open |
| 14 | **CRM** | arXiv 2403.05034, 8 Mar 2024; ECCV 2024 ⚠venue not re-verified | Convolutional (UNet) reconstruction model using six orthographic views + canonical coordinate maps → tri-plane, exploiting pixel-to-triplane alignment. | [arXiv](https://arxiv.org/abs/2403.05034) | Open |
| 15 | **SF3D** (Stability AI) | arXiv 2408.00653; CVPR 2025 | Feed-forward model that outputs mesh **plus UV atlas plus PBR materials plus delit albedo** in ~0.5s. Closest 2024 approach to a "production-ready" feed-forward asset. | [arXiv](https://arxiv.org/abs/2408.00653) | Open |

### 1B. Generation 3 — native 3D latent diffusion (the current dominant paradigm)

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 16 | **3DShape2VecSet** (Zhang et al.) | arXiv 2301.11445, 26 Jan 2023; SIGGRAPH 2023 | Invented the **VecSet**: represent a shape as an unordered *set* of latent vectors cross-attended into an occupancy field. The latent format that almost the whole industry now uses. | [arXiv](https://arxiv.org/abs/2301.11445) | Open |
| 17 | **Shap-E** (OpenAI) | arXiv 2305.02463, 3 May 2023 | Diffusion over the *weights* of an implicit function (NeRF+SDF MLP). Fast, low quality, but the first widely-used native 3D generative model. | [arXiv](https://arxiv.org/abs/2305.02463) | Open |
| 18 | **Michelangelo** | arXiv 2306.17115; NeurIPS 2023 ⚠venue not re-verified | Shape–image–text aligned latent space (a CLIP analogue for shapes) before diffusion, improving multimodal conditioning. | [arXiv](https://arxiv.org/abs/2306.17115) | Open |
| 19 | **CLAY** (Zhang et al.) | arXiv 2406.13897; SIGGRAPH 2024 | Scaled the VecSet recipe to a 1.5B DiT with a multi-resolution VAE and rich conditioning (text/image/voxel/point/box). The template every Chinese lab then copied. | [arXiv](https://arxiv.org/abs/2406.13897) | Closed (Rodin/Deemos) |
| 20 | **TRELLIS** (Xiang et al., MSRA) | arXiv 2412.01506, 2 Dec 2024; CVPR 2025 | **Structured LATents (SLAT)**: latent codes attached to *active sparse voxels*, decodable to mesh, 3DGS, or radiance field from one generation. Rectified-flow transformer. The other dominant latent format. | [arXiv](https://arxiv.org/abs/2412.01506) | Open (MIT) |
| 21 | **Hunyuan3D 2.0** (Tencent) | arXiv 2501.12202, 21 Jan 2025 | Two-stage open system: Hunyuan3D-DiT (flow-matching over a ShapeVAE SDF latent) + Hunyuan3D-Paint (mesh-conditioned MV texture model). The reference open pipeline of 2025. | [arXiv](https://arxiv.org/abs/2501.12202) | Open (community license) |
| 22 | **TripoSG** (Tripo/VAST) | arXiv 2502.06608, 10 Feb 2025 | Large-scale **rectified flow** over a VecSet SDF latent, 4B params, trained on a heavily curated 2M-asset corpus; showed data curation matters as much as architecture. | [arXiv](https://arxiv.org/abs/2502.06608) | Open |
| 23 | **Direct3D-S2** | arXiv 2505.17412, May 2025 | **Spatial Sparse Attention** kernel for sparse-volume DiTs: 3.9× forward / 9.6× backward speedup, enabling 1024³ training **on 8 GPUs** where prior work needed ≥32 GPUs for 256³. | [arXiv](https://arxiv.org/abs/2505.17412) | Open |
| 24 | **Sparc3D** | arXiv 2505.14521, May 2025; NeurIPS 2025 | *Sparcubes* (sparse deformable marching cubes; raw mesh → watertight 1024³ in ~30s, 3× faster than prior remeshers) + *Sparconv-VAE*, the first fully sparse-convolutional, modality-consistent 3D VAE. | [arXiv](https://arxiv.org/abs/2505.14521) | Open |
| 25 | **Step1X-3D** (StepFun) | arXiv 2505.07747, May 2025 | Fully open two-stage system (800K curated assets, 1.3B geometry DiT + texture model) released with data pipeline and training code — the most *reproducible* of the 2025 systems. | [arXiv](https://arxiv.org/abs/2505.07747) | Open |
| 26 | **Hunyuan3D 2.1** (Tencent) | arXiv 2506.15442, 18 Jun 2025 | First fully open-source release with **weights and training code** for both stages, and PBR (albedo/metallic/roughness) rather than baked RGB. Shape 3.3B + Paint 2B. | [arXiv](https://arxiv.org/abs/2506.15442) | Open (community license) |
| 27 | **Hunyuan3D 2.5 / LATTICE** | arXiv 2506.16504, Jun 2025; LATTICE report arXiv 2512.03052, Dec 2025 | Scaled the shape foundation model to **10B parameters** ("LATTICE"), explicitly framed as closing the gap to handcrafted meshes. 2.5 itself is API-only. | [2506.16504](https://arxiv.org/abs/2506.16504) · [2512.03052](https://arxiv.org/abs/2512.03052) | Closed (2.5) |
| 28 | **Hunyuan3D-Omni** (Tencent) | arXiv 2509.21245, 25 Sep 2025 | Unified control interface over the 2.1 backbone: point clouds, voxels, bounding boxes, **and skeletal pose** as conditioning, with difficulty-aware modality sampling. | [arXiv](https://arxiv.org/abs/2509.21245) | Open |
| 29 | **Hunyuan3D Studio** (Tencent) | arXiv 2509.12815, 16 Sep 2025 | Not a model but a *pipeline*: part-level generation + polygon (retopology) generation + semantic UV + PBR, chained into one game-ready asset system. | [arXiv](https://arxiv.org/abs/2509.12815) | Partly closed |
| 30 | **Seed3D 1.0** (ByteDance Seed) | arXiv 2510.19944, 22 Oct 2025 | 1.5B DiT targeting **simulation-ready** assets (physics-engine-importable with minimal config) for embodied-AI world simulators; claims to beat 3B Hunyuan3D-2.1 at half the size. | [arXiv](https://arxiv.org/abs/2510.19944) | Closed (API) |
| 31 | **TRELLIS.2** (Xiang et al., MSRA) | arXiv 2512.14692, Dec 2025; ⚠listed as CVPR 2026 by a third-party paper-notes site, arXiv metadata says tech report | **O-Voxel**: a "field-free" omni-voxel latent handling open, non-manifold and enclosed surfaces *and* PBR materials, with a 16× spatially compressing sparse VAE (1024³ textured asset → ~9.6K tokens) and a 4B flow model. 3s at 512³, 17s at 1024³, 60s at 1536³ on one H100. Current open-weight SOTA. | [arXiv](https://arxiv.org/abs/2512.14692) · [HF](https://huggingface.co/microsoft/TRELLIS.2-4B) | **Open (MIT)** |
| 32 | **EVA01** | arXiv 2605.16745, May 2026 | Mixture-of-Transformers MLLM that treats 3D mesh as a *native modality* — understanding, generation, and multi-turn context-aware editing in one model. The clearest 2026 attempt at breaking the "stateless reconstructor" framing. | [arXiv](https://arxiv.org/abs/2605.16745) | ? |

### 1C. Mesh generation as sequence modeling

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 33 | **MeshGPT** (Siddiqui et al.) | arXiv 2311.15475, 26 Nov 2023; CVPR 2024 | Learn a VQ-VAE vocabulary of triangle embeddings via graph convolutions, then a decoder-only transformer predicts meshes as token sequences. "Teach GPT to speak mesh." | [arXiv](https://arxiv.org/abs/2311.15475) | Open |
| 34 | **MeshAnything / V2** (Chen et al.) | V1: arXiv 2406.10163, ICLR 2025 · V2: arXiv 2408.02555, ICCV 2025 | Shape-conditioned artist-mesh generation (any 3D representation → artist mesh). V2's **Adjacent Mesh Tokenization** encodes a face with one vertex where possible, halving sequence length and doubling the face limit. | [V1](https://arxiv.org/abs/2406.10163) · [V2](https://arxiv.org/abs/2408.02555) | Open |
| 35 | **EdgeRunner** | arXiv 2409.18114, Sep 2024 | EdgeBreaker-derived traversal as the tokenization, giving a compact, locality-respecting serialization of connectivity. | [arXiv](https://arxiv.org/abs/2409.18114) | Open |
| 36 | **BPT** — Blocked and Patchified Tokenization | arXiv 2411.07025, Nov 2024; CVPR 2025 | Block-wise coordinate indexing + patch aggregation cuts sequence length ~75%, unlocking meshes >8K faces. The compression baseline everything since is measured against. | [arXiv](https://arxiv.org/abs/2411.07025) | Open |
| 37 | **Meshtron** (NVIDIA) | arXiv 2412.09548, Dec 2024; ⚠venue (OpenReview `mhzDv7UAMu`) not confirmed | Hourglass transformer + truncated-sequence training + sliding-window inference: **64K faces at 1024-level coordinate resolution**, 1.1B params, constant memory regardless of mesh size. | [arXiv](https://arxiv.org/abs/2412.09548) | Closed ⚠ |
| 38 | **LLaMA-Mesh** (NVIDIA) | arXiv 2411.09595, Nov 2024 | Fine-tune an LLM to read/write OBJ text directly, unifying language and mesh in one token space — no new tokenizer, at the cost of very long sequences. | [arXiv](https://arxiv.org/abs/2411.09595) | Open |
| 39 | **DeepMesh** | arXiv 2503.15265, Mar 2025; ICCV 2025 | First to apply **preference optimization (DPO)** to mesh generation, with a reward combining human preference and 3D alignment; improved block-wise indexing for high resolution. | [arXiv](https://arxiv.org/abs/2503.15265) | Open |
| 40 | **MeshMosaic** | arXiv 2509.19995, 24 Sep 2025; CVPR 2026 | Local-to-global assembly: segment the shape into patches (via PartField), generate each patch autoregressively with shared boundary conditions and *per-patch* quantization. **>100K triangles with a 0.5B model.** | [arXiv](https://arxiv.org/abs/2509.19995) | Code ⚠ |
| 41 | **QuadGPT** | arXiv 2509.21420, Sep 2025 | First *native* quad-dominant mesh generator (mixed tri/quad tokenization + hourglass transformer + truncated DPO), instead of generating triangles and merging them. | [arXiv](https://arxiv.org/abs/2509.21420) | ? |
| 42 | **MeshRipple** | CVPR 2026, pp. 12706–12718 | Frontier-aware BFS tokenization + sparse-attention global memory, fixing the holes and fragmentation caused by sliding-window inference breaking long-range topological dependencies. | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_MeshRipple_Structured_Autoregressive_Generation_of_Artist-Meshes_CVPR_2026_paper.html) | ? |
| 43 | **Mesh-Pro** (Tencent + CASIA) | arXiv 2603.00526, 28 Feb 2026; **CVPR 2026** | First **asynchronous online RL** for mesh generation (3.75× faster than synchronous), with Advantage-guided Ranking Preference Optimization, diagonal-aware tri/quad tokenization, and a ray-based geometric-integrity reward. | [arXiv](https://arxiv.org/abs/2603.00526) | ? |

### 1D. Texture, PBR materials, and UV

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 44 | **TEXTure** | arXiv 2302.01721, Feb 2023; SIGGRAPH 2023 ⚠venue not re-verified | Iterative depth-conditioned inpainting around the object with a trimap of "keep/refine/generate" regions. Founded diffusion texturing. | [arXiv](https://arxiv.org/abs/2302.01721) | Open |
| 45 | **Text2Tex** | arXiv 2303.11396; ICCV 2023 | Same iterative-inpainting family with automatic view selection to close holes. | [arXiv](https://arxiv.org/abs/2303.11396) | Open |
| 46 | **SyncMVD** | arXiv 2311.12891; SIGGRAPH Asia 2023 ⚠venue not re-verified | Synchronize the *latents* of multiple views through the UV texture space at each denoising step, instead of sequential inpainting. Removed most seam artifacts. | [arXiv](https://arxiv.org/abs/2311.12891) | Open |
| 47 | **Paint3D** | arXiv 2312.13913; CVPR 2024 | Explicitly **lighting-less** texture diffusion in UV space — the first serious attack on baked illumination in generated textures. | [arXiv](https://arxiv.org/abs/2312.13913) | Open |
| 48 | **MVPaint** | arXiv 2411.02336, Nov 2024 | Synchronized multi-view generation + 3D spatial-aware refinement + UV inpainting, robust to bad UV layouts. | [arXiv](https://arxiv.org/abs/2411.02336) | Open |
| 49 | **Material Anything** | arXiv 2411.15138, Nov 2024 | Material diffusion with a confidence mask to unify textured and untextured, lit and unlit inputs into one PBR estimator. | [arXiv](https://arxiv.org/abs/2411.15138) | Open |
| 50 | **TexGaussian** | arXiv 2411.19654, Nov 2024 | Octree-based 3D Gaussians as the texturing substrate, producing PBR maps in a single feed-forward pass (no per-asset optimization). | [arXiv](https://arxiv.org/abs/2411.19654) | Open |
| 51 | **MaterialMVP** (Tencent) | arXiv 2503.10289, Mar 2025; ICCV 2025 | Illumination-invariant multi-view PBR diffusion with dual-channel material supervision and consistency-regularized training; the PBR engine behind Hunyuan3D-Paint 2.1. | [arXiv](https://arxiv.org/abs/2503.10289) | Open |
| 52 | **RomanTex** (Tencent) | arXiv 2503.19011, Mar 2025 | 3D-aware RoPE + decoupled multi-attention to bind the 2D texture generator to the underlying geometry, reducing view drift. | [arXiv](https://arxiv.org/abs/2503.19011) | Open |
| 53 | **UniTEX** | arXiv 2505.23253, May 2025; CVPR 2026, pp. 19917–19927 | **Bypasses UV entirely** in stage 2: a Large Texturing Model regresses texture in a continuous volumetric "Texture Function" field, sidestepping UV topological ambiguity. | [arXiv](https://arxiv.org/abs/2505.23253) · [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_UniTEX_Universal_High_Fidelity_Generative_Texturing_for_3D_Shapes_CVPR_2026_paper.html) | Open |
| 54 | **ArtUV** + **SeamCrafter** (Tencent) | arXiv 2509.20710 and 2509.20725, 25 Sep 2025 | Learned **UV unwrapping**: SeamCrafter is a GPT-style seam predictor fine-tuned with DPO against a distortion/fragmentation preference model; ArtUV adds an autoencoder that refines a software UV layout into artist-style islands. | [ArtUV](https://arxiv.org/abs/2509.20710) · [SeamCrafter](https://arxiv.org/abs/2509.20725) | Code ⚠ |
| 55 | **MatLat** (KAIST) | arXiv 2512.17302, 19 Dec 2025; CVPR 2026 Highlight ⚠venue from project page, not arXiv metadata | Fine-tunes the *VAE* (not just the diffusion model) so roughness/metallic channels live in a latent space aligned with the pretrained RGB prior, plus a locality regularizer for cross-view consistency. Objaverse-XL shaded FID 6.309 → 3.083, albedo FID 9.630 → 4.599. | [arXiv](https://arxiv.org/abs/2512.17302) · [code](https://github.com/KAIST-Visual-AI-Group/MatLat) | **Open** |
| 56 | **VideoMatGen** (NVIDIA) | arXiv 2603.16566, Mar 2026 | Repurposes a **video** diffusion transformer (Cosmos-Predict) as the multi-view material generator, with a custom VAE packing base color + roughness + metallic + height into one latent. | [arXiv](https://arxiv.org/abs/2603.16566) | ? |
| 57 | **Toward Richer Material Generation via Procedural Data Enhancement** (NVIDIA/Cornell) | arXiv 2606.14988, Jun 2026; **SIGGRAPH 2026** | Attacks the *data* limit rather than the model: procedurally uplifts single-lobe GGX PBR into multi-lobe layered BSDFs (clearcoat, dust, scattering), compressed into a 6D neural-material latent over 10,640 Objaverse/BlenderVault assets. | [arXiv](https://arxiv.org/abs/2606.14988) | Dataset ⚠ |

### 1E. Part-level, structured, and editable generation

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 58 | **SAMPart3D** | arXiv 2411.07184, Nov 2024 | Zero-shot **multi-granularity** 3D part segmentation by distilling 2D foundation-model features into a 3D field, without text prompts. | [arXiv](https://arxiv.org/abs/2411.07184) | Open |
| 59 | **PartGen** (Meta) | arXiv 2412.18608, Dec 2024; CVPR 2025 | Multi-view diffusion that first segments parts consistently across views, then *completes* each part amodally (including unseen interiors). | [arXiv](https://arxiv.org/abs/2412.18608) | Closed ⚠ |
| 60 | **PartField** (NVIDIA) | arXiv 2504.11451, Apr 2025 | Feed-forward per-point *part feature field* — clustering it yields a segmentation hierarchy and cross-shape correspondence. Now used as infrastructure (e.g. MeshMosaic's patch splitter). | [arXiv](https://arxiv.org/abs/2504.11451) | Open |
| 61 | **HoloPart** | arXiv 2504.07943, Apr 2025 | Generative **part amodal completion**: turn incomplete surface segments into complete, individually watertight parts using local+global attention. | [arXiv](https://arxiv.org/abs/2504.07943) | Open |
| 62 | **PartCrafter** | arXiv 2506.05573, Jun 2025 | First *end-to-end* structured generator: one image → N part meshes denoised simultaneously, with a compositional latent space and hierarchical (intra-part / inter-part) attention. No pre-segmentation. | [arXiv](https://arxiv.org/abs/2506.05573) | Open |
| 63 | **OmniPart** | arXiv 2507.06165, Jul 2025; **SIGGRAPH Asia 2025** | Two-stage decoupling: an autoregressive planner emits part bounding boxes, then TRELLIS is fine-tuned into a spatially-conditioned part synthesizer with a voxel-discarding mechanism for clean interfaces. | [arXiv](https://arxiv.org/abs/2507.06165) · [ACM](https://dl.acm.org/doi/10.1145/3757377.3763872) | Open |
| 64 | **X-Part** (Tencent) | arXiv 2509.08643, Sep 2025 | Controllable, structure-coherent shape decomposition with bounding-box prompts and point-level semantics; the part module inside Hunyuan3D Studio. | [arXiv](https://arxiv.org/abs/2509.08643) | Open |
| 65 | **Nano3D** | arXiv 2510.15019, 16 Oct 2025; **ICLR 2026** | Training-free, mask-free part-level 3D editing by running FlowEdit inside TRELLIS with Voxel/Slat-Merge; ships **Nano3D-Edit-100k**, the first large-scale paired 3D-editing dataset. | [arXiv](https://arxiv.org/abs/2510.15019) · [data](https://huggingface.co/datasets/yejunliang23/Nano3D-Edit-100k) | Open |
| 66 | **DreamPartGen** | arXiv 2603.19216, Mar 2026 | Part-aware *text*-to-3D: Duplex Part Latents (geometry+appearance per part) co-denoised with Relational Semantic Latents derived from language, so parts stay text-grounded rather than merely geometric. | [arXiv](https://arxiv.org/abs/2603.19216) | ? |

### 1F. Articulated and riggable generation

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 67 | **Articulate-Anything** | arXiv 2410.13882, Oct 2024; ICLR 2025 ⚠venue not re-verified | VLM-driven *program synthesis*: an actor–critic loop writes and iteratively repairs Python that assembles a URDF from retrieved/generated part meshes, from text, image, or video input. | [arXiv](https://arxiv.org/abs/2410.13882) | Open |
| 68 | **RigAnything** | arXiv 2502.09615, Feb 2025; **ACM TOG 44(4) / SIGGRAPH 2025** | Template-free autoregressive skeleton generation with a joint-diffusion head; handles arbitrary input pose (not just rest pose) and rigs in <2s, ~20× faster than template methods. | [arXiv](https://arxiv.org/abs/2502.09615) | Open |
| 69 | **MagicArticulate** (ByteDance/NTU) | arXiv 2502.12135, Feb 2025; **CVPR 2025**, pp. 15998–16007 | Reframed skeleton generation as sequence modeling, and released **Articulation-XL** (33K+ rigged models mined from Objaverse-XL) — the dataset that made this subfield possible. | [arXiv](https://arxiv.org/abs/2502.12135) · [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Song_MagicArticulate_Make_Your_3D_Models_Articulation-Ready_CVPR_2025_paper.html) | Open |
| 70 | **UniRig** (Tsinghua + Tripo) | arXiv 2504.12451, Apr 2025; **SIGGRAPH 2025 (TOG)** | Skeleton Tree Tokenization + bone-point cross-attention for skeleton *and* skinning in one framework; **Rig-XL** (14K+ rigged models). Reports +215% rigging / +194% motion accuracy over prior SOTA. | [arXiv](https://arxiv.org/abs/2504.12451) · [code](https://github.com/VAST-AI-Research/UniRig) | Open |
| 71 | **Anymate** | arXiv 2505.06227, May 2025; SIGGRAPH 2025 ⚠venue not re-verified | A dataset (230K+ rigged assets) and a modular joint/connectivity/skinning baseline suite — the closest thing to a standard benchmark for auto-rigging. | [arXiv](https://arxiv.org/abs/2505.06227) | Open |
| 72 | **Puppeteer** (ByteDance Seed + NTU) | arXiv 2508.10898, Aug 2025; **NeurIPS 2025 Spotlight** | Joint-based tokenization + hierarchical ordering with stochastic perturbation for skeletons, topology-aware joint attention for skinning, and a *network-free* differentiable optimizer that animates from reference video. **Articulation-XL2.0: 59.4K rigged models incl. 11.4K diverse-pose.** | [arXiv](https://arxiv.org/abs/2508.10898) · [code](https://github.com/Seed3D/Puppeteer) | Open |
| 73 | **URDF-Anything** → **URDF-Anything+** | arXiv 2511.00940 (NeurIPS 2025) → arXiv 2603.14010, Mar 2026 | From MLLM-based URDF reconstruction (+17% mIoU, −29% joint error, +50% physical executability) to an end-to-end autoregressive **diffusion** model emitting part geometry + joint parameters from one RGB image, with a termination token for variable part counts. Enables sim-trained policies to transfer without online adaptation. | [v1](https://arxiv.org/abs/2511.00940) · [v2](https://arxiv.org/abs/2603.14010) | ? |
| 74 | **Rigel3D** | arXiv 2605.13129, May 2026 | Generates geometry **and rig jointly** rather than post-hoc: coupled *surface SLat* and *skeleton SLat* decoded into mesh + skeleton topology + joint coordinates + skinning weights, plus open-vocabulary joint labels for retargeting. | [arXiv](https://arxiv.org/abs/2605.13129) | ? |
| 75 | **AniGen** (VAST) | **SIGGRAPH 2026** | Unified *S³ Fields* (Shape, Skeleton, Skin) over one spatial domain, with a confidence-decaying skeleton field for Voronoi-boundary ambiguity and a dual skin field decoupling weights from joint count. | [code](https://github.com/VAST-AI-Research/AniGen) | Announced ⚠ |
| 76 | **SPARK** | **CVPR 2026** | VLM extracts coarse URDF + part reference images; a DiT synthesizes consistent parts; a differentiable forward-kinematics module refines continuous joint parameters under VLM-generated open-state supervision. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/He_SPARK_Sim-ready_Part-level_Articulated_Reconstruction_with_VLM_Knowledge_CVPR_2026_paper.pdf) | ? |

### 1G. CAD and parametric generation

| # | Name | Venue + date | One-line contribution | Link | Weights |
|---|---|---|---|---|---|
| 77 | **DeepCAD** | arXiv 2105.09492, May 2021; ICCV 2021 ⚠venue not re-verified | Transformer over sketch-and-extrude command sequences + the 160K-model dataset that every subsequent CAD paper trains and evaluates on. | [arXiv](https://arxiv.org/abs/2105.09492) | Open |
| 78 | **BrepGen** | arXiv 2401.15563, Jan 2024; SIGGRAPH 2024 ⚠venue not re-verified | Diffusion directly over the **B-rep** structured latent tree (faces → edges → vertices) with deduplication to enforce topological validity. | [arXiv](https://arxiv.org/abs/2401.15563) | Open |
| 79 | **Text2CAD** | arXiv 2409.17106, Sep 2024; NeurIPS 2024 ⚠venue not re-verified | Text → parametric CAD construction sequence, with an LLM/VLM-built annotation pipeline spanning beginner-to-expert prompt styles over DeepCAD. | [arXiv](https://arxiv.org/abs/2409.17106) | Open |
| 80 | **CAD-MLLM** | arXiv 2411.04954, Nov 2024 | Multimodality-conditioned CAD generation (text/image/point cloud/command) in one MLLM, plus topology-quality metrics that geometry-only metrics miss. | [arXiv](https://arxiv.org/abs/2411.04954) | Open |
| 81 | **CAD-Recode** | arXiv 2412.14042, Dec 2024; **ICCV 2025** | Point cloud → executable **CadQuery Python**, decoded by a Qwen2-1.5B with a single-layer point projector, trained on 1M procedurally generated programs. DeepCAD mean CD 0.30, IoU 92.0, invalidity 0.4%; Fusion360 mean CD 0.35, IoU 87.8 — ~10× lower mean CD than prior SOTA. | [arXiv](https://arxiv.org/abs/2412.14042) · [code](https://github.com/filaPro/cad-recode) | Open |
| 82 | **HoLa** | arXiv 2504.14257, Apr 2025; SIGGRAPH 2025 ⚠venue not re-verified | *Holistic* B-rep latent: one latent per solid from which all faces/edges are derived, avoiding the per-primitive latents that cause topological inconsistency. | [arXiv](https://arxiv.org/abs/2504.14257) | Open |
| 83 | **cadrille** | arXiv 2505.22914, May 2025 | Multi-modal (point cloud + image + text) CAD reconstruction and the **first to apply online RL fine-tuning** to CAD program generation. | [arXiv](https://arxiv.org/abs/2505.22914) | Open |
| 84 | **BrepForge** | arXiv 2605.19411, May 2026 | Exploits a structural asymmetry in B-reps: wireframe composition is high-entropy, surface interiors are not. Factorizes into wireframe generation + boundary-conditioned surface instantiation; reconstructs from only 4,096 points. | [arXiv](https://arxiv.org/abs/2605.19411) | ? |
| 85 | **DualBrep** | arXiv 2606.31579, Jun 2026 | Encodes a CAD solid as *dual continuous fields* — an SDF for geometry and a UDF whose Voronoi partition implicitly encodes topology — into one latent, so flow matching samples geometry and topology jointly instead of sequentially. | [arXiv](https://arxiv.org/abs/2606.31579) | ? |
| 86 | **ParaCAD** — *Autoregressive B-Rep Shape Generation with Parametric Surfaces* | arXiv 2607.17093, 19 Jul 2026; SIGGRAPH 2026 ⚠venue not in arXiv metadata | Autoregressive B-rep generation on **native parametric surfaces** (exact surface type + continuous parameters per face), recovering edges/vertices by global surface intersection rather than sampling a discretized proxy. | [arXiv](https://arxiv.org/abs/2607.17093) · [code](https://github.com/dafei-qin/ParaCAD) | Code ⚠ |
| 87 | **CADENA** | arXiv 2608.00799, **1 Aug 2026** | *Stepwise* reverse engineering: emit one CAD operation, execute it, look at the residual against the target, emit the next. RL against volumetric IoU with no ground-truth program. Introduces CADENA-Bench (real mechanical parts) and GMS, a normal-aware, watertightness-free metric. | [arXiv](https://arxiv.org/abs/2608.00799) · [code](https://github.com/zhemdi/cadena) · [weights](https://huggingface.co/kulibinai/cadena) · [bench](https://huggingface.co/datasets/kulibinai/cadena-bench) | **Open** |

### 1H. Evaluation

| Name | Venue + date | What it measures | Link |
|---|---|---|---|
| **ULIP-2** | arXiv 2305.08275; CVPR 2024 ⚠venue not re-verified | Scalable tri-modal (3D/image/text) pretraining; the source of "ULIP score" semantic-alignment metrics | [arXiv](https://arxiv.org/abs/2305.08275) |
| **Uni3D** | arXiv 2310.06773; ICLR 2024 ⚠venue not re-verified | Billion-scale unified 3D representation; the other standard backbone for text–shape alignment scoring | [arXiv](https://arxiv.org/abs/2310.06773) |
| **T³Bench** | arXiv 2310.02977, Oct 2023 | First structured text-to-3D suite: quality + alignment across three prompt difficulty tiers | [arXiv](https://arxiv.org/abs/2310.02977) |
| **GPTEval3D** | arXiv 2401.04092; **CVPR 2024** | GPT-4V as a pairwise judge over user-defined criteria → Elo ranking. The de facto automatic metric | [arXiv](https://arxiv.org/abs/2401.04092) |
| **3DGen-Bench** | arXiv 2503.21745, Mar 2025 | Closed expert annotation (13.8K votes) over text-to-3D *and* image-to-3D, plus a trained automatic scorer | [arXiv](https://arxiv.org/abs/2503.21745) |
| **3D Arena** | arXiv 2506.18787, Jun 2025 | Open human-preference platform: **123,243 votes / 8,096 users / 19 models** since Jun 2024; iso3d prompt set; 99.75% authenticity via binomial fraud detection | [arXiv](https://arxiv.org/abs/2506.18787) |
| **Hi3DEval** | arXiv 2508.05609, Aug 2025; **NeurIPS 2025 D&B** | Hierarchical: object-level **and part-level** quality, plus explicit material-realism scoring (albedo, saturation, metallicness) from reflectance under varied lighting | [arXiv](https://arxiv.org/abs/2508.05609) |
| **CADENA-Bench + GMS** | arXiv 2608.00799, 1 Aug 2026 | Real mechanical parts, per-family reporting, and a metric defined on open geometry that matches surfaces by *normal* as well as position; benchmark publicly released | [arXiv](https://arxiv.org/abs/2608.00799) · [bench](https://huggingface.co/datasets/kulibinai/cadena-bench) |

---

## 2. The twelve that matter most, explained plainly

*Written for someone comfortable with diffusion models and transformers but not with graphics. Jargon is defined on first use.*

**Jargon primer.** A **mesh** is a list of 3D vertices plus a list of triangles (or quads) connecting them; **topology** means the connectivity pattern — how the triangles are laid out — as opposed to the shape they trace. An **SDF (signed distance function)** is a function from a 3D point to its distance from the surface, negative inside; the surface is the zero level set. **Marching cubes** is the classical algorithm that converts such a function, sampled on a grid, into a mesh; it produces dense, uniform, ugly triangles ("triangle soup") regardless of how clean the underlying shape is. **UV unwrapping** is the act of cutting a 3D surface along **seams** and flattening it into a 2D square so you can paint a texture image on it. **PBR (physically based rendering)** materials store *what the surface is made of* — albedo (base color with no lighting), roughness, metallic — rather than *what it looked like under one particular light*; only PBR relights correctly in a game engine. **B-rep (boundary representation)** is the CAD format: exact analytic surfaces (planes, cylinders, splines) glued along exact curves, not an approximating mesh.

---

**1. DreamFusion (2209.14988) — the trick that started it all, and the trick that failed.**
There was almost no 3D training data in 2022, but there were excellent 2D text-to-image diffusion models. DreamFusion's idea, **Score Distillation Sampling (SDS)**, is to make the 2D model act as a critic: parameterize a NeRF, render it from a random camera, add noise to the render, ask the frozen diffusion model to denoise it, and backpropagate the difference between predicted and injected noise into the NeRF's weights. Repeat for a few hours and you get a 3D thing whose every view is a plausible image of your prompt. It requires zero 3D data. That is why it mattered, and it is also why it died: because the 2D critic has no idea what "the back of this object" means, it happily approves a bunny with a face on both ends (the **Janus problem**); because SDS is effectively mode-seeking and needs enormous classifier-free guidance, outputs are over-saturated and low-diversity; and because it is per-prompt optimization, every asset costs GPU-hours. ProlificDreamer's Variational Score Distillation (2305.16213) fixed the saturation and diversity by treating the 3D scene as a distribution optimized by particle variational inference, but could not fix the cost. Both surveys note the surviving structural defect independently of quality: SDS geometry comes out noisy with **lighting baked into the albedo channel**, which makes it useless downstream.

**2. Zero-1-to-3 (2303.11328) and MVDream (2308.16512) — teaching 2D models about cameras.**
The generation-2 insight was that you do not need the 2D model to *be* 3D-aware, you just need it to be *camera-conditioned*. Zero-1-to-3 fine-tunes Stable Diffusion on Objaverse renders so that, given an image and a relative camera rotation, it produces the image from that new viewpoint. Now you can hallucinate the back of an object directly instead of optimizing your way to it. MVDream goes one step further: rather than generating views one at a time and hoping they agree, it generates four views *jointly* by replacing 2D self-attention with attention that spans all four views, so the network can enforce consistency internally. This is the single cleanest fix for the Janus problem, and the same "make attention span the views" move reappears everywhere later, including in every modern texture generator.

**3. LRM (2311.04400) — amortization.**
LRM asked the obvious question: if we're going to reconstruct 3D from images a million times, why optimize each one? It trains a ~500M-parameter transformer to map image tokens directly to a **tri-plane** (three axis-aligned 2D feature grids whose bilinear samples, concatenated, define a 3D field — a cheap way to get 3D capacity at 2D memory cost), decoded as a NeRF. Inference is ~5 seconds. Instant3D then chained a multi-view generator into an LRM, giving the recipe that dominated 2024: *text → 4 views → transformer → 3D*. TripoSR made it open and sub-second; InstantMesh swapped the NeRF head for **FlexiCubes**, a differentiable iso-surfacing layer, so the model could be trained against mesh geometry directly. The limitation of this whole family is inherited: the geometry is only as good as what the 2D views imply, so thin structures, concavities, and object interiors are systematically wrong.

**4. 3DShape2VecSet (2301.11445) — the VecSet, and why it won.**
This is the most quietly influential paper in the whole survey. The problem with 3D latents is choosing a data structure: voxel grids waste memory on empty space and scale as N³; point clouds have no connectivity; meshes are irregular graphs that neural nets hate. 3DShape2VecSet's answer: represent a shape as an **unordered set of, say, 512 latent vectors**, with no spatial index at all. To evaluate occupancy at a query point, cross-attend the query against the set. Because the latent is a set of tokens, a vanilla transformer can denoise it — no sparse convolutions, no custom kernels, no resolution ceiling. This is called a **VecSet**, and it is what CLAY, Hunyuan3D, TripoSG, Step1X-3D, and Michelangelo all generate. The tradeoff is that the tokens are not spatially grounded, so you cannot easily condition a specific token on a specific region of space.

**5. TRELLIS (2412.01506) — the structured latent, the other half of the field.**
TRELLIS makes the opposite choice. It first predicts *which* voxels on a coarse grid the surface passes through (the "sparse structure"), then attaches a latent code to each of those active voxels — **SLAT, Structured LATents**. The latent is therefore spatially indexed: token *i* means "the surface detail near this specific location". That buys you two things a VecSet cannot easily give you: (a) you can decode the same latent into a mesh, a 3D Gaussian splat, or a radiance field, because the geometry is already localized; (b) you can condition, edit, or segment *regionally*, which is exactly why almost every 2025–2026 part-level and editing paper (OmniPart, Nano3D, Rigel3D) is built on TRELLIS rather than on a VecSet model. The cost is that you now need sparse-convolution / sparse-attention machinery.

**6. Hunyuan3D 2.1 (2506.15442) — the open reference implementation.**
Hunyuan3D's contribution is less a single idea than a complete, released, two-stage system that everybody can build on: **Hunyuan3D-DiT**, a flow-matching diffusion transformer over a ShapeVAE latent (importance-sampled on the mesh surface so detail-dense regions get more tokens), producing an SDF that marching cubes turns into a mesh; then **Hunyuan3D-Paint**, a mesh-conditioned multi-view diffusion model that paints it. Version 2.1 is the inflection point because it is the first in the series to release **both weights and training code**, and because it switched the paint stage from baked RGB to **PBR** — albedo, metallic, roughness as separate channels — which is the difference between an asset that looks right in the demo render and an asset that looks right in Unreal. (Note the trend the 2604.23629 survey flags: 2.5, 3.0, and 3.1 went back behind an API.)

**7. TRELLIS.2 (2512.14692) — the current open-weight frontier, and the clearest statement of what a 3D latent should be.**
Every SDF-based model shares a hidden constraint: an SDF assumes the surface is a closed boundary between inside and outside. Real assets are not like that — a leaf is an open surface, a mechanical assembly is non-manifold, a cockpit has an enclosed interior. TRELLIS.2 replaces the field with **O-Voxel**, a "field-free" sparse voxel structure using a flexible dual grid: one vertex per occupied cell, positioned by solving a quadratic error function against the actual mesh triangles, with extra terms that pull vertices onto boundary edges. It handles open, non-manifold, and enclosed geometry, converts to and from a mesh in seconds/milliseconds with no optimization or rendering in the loop, and carries **six channels of PBR material per voxel including opacity** so glass and translucent surfaces work. On top of it sits a sparse-convolutional VAE with 16× spatial downsampling — a 1024³ fully-textured asset becomes ~9.6K latent tokens — and a 4B flow-matching DiT. Result: 3s at 512³, 17s at 1024³, 60s at 1536³ on a single H100, MIT-licensed, weights on Hugging Face. If you need one open model as a baseline in mid-2026, this is it.

**8. Direct3D-S2 (2505.17412) — the paper that put high-resolution 3D within reach of a normal lab.**
Worth calling out separately because of what it means for an academic budget. Sparse-volume DiTs choke because attention over hundreds of thousands of active voxels is quadratic. Direct3D-S2's **Spatial Sparse Attention** partitions tokens spatially so attention is computed within and across compressed spatial blocks, with a custom GPU kernel: 3.9× faster forward, 9.6× faster backward than FlashAttention-2 at 1024³. The headline claim is the one that matters here — training at **1024³ on 8 GPUs**, where prior volumetric methods needed 32+ GPUs to train at 256³. Sparc3D (2505.14521) attacks the same wall from the data side, with a sparse deformable marching-cubes remesher that turns a raw mesh into a watertight 1024³ surface in ~30s (3× faster than prior watertighting) and a fully sparse-convolutional VAE that removes the modality conversion step entirely.

**9. MeshGPT (2311.15475) → MeshMosaic (2509.19995) — mesh generation as language modeling, and the scaling story.**
Marching cubes gives you 100K uniform triangles for a shape an artist would model with 800 well-placed ones. Artist topology matters because edge loops determine how a mesh deforms when animated and how efficiently it renders. MeshGPT's move: learn a **VQ-VAE vocabulary of triangle embeddings** (using graph convolutions so each token knows its local neighborhood), flatten the mesh into a token sequence, and train a decoder-only transformer to predict the next token. It works, and it produces genuinely artist-like compact meshes — but the sequence length explodes with face count, so the entire subsequent literature is about **tokenization compression**: MeshAnything V2's Adjacent Mesh Tokenization (one vertex per face where possible, ~half the tokens), EdgeRunner's EdgeBreaker traversal, BPT's blocked indexing plus patch aggregation (~75% shorter). Meshtron then attacked it architecturally — an hourglass transformer that merges tokens internally, plus sliding-window inference giving constant memory — reaching **64K faces at 1024-level coordinate resolution**. MeshMosaic finally sidesteps the problem: segment the shape into patches, generate each patch autoregressively with shared boundary conditions, and quantize each patch in its *own* normalized coordinate frame. Sequence length now depends on patch size, not total face count, and effective quantization resolution goes up. It reaches **>100K triangles with a 0.5B model**.

**10. PartCrafter (2506.05573) — why part-level is the real unlock.**
A generated object is a single fused blob. You cannot open the drawer, swap the wheel, assign a different material to the handle, or attach a joint. Every prior part-aware method was two-stage — segment the image, reconstruct each segment — which fails whenever a part is occluded. PartCrafter takes a pretrained whole-object shape DiT and *restructures* it: each part gets its own disentangled set of latent tokens, all parts are denoised simultaneously, and a hierarchical attention mechanism lets tokens attend within their part (local detail) and across all parts (global coherence). Because it inherits the whole-object generative prior, it can hallucinate parts that are not visible at all in the input image. Its own stated limitation is the field's: it trains on ~50K part-annotated assets versus the millions used for whole objects. OmniPart (SIGGRAPH Asia 2025) offers the complementary design — plan part bounding boxes autoregressively first, then fine-tune TRELLIS into a spatially-conditioned part synthesizer, with a voxel-discarding mechanism to keep part interfaces clean.

**11. Puppeteer (2508.10898) / UniRig (2504.12451) — rigging stopped being the unsolved stage.**
A **rig** is a skeleton of joints plus, for every mesh vertex, a set of weights saying how much each bone influences it (**skinning weights**); without one, a generated model is a statue. Classical learned rigging (RigNet) used graph convolutions and did not survive contact with the diversity of AI-generated meshes. The 2025 fix was to make it a sequence problem: UniRig introduces **Skeleton Tree Tokenization** so a bone hierarchy becomes a valid token sequence by construction, with bone-point cross-attention for skinning; Puppeteer uses joint-based tokenization with hierarchical ordering plus stochastic perturbation (so the model learns bidirectionally rather than memorizing one traversal), and topology-aware joint attention that encodes skeletal graph distance when predicting weights. The other half of the story is data: MagicArticulate mined **Articulation-XL** (33K rigged models) from Objaverse-XL, UniRig built **Rig-XL** (14K), Anymate reports 230K+, and Puppeteer expanded to **Articulation-XL2.0 (59.4K, including 11.4K non-rest-pose examples)** — that pose diversity is what makes the models work on in-the-wild inputs, and Puppeteer's ablations show MagicArticulate degrading specifically on the diverse-pose subset because it trained on rest poses only. The 2026 direction is to stop bolting the rig on afterwards: Rigel3D generates coupled surface and skeleton latents, AniGen generates shape, skeleton, and skin as three fields over one spatial domain.

**12. CAD-Recode (2412.14042) — the "just write code" result.**
CAD is a different problem from mesh generation because the ground truth is a *program*: sketch a rectangle, extrude it 20mm, fillet these edges. Prior work built bespoke command tokenizers and transformers over them. CAD-Recode's observation is that pretrained LLMs already know Python, and **CadQuery** is a Python CAD library — so emit CadQuery source, decoded by a Qwen2-1.5B with a single linear layer projecting point-cloud tokens into its embedding space, trained on 1M procedurally generated programs. It beats the specialized architectures by roughly an order of magnitude in mean Chamfer distance (DeepCAD test: 0.30 mean CD, 92.0 IoU, 0.4% invalid, versus CAD-SIGNet's 3.43 / 77.6 / 0.9), the output is human-readable and editable, and off-the-shelf LLMs can answer questions about it. The 2026 follow-ups add the loop CAD-Recode lacks: cadrille adds online RL; CADENA (2608.00799) emits **one operation at a time and executes it**, comparing the partial build to the target before deciding the next step, with RL against volumetric IoU requiring no ground-truth program at all.

---

## 3. Generation-by-generation: what actually happened

### 3.1 Generation 1 (2022–2023): SDS optimization — and precisely why it died

DreamFusion → Magic3D → Fantasia3D → ProlificDreamer → RichDreamer is a clean progression of increasingly sophisticated distillation objectives (SDS → CSD → ISM → VSD → ASD; the 2504.11734 survey has the full taxonomy). It ended for five compounding reasons, four of which are quality and one of which is decisive:

1. **Cost.** Per-prompt optimization, GPU-hours per asset. Non-negotiable for any production or dataset-scale use.
2. **Janus / multi-face artifacts.** A 2D critic evaluating one view at a time cannot enforce global 3D consistency.
3. **Over-saturation and low diversity.** SDS's dependence on very large classifier-free guidance is mode-seeking. VSD fixed this but at the cost of LoRA-based score estimation, which ScaleDreamer later showed compromises generalization across prompts and causes training instability.
4. **Baked lighting.** Per 2604.23629 §1: SDS "frequently yields noisy surfaces with lighting entangled in the albedo channel." An asset you cannot relight is not an asset.
5. **Data arrived.** Objaverse (800K) and Objaverse-XL (10.2M) removed the premise. Once you can train on 3D directly, distilling from 2D is strictly worse.

SDS is not entirely gone — it survives in avatar generation and in low-data regimes — but no frontier object-level system in 2025–2026 uses it for geometry.

### 3.2 Generation 2 (2023–2024): multi-view diffusion + reconstruction

Two sub-lineages, usually chained. **Multi-view generators**: Zero-1-to-3 (camera conditioning) → SyncDreamer (shared 3D feature volume) → MVDream (joint cross-view attention) → Zero123++ (engineering: tiled 6-view layout, reference attention) → Wonder3D (color + normals) → ImageDream (image-prompted). **Reconstructors**: LRM → Instant3D → TripoSR → InstantMesh / CRM / LGM → SF3D.

This generation is not dead — it *moved*. The multi-view machinery is now the standard **texturing** stage (Hunyuan3D-Paint, MaterialMVP, RomanTex, MatLat are all multi-view diffusion models), while geometry moved to native 3D. The reason is structural: 2D views underdetermine interiors, thin structures, and occluded geometry, and no amount of view consistency fixes that. TRELLIS.2 states it directly about its predecessor: SLAT's "reliance on multiview 2D image feature input and pure rendering-based supervision leads to deficiencies in capturing complex structures and materials."

### 3.3 Generation 3 (2024–present): native 3D latent diffusion

**Two latent families, and you should know which one you're looking at.**

- **VecSet** (3DShape2VecSet → Michelangelo → CLAY → Hunyuan3D-DiT → TripoSG → Step1X-3D → LATTICE). An *unordered set* of latent tokens; occupancy/SDF at a query point is obtained by cross-attention. Advantage: a plain transformer works, scaling is trivial, no custom kernels. Disadvantage: no spatial grounding, so regional conditioning and part-level control are awkward.
- **Structured / sparse-voxel latent** (XCube → TRELLIS → Direct3D-S2 → Sparc3D → SparseFlex → TRELLIS.2). Latents attached to *active voxels* in a sparse grid. Advantage: spatially indexed, multi-format decoding, natural regional control, and the resolution scales with the sparse structure rather than the token budget. Disadvantage: needs sparse convolution / sparse attention infrastructure.

Both are trained with **flow matching / rectified flow** rather than DDPM — a straight-line probability path between noise and data, which is simpler and converges faster; this is now universal in the area. Both are typically **two-stage**: geometry first, then texture/material, because it decouples two hard problems and lets you texture handcrafted meshes too. Both are conditioned primarily on a **single image** (with DINOv2/DINOv3 features), with text-to-3D handled by putting a text-to-image model in front — genuine text-conditioned native 3D is comparatively rare and mostly worse.

**The 2026 state.** TRELLIS.2's O-Voxel is the most complete answer to the representation question so far, because it is the first to handle open/non-manifold/enclosed geometry *and* PBR *and* achieve high compression, and because the whole thing is MIT-licensed with training code. The unresolved axis is unification. UniLat3D argues for a single geometry-appearance latent; Native3D (2606.07117 — note this one is *scene*-level, not object-level) proposes a unified mesh-texture joint representation that "completely bypasses 2D intermediate representations," on the argument that adapting 3D to the 2D domain "inevitably introduces domain adaptation issues including geometric structural distortion and texture detail degradation"; EVA01 (2605.16745) goes further and makes mesh a native MLLM modality via a Mixture-of-Transformers with separate understanding and generation experts, targeting multi-turn editing with identity preservation. That argument is not settled.

---

## 4. Mesh generation as sequence modeling: what triangle counts are real, and is topology solved?

**Documented face-count ceilings, in order:**

| System | Claimed max | Mechanism |
|---|---|---|
| MeshGPT (2023) | ~800 faces (ShapeNet categories) ⚠ approximate | VQ-VAE + decoder-only transformer |
| MeshAnything V1 | ~800 faces | Shape-conditioned; "hundreds of times fewer faces" than marching cubes |
| MeshAnything V2 | ~1,600 faces | Adjacent Mesh Tokenization (~½ sequence length) |
| BPT (CVPR 2025) | >8,000 faces | ~75% sequence compression via blocked indexing + patchification |
| Meshtron (NVIDIA) | **64,000 faces @ 1024³ coord resolution** | Hourglass transformer, truncated training, sliding-window inference, 1.1B params |
| MeshMosaic (CVPR 2026) | **>100,000 triangles** | Patch decomposition, per-patch quantization, boundary conditioning; only 0.5B params |

**Is artist-quality topology solved? No, and the field says so in three distinct ways.**

*Scale is close to solved; structure is not.* MeshMosaic notes prior methods "typically handle only around 8K faces" — so the jump to 100K is real and recent. But MeshRipple (CVPR 2026) identifies a failure that scale alone does not fix: training on truncated segments with sliding-window inference "breaks long-range geometric dependencies, producing holes and fragmented components." Its fix — frontier-aware BFS tokenization plus sparse-attention global memory — is an admission that the sequence formulation fights the geometry.

*Triangles are not what artists want.* Professional topology is **quad-dominant** with clean edge loops. Until Sep 2025, quad meshes were produced by generating triangles and merging pairs, which QuadGPT says "typically produces quad meshes with poor topology." QuadGPT is the first native quad-dominant autoregressive generator; Mesh-Pro (Mar 2026) refines its tokenization (declaring face type with a leading token "precludes a truly consistent canonical ordering, resulting in geometric artifacts and structural defects") and adds a ray-based reward specifically to "reduce broken ratio."

*The objective is wrong, so RL is being bolted on.* Imitation of artist meshes optimizes local plausibility, not the non-local properties (deformation-aware edge loops, controlled density gradients) that make topology good. Hence a rapid RL turn: DeepMesh (DPO), Mesh-RFT (masked DPO for localized defect correction), QuadGPT (tDPO), Mesh-Pro (asynchronous online ARPO, 3.75× faster than synchronous, explicitly claiming better generalization than DPO and GRPO). The 2604.23629 survey's assessment: data-imitation methods "do not explicitly optimize non-local topological qualities."

*A minority report.* Diffusion-based topology generation is underexplored — PolyDiff (discrete denoising over quantized triangle soups), MeshCraft (continuous face tokens + flow DiT, parallel decoding with face-count control), and SpaceMesh (SIGGRAPH Asia 2024; continuous halfedge latents with a Sinkhorn-normalized connectivity network, "orders of magnitude faster than autoregressive methods"). The survey's stated reason for the imbalance: "corrupting and denoising irregular polygonal connectivity without destroying mesh validity is considerably harder than prefix-conditioned sequence prediction."

---

## 5. Texture and PBR materials

The arc is: **sequential inpainting → synchronized multi-view → UV-space / 3D-native → PBR-first**.

- **Sequential inpainting (2023).** TEXTure, Text2Tex: paint from one view, project onto the mesh, rotate, inpaint the newly visible region. Simple and fundamentally lossy — error accumulates and you get visible seams at every view boundary.
- **Synchronized multi-view (2023–2025).** SyncMVD synchronizes latents through UV space at each denoising step; MVPaint adds 3D-aware refinement and UV inpainting; RomanTex binds the 2D generator to the geometry with 3D-aware rotary embeddings; Hunyuan3D-Paint is the productized version. This is the current workhorse.
- **UV-space generation.** TEXGen (SIGGRAPH Asia 2024) generates directly in the UV domain rather than in views, avoiding reprojection entirely — but inherits whatever the UV atlas gives it.
- **Bypassing UV.** UniTEX (CVPR 2026) is the most interesting recent move: rather than fight UV's topological ambiguity, regress texture in a continuous volumetric **Texture Function** field, supervised by extending surface textures into a volume. TRELLIS.2 makes the same argument from the latent side, storing material per voxel and reasoning about appearance natively in 3D: "multi-view approaches often suffer from inconsistencies… UV-based methods suffer from ambiguous UV charts and seam artifacts."

**PBR specifically.** The transition from baked RGB to albedo/roughness/metallic is the single most important production change of the period. Paint3D (CVPR 2024) started it with lighting-less texture diffusion; Material Anything, TexGaussian, MaterialMVP, and PBR3DGen made material decomposition a first-class output; Hunyuan3D 2.1 (Jun 2025) shipped it in an open system. Three 2026 papers deserve attention:

- **MatLat** (CVPR 2026 Highlight) makes the sharpest technical point: everyone else freezes the pretrained VAE and stuffs roughness/metallic into extra channels, which causes a latent distribution shift that hurts convergence. MatLat fine-tunes the encoder with *residual latent prediction* and KL-regularizes the posterior back toward the original RGB latent distribution, plus a locality regularizer (crop latent patches, decode, align to image regions) because correspondence-aware attention alone is insufficient for view consistency unless the latent-to-image map preserves spatial locality. Measured: shaded FID 6.309 → 3.083, albedo FID 9.630 → 4.599 on Objaverse-XL.
- **VideoMatGen** (NVIDIA, Mar 2026) uses a **video** diffusion transformer (Cosmos-Predict 1-7B) as the multi-view material generator, with a custom VAE packing base color, roughness, metallicity, and height into one latent so extra channels cost no extra tokens.
- **Toward Richer Material Generation** (SIGGRAPH 2026) attacks the data ceiling instead: standard PBR is a diffuse term plus a *single* GGX specular lobe, which cannot express clearcoat, dust, or layered scattering. They procedurally uplift simple PBR into multi-lobe layered BSDFs (22 parameters), compress into a 6D neural-material latent (two RGB latent textures + a universal decoder MLP), and build the dataset over 10,640 Objaverse/BlenderVault assets — 22 GPU-hours of latent optimization on 80 L40/L40S.

**UV unwrapping became a learning problem in Sep 2025.** SeamCrafter is a GPT-style seam generator over point clouds with a dual-branch encoder (surface points for geometry, vertex-edge skeleton points for topology), DPO-fine-tuned against a preference model scoring UV distortion and fragmentation. ArtUV chains SeamGPT-style seam prediction with an autoencoder that refines a software UV layout into artist-style islands. This is a genuinely new capability — before this, xatlas was the answer.

---

## 6. Part-level and structured generation: why it matters

**Why it matters, concretely.** A monolithic mesh cannot be edited (change the handle without regenerating the mug), cannot carry per-part materials, cannot be rigged with an articulated joint, cannot be given per-part collision geometry, and cannot be reused compositionally. The 2604.23629 survey names controllability as one of three coupled bottlenecks: "Current methods overwhelmingly operate in a one-shot paradigm — the user provides a prompt and receives a complete asset with no opportunity for iterative refinement. This conflicts with typical professional workflows."

**State of the art, in two families.**

*Decompose an existing shape.* SAMPart3D (multi-granularity zero-shot segmentation by distilling 2D features into a 3D field), PartField (feed-forward part feature fields that give a segmentation hierarchy *and* cross-shape correspondence — now used as infrastructure by other systems including MeshMosaic), HoloPart (generative amodal completion of segmented pieces into individually watertight parts), X-Part (controllable decomposition with box prompts, shipped inside Hunyuan3D Studio).

*Generate parts directly.* PartGen (multi-view segmentation then amodal part reconstruction), PartCrafter (simultaneous denoising of per-part latent sets with hierarchical attention, no pre-segmentation), OmniPart (autoregressive box planning + spatially-conditioned TRELLIS fine-tuning), DreamPartGen (2026; text-grounded parts via Relational Semantic Latents co-denoised with per-part Duplex Latents), MoCA (Mixture-of-Components Attention with importance-based routing, scaling to 32 components per asset — twice prior capacity).

**The honest state.** Part-level generation works and is being productized, but it is data-starved and the semantics are shallow. PartCrafter states the ceiling plainly: 50K part-annotated training assets versus millions for whole objects. Nano3D's contribution is telling — it builds the first 100K-pair 3D *editing* dataset by using a training-free pipeline, precisely because no such dataset existed. And the parts produced are geometric, not functional: DreamPartGen's framing is that prior part-aware work "remain[s] largely geometry-focused, lacking semantic grounding."

---

## 7. Articulated and riggable generation

Three distinct problems get conflated here; keep them apart.

**(a) Auto-rigging a character-like mesh** — predict a skeleton and per-vertex skinning weights. RigNet (2020) → MagicArticulate / RigAnything / UniRig (early 2025) → Puppeteer (NeurIPS 2025 Spotlight). Autoregressive skeleton tokenization won decisively; the differentiators are now tokenization scheme (Skeleton Tree Tokenization vs joint-based with hierarchical randomization), skinning architecture (bone-point cross-attention vs topology-aware joint attention over graph distance), and **pose diversity in the training data**. Puppeteer's ablation is the useful evidence: MagicArticulate matches it on rest-pose benchmarks but "degrades on the diverse-pose subset, since it was trained only on predominantly rest-pose data."

**(b) Articulated *objects* with kinematic joints** (drawers, laptops, doors) — the embodied-AI-facing problem, where the deliverable is a **URDF** (the XML robotics format specifying links, joints, axes, and limits) that imports into Isaac Sim or MuJoCo. Articulate-Anything (ICLR 2025) does it by VLM program synthesis with an actor–critic repair loop. URDF-Anything (NeurIPS 2025) uses a 3D MLLM with a `[SEG]` token that jointly optimizes segmentation and kinematic parameters (+17% mIoU, −29% joint parameter error, +50% physical executability over baselines). URDF-Anything+ (Mar 2026) replaces the multi-stage pipeline with end-to-end autoregressive diffusion in a structured latent space, emitting part geometry and joint parameters part-by-part with a termination token, and demonstrates a **"Real-Follow-Sim"** loop where policies trained purely in simulation on the generated twin transfer without online adaptation. SPARK (CVPR 2026) adds differentiable forward kinematics optimized under VLM-generated open-state supervision.

**(c) Generating geometry and rig jointly** — the 2026 frontier, and the right framing. Rigel3D learns coupled surface and skeleton SLats decoded into mesh + skeleton topology + joint coordinates + skinning weights, plus open-vocabulary joint labels embedded in a vision-language space so the rig can be retargeted to arbitrary animation templates. AniGen (SIGGRAPH 2026) represents shape, skeleton, and skin as three mutually consistent fields over a shared spatial domain, with a confidence-decaying skeleton field to handle the geometric ambiguity of bone position at Voronoi boundaries and a dual skin feature field decoupling skinning from a fixed joint count. TokenRig/SkinTokens goes the other way — an FSQ-CVAE compresses skinning weights into a discrete vocabulary so a Qwen3-0.6B can emit skeleton and skin as one interleaved token sequence, GRPO-refined against volumetric joint coverage, bone-mesh containment, skinning sparsity, and deformation smoothness.

**Datasets are the real story here.** Articulation-XL (33K) → Rig-XL (14K) → Anymate (230K+) → Articulation-XL2.0 (59.4K with 11.4K diverse poses). Two years ago there was essentially nothing.

---

## 8. CAD and parametric generation

CAD is a separate world because the target is exact: a **B-rep** with analytic surfaces, or the **program** that constructs one. Chamfer distance on a sampled mesh is a poor proxy for either.

**Three approaches, and which is winning.**

1. **Command-sequence transformers** (DeepCAD, SkexGen, Text2CAD, CAD-MLLM). Restricted to sketch-and-extrude; the vocabulary cannot express fillets or chamfers without B-rep entity references.
2. **Direct B-rep generation** (BrepGen → HoLa → BrepForge → DualBrep → ParaCAD). The difficulty is that a B-rep couples discrete topology to continuous geometry. BrepGen diffuses over a structured latent tree with deduplication; HoLa derives everything from one holistic per-solid latent to avoid inter-primitive inconsistency; BrepForge (2026) factorizes on the observation that "wireframe composition involves high-entropy structural decisions" while surface interiors do not; DualBrep (2026) encodes geometry as an SDF and topology as a UDF whose Voronoi partition implicitly segments faces, so flow matching samples both jointly and avoids the error accumulation of sequential predictors; ParaCAD (SIGGRAPH 2026) tokenizes each face by its **exact surface type and continuous parameters** and recovers edges/vertices by global intersection — the first to keep native parametric surfaces end-to-end.
3. **Code generation** (CAD-Recode → cadrille → CADEvolve → CADENA). Currently the strongest on reconstruction benchmarks, and the most useful, because a CadQuery program is editable, diffable, and LLM-inspectable.

**The 2026 story is the loop and the benchmark.** cadrille was first with online RL. CADENA (Aug 2026) makes execution part of generation: emit one operation, execute it, render target-vs-build as a red/green channel overlay from eight canonical viewpoints, decide the next operation from the residual. RL rewards volumetric IoU on the executed geometry, which "requires the target mesh and nothing else, in particular no ground-truth program," and makes validity part of the objective rather than a post-hoc filter.

**CADENA's benchmark critique is the most important CAD result of the year** and applies well beyond CAD (see §11).

---

## 9. Evaluation: how it's done, and what's documented to be wrong with it

**What people actually report.** Geometry: Chamfer Distance, Earth Mover's Distance, F-Score@τ, Normal Consistency; distribution-level Coverage / MMD / 1-NNA. Appearance: PSNR / SSIM / LPIPS against renders, FID / KID, CLIP Score and CLIP R-Precision. Semantic alignment: ULIP-2 and Uni3D embeddings for text–shape similarity. Automatic holistic: GPTEval3D (GPT-4V pairwise → Elo). Human: A/B preference, Likert dimensions, and now 3D Arena's open leaderboard.

**Documented problems, with attribution.**

*GPTEval3D's own authors list four.* "GPT-4V's responses are not always true… hallucinations"; "systematic errors, such as bias toward certain image positions"; "a good metric should be 'un-gamable'. However one could potentially construct adversarial patterns to attack GPT-4V… one might gain a high score without needing to produce high-quality 3D assets"; and cost — "a quadratically growing number of comparisons." (arXiv 2401.04092, Limitations.)

*Human preference measures the wrong thing — and this is measured, not asserted.* 3D Arena, across 123,243 votes: Gaussian splats beat meshes by **16.6 ELO**, and textured beats untextured by **144.1 ELO** (56.9% vs 32.4% win rate, p < 1.4×10⁻¹⁰⁴). The cleanest datum is that TRELLIS versus TRELLIS-3DGS — *the same model*, different render format — differ by **78 ELO**. Their diagnosis: "voting patterns systematically favor visual impact through vibrant rendering and aesthetic appeal over downstream utility," which they attribute to surface features being processed in 150–200ms while geometry requires deliberate analysis. Also notable: the lowest-polygon-count bracket (dominated by a topology-aware InstantMesh+MeshAnything hybrid) sits at the *bottom* of the leaderboard at 1016 ELO — the arena actively penalizes clean topology.

*Object-level scoring hides part-level failure.* Hi3DEval (NeurIPS 2025 D&B): existing methods "rely on image-based metrics and operate solely at the object level, limiting their ability to capture spatial coherence, material authenticity, and high-fidelity local details." Their part-level protocol exists specifically to "isolate and assess localized imperfections, such as collapses, distortions, or multi-faces, that are often obscured in holistic object-level evaluations."

*No metric measures deployability at all.* From 2604.23629 §3.4 and §7.2: "Common metrics borrowed from image generation (FID, CLIP similarity) assess appearance fidelity but are silent on topology validity, UV-seam density, PBR accuracy, and engine import success rate." And: "no widely adopted standardized metric yet exists for material separation quality"; "a comprehensive UV benchmark against production standards does not yet exist"; "no standardized topology evaluation suite exists."

*Benchmark saturation and survivorship bias (CAD, but generalizable).* CADENA: "On DeepCAD every method lands within a few points of every other: CD between 0.15 and 0.18, IoU between 89.7 and 96.1, GMS between 89.8 and 97.0. The ranking is decided by differences comparable to the noise of the evaluation itself." Moving to real mechanical parts costs every learned method roughly half its score (cadrille 94.8 → 49.8; CAD-Recode 92.9 → 48.9; CADEvolve 95.3 → 52.9). They also show a **protocol effect larger than most method effects**: cadrille's MCB IoU moves from 47.6 to 67.0 purely by changing which parts the average is taken over. And on metric validity: "IoU scores how far the prediction fills the target's volume and Chamfer distance how close the sampled points lie, and a reconstruction assembled from the wrong primitives satisfies either."

---

## 10. Technical trends: what converged, and the open/closed landscape

### What converged (high confidence)

1. **Flow matching / rectified flow over a learned 3D latent, conditioned on a single image, with DINO-family image features.** Universal across Hunyuan3D, TripoSG, TRELLIS(.2), Direct3D-S2, Step1X-3D, Seed3D, Sparc3D.
2. **Two stages: geometry then material.** Preserved even by TRELLIS.2, whose third stage doubles as a standalone "mesh + reference image → PBR texture" model. UniLat3D, Native3D, and EVA01 dissent.
3. **Sparse everything.** Sparse voxels, sparse convolution, sparse attention, per-voxel latents. Dense volumetric is gone. The efficiency payoff is concrete (Direct3D-S2: 1024³ on 8 GPUs).
4. **Text-to-3D is a wrapper.** Nearly all "text-to-3D" is text-to-image-to-3D. Native text-conditioned 3D exists but is not competitive.
5. **PBR replaced baked RGB** as the appearance target, ~mid-2025.
6. **Autoregressive sequence modeling owns mesh topology**, and **RL/preference optimization** is now standard on top of it (DeepMesh, Mesh-RFT, QuadGPT, Mesh-Pro, and in rigging, TokenRig).
7. **Everything downstream is being rebuilt on TRELLIS-style structured latents**, because spatial indexing is what part-level control, editing, and rig coupling require. OmniPart, Nano3D, Rigel3D, URDF-Anything+ are all built this way.
8. **LLMs/MLLMs are the interface for symbolic outputs** — CAD programs (CAD-Recode, cadrille), URDFs (URDF-Anything), skinning tokens (TokenRig), scene scripts (SceneCraft, UnrealLLM).

### Open-weight vs closed-weight

Broadly, **the open frontier caught up and then partially inverted the usual pattern**, but with an important qualifier.

*Genuinely open, current, and strong:* **TRELLIS.2-4B** (MIT, weights + inference + training code + dataset, 512³–1536³ PBR), **Hunyuan3D 2.1** (weights + training code, Tencent community license), **Step1X-3D**, **TripoSG**, **Direct3D-S2**, **Sparc3D**, **TRELLIS**, **TripoSR/SF3D**, plus most of the part-level, rigging, and CAD research code (PartCrafter, OmniPart, HoloPart, PartField, UniRig, Puppeteer, CAD-Recode, MatLat).

*Closed:* **CLAY** (productized as Rodin), **Hunyuan3D 2.5 / 3.0 / 3.1** (API), **Seed3D 1.0** (ByteDance, API only despite a public technical report), **Meshtron** ⚠, **Tripo / Meshy / Rodin / CSM / Kaedim** as products.

*The trend the survey flags, and I would take seriously:* 2604.23629 §8.4 — "Leading systems increasingly migrate from open-weight releases to proprietary APIs — Hunyuan3D exemplifies this pattern — making independent replication and fair comparison progressively harder." That is exactly the Hunyuan3D trajectory: 1.0 open, 2.0 open, 2.1 fully open with training code, 2.5/3.0/3.1 API-only. Microsoft cutting against this with an MIT-licensed 4B model is the notable counterexample of the period, though note the TRELLIS.2 project page states the materials are "not intended for commercial exploitation," which sits awkwardly with the MIT license on the GitHub and HF cards — worth checking before commercial use.

*Where closed still wins:* the 2604.23629 survey's own qualitative comparison (Figure 4) concludes that "closed-source systems generally achieve superior geometric fidelity and surface detail over their open-source counterparts," comparing Rodin Gen 1.5, Tripo V2.5, Hunyuan3D 3.1, and Meshy 5 against TRELLIS.2, SF3D, InstantMesh, and TripoSG. Treat that as one team's visual judgment, not a measurement.

---

## 11. Explicitly stated open problems and failure modes (attributed)

**On the assetization bottleneck — the field's central practical problem.**
> "Even when a method produces a visually compelling shape, the gap between its output and a deployable asset encompasses retopology, UV unwrapping, PBR separation, rigging, LOD creation, and collision fitting… collision mesh generation, multi-level LOD hierarchies, and physics parameter assignment remain outside the scope of any current automated system and still require manual authoring."
> — *Production-Ready 3D Asset Generation*, arXiv 2604.23629, §8.1

**On data — no dataset has everything.**
> "To our knowledge, no existing dataset simultaneously provides manifold topology, artist-quality UV layouts, complete PBR channels, skeletal rigs, and natural-language descriptions."
> And: "Only GSO, ABO, ScanNet++, Hypersim, and Infinigen Indoors provide complete PBR channels… Their combined scale falls far short of Objaverse-XL's 10.2M instances, preventing PBR estimation and geometry generation from being co-trained at internet scale."
> Also: "No large-scale 3D dataset provides explicit annotations for production topology quality — face-aligned edge loops, quad-dominant structure, or density gradients. Without ground-truth topology labels, neither supervised retopology nor evaluation metrics for topological fidelity can be established."
> — arXiv 2604.23629, §8.1 and §3.4

**On evaluation being silent on everything that matters for deployment.**
> "Common metrics borrowed from image generation (FID, CLIP similarity) assess appearance fidelity but are silent on topology validity, UV-seam density, PBR accuracy, and engine import success rate."
> And: "Professional evaluation by technical artists — judging topology, edge flow, UV layout, material correctness, and estimated cleanup time — would be the most informative signal for production readiness but remains largely absent from the literature."
> — arXiv 2604.23629, §3.4 and §7.3

**On human preference actively rewarding the wrong thing.**
> "Voting patterns systematically favor visual impact through vibrant rendering and aesthetic appeal over downstream utility. This disconnect manifests clearly in the systematic advantages of splats over meshes (16.6 ELO points) and textured over untextured models (144.1 ELO points), despite widespread industry recognition that clean mesh topology is essential for professional workflows."
> — 3D Arena, arXiv 2506.18787, §4.3

**On automatic VLM judges.**
> "GPT-4V's responses are not always true… GPT-4V can also process some systematic errors, such as bias toward certain image positions… a good metric should be 'un-gamable'. However one could potentially construct adversarial patterns to attack GPT-4V. This way one might gain a high score without needing to produce high-quality 3D assets."
> — GPTEval3D, arXiv 2401.04092, Limitations

**On benchmarks being saturated and unrepresentative (CAD, generalizable).**
> "On DeepCAD every method lands within a few points of every other… The ranking is decided by differences comparable to the noise of the evaluation itself, so a method that reconstructs mechanical parts poorly is nearly indistinguishable from one that does it well."
> And: "Test sets built from sketch–extrude corpora understate the difficulty of the task they are taken to measure."
> — CADENA, arXiv 2608.00799, §5

**On autoregressive mesh generation's structural failure.**
> "Autoregressive mesh generators serialize faces into sequences and train on truncated segments with sliding-window inference to cope with memory limits. However, this mismatch breaks long-range geometric dependencies, producing holes and fragmented components."
> — MeshRipple, CVPR 2026

**On quad tokenization being subtly broken.**
> "Prior methods declare the face type (triangle or quad) with a leading special token and preclude a truly consistent canonical ordering, resulting in geometric artifacts and structural defects."
> — Mesh-Pro, arXiv 2603.00526

**On part-level data scarcity.**
> "PartCrafter is trained on 50K part-level data, which is relatively small compared to that used to train 3D object generation models (typically millions)."
> — PartCrafter, arXiv 2506.05573, Limitations

**On rigging quality under motion.**
> "Generated rigs may still contain missing joints, spurious branches, or imperfect connectivity. Because skinning is not supervised by long motion sequences, deformations can be less natural under extreme poses or specific animation intents. Open-vocabulary labels may be ambiguous for repeated structures such as fingers, tails, or decorative appendages."
> — Rigel3D, arXiv 2605.13129, §6

**On rig evaluation.**
> "RigNet introduced a basic evaluation protocol, but it covers only a narrow range of skeleton types and does not assess deformation under motion."
> — arXiv 2604.23629, §7.2

**On the resolution floor of voxel latents.**
> "O-Voxel's representation power is bounded by its spatial resolution. For detailed geometric features smaller than the voxel size, the Flexible Dual Grid formulation could produce aliasing artifacts. For example, when two parallel surfaces that are very close to each other intersect the same voxel, the QEF solver, by design, will place the dual vertex at a position that minimizes the error to both surfaces… the volumetric material attributes in such a voxel will be an average of the properties of both surfaces, leading to blurred appearance."
> — TRELLIS.2, arXiv 2512.14692, Appendix F

**On editing being bounded by the base model.**
> "It supports only localized edits; the VAE in TRELLIS introduces reconstruction loss; and the overall performance is constrained by TRELLIS's generative capacity."
> — Nano3D, arXiv 2510.15019

**On physics being entirely absent.**
> "Most 3D generation methods optimize exclusively for visual plausibility, producing shapes and scenes that carry no information about mass, friction, elasticity, or articulation constraints… current methods address placement-level physics rather than intrinsic material properties."
> — arXiv 2604.23629, §8.2

**On material expressivity being data-limited, not model-limited.**
> "Generative models for material creation are fundamentally limited by the quality and expressivity of available training data. Simple PBR materials, which combine a diffuse term with a single-lobe specular component, are commonly used for training but are insufficient to capture many important visual effects present in real materials."
> — *Toward Richer Material Generation*, arXiv 2606.14988

---

## 12. Under-explored gaps: what an 8–32 GPU academic lab can actually do

You cannot beat Tencent, ByteDance, or MSRA at training a 4–10B image-to-3D foundation model on a proprietary curated corpus. Do not try. Every direction below is chosen because it is bottlenecked by *ideas, careful experiment design, or data curation* rather than by scale, and because at least one 2025–2026 primary source explicitly names it as open.

### 12.1 The highest-leverage gap: a production-readiness benchmark

This is the field's loudest unmet need and it is a pure engineering + curation project requiring almost no training compute. 2604.23629 asks for it in its conclusion; §3.4 and §7.2 enumerate exactly what is missing: **topology validity** (manifoldness, watertightness, genus correctness, quad ratio, edge-flow alignment), **UV quality** (stretch/angular distortion, seam length and visibility, chart packing efficiency, overlap detection), **PBR separation quality** (relighting consistency — currently only paper-specific protocols from Paint3D and MaterialMVP), **rig robustness** (skinning smoothness, self-intersection under extreme poses, joint-limit violations, retargeting success), and **engine import success** (does it load into Unreal/Unity/Blender and render at interactive rates without manual repair; triangle budget compliance; draw calls; collision mesh validity).

Every one of these is computable with existing open tooling (trimesh, xatlas, Blender's Python API, headless UE5/Unity import). Run it over the ~19 models already on 3D Arena plus the open checkpoints (TRELLIS.2, Hunyuan3D 2.1, TripoSG, Step1X-3D, Direct3D-S2, Sparc3D). The predictable and publishable result: the ranking will not match the ELO ranking. 3D Arena's own data already hints at it — their lowest-polygon, topology-aware entry sits *last*.

**Sharpen it with the two methodological findings nobody has generalized.** (a) 3D Arena proposes but does not implement a **separate topology ELO** collected from wireframe-only views with polygon counts; running that experiment is cheap and would directly measure the aesthetic-usability confound. (b) CADENA shows that averaging convention alone can move a published number from 47.6 to 67.0 IoU. Nobody has audited protocol sensitivity in general 3D generation, and I would bet it is at least as large there.

### 12.2 Metrics that survive the "wrong primitives" test

CADENA's critique is deeper than CAD: *"a reconstruction assembled from the wrong primitives satisfies either [IoU or Chamfer] — a cylinder approximated by an extruded polygon, or in the limit a shape packed with voxels, scores well while the program that produced it is wrong."* Their answer, **GMS**, matches points by **normal as well as position**, is scale-normalized, sampling-density tolerant, and defined on open non-watertight geometry — so no part gets dropped as unmeasurable. Nobody has ported this idea to general mesh generation, where the analogous failure (a marching-cubes blob that scores well on Chamfer despite terrible topology) is endemic. A normal-aware, curvature-aware, open-geometry-valid geometric metric for general 3D generation, validated against technical-artist judgments, is a well-scoped contribution.

Also note CADENA's negative result, which is worth replicating rather than assuming: they tried GMS as an RL reward and *"the outcome was self-defeating: the resulting policy improved on GMS while leaving Chamfer distance and IoU no better, and markedly worse on the MCB subsample."* Reward-hacking of geometric metrics is real and undocumented.

### 12.3 Part-level and articulation data — the acknowledged hole

PartCrafter trains on 50K part-annotated assets against millions for whole objects; the survey says no dataset jointly provides topology, UV, PBR, rig, and language. This is a **data contribution opportunity** that needs curation and CPU time, not a GPU cluster. Two specific angles:

- **Mine parts and rigs from game/engine sources**, which the survey explicitly names as an under-used strategy ("annotation harvesting from game engines"). Articulation-XL2.0's value came precisely from adding pose diversity, not raw count.
- **Build the counterfactual/paired data that supervision needs.** Nano3D showed the way for editing: use a *training-free* pipeline (FlowEdit inside TRELLIS) to manufacture 100K paired before/after 3D examples, then train a feed-forward model on them. The same bootstrap is unexploited for part decomposition, retopology (dense mesh → artist mesh pairs), and UV (software UV → artist UV pairs — ArtUV formulates its task exactly as "learning the discrepancy between traditional software-generated UV maps and artist-optimized UV maps", which is a supervised problem waiting for a public dataset).

### 12.4 Topology-quality supervision and learned retopology

Direct quote from 2604.23629 §3.4: "Without ground-truth topology labels, neither supervised retopology nor evaluation metrics for topological fidelity can be established." That is a solvable data problem, not a scale problem. Curate a corpus of artist meshes with derived topology-quality labels (quad ratio, edge-loop alignment to curvature principal directions, density-gradient smoothness, pole counts), then either (a) train a topology *critic* usable as an RL reward — which is the missing piece in DeepMesh / Mesh-RFT / Mesh-Pro, all of which currently use hand-designed or preference-derived rewards — or (b) train supervised retopology directly. A learned topology critic that correlates with technical-artist judgment would be immediately adopted by every RL mesh paper.

### 12.5 Diffusion-based topology generation (the minority branch)

Autoregressive dominates because, per the survey, "corrupting and denoising irregular polygonal connectivity without destroying mesh validity is considerably harder than prefix-conditioned sequence prediction." But SpaceMesh's continuous halfedge latents with a Sinkhorn-normalized connectivity network are reported "orders of magnitude faster than autoregressive methods," and MeshCraft's flow-DiT over face tokens gives parallel decoding with explicit face-count control. Three papers in three years versus dozens on the AR side. This is an under-populated branch with a clear efficiency argument, tractable at 8–32 GPUs (MeshMosaic reached 100K triangles with a **0.5B** model — this subfield is not compute-bound).

### 12.6 The geometry/texture split: is two-stage actually necessary?

Everyone does geometry-then-texture. UniLat3D, Native3D (at scene level), and EVA01 all argue the split causes misalignment; TRELLIS.2 keeps the split but stores material in the same latent structure. Nobody has run the controlled experiment at object level: same latent, same data, same parameter budget, single-stage joint generation versus cascaded. This is a 32-GPU ablation on an open backbone (TRELLIS.2 released training code) and it settles an architectural question the whole field has assumed rather than tested.

### 12.7 Editing and iterative refinement as a first-class objective

2604.23629 §8.1: "a fully interactive generation loop that maintains a persistent, editable asset state remains open." Nano3D is training-free and admits it "supports only localized edits" and is "constrained by TRELLIS's generative capacity." EVA01's pitch is that diffusion LRMs are "stateless reconstructors." With Nano3D-Edit-100k now public, training a **feed-forward** instruction-conditioned 3D editor on a TRELLIS-scale backbone is directly tractable — Nano3D's authors say so explicitly ("laying the groundwork for the development of feed-forward 3D editing models"). The genuinely open part is *multi-turn* editing with identity preservation across turns, where no benchmark exists.

### 12.8 Material expressivity and PBR data

The SIGGRAPH 2026 procedural-uplift paper builds its neural-material dataset with **22 hours on 80 L40/L40S** — inside a large academic budget — and it is a *data* method. Extending procedural uplift to other under-modeled effects (anisotropy, subsurface scattering, thin-film, fabric sheen), or applying MatLat's key insight (fine-tune the *VAE* with residual prediction + KL alignment to the RGB prior rather than freezing it) to other auxiliary-channel problems, are both small-compute, high-clarity contributions. MatLat's ablation showing that *both* distributional alignment and locality preservation are individually necessary is the kind of result that generalizes.

### 12.9 Physics parameters as a generation output

Named as open by 2604.23629 §8.2: mass, friction, restitution, articulation limits, collision geometry are assigned manually or not at all. URDF-Anything+ produces joints but the paper's framing is kinematics, not dynamics. There is no dataset of generated assets with physical properties and no metric for "was the predicted mass plausible." Building one using Infinigen Indoors / ScanNet++ / ABO material labels plus a differentiable or sampling-based sim check (does the object rest stably, does the drawer slide) is a well-defined project that connects directly to the embodied-AI demand that Seed3D and URDF-Anything+ are chasing.

### 12.10 Things to avoid

- **Training a new foundation-scale image-to-3D model.** You will lose to a 4B MIT-licensed model that already exists.
- **Reporting Chamfer/F-Score on Toys4K or GSO as your headline.** Saturated and, per CADENA's parallel finding in CAD, decided by evaluation noise.
- **Reporting a GPTEval3D Elo without acknowledging the four documented failure modes** its own authors list.
- **Optimizing for human A/B preference as your primary target.** 3D Arena measured that this rewards splats and saturated textures over clean geometry, with a 78-ELO gap between two renderings of the *same model*.
- **Assuming a DeepCAD/Fusion360 win transfers.** Every learned method loses roughly half its score on real mechanical parts.

---

## 13. Verification caveats

Items I could not confirm from a primary source in this session, listed so they can be checked before citation.

**Incorrect arXiv IDs in the 2604.23629 survey.** I checked every ID I reused. Three in that survey's reference list resolve to unrelated papers:
- **ArtUV** is cited as arXiv 2504.09914 → that ID is *"Improving Multimodal Hateful Meme Detection…"*. The correct ID is **2509.20710**.
- **PartUV** is cited as arXiv 2506.04173 → that ID is a synthetic-MRI paper. Correct ID **not found**; I have therefore mentioned PartUV only in passing and without an ID.
- **SeamCrafter** is cited as arXiv 2504.12256 → that ID is *"FLIP Reasoning Challenge"*. The correct ID is **2509.20725**.
Treat other IDs sourced from that survey's bibliography with corresponding suspicion.

**Venues I did not independently re-verify** (taken from the surveys, project pages, or general knowledge, not from a proceedings listing this session): DreamFusion ICLR 2023, Magic3D CVPR 2023, ProlificDreamer NeurIPS 2023, CRM ECCV 2024, Michelangelo NeurIPS 2023, TEXTure SIGGRAPH 2023, SyncMVD SIGGRAPH Asia 2023, Articulate-Anything ICLR 2025, Anymate SIGGRAPH 2025, DeepCAD ICCV 2021, BrepGen SIGGRAPH 2024, Text2CAD NeurIPS 2024, HoLa SIGGRAPH 2025, ULIP-2 CVPR 2024, Uni3D ICLR 2024, Meshtron's venue (an OpenReview ID exists, `mhzDv7UAMu`, but I did not confirm acceptance), LLaMA-Mesh's venue.

**TRELLIS.2's venue is ambiguous.** arXiv metadata and the project page call it a tech report (`journal={Tech report}`); the 2604.23629 survey cites it as `CoRR abs/2512.14692`; a third-party paper-notes site files it under CVPR 2026. I have not seen it on a CVF listing. Do not cite it as CVPR 2026 without checking.

**TRELLIS.2's licensing is internally inconsistent.** GitHub and the Hugging Face model card both state MIT for code and weights; the project page states the materials "are not intended for commercial exploitation or use." Resolve this before any commercial deployment.

**Hunyuan3D 3.0 and 3.1 have no technical report I could find.** All claims about 3.0 (1536³ geometric resolution, 3.6 billion voxels, "3× accuracy improvement," 3D-DiT hierarchical sculpting, September 2025 announcement) come from secondary blog coverage of a Tencent summit, not from a paper or an official technical document. I have kept them out of the main tables except as products. Hunyuan3D 3.1 appears in the 2604.23629 survey only as a cited *URL* to a cloud service, not a paper.

**Venues asserted from project pages rather than arXiv metadata or a proceedings listing:** MatLat (CVPR 2026 Highlight) and ParaCAD (SIGGRAPH 2026). Mesh-Pro's CVPR 2026 acceptance *is* in its arXiv comment field, and CADENA's code, weights, and benchmark links *are* in its arXiv comment field, so those two are solid.

**Weights columns marked "?" or "⚠"** are cases where I did not open the repository or Hugging Face page during this session. Specifically unverified: Meshtron (I believe closed, but did not confirm), QuadGPT, MeshRipple, Mesh-Pro, VideoMatGen, DreamPartGen, Rigel3D, URDF-Anything(+), SPARK, BrepForge, DualBrep, PartGen. AniGen and TokenRig/SkinTokens have public GitHub repos but I did not verify that checkpoints are actually downloadable (SkinTokens lists Hugging Face download links; AniGen's release state is unclear).

**Face-count figures** are as-claimed by each paper, not independently reproduced, and the definitions differ (max generated versus max reliably generated versus training cap). MeshGPT's ~800-face figure in §4 is my characterization from the MeshAnything comparison rather than a number MeshGPT states that way — treat as approximate.

**3D Arena's leaderboard is a moving target.** The ELO figures quoted are from the June 2025 paper snapshot (19 models, 123,243 votes). The live leaderboard will have changed; notably it did not include TRELLIS.2 at the time of that snapshot.

**Third-party product comparisons** (relative quality of Tripo vs Meshy vs Rodin, per-asset pricing, the 3,000–5,000 assets/month self-hosting break-even) come from commercial review blogs, not from measurement. I have used them only for the shape of the market, not for any quantitative claim.

**Anymate's dataset size (230K+)** is from secondary description; I verified the paper's existence and title (arXiv 2505.06227) but not the exact scale figure.

**The CADENA related-work citations to 2026 CAD papers** (CADEvolve, CADReasoner, CADFit, IterCAD, HistCAD, Pointer-CAD, CADFS, Zero-to-CAD, BenchCAD) are reported *from CADENA's text*. I did not resolve their arXiv IDs, and I have deliberately not assigned IDs to them anywhere in this report.
