# 3D Reconstruction & Neural Rendering
### Literature survey, 2023 – August 2026 (emphasis on 2025 – mid-2026)
*Compiled 5 August 2026. Claims are traced to primary pages / OA PDFs / GitHub where possible; uncertain items are flagged with ⚠. arXiv IDs appear only when verified from an official page or arXiv listing in this session.*

---

## 0. How to read this report

Three shifts organize the period:

1. **Representation:** NeRF-quality novel view synthesis moved from volumetric MLPs to **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023). By 2025 industry tooling (DCC, GIS, game engines) treated splats as a first-class imaging medium; research volume exploded (Kerbl’s own 2025 outlook notes the follow-on literature is already too large for one complete review).
2. **Geometry prior:** Classical SfM/MVS was joined—and often *initialized*—by **feed-forward pointmap / visual-geometry transformers** (DUSt3R → MASt3R → Spann3R / CUT3R / Fast3R → **VGGT**, CVPR 2025 Best Paper → **Depth Anything 3**, arXiv Nov 2025 / ICLR 2026 academic paper). Pose-free and sparse-view reconstruction became default research settings.
3. **Geometry vs appearance:** Photorealistic rendering (3DGS) and accurate surfaces (NeuS / Neuralangelo / 2DGS / SuGaR / Gaussian Surfels) remain partially decoupled. Mid-2026 practice is hybrid: foundation geometry → Gaussian or mesh refinement.

**Anchor surveys / maps used this session**
- Kerbl, *The Impact and Outlook of 3D Gaussian Splatting* — [arXiv 2510.26694](https://arxiv.org/abs/2510.26694) (Oct 2025 outlook).
- Community list: [ruili3/awesome-dust3r](https://github.com/ruili3/awesome-dust3r) (DUSt3R lineage; useful for 2025 names, not a peer-reviewed taxonomy).
- [3D-Vision-World/awesome-NeRF-and-3DGS-SLAM](https://github.com/3D-Vision-World/awesome-NeRF-and-3DGS-SLAM) (SLAM index).

---

## 1. Taxonomy table (~30 key papers)

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 1 | **Instant-NGP** (Müller et al.) | **SIGGRAPH / TOG**, Jul 2022 | Multiresolution hash encoding + tiny MLP; seconds-scale NeRF training; substrate for many later systems. | [project](https://nvlabs.github.io/instant-ngp) · [code](https://github.com/NVlabs/instant-ngp) |
| 2 | **Mip-NeRF 360** (Barron et al.) | **CVPR 2022** | Unbounded anti-aliased NeRF; quality baseline that 3DGS papers still compare to. | [project](https://jonbarron.info/mipnerf360/) |
| 3 | **Zip-NeRF** (Barron et al.) | **ICCV 2023** | Anti-aliased *grid-based* NeRF; closes much of the quality gap vs Mip-NeRF 360 at Instant-NGP-like speed. | [project](https://jonbarron.info/zipnerf/) · [arXiv 2304.06706](https://arxiv.org/abs/2304.06706) |
| 4 | **3D Gaussian Splatting** (Kerbl et al.) | **SIGGRAPH / TOG**, Jul 2023 | Anisotropic 3D Gaussians + interleaved densification + tile rasterizer; real-time 1080p NVS with NeRF-level quality. | [project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |
| 5 | **NeuS** (Wang et al.) | **NeurIPS 2021** | Volume rendering of neural SDFs; founding neural surface line. | [project](https://lingjie0206.github.io/papers/NeuS/) |
| 6 | **VolSDF** (Yariv et al.) | **ICLR 2021** | Hybrid volume–surface density from SDF; base for BakedSDF. | [OpenReview](https://openreview.net/forum?id=KGzYcEwS_GM) |
| 7 | **Neuralangelo** (Li et al.) | **CVPR 2023** | Instant-NGP hash grids + numerical gradients + progressive LOD for high-fidelity multi-view surfaces. | [arXiv 2306.03092](https://arxiv.org/abs/2306.03092) |
| 8 | **BakedSDF** (Yariv et al.) | **SIGGRAPH 2023** | Bake neural SDF of unbounded scenes to meshes + spherical Gaussians for real-time view synthesis. | [arXiv 2302.14859](https://arxiv.org/abs/2302.14859) |
| 9 | **FlexiCubes** (Shen et al.) | **SIGGRAPH / TOG**, Jul 2023 | Differentiable isosurface extraction with extra local DOFs for gradient-based mesh optimization. | [project](https://research.nvidia.com/labs/toronto-ai/flexicubes/) |
| 10 | **GLOMAP** (Pan et al.) | **ECCV 2024** | Global SfM with joint camera+point positioning; COLMAP-level accuracy, orders-of-magnitude faster; later folded into COLMAP 4.0 as `global_mapper`. | [project](https://lpanaf.github.io/eccv24_glomap/) · [arXiv 2407.20219](https://arxiv.org/abs/2407.20219) · [COLMAP 4.0](https://github.com/colmap/colmap/releases/tag/4.0.0) |
| 11 | **Depth Anything** (Yang et al.) | **CVPR 2024** | Relative MDE foundation model via 1.5M labeled + 62M+ unlabeled images. | [arXiv 2401.10891](https://arxiv.org/abs/2401.10891) · [code](https://github.com/LiheYoung/Depth-Anything) |
| 12 | **Depth Anything V2** (Yang et al.) | **NeurIPS 2024** | Synthetic-teacher → massive real pseudo-labels; sharper relative depth; metric fine-tunes. | [project](https://depth-anything-v2.github.io/) · [arXiv 2406.09414](https://arxiv.org/abs/2406.09414) |
| 13 | **Metric3D / Metric3D v2** | **ICCV 2023** / **TPAMI 2024** | Canonical-camera transform for zero-shot *metric* depth (+ normals in v2); 16M-image training scale claimed for v2. | [v1 ICCV](https://openaccess.thecvf.com/content/ICCV2023/html/Yin_Metric3D_Towards_Zero-shot_Metric_3D_Prediction_from_A_Single_Image_ICCV_2023_paper.html) · [v2 arXiv 2404.15506](https://arxiv.org/abs/2404.15506) |
| 14 | **UniDepth** (Piccinelli et al.) | **CVPR 2024** (Highlight); **UniDepthV2** arXiv Feb 2025 | Universal monocular *metric* depth without test-time camera info; V2 simplifies design + edge-guided loss. | [code](https://github.com/lpiccinelli-eth/unidepth) · [V1 arXiv 2403.18913](https://arxiv.org/abs/2403.18913) · [V2 arXiv 2502.20110](https://arxiv.org/abs/2502.20110) |
| 15 | **DUSt3R** (Wang et al.) | **CVPR 2024** | Pairwise *pointmap* regression without known poses/intrinsics; unifies mono/multi-view geometry + global alignment. | [OA PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf) · [code](https://github.com/naver/dust3r) |
| 16 | **MASt3R** (Leroy et al.) | **ECCV 2024** | DUSt3R + dense local features + fast reciprocal matching; strong Map-free localization. | [arXiv 2406.09756](https://arxiv.org/abs/2406.09756) · [code](https://github.com/naver/mast3r) |
| 17 | **MASt3R-SfM** (Duisterhof et al.) | **3DV 2025** | Fully integrated unconstrained SfM on MASt3R matches. | [OpenReview](https://openreview.net/forum?id=5uw1GRBFoT) |
| 18 | **Spann3R** (Wang & Agapito) | **3DV 2025** | Spatial memory → global pointmaps without pairwise global alignment. | [arXiv 2408.16061](https://arxiv.org/abs/2408.16061) · [project](https://hengyiwang.github.io/projects/spanner) |
| 19 | **CUT3R** (Wang et al.) | **CVPR 2025 Oral** | Recurrent stateful transformer; online / streaming metric pointmaps. | [arXiv 2501.12387](https://arxiv.org/abs/2501.12387) · [project](https://cut3r.github.io/) |
| 20 | **Fast3R** (Yang et al., Meta FAIR) | **CVPR 2025** | Many-view (1000+) reconstruction in one forward pass; drops iterative global alignment. | [arXiv 2501.13928](https://arxiv.org/abs/2501.13928) · [project](https://fast3r-3d.github.io/) · [code](https://github.com/facebookresearch/fast3r) |
| 21 | **VGGT** (Wang et al.) | **CVPR 2025 Best Paper** | Single feed-forward net: cameras, depth, pointmaps, tracks from 1–hundreds of views; under ~1 s; beats many optimize-then-align pipelines. | [project](https://vgg-t.github.io/) · [OA](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) · [arXiv 2503.11651](https://arxiv.org/abs/2503.11651) |
| 22 | **MoGe / MoGe-2** | **CVPR 2025 Oral** / arXiv Jul 2025 | Open-domain monocular pointmaps (affine-invariant → metric + normals + sharpness in MoGe-2). | [OA MoGe](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_CVPR_2025_paper.html) · [MoGe-2 arXiv 2507.02546](https://arxiv.org/abs/2507.02546) · [code](https://github.com/microsoft/MoGe) |
| 23 | **Depth Anything 3 (DA3)** | arXiv **2511.10647**, Nov 2025; academic paper **ICLR 2026** (per project page) | Plain transformer + depth–ray target for any-view geometry ± poses; claims large gains vs VGGT on pose/geometry; feed-forward 3DGS head; nested metric models. | [project](https://depth-anything-3.github.io/) · [arXiv](https://arxiv.org/abs/2511.10647) |
| 24 | **Mip-Splatting** (Yu et al.) | **CVPR 2024 Best Student Paper** | 3D smoothing + 2D mip filter → alias-free multi-scale 3DGS. | [code](https://github.com/autonomousvision/mip-splatting) |
| 25 | **Scaffold-GS** (Lu et al.) | **CVPR 2024 Highlight** | Anchor/neural Gaussians for structured, view-adaptive, storage-efficient scenes. | [code](https://github.com/city-super/scaffold-gs) |
| 26 | **2DGS** (Huang et al.) | **SIGGRAPH 2024** | Flatten to 2D Gaussian disks + perspective-correct rasterization; strong surfaces + TSDF mesh extraction. | [project](https://surfsplatting.github.io/) · [arXiv 2403.17888](https://arxiv.org/abs/2403.17888) |
| 27 | **SuGaR** (Guédon & Lepetit) | **CVPR 2024** | Surface-aligned Gaussians → Poisson mesh in minutes; editable hybrid mesh+Gaussians. | [project](https://anttwo.github.io/sugar/) |
| 28 | **Gaussian Surfels** (Dai et al.) | **SIGGRAPH 2024** | Zero-thickness Gaussian ellipses + normal–depth consistency; Poisson meshing. | [project](https://turandai.github.io/projects/gaussian_surfels/) |
| 29 | **Spec-Gaussian** (Yang et al.) | **NeurIPS 2024** | Anisotropic spherical Gaussians for specular / anisotropic appearance beyond SH. | [project](https://ingra14m.github.io/Spec-Gaussian-website/) · [arXiv 2402.15870](https://arxiv.org/abs/2402.15870) |
| 30 | **GaussianPro** (Cheng et al.) | **ICML 2024** | MVS-style progressive propagation densification for textureless / large outdoor scenes. | [PMLR](https://proceedings.mlr.press/v235/cheng24f.html) · [arXiv 2402.14650](https://arxiv.org/abs/2402.14650) |
| 31 | **AbsGS** (Ye et al.) | **ACM MM 2024** | Homodirectional view-space gradients fix densification “gradient collision”; recovers fine detail. | [arXiv 2404.10484](https://arxiv.org/abs/2404.10484) · [code](https://github.com/TY424/AbsGS) |
| 32 | **VastGaussian** (Lin et al.) | **CVPR 2024** | Progressive cell partition + appearance decoupling for vast scenes. | [OA](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_VastGaussian_Vast_3D_Gaussians_for_Large_Scene_Reconstruction_CVPR_2024_paper.html) |
| 33 | **Hierarchical 3DGS** (Kerbl et al.) | **SIGGRAPH / TOG**, Jul 2024 | Chunk training + Gaussian LOD hierarchy for km-scale captures / tens of thousands of images. | [project](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) |
| 34 | **CityGaussian / CityGaussianV2** | **ECCV 2024** / **ICLR 2025** | Divide-and-conquer + LoD for city-scale real-time 3DGS; V2 improves geometry/compression. | [ECCV poster](https://eccv.ecva.net/virtual/2024/poster/178) · [V2 OpenReview](https://openreview.net/forum?id=a3ptUbuzbW) |
| 35 | **4D-GS** (Wu et al.) | **CVPR 2024** | Canonical 3D Gaussians + HexPlane-inspired deformation field; real-time dynamic NVS. | [OA](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) · [arXiv 2310.08528](https://arxiv.org/abs/2310.08528) |
| 36 | **Deformable 3DGS** (Yang et al.) | **CVPR 2024** | Canonical Gaussians + deformation MLP for monocular dynamic scenes; real-time. | [project](https://ingra14m.github.io/Deformable-Gaussians/) |
| 37 | **Ex4DGS** (Lee et al.) | **NeurIPS 2024** | Fully *explicit* 4DGS: static/dynamic split, keyframe interpolation, progressive training, point backtracking. | [project](https://leejunoh.com/Ex4DGS/) |
| 38 | **MoSca** (Lei et al.) | **CVPR 2025** | 4D motion scaffolds from foundation priors; monocular casual-video dynamic Gaussian fusion; BA for poses/focal. | [OA](https://openaccess.thecvf.com/content/CVPR2025/html/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.html) |
| 39 | **Shape of Motion** (Wang et al.) | **ICCV 2025** | Monocular 4D recon with SE(3) motion bases + depth/track priors; long-range 3D tracks + NVS. | [project](https://shape-of-motion.github.io/) · [OA PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.pdf) · [arXiv 2407.13764](https://arxiv.org/abs/2407.13764) |
| 40 | **GS-SLAM / SplaTAM / MonoGS / Photo-SLAM** | all **CVPR 2024** | First wave of dense SLAM with 3DGS maps (RGB-D tracking+map; silhouette-guided RGB-D; monocular GS SLAM; ORB-style tracking + GS mapping). | [SplaTAM](https://spla-tam.github.io/) · [MonoGS](https://github.com/muskie82/MonoGS) · [Photo-SLAM](https://huajianup.github.io/research/Photo-SLAM/) · [GS-SLAM arXiv 2311.11700](https://arxiv.org/abs/2311.11700) |
| 41 | **WildGS-SLAM** (Zheng et al.) | **CVPR 2025** | Monocular GS-SLAM in *dynamic* environments (beyond static-scene assumption of 2024 systems). | [OA PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zheng_WildGS-SLAM_Monocular_Gaussian_Splatting_SLAM_in_Dynamic_Environments_CVPR_2025_paper.pdf) |

**Also important (not given dedicated table rows)**
- **DyNeRF** (Li et al., **CVPR 2022**): multi-view video dynamic NeRF baseline that 4DGS papers compare against.
- **InstantSplat** ([arXiv 2403.20309](https://arxiv.org/abs/2403.20309); project [instantsplat.github.io](https://instantsplat.github.io/)): DUSt3R init + fast pose-free sparse-view 3DGS. ⚠ Venue: NVIDIA AVG lists “CVPR NRI 2024”; not confirmed here as CVPR *main track*.
- **Octree-GS** (Ren et al.): LOD-structured anchors; journal version **TPAMI 2025** ([IEEE](https://doi.org/10.1109/tpami.2025.3568201)).
- **DiffMC / DiffDMC** (Wei et al., DISO library; cited with NeuManifold **WACV 2025**): CUDA differentiable Marching Cubes / Dual MC. Related to—but distinct from—**FlexiCubes**. User’s “DiffMCubes” maps here; do not confuse with Liao et al. Deep Marching Cubes (CVPR 2018).
- **Frosting** (Guédon et al., **ECCV 2024**): adaptive Gaussian “frosting” layer on meshes (SuGaR successor).

---

## 2. What converged by mid-2026

### 2.1 Classical SfM/MVS is not dead—it got a fast sibling
- **COLMAP** remains the default calibration backend for offline 3DGS.
- **GLOMAP (ECCV 2024)** showed global SfM can match incremental accuracy while being far faster; **COLMAP 4.0** absorbed it as a first-class `global_mapper` (stand-alone glomap repo deprecated).
- Learning-based matching / pointmaps (MASt3R-SfM, VGGT, DA3-Long demos) increasingly *replace or bootstrap* feature matching + BA, especially when coverage is sparse or internet-photo-like.

### 2.2 Feed-forward visual geometry is the new front-end
Consensus trajectory (verified venues):

`DUSt3R (CVPR’24) → MASt3R (ECCV’24) → Spann3R / CUT3R / Fast3R (3DV/CVPR’25) → VGGT (CVPR’25 Best Paper) → Depth Anything 3 (arXiv Nov’25 / ICLR’26 paper)`

Converged ideas:
- **Pointmaps / depth–rays** as a unified 3D output, not just depth maps.
- **Pose-optional** inference (estimate cameras jointly).
- **Multi-view transformers** (frame-wise + global attention) over pairwise-only graphs when N is large.
- **Foundation depth** (Depth Anything, Metric3D, UniDepth, MoGe) as plug-in priors for SLAM, 4D, and GS init.

Open tension (not resolved): DA3’s project claims large average gains over VGGT on a new visual-geometry benchmark; VGGT remains the 2025 conference landmark. Treat head-to-head SOTA as **benchmark-dependent** until a community suite settles.

### 2.3 3DGS is the default radiance representation
By Kerbl’s Oct 2025 outlook and 2025 industry wrap-ups:
- Research moved from “can we match NeRF?” to **compression, LOD, mobile/VR, feed-forward GS, math of splatting, and city scale**.
- Quality forks converged into a shortlist of *practical* fixes: **Mip-Splatting** (aliasing), **AbsGS / densification fixes**, **Scaffold / Octree** (structure + LOD), **Spec-Gaussian** (specular), **2DGS / SuGaR / Surfels** (geometry).
- Large scenes: **VastGaussian, Hierarchical 3DGS, CityGaussian(+V2), Octree-GS**—shared recipe = *partition + merge + LoD*, not one monolithic optimize.

### 2.4 Surfaces: from hours of Neuralangelo to minutes of GS meshing
- Neural SDF line (NeuS → VolSDF → Neuralangelo / BakedSDF) still wins some high-fidelity object metrics but is slow for scenes.
- **2DGS, SuGaR, Gaussian Surfels, GOF**-class methods made **mesh extraction from radiance captures** practical for content pipelines.
- Differentiable meshing tools (**FlexiCubes**, DiffMC) matter more for *generative / inverse* mesh optimization than for photogrammetry end-users.

### 2.5 Dynamic / 4D: two regimes
1. **Multi-view / studio dynamic** → deformable / HexPlane 4DGS (Wu, Yang; CVPR 2024) and explicit keyframe 4DGS (Ex4DGS, NeurIPS 2024): real-time playback is solved for many clips.
2. **Casual monocular video** → motion factorization + foundation priors (**Shape of Motion**, ICCV 2025; **MoSca**, CVPR 2025): still optimization-heavy, prior-dependent, and failure-prone under extreme non-rigidity / occlusion.

### 2.6 GS-SLAM: dense photoreal maps are possible; robustness is not
- 2024 CVPR cohort proved RGB-D and even monocular GS-SLAM.
- 2025–26 focus: **dynamics** (WildGS-SLAM), large drift, and marrying feed-forward geometry (VGGT/DA3) with online BA—still an active systems problem, not a settled stack.

---

## 3. Open problems (with attribution)

| Problem | Who articulates it / evidence | Status as of Aug 2026 |
|---|---|---|
| **Geometry ≠ appearance** | 2DGS / SuGaR / Neuralangelo papers: high PSNR can coexist with wrong surfaces; reverse also true. | Partially mitigated by surface-aware primitives; no single loss that guarantees both. |
| **Anti-aliasing & multi-scale consistency** | Mip-Splatting (CVPR 2024 Best Student): vanilla 3DGS breaks under focal/distance changes. | Filters help; city-scale + foveated + streaming still expose popping / LOD seams (Hierarchical 3DGS, Octree-GS discussions). |
| **Densification pathology** | AbsGS (ACM MM 2024): “gradient collision” leaves large floaters / blur. | Many patches; not a unified densification theory. |
| **Sparse / pose-free / in-the-wild capture** | InstantSplat; DUSt3R/VGGT lines; Hierarchical 3DGS notes sparse coverage failures of vanilla densify. | Feed-forward init helps; metric scale + long-term consistency still fragile. |
| **City / planet scale under resource budgets** | VastGaussian, CityGaussian, Hierarchical 3DGS, Kerbl 2025 outlook § large-scale & mobile. | Partition+LOD works; seam / appearance / out-of-core training remain engineering-heavy (2026 “LoD of Gaussians” / BlitzGS-style follow-ons). |
| **Casual monocular 4D** | Shape of Motion; MoSca: ill-posed without strong priors; concurrent methods proliferate without shared eval. | SE(3) bases / scaffolds help; topology changes, fluids, crowds open. |
| **Dynamic SLAM with photoreal maps** | WildGS-SLAM (CVPR 2025): static-scene GS-SLAM collapses with movers. | Masking + motion priors; semantic dependence. |
| **Physically based / relightable GS** | Spec-Gaussian; industry push for deferred/relight (Kerbl outlook; radiancefields 2025 wrap). | Specularity better; full BRDF + exposure + editability incomplete. |
| **Evaluation mismatch** | DA3 introduces a new visual-geometry + rendering bench claiming large wins vs VGGT; WorldLens-style critiques in adjacent WM literature warn that visual metrics ≠ task utility. | No single community “geometry+render+downstream” suite dominates. |
| **Theoretical fidelity of splat volume rendering** | Kerbl outlook cites analyses asking whether 3DGS is accurate volumetric rendering (Celarek et al., CGF 2025, per that survey). | Practical rasterization ≠ continuous volume integral; artifacts remain under extreme FOV / transparency. |

---

## 4. Under-explored academic-lab gaps

These are areas where *industry/lab demos* outrun *clean, publishable, ablatable science*—good PhD / academic-lab targets:

1. **Principled densification & capacity control.** Many one-off heuristic patches (AbsGS, Mini-Splatting, Taming3DGS, …). Missing: theory linking view-space gradients, coverage, and reconstruction error with budgets.
2. **Unified feed-forward → refine contracts.** InstantSplat / DA3-GS heads / Splatt3R-style pipelines exist, but few papers rigorously characterize *when* to stop optimizing, how to calibrate uncertainty into BA, or how errors compound over kilometers (DA3-Long vs COLMAP is a teaser, not a theory).
3. **Metric, temporally consistent geometry for robotics—not just NVS.** Depth Anything / Metric3D / UniDepth / MoGe optimize image metrics; closed-loop control cares about **drift, latency, and calibration under rolling shutter / HDR / rain**. Thin academic coverage relative to NVS papers.
4. **Cross-illumination large scenes.** VastGaussian’s appearance decoupling is a start; city captures with diurnal lighting, auto-exposure, and multi-season revisits are still poorly formalized.
5. **Dynamic scenes with topology change.** Shape of Motion / MoSca assume persistent Gaussian / rigid-ish groups. Fluids, cloth tear, object entry/exit, multi-person contact need representations beyond SE(3) bases.
6. **GS-SLAM × foundation models without training-time cheating.** Many systems silently depend on offline depth/pose oracles or offline BA. Fair online protocols (sensor clock, compute envelope, no future frames) are scarce.
7. **Relightable, editable, mesh-compatible assets for DCC.** SuGaR/Frosting/2DGS get meshes out; PBR materials, UV atlases, and artist-grade topology remain mostly product work, not CV benchmarks.
8. **Compression + streaming standards.** Kerbl outlook and industry wraps stress mobile/VR; academic papers under-publish bitrates, progressive transmission, and perceptual quality under bandwidth caps.
9. **Safety / failure detection for geometry foundation models.** VGGT/DA3 can be confidently wrong under specular, transparent, or deferred-sky scenes. Calibrated failure flags for downstream autonomy are rare.
10. **Fair long-horizon 4D benchmarks.** iPhone / NVIDIA Dynamic / HyperNeRF suites are short. Minute-scale casual video with GT 3D tracks (Shape of Motion’s direction) needs public scale-up.

---

## 5. Suggested reading order (if starting cold)

1. **3DGS** (Kerbl SIGGRAPH 2023) + **Mip-Splatting** + **2DGS** — representation + aliasing + geometry.
2. **DUSt3R → VGGT → Depth Anything 3** — modern geometry front-end.
3. **GLOMAP + COLMAP 4.0 notes** — where classical SfM sits now.
4. **Hierarchical 3DGS or CityGaussian** — scale.
5. **4D-GS + Shape of Motion / MoSca** — dynamic split (studio vs casual).
6. **SplaTAM + MonoGS + WildGS-SLAM** — online mapping arc.
7. **Kerbl 2025 outlook (2510.26694)** — map of the splat ecosystem.

---

## 6. Uncertainty ledger

| Item | Flag |
|---|---|
| InstantSplat conference track | ⚠ Project cites arXiv; NVIDIA page says CVPR NRI 2024 — not verified as CVPR main. |
| Depth Anything 3 ICLR 2026 | ⚠ Stated on [project page](https://depth-anything-3.github.io/) (academic paper vs Seed tech report); not independently verified via ICLR proceedings in this session. |
| DiffMCubes naming | Community “DiffMC” (Wei/DISO) ≠ Deep Marching Cubes 2018; FlexiCubes is the SIGGRAPH 2023 mesh-opt standard. |
| BlitzGS / 2026 ultra-large LoD papers | Exist on arXiv (e.g. BlitzGS [2605.13794](https://arxiv.org/abs/2605.13794) seen this session) but are too new for stable venue assignment—use cautiously. |
| Any arXiv ID **not** linked above | **Not invented**; omitted on purpose. |

---

*End of survey. Companion notes in this folder: `01-related-work.md`, `02-idea-candidates.md`, `03-survey-embodied-driving.md`.*
