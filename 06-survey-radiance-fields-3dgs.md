# Radiance Fields and 3D Gaussian Splatting — Literature Survey

**Coverage:** 2024 → August 2026, weighted toward **Aug 2025 – Aug 2026**
**Compiled:** 5 August 2026
**Scope:** representations, novel view synthesis (NVS), surface reconstruction, dynamic/4D, feed-forward/sparse-view, efficiency/compression, inverse rendering, large-scale scenes

---

## 0. How this was built, and how to read it

Every arXiv identifier in this document was checked against the **arXiv API** (`export.arxiv.org/api/query`) by ID or exact title on 5 Aug 2026; the returned title and first-submission date are what appear in the table. **No arXiv ID here is from memory.** Venues were taken, in order of preference, from (a) the arXiv `journal_ref` or author `comment` field, (b) an official conference/publisher page, (c) the project page. Where a venue rests only on my prior knowledge and I could not re-confirm it today, it is marked **†** and re-listed in §9 (Verification caveats).

A methodological warning that colours everything below: **two 2026 papers argue that the field's standard benchmarks are measuring the wrong thing.** *Mind the Gap* (arXiv [2607.01556](https://arxiv.org/abs/2607.01556)) shows the Mip-NeRF 360 "every 8th frame" holdout measures near-trajectory *interpolation*, and that switching to a spatial-sector holdout costs **3–12 dB PSNR** — several times larger than the differences between competing methods, and large enough to flip method rankings. *From Blobs to Spokes* (arXiv [2604.07337](https://arxiv.org/abs/2604.07337)) shows that DTU Chamfer Distance computed on mesh *vertices* rewards denser tessellation, and that PGSR's strong DTU/T&T numbers partly reflect its non-watertight meshes having holes that coincide with holes in the ground-truth scan. Treat all leaderboard deltas below 0.5 dB, and all Chamfer deltas below ~0.05 mm, as noise.

---

## 1. Executive summary — direct answers to the seven questions

**1. Is 3DGS still the representation, and has anything displaced it?**
3DGS is still the default, but 2026 is the first year in which that is a *considered* choice rather than an automatic one. Three things have changed. (i) **The primitive is now a free variable.** Beta kernels (Deformable Beta Splatting, SIGGRAPH 2025; Universal Beta Splatting, ICLR 2026), Student-t kernels with signed "scooping" (3D Student Splatting and Scooping), smooth convexes (3D Convex Splatting, CVPR 2025), sparse voxels (SVRaster, CVPR 2025), Voronoi foam (Radiant Foam), and **triangles** (Triangle Splatting, 3DV 2026) all match or beat Gaussians on Mip-NeRF 360 with fewer parameters. (ii) **The renderer is now a free variable.** 3DGRT (SIGGRAPH Asia 2024) and 3DGUT (CVPR 2025) replace the EWA-splatting approximation with ray tracing and an unscented transform, enabling rolling shutter, fisheye, and secondary rays; StochasticSplats (ICCV 2025) removes the global sort. (iii) **In feed-forward 4D, something genuinely non-Gaussian won.** The **CVPR 2026 Best Paper, D4RT**, reconstructs dynamic scenes with a query-based pointmap transformer that has no Gaussians in it at all, and is ~9× faster than VGGT at pose and ~100× faster than MegaSaM. The honest summary: *3DGS won the per-scene optimization slot and is being standardized (glTF, MPEG) as a delivery format; it did not win the feed-forward slot, where transformer pointmaps are ahead.*

**2. NeRF vs 3DGS in 2026.** Not a competition anymore; the two lineages merged conceptually. Geometry-Grounded Gaussian Splatting (arXiv [2601.17835](https://arxiv.org/abs/2601.17835)) proves Gaussian primitives *are* stochastic solids in the Miller et al. "Objects as Volumes" sense, which unifies the 3DGS and NeRF rendering equations and lets you derive a NeRF-style geometric field from Gaussians. Practically: NeRF/MLP survives only where ray-marching buys something splatting cannot cheaply fake — Zip-NeRF-class anti-aliasing, and view-dependent reflections (NeRF-Casting). Ray-marched *Gaussian* hybrids (RayGaussX, arXiv [2509.07782](https://arxiv.org/abs/2509.07782)) now beat both on Mip-NeRF 360 (25.24 outdoor / 32.43 indoor PSNR vs 3DGS's 24.67 / 30.96 in a common protocol).

**3. Surface reconstruction.** This is the area with the clearest measurable progress. DTU mean Chamfer went 3DGS 1.96 mm → 2DGS 0.80 → GOF 0.74 → RaDe-GS 0.68 → PGSR 0.52 → **GeoSVR / GGGS 0.47 mm**, now beating Neuralangelo (0.61 mm) at ~15 minutes instead of >12 hours. Tanks & Temples F1 went 2DGS 0.30 → PGSR 0.52 → **GGGS 0.60**. Two 2025–26 ideas dominate: put the mesh *inside* the training loop (MILo, SIGGRAPH Asia 2025) and give each Gaussian a principled geometric field instead of a depth heuristic (GGGS, Gaussian Wrapping). **But** the field simultaneously discovered its own metrics are biased, so the "solved" claim is weaker than the numbers suggest.

**4. Feed-forward vs per-scene.** The boundary moved decisively in 2025–26 but did not disappear. Feed-forward models now handle **unposed, arbitrary-count, scene-level** input in seconds (AnySplat, YoNoSplat, Depth Anything 3, ZipSplat). What remains true, stated bluntly by ForeSplat (arXiv [2605.22020](https://arxiv.org/abs/2605.22020)): *"when the prediction is used as-is, a persistent quality gap to a full per-scene 3DGS optimization remains."* The dominant 2026 pattern is therefore **predict-then-refine**: feed-forward for initialization + geometry/pose, then seconds-to-minutes of test-time optimization. The 2026 research frontier is making the feed-forward model *aware* it will be refined (ForeSplat, iSplat, GIFSplat, Diff3R).

**5. Compression.** Roughly **100× at no quality loss** is now routine and standardized. HAC++ (TPAMI 2025) reports >100× vs vanilla 3DGS *with improved fidelity*; SpeedyGS (Jul 2026) reports up to 160× with the additional and now-decisive property of near-zero decode latency. The centre of gravity has shifted from ratio to **decode time, streaming, and interoperability**: Khronos issued the **KHR_gaussian_splatting** glTF release candidate on 3 Feb 2026 (ratification targeted Q2 2026), and **MPEG has an active Gaussian Splat Coding exploration** (Part 45), though no standard exists yet.

**6. Inverse rendering.** The least solved of the seven. 2026 work (IRGS++, RadioGS, PTIR-GS, MaterialClusterGS) has converged on differentiable **ray tracing of Gaussians** with the full rendering equation, replacing 2024's screen-space G-buffer approximations. But the underlying ill-posedness is untouched: MaterialClusterGS states that per-primitive BRDF fitting is "highly under-constrained: shadows, indirect illumination, geometric errors, and visibility residuals can be absorbed into thousands of slightly different local material estimates." The honest state: **relighting under novel illumination looks plausible; the recovered albedo/roughness are not trustworthy physical quantities**, and the strongest results now come from *learned priors* (DiffusionRenderer, Neural Gaffer, UniRelight) rather than from optimization.

**7. Large scale.** Divide-and-conquer (VastGaussian, CityGaussian, Hierarchical 3DGS) is being replaced by **out-of-core, unpartitioned** training: A LoD of Gaussians (SIGGRAPH 2026) trains 150M+ Gaussians on a single ≤24 GB GPU by keeping the model in CPU memory and streaming, explicitly to remove chunk-boundary artifacts. On-the-fly reconstruction from unposed images (TOG 44(4), Aug 2025) removes the offline COLMAP stage.

---

## 2. The key papers

*A note on scope:* the brief asked for ~35 papers. I list 82, because seven sub-areas do not compress into 35 rows without dropping whole threads (all of compression, or all of inverse rendering). The **~14 papers that get a plain-language deep dive in §3** are the ones I would call essential; the rest are the supporting scaffolding you need to read them in context. Rows in **bold** in the table are the essential set.

Legend: **†** = venue from prior knowledge / secondary source, not re-verified against a primary listing today (see §9). Dates are **arXiv first-submission** dates verified today unless a journal reference is given.

### A. Representations and the NeRF → 3DGS transition

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 1 | NeRF | ECCV 2020† · arXiv 2020-03-19 | Scene as an MLP mapping (position, direction) → (colour, density), rendered by ray-marching; started the field. | [2003.08934](https://arxiv.org/abs/2003.08934) |
| 2 | Instant-NGP | ACM TOG 41(4), SIGGRAPH 2022† · arXiv 2022-01-16 | Multiresolution hash encoding cuts NeRF training from hours to seconds. | [2201.05989](https://arxiv.org/abs/2201.05989) |
| 3 | Mip-NeRF 360 | CVPR 2022† · arXiv 2021-11-23 | Unbounded scenes + scale-aware anti-aliasing; defined the benchmark everyone still uses. | [2111.12077](https://arxiv.org/abs/2111.12077) |
| 4 | TensoRF | ECCV 2022† · arXiv 2022-03-17 | Factorizes the radiance volume into low-rank tensor components — the "explicit grid" turn. | [2203.09517](https://arxiv.org/abs/2203.09517) |
| 5 | Zip-NeRF | ICCV 2023† · arXiv 2023-04-13 | Combines hash grids with mip-NeRF cone sampling; still the NeRF-side quality reference. | [2304.06706](https://arxiv.org/abs/2304.06706) |
| 6 | **3D Gaussian Splatting** | **ACM TOG 42(4), July 2023 (SIGGRAPH 2023)** | Millions of anisotropic 3D Gaussians + tile-based differentiable rasterizer: NeRF-quality at 100+ FPS. | [2308.04079](https://arxiv.org/abs/2308.04079) |
| 7 | Mip-Splatting | CVPR 2024† · arXiv 2023-11-27 | 3D smoothing filter + 2D mip filter remove 3DGS's zoom-in/zoom-out aliasing. | [2311.16493](https://arxiv.org/abs/2311.16493) |
| 8 | Scaffold-GS | CVPR 2024† · arXiv 2023-11-30 | Anchor points spawn view-dependent neural Gaussians on the fly — the ancestor of most compression work. | [2312.00109](https://arxiv.org/abs/2312.00109) |
| 9 | Octree-GS | arXiv 2024-03-26 (venue †/unclear) | Octree-structured LOD over anchors for consistent real-time rendering at varying distance. | [2403.17898](https://arxiv.org/abs/2403.17898) |
| 10 | 3DGS-MCMC | NeurIPS 2024† · arXiv 2024-04-15 | Reinterprets densification as SGLD sampling; removes the heuristic clone/split rules. | [2404.09591](https://arxiv.org/abs/2404.09591) |
| 11 | 3DGRT (Gaussian Ray Tracing) | **SIGGRAPH Asia 2024** (author-stated) · arXiv 2024-07-09 | Ray-traces Gaussian particles via BVH: enables reflections, refraction, rolling shutter. | [2407.07090](https://arxiv.org/abs/2407.07090) |
| 12 | 3DGUT | **CVPR 2025** (author-stated) · arXiv 2024-12-17 | Unscented transform replaces the local-affine projection, supporting distorted cameras + secondary rays inside rasterization. | [2412.12507](https://arxiv.org/abs/2412.12507) |
| 13 | SVRaster (Sparse Voxels Rasterization) | **CVPR 2025** (author-stated) · arXiv 2024-12-05 | Shows sparse voxels, not Gaussians, suffice for real-time high-fidelity radiance fields. | [2412.04459](https://arxiv.org/abs/2412.04459) |
| 14 | Deformable Beta Splatting | **SIGGRAPH 2025** (author-stated) · arXiv 2025-01-27 | Beta kernels with bounded support replace Gaussians: SOTA quality at 45% of 3DGS-MCMC parameters, 1.5× faster. | [2501.18630](https://arxiv.org/abs/2501.18630) |
| 15 | Triangle Splatting | **3DV 2026**, Vancouver, 20–23 Mar 2026, pp. 1248–1257 | Directly optimizes triangles as differentiable splats; >2,400 FPS in an off-the-shelf mesh renderer. | [2505.19175](https://arxiv.org/abs/2505.19175) |
| 16 | Triangle Splatting+ | arXiv 2025-09-29 (venue unconfirmed) | Adds shared-vertex connectivity and enforces *opaque* triangles — output drops straight into a game engine. | [2509.25122](https://arxiv.org/abs/2509.25122) |
| 17 | Universal Beta Splatting | **ICLR 2026** (author-stated) · arXiv 2025-09-30 | Generalizes Beta kernels to a unified controllable primitive family. | [2510.03312](https://arxiv.org/abs/2510.03312) |
| 18 | RayGaussX | arXiv 2025-09-09 (venue unconfirmed) | Accelerated Gaussian *ray marching*; tops Mip-NeRF 360 among the methods tabulated in GGGS. | [2509.07782](https://arxiv.org/abs/2509.07782) |
| 19 | Mind the Gap | arXiv 2026-07-02 (preprint) | Shows standard 3DGS evaluation measures near-trajectory interpolation; 3–12 dB interpolation/extrapolation gap. | [2607.01556](https://arxiv.org/abs/2607.01556) |

### B. Surface / mesh reconstruction

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 20 | NeuS | NeurIPS 2021† · arXiv 2021-06-20 | Unbiased conversion of an SDF into volume-rendering density — the implicit-surface baseline. | [2106.10689](https://arxiv.org/abs/2106.10689) |
| 20b | VolSDF | NeurIPS 2021† · arXiv 2021-06-22 | Models density as a transformed SDF, giving volume rendering a surface to be consistent with. DTU 0.86 mm. | [2106.12052](https://arxiv.org/abs/2106.12052) |
| 21 | Neuralangelo | CVPR 2023† · arXiv 2023-06-05 | Hash-grid SDF with numerical gradients + coarse-to-fine; the pre-3DGS quality ceiling (DTU 0.61 mm, >12 h). | [2306.03092](https://arxiv.org/abs/2306.03092) |
| 25b | Gaussian Surfels | SIGGRAPH 2024† · arXiv 2024-04-27 | Zeroes the third scale axis to make Gaussians surfels, with depth-normal consistency — the parallel discovery to 2DGS. | [2404.17774](https://arxiv.org/abs/2404.17774) |
| 22 | SuGaR | CVPR 2024† · arXiv 2023-11-21 | First practical mesh extraction from 3DGS: regularize Gaussians onto surfaces, then Poisson-reconstruct. | [2311.12775](https://arxiv.org/abs/2311.12775) |
| 23 | **2D Gaussian Splatting** | SIGGRAPH 2024† · arXiv 2024-03-26 | Collapses each 3D Gaussian to an oriented 2D disk, making the primitive an actual surface element. DTU 0.80 mm. | [2403.17888](https://arxiv.org/abs/2403.17888) |
| 24 | Gaussian Opacity Fields | SIGGRAPH Asia 2024† · arXiv 2024-04-16 | Defines a ray-tracing-based opacity field over 3D Gaussians → tetrahedra-based adaptive meshing in unbounded scenes. | [2404.10772](https://arxiv.org/abs/2404.10772) |
| 25 | PGSR | IEEE TVCG† · arXiv 2024-06-10 | Flattens Gaussians into planes + multi-view photometric/geometric consistency. DTU 0.52 mm. | [2406.06521](https://arxiv.org/abs/2406.06521) |
| 26 | **MILo** | **ACM TOG 44(6), Dec 2025 (SIGGRAPH Asia 2025)**, DOI [10.1145/3763339](https://doi.org/10.1145/3763339) | Differentiably extracts a mesh (vertices *and* connectivity) at **every** training iteration; ~10× fewer vertices. | [2506.24096](https://arxiv.org/abs/2506.24096) |
| 27 | GeoSVR | **NeurIPS 2025 Spotlight** (author-stated) · arXiv 2025-09-22 | Sparse-voxel surface reconstruction; DTU 0.47 mm, T&T F1 0.56 — beats all Gaussian methods of its date. | [2509.18090](https://arxiv.org/abs/2509.18090) |
| 28 | **Geometry-Grounded Gaussian Splatting (GGGS)** | arXiv 2026-01-25 (preprint) | Proves Gaussians are *stochastic solids*; median-transmittance depth gives DTU 0.47 mm, T&T F1 **0.60** in 15 min. | [2601.17835](https://arxiv.org/abs/2601.17835) |
| 29 | From Blobs to Spokes (Gaussian Wrapping) | arXiv 2026-04-08 (preprint) | One learnable oriented normal per Gaussian → closed-form vacancy field, watertight meshes from 2 pivots; exposes evaluation bias. | [2604.07337](https://arxiv.org/abs/2604.07337) |

### C. Dynamic / 4D

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 30 | D-NeRF | CVPR 2021† · arXiv 2020-11-27 | Canonical NeRF + learned deformation field; still the (saturated) synthetic dynamic benchmark. | [2011.13961](https://arxiv.org/abs/2011.13961) |
| 30b | HyperNeRF | ACM TOG / SIGGRAPH Asia 2021† · arXiv 2021-06-24 | Lifts the canonical space to higher dimensions so topology changes (a mouth opening) stop breaking the deformation field. | [2106.13228](https://arxiv.org/abs/2106.13228) |
| 30c | Deformable 3DGS | CVPR 2024† · arXiv 2023-09-22 | Canonical Gaussians + an MLP deformation field: the direct 3DGS analogue of D-NeRF, for monocular dynamic scenes. | [2309.13101](https://arxiv.org/abs/2309.13101) |
| 31 | 4D Gaussian Splatting | CVPR 2024† · arXiv 2023-10-12 | HexPlane-encoded deformation field drives a canonical Gaussian set — real-time dynamic rendering. | [2310.08528](https://arxiv.org/abs/2310.08528) |
| 32 | Spacetime Gaussian Feature Splatting | **CVPR 2024** (author-stated) · arXiv 2023-12-28 | Temporal opacity + polynomial motion + splatted features; the strong multi-view dynamic baseline. | [2312.16812](https://arxiv.org/abs/2312.16812) |
| 33 | **Shape of Motion** | **ICCV 2025** (author-stated) · arXiv 2024-07-18 | Monocular 4D via persistent Gaussians whose trajectories are linear combinations of a few global SE(3) motion bases. | [2407.13764](https://arxiv.org/abs/2407.13764) |
| 34 | MoSca | arXiv 2024-05-27 (venue unconfirmed) | "4D motion scaffolds": a deformation graph from 2D tracks that regularizes casual-video Gaussian fusion. | [2405.17421](https://arxiv.org/abs/2405.17421) |
| 34b | Dynamic Gaussian Marbles | arXiv 2024-06-26 (venue unconfirmed) | Isotropic "marbles" + a divide-and-conquer trajectory-learning curriculum for casual monocular video. | [2406.18717](https://arxiv.org/abs/2406.18717) |
| 35 | 4DGT | **NeurIPS 2025 Spotlight** (author-stated) · arXiv 2025-06-09 | A 4D Gaussian *transformer* trained on real monocular video — feed-forward, no per-scene optimization. | [2506.08015](https://arxiv.org/abs/2506.08015) |
| 36 | MoVieS | **CVPR 2026** (author-stated) · arXiv 2025-07-14 | Motion-aware 4D dynamic view synthesis in ~1 second. | [2507.10065](https://arxiv.org/abs/2507.10065) |
| 37 | **D4RT** | **CVPR 2026 — Best Paper Award** | One transformer + one "query the 3D position of any spacetime point" interface replaces depth/tracking/pose pipelines. **No Gaussians.** | [Project page](https://d4rt-paper.github.io/) · [CVPR announcement](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers) |
| 38 | C4G | arXiv 2026-05-29 (preprint) | ~2048 timestamp-conditioned query tokens instead of per-pixel Gaussians: fixes duplication/ghosting in feed-forward 4D. | [2605.31595](https://arxiv.org/abs/2605.31595) |

### D. Feed-forward / sparse-view

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 39 | pixelSplat | CVPR 2024† · arXiv 2023-12-19 | Predicts 3D Gaussians from an image *pair* in one pass; solved scale ambiguity with a sampled depth distribution. | [2312.12337](https://arxiv.org/abs/2312.12337) |
| 40 | MVSplat | ECCV 2024† · arXiv 2024-03-21 | Cost-volume multi-view stereo backbone → far fewer parameters than pixelSplat at higher quality. | [2403.14627](https://arxiv.org/abs/2403.14627) |
| 40b | latentSplat | ECCV 2024† · arXiv 2024-03-24 | Variational Gaussians in a latent space, decoded by a GAN-style head — reconstruction plus a generative prior for unseen regions. | [2403.16292](https://arxiv.org/abs/2403.16292) |
| 40c | DUSt3R | CVPR 2024† · arXiv 2023-12-21 | Regresses aligned pointmaps from an uncalibrated image pair; the ancestor of all pose-free feed-forward 3D. | [2312.14132](https://arxiv.org/abs/2312.14132) |
| 41 | GS-LRM | ECCV 2024† · arXiv 2024-04-30 | Plain transformer maps posed image tokens directly to per-pixel Gaussians; the "scale it up" recipe. | [2404.19702](https://arxiv.org/abs/2404.19702) |
| 42 | NoPoSplat | ICLR 2025† · arXiv 2024-10-31 | Drops camera poses entirely by predicting Gaussians in a canonical frame anchored to the first view. | [2410.24207](https://arxiv.org/abs/2410.24207) |
| 42b | Splatt3R | arXiv 2024-08-25 (venue unconfirmed) | Bolts a Gaussian head onto MASt3R: zero-shot splats from an uncalibrated pair with no depth or pose input. | [2408.13912](https://arxiv.org/abs/2408.13912) |
| 43 | Long-LRM | **ICCV 2025**‡ · arXiv 2024-10-16 | Mamba2 + transformer blocks scale feed-forward reconstruction to 32 wide-coverage input views. | [2410.12781](https://arxiv.org/abs/2410.12781) |
| 44 | **VGGT** | CVPR 2025† (Best Paper) · arXiv 2025-03-14 | A single feed-forward transformer for pose, depth, pointmaps and tracks — became the standard 3D backbone. | [2503.11651](https://arxiv.org/abs/2503.11651) |
| 45 | Bolt3D | **ICCV 2025** (author-stated) · arXiv 2025-03-18 | Multi-view latent diffusion generates a full 3D scene (splats) in seconds. | [2503.14445](https://arxiv.org/abs/2503.14445) |
| 46 | RayZer | ICCV 2025 Best Student Paper Hon. Mention‡ · arXiv 2025-05-01 | Self-supervised large view-synthesis model trained with **no** camera annotation. | [2505.00702](https://arxiv.org/abs/2505.00702) |
| 47 | AnySplat | arXiv 2025-05-29 (venue unconfirmed) | Feed-forward splats from unconstrained, uncalibrated, arbitrary-count views; voxelizes away redundant pixel-Gaussians. | [2505.23716](https://arxiv.org/abs/2505.23716) |
| 48 | Depth Anything 3 | arXiv 2025-11-13 (venue unconfirmed) | Geometry foundation model now used as the default feed-forward backbone in 2026 splatting pipelines. | [2511.10647](https://arxiv.org/abs/2511.10647) |
| 49 | ZipSplat | arXiv 2026-06-03 (preprint) | Decouples primitives from the pixel grid via k-means scene tokens: pose-free SOTA with ~6× fewer Gaussians. | [2606.05102](https://arxiv.org/abs/2606.05102) |
| 50 | ForeSplat | arXiv 2026-05-21 (preprint) | Trains the feed-forward model to *anticipate* subsequent per-scene refinement, fixing the train/deploy mismatch. | [2605.22020](https://arxiv.org/abs/2605.22020) |

### E. Efficiency / compression / deployment

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 51 | LightGaussian | **NeurIPS 2024** (author-stated) · arXiv 2023-11-28 | Prune by global significance + distil SH + vector-quantize: 15× smaller, 200+ FPS. | [2311.17245](https://arxiv.org/abs/2311.17245) |
| 52 | Compact 3D Gaussian Representation | CVPR 2024† · arXiv 2023-11-22 | Learnable masks + residual VQ + a grid for view-dependent colour. | [2311.13681](https://arxiv.org/abs/2311.13681) |
| 53 | Self-Organizing Gaussian Grids | ECCV 2024† · arXiv 2023-12-19 | Sorts Gaussians into a smooth 2D grid so off-the-shelf **image codecs (JPEG XL)** compress them. | [2312.13299](https://arxiv.org/abs/2312.13299) |
| 54 | HAC | **ECCV 2024** (journal_ref) · arXiv 2024-03-21 | Hash-grid context model over Scaffold-GS anchors makes arithmetic coding of splats work. | [2403.14530](https://arxiv.org/abs/2403.14530) |
| 55 | **HAC++** | **IEEE TPAMI 2025**, DOI [10.1109/TPAMI.2025.3594066](https://doi.org/10.1109/TPAMI.2025.3594066) · arXiv 2025-01-21 | Intra-anchor context + adaptive quantization: **>100× vs vanilla 3DGS with improved fidelity**, >20× vs Scaffold-GS. | [2501.12255](https://arxiv.org/abs/2501.12255) |
| 56 | FCGS | **ICLR 2025** (journal_ref) · arXiv 2024-10-10 | Feed-forward (optimization-free) splat compression — compress in one pass instead of retraining. | [2410.08017](https://arxiv.org/abs/2410.08017) |
| 57 | 3DGS.zip survey | arXiv 2024-06-17 (survey, venue †) | The reference taxonomy + rate-distortion comparison for splat compression. | [2407.09510](https://arxiv.org/abs/2407.09510) |
| 58 | LapisGS | arXiv 2024-08-27 (venue unconfirmed) | Layered progressive splats for adaptive *streaming*. | [2408.14823](https://arxiv.org/abs/2408.14823) |
| 59 | SpeedyGS | arXiv 2026-07-14 (preprint) | Rate-distortion-optimized quantization/pruning + octree tokens: up to **160×**, 9× faster optimization, near-zero decode latency. | [2607.12656](https://arxiv.org/abs/2607.12656) |
| 60 | CAGS | **SIGGRAPH Conference Papers '26**, 19–23 Jul 2026, DOI [10.1145/3799902.3811058](https://doi.org/10.1145/3799902.3811058) | Colour-adaptive volumetric-video streaming: +5–20 dB over prior adaptive streaming under bandwidth fluctuation. | [2605.09279](https://arxiv.org/abs/2605.09279) |
| 61 | MPEG GSC exploration | **DCC 2026**, DOI [10.1109/DCC66757.2026.00045](https://doi.org/10.1109/dcc66757.2026.00045), pub. 2026-03-24 | Official report on MPEG WG4/WG7 work toward standardized Gaussian splat coding (I-3DGS vs A-3DGS tracks). | (DOI) |
| 62 | KHR_gaussian_splatting | **Khronos release candidate, 3 Feb 2026**; ratification targeted Q2 2026 | glTF 2.0 extension storing splats as point primitives; SPZ (≈90% smaller than PLY) as the compression companion. | [Khronos](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release) · [spec](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) |

### F. Inverse rendering / relighting / materials

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 63 | TensoIR | CVPR 2023† · arXiv 2023-04-24 | Tensor-factorized joint radiance + physical-property estimation with efficient visibility/indirect light. | [2304.12461](https://arxiv.org/abs/2304.12461) |
| 64 | GS-IR | CVPR 2024† · arXiv 2023-11-26 | First 3DGS inverse-rendering pipeline: depth-derived normals + baked occlusion volumes. | [2311.16473](https://arxiv.org/abs/2311.16473) |
| 65 | Relightable 3D Gaussians | ECCV 2024† · arXiv 2023-11-27 | Per-Gaussian BRDF + point-based ray tracing for shadows/indirect light. | [2311.16043](https://arxiv.org/abs/2311.16043) |
| 66 | GaussianShader | CVPR 2024† · arXiv 2023-12-29 | Adds a shading function with normal estimation to handle reflective surfaces. | [2311.17977](https://arxiv.org/abs/2311.17977) |
| 67 | NeRF-Casting | SIGGRAPH Asia 2024† · arXiv 2024-05-23 | Casts reflection rays through the radiance field: consistent, high-frequency specular reflections. | [2405.14871](https://arxiv.org/abs/2405.14871) |
| 68 | IntrinsicAnything | ECCV 2024† · arXiv 2024-04-17 | Diffusion priors over albedo/specular resolve inverse-rendering ambiguity under unknown illumination. | [2404.11593](https://arxiv.org/abs/2404.11593) |
| 69 | Neural Gaffer | NeurIPS 2024† · arXiv 2024-06-11 | End-to-end diffusion relighting of any object conditioned on an environment map. | [2406.07520](https://arxiv.org/abs/2406.07520) |
| 70 | DiffusionLight | CVPR 2024† · arXiv 2023-12-14 | Inpaint a chrome ball with a diffusion model to get an HDR light probe from one photo. | [2312.09168](https://arxiv.org/abs/2312.09168) |
| 71 | **IRGS** | **CVPR 2025** (author-stated) · arXiv 2024-12-20 | 2D Gaussian **ray tracing** evaluates the full rendering equation with on-the-fly inter-reflection. | [2412.15867](https://arxiv.org/abs/2412.15867) |
| 72 | DiffusionRenderer | **CVPR 2025** (author-stated) · arXiv 2025-01-30 | Video-diffusion model does *both* inverse (G-buffers from video) and forward (relit video) rendering. | [2501.18590](https://arxiv.org/abs/2501.18590) |
| 73 | UniRelight | arXiv 2025-06-18 (venue unconfirmed) | Joint decomposition + synthesis for video relighting in one model. | [2506.15673](https://arxiv.org/abs/2506.15673) |
| 74 | RadioGS | arXiv 2026-03-02 (preprint) | Radiometric-consistency loss supervises *unobserved* views, fixing indirect-radiance queries; relight in <10 ms. | [2603.01491](https://arxiv.org/abs/2603.01491) |
| 75 | PTIR-GS | arXiv 2026-06-08 (preprint) | Splatting-free path-space inverse rendering with path-replay backpropagation and multi-bounce GI. | [2606.09606](https://arxiv.org/abs/2606.09606) |
| 76 | IRGS++ | arXiv 2026-07-24 (preprint) | Metallic-aware materials + MIS + a mesh-based relighting backend; deferred shading in image space. | [2607.22780](https://arxiv.org/abs/2607.22780) |

### G. Large-scale scenes

| # | Paper | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 77 | VastGaussian | **CVPR 2024** (author-stated) · arXiv 2024-02-27 | Progressive partitioning + decoupled appearance modelling for large scenes. | [2402.17427](https://arxiv.org/abs/2402.17427) |
| 78 | Hierarchical 3DGS | **ACM TOG 43(4), July 2024 (SIGGRAPH 2024)** (journal_ref) | A Gaussian hierarchy with smooth LOD interpolation and chunked training for very large datasets. | [2406.12080](https://arxiv.org/abs/2406.12080) |
| 79 | CityGaussianV2 | **ICLR 2025** (author-stated) · arXiv 2024-11-01 | Decomposed-gradient densification + elongation filter: geometrically accurate *and* efficient at city scale. | [2411.00771](https://arxiv.org/abs/2411.00771) |
| 80 | On-the-fly reconstruction | **ACM TOG 44(4), Aug 2025 (SIGGRAPH 2025)** (journal_ref) | Large-scale NVS from **unposed** images with no offline SfM stage. | [2506.05558](https://arxiv.org/abs/2506.05558) |
| 81 | CLM | arXiv 2025-11-07 (venue unconfirmed) | Removes the GPU-memory barrier for 3DGS via out-of-core training. | [2511.04951](https://arxiv.org/abs/2511.04951) |
| 82 | **A LoD of Gaussians** | **SIGGRAPH 2026** (journal_ref), DOI [10.1145/3799902.3811076](https://doi.org/10.1145/3799902.3811076) | Unpartitioned out-of-core training/rendering: **150M+ Gaussians on one ≤24 GB GPU**, no chunk-boundary artifacts. | [2507.01110](https://arxiv.org/abs/2507.01110) |

‡ Long-LRM (ICCV 2025) and RayZer (ICCV 2025 Best Student Paper Honorable Mention) venues come from the Paper Digest ICCV 2025 listing and the IEEE TCPAMI ICCV awards page respectively, not from the authors' own metadata.

---

## 3. Plain-language explanations of the most important work

*Jargon defined inline. Assumed background: deep learning, not graphics.*

**Splatting** means: instead of shooting a ray into the scene and integrating along it (what NeRF does), you take each 3D primitive, project it onto the image plane, and "smear" (splat) its footprint over the pixels it covers. It is a scatter operation rather than a gather operation, which maps far better onto GPUs.
**Alpha compositing** is the front-to-back blending rule `C = Σᵢ cᵢ αᵢ ∏_{j<i}(1−αⱼ)` — each primitive contributes its colour times its opacity times the light that survived everything in front of it. Both NeRF and 3DGS use this; they differ only in how they get the list of `(cᵢ, αᵢ)`.
**Spherical harmonics (SH)** are the Fourier basis on the sphere. Storing a few SH coefficients per primitive lets its colour vary with viewing direction, which is how splats fake specular highlights without any lighting model.
**Densification** is the process of adding primitives where reconstruction is poor and removing them where they are wasted — the discrete, non-differentiable heart of 3DGS optimization.
**SDF (signed distance function)** stores, at every point in space, the distance to the nearest surface, negative inside. Its zero level set *is* the surface, so meshing is well defined.
**BRDF (bidirectional reflectance distribution function)** is the function that says how much light arriving from direction ωᵢ leaves in direction ωₒ — i.e. "the material".
**Chamfer distance (CD)** measures point-cloud/mesh error: for every point in A find the nearest in B and average, then symmetrize. On DTU it is reported in millimetres; lower is better.

---

**1. 3D Gaussian Splatting** — *ACM TOG 42(4), SIGGRAPH 2023* — [2308.04079](https://arxiv.org/abs/2308.04079)
NeRF was beautiful but slow: every pixel required hundreds of MLP evaluations along a ray. 3DGS throws out the MLP and represents the scene as a few million explicit 3D Gaussian blobs, each with a position, a 3×3 covariance (an ellipsoid shape and orientation), an opacity, and SH colour coefficients. Because a 3D Gaussian projects to a 2D Gaussian under a local affine approximation of the camera, you can rasterize them in sorted tiles and alpha-composite — no ray marching, no network. The second half of the trick is *adaptive density control*: every ~100 iterations, Gaussians whose view-space position gradients are large get cloned (if small) or split (if large), and near-transparent ones get culled. This turned a 12-hour training + 1 FPS rendering problem into a ~20-minute training + 100+ FPS rendering problem, which is why the entire field pivoted within about six months.

**2. Mip-Splatting** — *CVPR 2024†* — [2311.16493](https://arxiv.org/abs/2311.16493)
Vanilla 3DGS bakes in the sampling rate of the training cameras. Zoom out and Gaussians that were sub-pixel start to alias and shimmer; zoom in and they become visible blobs. Mip-Splatting adds two filters: a **3D smoothing filter** that clamps each Gaussian's frequency content to the maximum rate any training camera could have observed, and a **2D mip filter** that replaces the ad-hoc screen-space dilation with a proper box filter matched to the render resolution. The word "mip" comes from mipmapping, the classic texture technique of pre-filtering at multiple scales. This is a small change with an outsized effect on perceived quality and it is now standard in essentially every serious 3DGS implementation.

**3. 3DGS-MCMC** — *NeurIPS 2024†* — [2404.09591](https://arxiv.org/abs/2404.09591)
The clone/split/prune heuristics in 3DGS are the ugliest part of the method: hand-tuned thresholds, sensitive to initialization, and they make the whole procedure depend on getting a good COLMAP point cloud first. This paper reinterprets the set of Gaussians as *samples from a probability distribution over scenes* and the optimization as **Stochastic Gradient Langevin Dynamics** (gradient descent plus calibrated noise, which asymptotically samples from a posterior rather than descending to a point). Densification becomes "relocate a dead sample to a live one in a way that preserves the distribution", derived rather than guessed. Practically, it removes the sensitivity to initialization and gives a *fixed Gaussian budget* knob, which is why almost every 2025–26 compression and feed-forward paper builds on MCMC rather than vanilla 3DGS.

**4. Scaffold-GS** — *CVPR 2024†* — [2312.00109](https://arxiv.org/abs/2312.00109)
Storing every attribute of every Gaussian explicitly is enormously redundant: neighbouring Gaussians on the same wall have nearly identical colours and shapes. Scaffold-GS puts a sparse grid of *anchor* points over the scene, each holding a learned feature vector, and at render time a tiny MLP decodes the anchor (conditioned on view direction and distance) into a handful of *neural Gaussians*. The Gaussians themselves are never stored. This is essentially the "latent code + decoder" trick from generative modelling applied to a radiance field, and it matters enormously downstream: HAC, HAC++, ContextGS, CAT-3DGS and most of the 100×-compression literature compress *anchors*, not Gaussians, because anchors live on a structured grid where entropy models actually work.

**5. 2D Gaussian Splatting** — *SIGGRAPH 2024†* — [2403.17888](https://arxiv.org/abs/2403.17888)
A 3D Gaussian is a fuzzy ellipsoid, and a fuzzy ellipsoid is a terrible model of a surface — it has volume, and where its "surface" is depends on which direction you look from, so the depth you render is not multi-view consistent. 2DGS collapses the third axis to zero, making each primitive an oriented elliptical *disk* (a surfel) with a well-defined normal. Rendering uses an explicit ray-splat intersection rather than the affine projection approximation, so the depth is exact and consistent across views. Adding a depth-distortion loss (concentrate weight along each ray) and a normal-consistency loss (align the disk normal with the depth-map gradient) gets you meshable geometry: DTU Chamfer 0.80 mm versus 1.96 mm for vanilla 3DGS. Everything in modern splat-based surface reconstruction descends from this.

**6. MILo** — *ACM TOG 44(6), SIGGRAPH Asia 2025* — [2506.24096](https://arxiv.org/abs/2506.24096)
Every prior method optimizes Gaussians and *then* extracts a mesh, which the paper argues is the fundamental error: "this postprocessing step collapses complex volumetric information into a surface", and nothing during training ever encouraged the Gaussians to be meshable. MILo extracts a mesh — vertices *and* connectivity — differentiably at **every single training iteration**: it spawns Delaunay pivot vertices from Gaussian parameters, maintains signed distance values at those vertices, runs GPU Marching Tetrahedra, then renders *both* the mesh and the Gaussians and penalizes disagreement in depth and normals. Gradients flow from the mesh back to the Gaussians. The result is roughly an order of magnitude fewer mesh vertices at equal or better fidelity (an entire bicycle scene with background at ~10× fewer vertices), which is what actually matters if you want to run physics or animation on the output.

**7. Geometry-Grounded Gaussian Splatting (GGGS)** — *arXiv 2026-01* — [2601.17835](https://arxiv.org/abs/2601.17835)
Every Gaussian meshing method up to this point used a *heuristic* for depth — alpha-weighted mean depth, or the depth of a fitted plane — and those heuristics are what produce floaters and view-inconsistent surfaces. GGGS derives the answer instead. Building on "Objects as Volumes" (Miller et al., 2023, arXiv [2312.15406](https://arxiv.org/abs/2312.15406)), it proves that a Gaussian primitive is exactly a **stochastic solid**: an object whose boundary is random, described by a *vacancy* field v(x) (the probability that x is empty), with attenuation σ(x,ω) = |ω·∇log v(x)|. This single result unifies the 3DGS and NeRF rendering equations and, for the first time, gives Gaussians a genuine geometric field. Depth is then defined properly as the **median** along the ray — the point where transmittance drops to 0.5 — found by binary search and differentiated in closed form. Result: DTU mean Chamfer **0.47 mm** and Tanks & Temples F1 **0.60**, the best of any explicit Gaussian method, in 15 minutes versus Neuralangelo's 0.61 mm in over 12 hours.

**8. Shape of Motion** — *ICCV 2025* — [2407.13764](https://arxiv.org/abs/2407.13764)
Reconstructing a moving 3D scene from a single moving camera is drastically under-determined: at each instant you see one view of a configuration that will never recur. The insight is that real scene motion is *low-dimensional* — a person walking is a handful of rigid parts, not a million independent point trajectories. Shape of Motion represents the scene as persistent 3D Gaussians that translate and rotate over time, and constrains each Gaussian's full trajectory to be a **linear combination of a small shared set of global SE(3) motion bases** (SE(3) = rigid rotation + translation). Per-Gaussian coefficients, the bases, and the canonical Gaussians are all optimized jointly against noisy off-the-shelf monocular depth and 2D point tracks. Because every Gaussian persists through the whole video, you get long-range 3D tracking for free — which is the property that made this the reference monocular-4D method for two years.

**9. pixelSplat** — *CVPR 2024†* — [2312.12337](https://arxiv.org/abs/2312.12337)
Per-scene optimization takes minutes; the obvious question is whether a network can just *predict* the splats. The hard part is scale ambiguity: from two images a network cannot tell a small nearby object from a large distant one, and naive regression collapses to a blurry mean depth. pixelSplat has the network output, per pixel, a *probability distribution over depth*, samples a depth from it, and places a Gaussian there — using a reparameterization trick so gradients flow through the sampling. This gives a real, multimodal 3D representation from an image pair in one forward pass, and it opened the entire feed-forward-splatting line (MVSplat, latentSplat, GS-LRM, NoPoSplat, Splatt3R, AnySplat, …).

**10. VGGT** — *CVPR 2025† (Best Paper)* — [2503.11651](https://arxiv.org/abs/2503.11651)
Classical 3D vision was a pipeline: feature matching → SfM → bundle adjustment → MVS, each stage with its own failure modes and its own optimization. VGGT replaces the whole thing with one large transformer that takes an arbitrary number of unposed images and directly emits camera intrinsics/extrinsics, depth maps, point maps and point tracks in a single forward pass. It is not conceptually deep — the contribution is that a big enough transformer trained on enough 3D-annotated data simply does this better and hundreds of times faster than optimization. Its practical impact on splatting is that "you need COLMAP first" stopped being true in 2025, and VGGT (and its successor Depth Anything 3) became the standard front end for pose-free feed-forward splatting.

**11. D4RT** — *CVPR 2026 Best Paper* — [project page](https://d4rt-paper.github.io/)
This is the most important single result of the survey period, and it is worth being precise about why. D4RT does for *dynamic* scenes what VGGT did for static ones, but with an architectural idea that goes further: instead of dense per-frame decoders for depth, tracks and pose, it encodes the video once into a fixed latent scene representation and exposes exactly one interface — **"where in 3D is the point that was at pixel (u,v) in frame t, at time t′?"** Every downstream task (depth, point cloud, 3D trajectory, camera intrinsics and extrinsics) is a different pattern of queries against that one interface. The payoff is efficiency, because you only decode what you ask for: at a 1 FPS target it produces 40,180 trajectories where SpatialTrackerV2 produces 2,290, and it is roughly 9× faster than VGGT at pose and two orders of magnitude faster than MegaSaM. It sets state of the art across 4D reconstruction and tracking benchmarks. **It contains no Gaussians, no radiance field, and no rendering.** The awards committee selected it out of 16,092 submissions (4,089 accepted). For anyone asking "is 3DGS being displaced", this is the concrete evidence that for *inferring* 4D structure the field has moved to query-based transformer pointmaps, and splats are increasingly a *rendering/delivery* format layered on top.

**12. HAC++** — *IEEE TPAMI 2025* — [2501.12255](https://arxiv.org/abs/2501.12255)
A raw 3DGS scene is hundreds of megabytes of unordered float32 attributes, which is fatal for web and mobile delivery. Standard entropy coding needs a *context model* — a way to predict the next symbol from already-decoded ones — and unordered point sets have no natural ordering to condition on. HAC's insight is to borrow structure from elsewhere: query a learned **hash grid** at each Scaffold-GS anchor's position, and use the mutual information between the anchor's attributes and the structured hash feature as the context for arithmetic coding. HAC++ adds intra-anchor context (attributes within one anchor predict each other), adaptive quantization, and masking of ineffective Gaussians. It reports **over 100× size reduction versus vanilla 3DGS while simultaneously improving fidelity** — the fidelity gain comes from the entropy loss acting as a regularizer that suppresses overfitting. This is the paper that made "a splat scene is a few megabytes" the default expectation.

**13. IRGS** — *CVPR 2025* — [2412.15867](https://arxiv.org/abs/2412.15867)
Gaussian inverse-rendering methods up to 2024 rendered a G-buffer (albedo, normal, roughness images) by rasterization and then did shading in screen space, approximating shadows and inter-reflections with precomputed or baked terms. That is fast but physically wrong, and the recovered materials end up entangled with the rasterizer. IRGS introduces **2D Gaussian ray tracing** and evaluates the *full rendering equation* — integrating incident radiance over the hemisphere at each shading point, with visibility and indirect radiance queried by tracing actual rays through the Gaussian field, differentiably, during optimization. This is the pivot that defines the 2026 inverse-rendering literature (RadioGS, PTIR-GS, IRGS++ all build on it); the trade-off is that you are now doing Monte Carlo integration inside the training loop, so sampling variance and cost become first-order engineering concerns.

**14. Triangle Splatting** — *3DV 2026* — [2505.19175](https://arxiv.org/abs/2505.19175)
The argument here is deliberately provocative: NeRF and 3DGS displaced triangles as the representation for photogrammetry, and that was a mistake, because triangles are what every GPU, game engine, and VR headset in the world is built to consume. The paper builds a differentiable renderer that optimizes triangles *directly* by rendering each one as a differentiable splat with a soft, learnable falloff toward its edges — combining the hardware-friendliness of triangles with the adaptive, unstructured density that made splatting work. Because triangles have hard edges while Gaussians have infinite smooth support, they capture creases and thin structures that Gaussians blur. On Mip-NeRF 360 it beats 3DGS, 2DGS and 3D Convex Splatting, and exceeds Zip-NeRF's perceptual quality on indoor scenes; the Garden scene renders at over **2,400 FPS at 1280×720 in an off-the-shelf mesh renderer** with no custom CUDA. Triangle Splatting+ (Sept 2025) then adds shared-vertex connectivity and forces the triangles opaque, so the output is a game-engine asset with zero post-processing.

---

## 4. Technical trends: what converged

**The 2026 dominant recipe for per-scene reconstruction.** If you started a project today, the consensus stack is:

1. **Front end:** a geometry foundation model (VGGT / Depth Anything 3 / MASt3R) for poses and an initial point cloud, replacing COLMAP. For large or casual captures, on-the-fly incremental reconstruction (TOG 44(4), 2025).
2. **Primitive:** Gaussians, but with the *understanding* that this is a choice. Beta kernels if you want fewer parameters; 2D disks or planar Gaussians if you want geometry; triangles if the output must enter a mesh pipeline.
3. **Densification:** MCMC/SGLD relocation with a fixed primitive budget, not clone/split heuristics. Structure-Aware Densification (SIGGRAPH 2026) adds a screen-space-extent vs local-texture-structure criterion for *when* to subdivide.
4. **Anti-aliasing:** Mip-Splatting's 3D + 2D filters. Non-negotiable.
5. **Structure:** anchors (Scaffold-GS) with octree LOD (Octree-GS) if the scene is large or you plan to compress.
6. **Regularization:** depth-distortion + normal-consistency (2DGS), multi-view photometric consistency (PGSR), and monocular normal/depth priors.
7. **Geometry extraction:** either mesh-in-the-loop (MILo) or a principled occupancy/vacancy field (GGGS, Gaussian Wrapping) — *not* TSDF fusion of blended depth maps, which is now regarded as the weak link.
8. **Delivery:** anchor-level entropy coding (HAC++ class) into SPZ / glTF `KHR_gaussian_splatting`.

**Six convergences worth naming explicitly.**

- **From heuristics to derivations.** Densification → SGLD (3DGS-MCMC). Depth heuristics → median transmittance of a stochastic solid (GGGS). Affine projection → unscented transform (3DGUT). Post-hoc meshing → in-the-loop differentiable meshing (MILo). The field is systematically replacing 2023's engineering hacks with principled equivalents, and in each case the principled version also performs better.
- **The primitive stopped being sacred.** 3DGS's real contribution, in retrospect, was the *differentiable tile rasterizer plus adaptive density control*, not the Gaussian. Once that was clear, the Gaussian became one option among many, and 2025–26 systematically explored the space: bounded-support kernels, signed kernels, convexes, voxels, foam, triangles.
- **The renderer stopped being sacred either.** Rasterization-with-global-sort is being unbundled: ray tracing (3DGRT, Radiant Foam), hybrid raster/ray (3DGUT), stochastic rasterization without sorting (StochasticSplats), and exact volumetric integration (EVER).
- **Pose-free is the default.** In 2024, "requires COLMAP poses" was an assumption; in 2026 it is a limitation. NoPoSplat, AnySplat, RayZer, YoNoSplat, ZipSplat and on-the-fly reconstruction all take unposed images.
- **Predict-then-refine is the settled architecture for sparse-view.** Not feed-forward *or* optimization, but feed-forward *then* seconds of optimization — with 2026 work (ForeSplat, iSplat, GIFSplat, Diff3R) making the network aware of the refinement it will receive, or folding refinement into the forward pass as learned recurrence.
- **Generative priors are now load-bearing, not decorative.** Difix3D+, VidSplat (SIGGRAPH 2026), GIFSplat, C4G and the entire relighting literature use diffusion models to supply information the observations do not contain. This is the single biggest methodological change since 2024, and it makes "reconstruction" and "generation" difficult to separate — with real consequences for evaluation.

**What has *not* converged.** Inverse rendering (every group has a different material model and a different notion of what "correct albedo" means); dynamic-scene benchmarks (D-NeRF is saturated and synthetic, DyCheck is tiny, and there is no agreed monocular-4D benchmark); and evaluation protocols generally, which two 2026 papers now argue are actively misleading.

---

## 5. Benchmark numbers

### Surface reconstruction — DTU, mean Chamfer distance over 15 scans (mm, lower better)

All figures below are from **Table 1 of GGGS** ([2601.17835](https://arxiv.org/abs/2601.17835)), which evaluates all Gaussian methods at half resolution under one protocol. This is a single-source table; see the caveat that follows.

| Method | Type | DTU CD ↓ | Optimization time |
|---|---|---|---|
| NeRF | implicit | 1.49 | >12 h |
| VolSDF | implicit | 0.86 | >12 h |
| NeuS | implicit | 0.84 | >12 h |
| Neuralangelo | implicit | 0.61 | >12 h |
| 3DGS | explicit | 1.96 | 7.8 min |
| 2DGS | explicit | 0.80 | 11.3 min |
| GOF | explicit | 0.74 | 52 min |
| 3DGSR | explicit | 0.70 | — |
| RaDe-GS | explicit | 0.68 | 8.2 min |
| GFSGS | explicit | 0.58 | 16.8 min |
| PGSR | explicit | 0.52 | 30.5 min |
| GeoSVR | sparse voxel | **0.47** | 53.3 min |
| **GGGS** | explicit | **0.47** | **15.0 min** |

### Surface reconstruction — Tanks & Temples, F1 over 6 scenes (higher better)

| Method | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | **Mean** | Time |
|---|---|---|---|---|---|---|---|---|
| NeuS | 0.29 | 0.29 | 0.17 | 0.83 | 0.24 | 0.45 | 0.38 | >24 h |
| Neuralangelo | 0.70 | 0.36 | 0.28 | 0.89 | 0.32 | 0.48 | 0.50 | >24 h |
| 2DGS | 0.36 | 0.23 | 0.13 | 0.44 | 0.16 | 0.26 | 0.30 | 15.5 min |
| GOF | 0.51 | 0.41 | 0.28 | 0.68 | 0.28 | 0.59 | 0.46 | 71.6 min |
| RaDe-GS | 0.49 | 0.36 | 0.27 | 0.72 | 0.27 | 0.61 | 0.45 | 12.1 min |
| PGSR | 0.66 | 0.44 | 0.20 | 0.81 | 0.33 | 0.66 | 0.52 | 42.9 min |
| GeoSVR | 0.68 | 0.49 | 0.34 | 0.83 | 0.37 | 0.66 | 0.56 | 66.4 min |
| **GGGS** | 0.70 | 0.56 | 0.38 | 0.81 | 0.42 | 0.70 | **0.60** | 32.1 min |

**Conflicting claim, unresolved:** GVGS ([2601.20331](https://arxiv.org/abs/2601.20331), Jan 2026) reports DTU mean CD **0.4933 mm** and claims best-in-class on 14 of 15 scans, "improving upon the best prior result by approximately 5%". That is *worse* than GGGS's 0.47 under GGGS's own table. Both use 2DGS-preprocessed DTU. I could not reconcile these from the papers alone — the likely cause is different mesh-extraction resolution or cropping, which is precisely the bias *From Blobs to Spokes* documents. **Do not treat either as definitive.**

**The protocol warning in full.** *From Blobs to Spokes* ([2604.07337](https://arxiv.org/abs/2604.07337)) states that the standard vertex-based Chamfer metric "is highly sensitive to tessellation density (e.g., extracting our meshes with 9 pivots instead of 2 artificially inflates scores)" and proposes two replacements — **Uniform Sampling** (sample points uniformly from the mesh surface, removing vertex-count bias) and **Virtual Scanning** (render depth maps from the original camera poses and back-project, mimicking the ground-truth acquisition). Under those protocols it finds PGSR is "an apparent outlier: its use of TSDF fusion and depth filtering yields non-watertight meshes whose holes coincide with those of the ground truth scan, an artifact of the acquisition process rather than genuine geometric quality." As a plug-in regularizer their normal-alignment loss + densification raises RaDe-GS's virtual-scan F1 from **0.39 → 0.48**.

### Novel view synthesis — Mip-NeRF 360, PSNR (outdoor / indoor)

From the appendix table of GGGS, one protocol across all methods.

| Method | Outdoor PSNR | Indoor PSNR |
|---|---|---|
| NeRF | 21.46 | 26.84 |
| Instant-NGP | 22.90 | 29.15 |
| Mip-NeRF 360 | 24.47 | 31.72 |
| 3DGS | 24.67 | 30.96 |
| SVRaster | 24.68 | 30.65 |
| **RayGaussX** | **25.24** | **32.43** |
| SuGaR | 22.93 | 29.43 |
| 2DGS | 24.34 | 30.40 |
| GOF | 24.82 | 30.79 |
| PGSR | 24.76 | 30.36 |
| GeoSVR | 24.83 | 30.46 |
| GGGS | 25.09 | 31.02 |

Public leaderboards show a top-of-table around **29.2–30.7 dB** scene-averaged on Mip-NeRF 360, dominated by MCMC-framework variants, but these mix evaluation protocols; NerfBaselines ([2406.17345](https://arxiv.org/abs/2406.17345)) demonstrated that simply switching between the two common image-downscaling conventions reorders the 3DGS-family rankings, and can make any 3DGS method except Scaffold-GS come last. **Cross-paper Mip-NeRF 360 PSNR comparisons at the 0.1–0.5 dB level are not meaningful.**

### Compression

| Method | Ratio vs vanilla 3DGS | Notes |
|---|---|---|
| LightGaussian | 15× | 200+ FPS; NeurIPS 2024 |
| HAC | ECCV 2024 baseline | anchor + hash-grid context |
| **HAC++** | **>100×** | *with improved fidelity*; >20× vs Scaffold-GS; TPAMI 2025 |
| FCGS | feed-forward | no per-scene optimization; ICLR 2025 |
| **SpeedyGS** | **up to 160×** | "negligible quality degradation"; 9× faster optimization; near-zero decode latency option; Jul 2026 preprint |
| SPZ format | ~90% smaller than PLY | Niantic Spatial, MIT licence; Khronos-endorsed |

The 2026 framing is explicit in SpeedyGS: "decoding with advanced 3DGS codecs still takes seconds, making them unsuitable for interactive applications." Ratio is solved; **latency and streamability are the live problems**, which is exactly what CAGS (SIGGRAPH 2026, +5–20 dB over prior adaptive streaming under bandwidth fluctuation) and LapisGS target.

### Standardization status (as of 5 Aug 2026)

- **Khronos glTF `KHR_gaussian_splatting`**: release candidate announced **3 Feb 2026**; ratification **targeted Q2 2026** — I could not confirm from a Khronos source whether ratification has actually completed. Splats are stored as point primitives (position, rotation, scale, opacity, SH), with graceful fallback to sparse point cloud. Contributors include Cesium/Bentley, Autodesk, Esri, Huawei, Niantic Spatial, NVIDIA, XGRIDS. Companion compression extensions proposed for Niantic's **SPZ** and Qualcomm's **L-GSC**. Cesium shipped support in CesiumJS 1.139 and Cesium for Unreal v2.23.0 in March 2026, ahead of ratification.
- **MPEG Gaussian Splat Coding (GSC)**: officially still at **Exploration** stage as *Part 45 – Gaussian splat coding: Implicit Neural Visual Representation*. Cross-group (WG 4 video, WG 5, WG 7 3D graphics/haptics), discussed at the 153rd MPEG meeting / 41st JVET meeting, 14–23 Jan 2026. Two tracks: short-term **I-3DGS** (minimal extensions to existing MPEG codecs for the INRIA 3DGS format) and long-term **A-3DGS** (alternative representations, including training-during-compression). **No Call for Proposals has been issued and no standard exists.**

---

## 6. Explicitly stated open problems and failure modes

*Attributed to the paper that states them. Quotations are from the cited works.*

**Optimization**

- **The "Blur Trap."** *Exploration Matters for Escaping the Blur Trap in 3DGS* (arXiv [2607.17965](https://arxiv.org/abs/2607.17965), Jul 2026) identifies a *gradient bias* in 3DGS's non-convex objective that traps optimization in blurry local optima, decomposed into a **Far-Side** and a **Near-Side** blur trap. The near-side variant arises because α-blending suppresses accumulated 2D gradients and so gates densification off exactly where splitting is needed. Their fix — random seeding and random splitting — is deliberately minimal, to show that *what 3DGS optimization is missing is explicit exploration*.
- **Two distinct uncertainty regimes conflict.** *Joint Modeling of Corruption-Driven and Information-Limited Uncertainty* (WACV 2026) names two systematic in-the-wild failure modes: **corruption-driven** (dynamic objects, exposure variation, motion blur → ghosting and spurious splats; aleatoric) and **information-limited** (insufficient multi-view coverage, small baselines, scene boundaries → floaters and over-smoothed surfaces; epistemic). It notes that treating them separately "can yield contradictory updates (e.g., pruning under-observed regions while retaining transient artifacts)".
- **Geometric "cheating."** MILo: NeRF and Gaussian methods "adjust their opacity and view-dependent colors independently of the geometry. This allows them to fit the training images more precisely, but often at the expense of geometric consistency. 'Cheating' leads to hallucinated structures such as floaters or cavities, which are particularly hard to resolve during mesh extraction."

**Evaluation**

- **The benchmark measures interpolation, not generalization.** *Mind the Gap* ([2607.01556](https://arxiv.org/abs/2607.01556)): every-8th-frame holdouts "have trained neighbors on both sides, so the metric measures near-trajectory interpolation rather than spatial generalization." The interpolation→extrapolation gap is **3–12 dB**, is "several times the differences typically reported between competing methods", flips at least two method rankings under multi-seed confirmation, and — importantly — persists across three representation families *including a non-Gaussian NeRF*, so it is a property of view coverage, not of Gaussians. Loss-side regularization gave "only marginal gains."
- **Surface metrics reward tessellation.** *From Blobs to Spokes*: vertex-based Chamfer "is highly sensitive to tessellation density"; PGSR's scores are "an artifact of the evaluation protocol rather than true reconstruction quality." It also notes T&T ground truth "is best to measure coarse geometry alignment, and fails to measure detail loyalty."
- **Protocol choice reorders leaderboards.** NerfBaselines ([2406.17345](https://arxiv.org/abs/2406.17345)): under the two common Mip-NeRF 360 downscaling conventions, "all 3DGS-based methods except for Scaffold-GS can become the worst."

**Surface reconstruction**

- GGGS's own limitations: median-depth binary search needs a fixed, wide initial depth interval, which is slow and "for large-scale scenes, the true median depth may even fall outside this preset range, hindering effective optimization"; and the Delaunay triangulation step "remains general-purpose", so thin or near-planar structures need very dense vertices.
- Gaussian Wrapping observes **surface erosion artifacts in GGGS** despite comparable Chamfer scores — i.e. the numbers do not capture the failure.

**Feed-forward**

- **The gap is real and persistent.** ForeSplat: "Across all these methods, the network is supervised to produce the final radiance field in one shot; when the prediction is used as-is, a persistent quality gap to a full per-scene 3DGS optimization remains."
- **Refinement can make things worse.** Diff3R ([2604.01030](https://arxiv.org/abs/2604.01030)) reports that on RealEstate10K with 4 unposed views, "AnySplat suffers from catastrophic overfitting (PSNR degrades by 0.2 dB) due to poor initialization" when test-time optimization is applied.
- **One-shot prediction cannot self-correct.** iSplat (CVPR 2026): one-shot feed-forward "establishes a brittle dependency on the initial depth scaffold, where any estimation errors are irreversibly propagated into the final Gaussian attributes… deprived of any mechanism for self-correction."
- **Pixel-aligned Gaussians are the wrong parameterization at scale.** AnchorSplat (CVPR 2026): "the number and distribution of their Gaussians remain tightly coupled to image resolution and viewpoint coverage, leading to a linear growth of total Gaussians according to the number of input views." ZipSplat makes the same point and notes per-pixel methods *degrade* as context views grow.
- **AnySplat's own stated failures:** "artifacts in challenging regions, such as skies, specular highlights, and thin structures; its reconstruction-based rendering loss may be less stable under dynamic scenes or varying illumination."

**Dynamic / 4D**

- **Pixel-wise feed-forward 4D duplicates and ghosts.** C4G: existing feed-forward methods "predict pixel-wise 3D Gaussians for each frame, suffering from duplicated Gaussians and view-dependent biases that lead to ghost artifacts and occlusion issues."
- **Monocular 4D remains fundamentally ill-posed.** Shape of Motion: "reconstruction of dynamic 3D Gaussians from a single video is severely ill-posed — at each point in time, [you observe only one view]." Every method's quality is bounded by the priors it imports (depth, tracks, diffusion), not by its optimizer.

**Inverse rendering**

- **Material decomposition is under-constrained, and per-primitive fitting makes it worse.** MaterialClusterGS ([2606.09018](https://arxiv.org/abs/2606.09018)): "this local fitting strategy makes material recovery highly under-constrained: shadows, indirect illumination, geometric errors, and visibility residuals can be absorbed into thousands of slightly different local material estimates." It adds that without shared material structure "editing one region does not propagate consistently to others of the same material, making per-primitive decompositions impractical for editing."
- **Materials recovered by rasterization are not valid for path tracing.** PTIR-GS: "naively integrating splatting-based inverted materials into a path-traced rendering pipeline leads to severe artifacts, revealing that the recovered properties remain tied to screen-space optimization rather than path-space transport."
- **Indirect radiance is queried from primitives that were never supervised for it.** RadioGS: existing methods "query indirect radiance from Gaussian primitives pre-trained for novel-view synthesis. However, these pre-trained Gaussian primitives are supervised only towards limited training viewpoints, thus lack supervision for modeling indirect radiances from unobserved views."
- **Three named obstacles for glossy/metallic.** IRGS++: (1) glossy, specular and metallic surfaces "require a more expressive material model and more reliable geometry priors; otherwise, specular highlights and reflective structures are easily misinterpreted during decomposition"; (2) "stratified sampling is inefficient for sharply peaked glossy integrands"; (3) "directly reusing [ray tracing's] training-time radiance queries for novel-illumination relighting is neither an efficient nor an appropriate relighting backend."

**Large scale**

- **Chunking is the problem, not the solution.** A LoD of Gaussians: partitioning "introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views." Its comparisons find CityGaussian "mainly optimized for aerial datasets, resulting in blurry reconstructions when other perspectives are introduced", and Hierarchical-3DGS suffering "chunking artifacts such as bleeding and ghosting when views at multiple scales are introduced."

**Compression / deployment**

- SpeedyGS: "decoding with advanced 3DGS codecs still takes seconds, making them unsuitable for interactive applications."
- MPEG/Ofinno readout: "The core question — how to provide efficient compression technologies to satisfy the various requirements and use cases in the current market — is still open," and the work "is officially in the Exploration stage — no formal standardization has been kicked off."

---

## 7. Under-explored gaps — tractable directions for a lab with 8–32 GPUs

*My analysis, not a claim from any paper. Ordered by (impact × tractability) / compute. All of these fit comfortably in 8–32 GPUs precisely because they are conceptual rather than scale-driven — do not compete with Google DeepMind on training a bigger 4D transformer.*

**1. A benchmark that measures what people actually claim. (Highest impact per GPU-hour in this list.)**
*Mind the Gap* found a 3–12 dB interpolation/extrapolation gap; *From Blobs to Spokes* found tessellation bias in Chamfer; NerfBaselines found protocol-dependent reordering. Nobody has combined these into a single, maintained, adopted benchmark. Concretely: take Mip-NeRF 360, T&T, DL3DV and ScanNet++; define spatial-sector holdouts alongside the standard ones; adopt uniform-sampling and virtual-scanning surface metrics; run every major method under multiple seeds and report confidence intervals; publish the harness. This is roughly 2–4 GPU-weeks of *compute* and mostly software engineering, and if adopted it becomes the most-cited artifact your lab produces this decade. The field is visibly ready for it — three independent 2026 papers converged on "our metrics are broken" without anyone fixing it.

**2. Extrapolation-aware regularization, evaluated on the extrapolation protocol.**
*Mind the Gap* reports that "loss-side regularization yields only marginal gains" against the extrapolation gap, and that the gap "tracks each view's angular distance to its nearest training view." That is a diagnosis, not a solution attempt. The obvious untried levers are *representational* rather than loss-based: opacity/scale priors that penalize primitives whose appearance is supported by a narrow angular wedge; explicit per-primitive angular-support statistics used to gate densification; or a generative prior applied specifically to high-angular-distance regions. A negative result here is publishable; a positive one is important. Fits on 8 GPUs.

**3. Cross-primitive controlled comparison under one optimizer.**
Gaussians, Beta kernels, Student-t, convexes, triangles, voxels and foam have each been benchmarked against *3DGS*, never against each other under a shared densification scheme, shared budget, shared anti-aliasing and shared evaluation protocol. Everyone in the field privately suspects most of the reported gains are optimizer and budget differences rather than primitive differences. Implement each primitive inside `gsplat` ([2409.06765](https://arxiv.org/abs/2409.06765)) with MCMC densification, sweep the primitive-count budget, and report quality/parameter and quality/FLOP Pareto fronts. Pure engineering, ~4 GPU-weeks, and it would settle a question the whole field is currently guessing at.

**4. Materials with shared structure, evaluated on relighting rather than reconstruction.**
MaterialClusterGS's palette idea (a small global set of BRDF prototypes plus a spatial assignment field) is exactly the right inductive bias for an under-constrained problem, and it is barely explored — one June 2026 preprint. Natural extensions: learn the palette size; hierarchical palettes (scene → object → part); cross-scene palettes transferred from a material library; and semantic priors from a 2D segmentation model to seed the assignment field. Critically, evaluate on *held-out illumination*, not on re-rendering under training illumination, which is what most inverse-rendering tables still do. Object-scale, so 8 GPUs is plenty.

**5. Compression targeting decode latency, not ratio.**
SpeedyGS names the real problem: seconds of decode time. But nobody has published a proper rate–*latency*–distortion study, and no one has looked seriously at *partial* or *progressive* decode (decode only the LOD and frustum you need, at interactive rates). Combining A LoD of Gaussians' out-of-core streaming machinery with HAC++-class entropy coding — so that the streamed unit is an entropy-coded LOD block — is an obvious, unclaimed combination with direct relevance to the glTF/MPEG standardization now in flight. Modest compute, high industrial relevance, and a natural path to standards-body contribution.

**6. Uncertainty-driven next-best-view for splatting.**
The WACV 2026 paper cleanly separates epistemic (coverage-limited) from aleatoric (corruption-driven) uncertainty in 3DGS, and *Mind the Gap* shows that angular distance to the nearest training view is a "zero-cost signal that also guides capture planning". Nobody has closed the loop: use the epistemic term to drive active capture, either in simulation or with a robot/drone, and show that N actively chosen views beat 2N passively captured ones. Highly tractable, directly useful, and connects the radiance-field literature to robotics.

**7. Test-time optimization as a learned operator (the honest version).**
ForeSplat, iSplat, GIFSplat and Diff3R are all circling the same idea in 2026, so this is competitive — but the *analysis* is missing. What is the actual Pareto frontier of (feed-forward inference time + K refinement steps) vs quality, across DL3DV / RE10K / ScanNet++ / Mip-NeRF 360, with modern backbones? Where exactly does the crossover with full per-scene optimization sit in 2026, and how does it move with view count? Diff3R's finding that AnySplat *degrades* under naive TTO suggests the frontier is non-monotonic and poorly understood. A careful empirical paper here would be widely cited and needs no new architecture.

**8. Dynamic-scene benchmark reform.**
D-NeRF is synthetic and saturated; DyCheck is small; every monocular-4D paper evaluates on a slightly different subset. Meanwhile D4RT-class query-based models and Gaussian-based 4D models are not even evaluated on the same axes (tracking metrics vs NVS metrics). A benchmark that scores *both* — 3D trajectory accuracy and novel-view quality, on the same casual monocular captures, with an extrapolation holdout — would define the field's next two years. Same argument as #1, applied to 4D.

**9. Non-Gaussian primitives for *surface* reconstruction.**
GeoSVR shows sparse voxels reach DTU 0.47 mm, matching the best Gaussian method. Triangle Splatting+ produces meshes directly. But nobody has combined GGGS's stochastic-solid formulation with a non-Gaussian kernel — the derivation is kernel-agnostic in principle (Deformable Beta Splatting already proved the analogous point for MCMC densification: "adjusting regularized opacity alone ensures distribution-preserved MCMC, independent of the splatting kernel type"). A bounded-support kernel with a closed-form vacancy field should give better thin structures than Gaussians *and* a principled geometric field. This is a well-scoped theory+implementation paper.

**Anti-recommendations.** Do not build a bigger feed-forward reconstruction transformer (VGGT/D4RT/DA3 scale is out of reach and the incumbents iterate fast). Do not build another "3DGS + application domain X" paper. Do not build another compression method reporting a bigger ratio at similar decode cost — that axis is saturated and about to be frozen by standards.

---

## 8. What is genuinely solved vs. not

| Task | Status (Aug 2026) |
|---|---|
| Dense multi-view NVS, static, bounded, interpolated views | **Solved.** ~30 dB, real-time, minutes to train, standardized delivery format. |
| Dense multi-view NVS, *extrapolated* views | **Not solved.** 3–12 dB penalty; nobody targets it. |
| Anti-aliasing / LOD | **Solved in principle** (Mip-Splatting, Octree-GS); still engineering work at city scale. |
| Object-scale surface reconstruction (DTU-like) | **Effectively solved.** 0.47 mm in 15 min beats 12-hour implicit methods. |
| Scene-scale watertight meshes with thin structures | **Nearly solved** (MILo, Gaussian Wrapping) but the metrics used to claim it are unreliable. |
| Multi-view dynamic capture (studio rigs) | **Solved.** Spacetime Gaussians class, real-time playback, streaming standardized enough for SIGGRAPH 2026 systems papers. |
| Monocular casual-video 4D — *geometry and tracking* | **Largely solved feed-forward** by D4RT/VGGT-class models. |
| Monocular casual-video 4D — *photorealistic free-viewpoint rendering* | **Not solved.** Disocclusions require generation, not reconstruction. |
| Sparse-view (2–8 images) reconstruction | **Solved to "good enough"** with feed-forward + refinement; a persistent gap to per-scene optimization remains. |
| Pose-free reconstruction | **Solved** for most captures. |
| Compression ratio | **Solved** (100–160×). |
| Interactive decode / streaming | **Open**, and now the active problem. |
| Relighting under novel illumination (plausible-looking) | **Mostly solved** via diffusion priors. |
| Physically correct geometry/material/lighting decomposition | **Not solved, and not close.** Under-constrained by construction. |
| City-scale reconstruction | **Solved on one consumer GPU** (150M+ Gaussians, out-of-core) as of SIGGRAPH 2026. |

---

## 9. Verification caveats

**Read this section before citing anything above.**

**Venues I could NOT confirm from a primary source today (marked † in the tables).** For these, the arXiv metadata carried no `journal_ref` and no venue statement in the author `comment`, and I did not separately open the conference's accepted-papers listing. They come from my prior knowledge and should be re-checked before you cite them:
NeRF (ECCV 2020), Instant-NGP (SIGGRAPH 2022), Mip-NeRF 360 (CVPR 2022), TensoRF (ECCV 2022), Zip-NeRF (ICCV 2023), Mip-Splatting (CVPR 2024), Scaffold-GS (CVPR 2024), 3DGS-MCMC (NeurIPS 2024), SuGaR (CVPR 2024), 2DGS (SIGGRAPH 2024), GOF (SIGGRAPH Asia 2024), PGSR (IEEE TVCG), Gaussian Surfels (SIGGRAPH 2024), NeuS (NeurIPS 2021), VolSDF (NeurIPS 2021), Neuralangelo (CVPR 2023), D-NeRF (CVPR 2021), HyperNeRF (SIGGRAPH Asia 2021), Deformable 3DGS (CVPR 2024), 4D Gaussian Splatting (CVPR 2024), pixelSplat (CVPR 2024), MVSplat (ECCV 2024), latentSplat (ECCV 2024), DUSt3R (CVPR 2024), GS-LRM (ECCV 2024), NoPoSplat (ICLR 2025), VGGT (CVPR 2025 Best Paper), Compact 3D Gaussian Representation (CVPR 2024), Self-Organizing Gaussian Grids (ECCV 2024), TensoIR (CVPR 2023), GS-IR (CVPR 2024), Relightable 3D Gaussians (ECCV 2024), GaussianShader (CVPR 2024), NeRF-Casting (SIGGRAPH Asia 2024), IntrinsicAnything (ECCV 2024), Neural Gaffer (NeurIPS 2024), DiffusionLight (CVPR 2024), 3DGS.zip (survey venue).

**Papers whose venue is genuinely unknown to me.** Octree-GS, Triangle Splatting+, RayGaussX, MoSca, AnySplat, Depth Anything 3, LapisGS, UniRelight, CLM, Dynamic Gaussian Marbles, Splatt3R, EVER, Radiant Foam, 3D Student Splatting and Scooping, RaDe-GS. Note that **RaDe-GS appears in the SIGGRAPH 2026 technical-papers schedule** despite its arXiv posting being June 2024 — I could not determine whether that is the original publication or a later journal presentation, so I left it unlabelled.

**Secondary-source venues (flagged ‡).** Long-LRM at ICCV 2025 comes from the Paper Digest ICCV 2025 listing. RayZer as ICCV 2025 Best Student Paper Honorable Mention comes from the IEEE TCPAMI ICCV awards table. Neither was confirmed against the ICCV proceedings.

**D4RT has no arXiv ID that I could find.** I verified the paper, its authors, its award and its claims from the CVPR 2026 official Best Papers announcement, the Oxford Engineering news page, the project page (`d4rt-paper.github.io`), and a paper-notes summary. If an arXiv preprint exists, I did not locate it, and I have deliberately not guessed an identifier. The efficiency figures (40,180 vs 2,290 trajectories at 1 FPS; 9× faster than VGGT at pose; ~100× faster than MegaSaM; 18–300× faster tracking) come from a third-party paper summary, **not** from the paper's own text, which I did not read in full — treat them as approximate.

**Numbers I did not independently verify.**
- All DTU/T&T/Mip-NeRF 360 figures in §5 are transcribed from **one paper's tables** (GGGS). Author-run baselines are systematically optimistic for the authoring method. The GVGS discrepancy (0.49 mm claimed as best-in-class, vs GGGS's 0.47) is unresolved and is itself evidence that these numbers are not protocol-stable.
- HAC++'s ">100× with improved fidelity" and SpeedyGS's "160×" are author claims from their abstracts. I did not inspect their rate-distortion curves, and the two use different baselines and datasets, so **they are not directly comparable**.
- SPZ's "up to 90% smaller than PLY" is a Khronos/Niantic marketing claim, not a measured result from a peer-reviewed source.
- CAGS's "+5–20 dB under fluctuating bandwidth" is against adaptive-streaming baselines under the authors' own network conditions, not a like-for-like codec comparison.
- Triangle Splatting's "2,400 FPS at 1280×720" is for the single Garden scene on unspecified hardware.
- Mip-NeRF 360 leaderboard figures (29.2–30.7 dB) come from an aggregator site mirroring PapersWithCode. Given NerfBaselines' protocol findings, I would not cite these at all.

**Things I looked for and could not establish.**
- Whether Khronos `KHR_gaussian_splatting` has actually been **ratified** as of 5 Aug 2026. The Q2 2026 target has passed; the specification repository still says "TODO: Add known implementations before final ratification", which suggests it had not completed at the time that text was written, but I found no announcement either way.
- The **complete CVPR 2026 accepted-paper list** for radiance fields. I sampled it via search and confirmed individual papers (GIFSplat, iSplat, AnchorSplat, MoVieS) through openaccess.thecvf.com, but I did not enumerate all 4,089 accepted papers, so this survey's CVPR 2026 coverage is a sample, not a census. A systematic arXiv sweep of Aug 2025 – Aug 2026 was attempted and abandoned after repeated HTTP 429 rate-limits from the arXiv API.
- **NeurIPS 2026** and **ECCV 2026** outcomes. Given the date, NeurIPS 2026 decisions may not be public and ECCV 2026 (an even year) should be relevant, but I found nothing and did not include either.
- Whether anything at SIGGRAPH Asia 2025 beyond MILo is significant for this survey. I confirmed MILo's venue precisely, but did not review the full SIGGRAPH Asia 2025 programme; the SIGGRAPH 2026 programme I did review (via the Ke-Sen Huang and Keenan Crane listings, which are community-maintained mirrors rather than ACM's official list).
- **Bolt3D, latentSplat, Splatt3R, DUSt3R, Dynamic Gaussian Marbles, HyperNeRF, Deformable 3DGS, VolSDF, Gaussian Surfels, TensoIR, GS-IR, GaussianShader, IntrinsicAnything, Neural Gaffer, DiffusionLight, LapisGS** and several others are listed for completeness of the narrative but I did not read their full texts during this survey; their one-line descriptions rest on abstracts and prior familiarity.

**One more thing I could not do.** The brief asked about **RGB↔X** (Zeng et al., SIGGRAPH 2024) specifically. Searching arXiv for that title returned only unrelated RGB-X sensor-fusion papers, and I could not locate a primary record for it in the time available. It is therefore **absent from this survey**, not judged unimportant. Same for **Compact3D** — I include *Compact 3D Gaussian Representation* (Lee et al., [2311.13681](https://arxiv.org/abs/2311.13681)), which may or may not be the paper the brief meant; there is a separately-named "Compact3D" (Navaneet et al.) on K-means vector quantization that I did not verify.

**Structural bias to be aware of.** The Aug 2025 – Aug 2026 portion of this survey is weighted toward papers that (a) rank well in web search and (b) have HTML on arXiv. Recent preprints are over-represented relative to their eventual importance, and non-English-language or non-arXiv venues are absent. Roughly a third of the 2026 entries are unrefereed preprints, marked as such.
