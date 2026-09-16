# 3D Geometry Foundation Models, Feed-Forward Reconstruction, SfM, SLAM, and Monocular Depth
### Literature survey, 2024 – August 2026 (strong emphasis on Aug 2025 – Aug 2026)
*Compiled 5 August 2026. Every arXiv ID and venue below was read off a primary source (arXiv abstract page, CVF Open Access, OpenReview, NeurIPS/ICLR proceedings, or the authors' own repo/project page) during this session, except where flagged with ⚠. Items I could not confirm are collected in §8 and are never given invented IDs.*

---

## 0. How to read this report

Four things changed between August 2025 and August 2026, and they organize everything below.

1. **Feed-forward reconstruction started behaving like a foundation model.** VGGT-Ω (CVPR 2026 Best Paper Finalist) is the first paper in this literature to demonstrate clean power-law scaling: 3D point error falls monotonically from 0.107 → 0.046 as parameters go 0.2B → 10B, and 0.275 → 0.073 as data goes 2K → 2M sequences. Before this, "scale it up" was an assumption; now it is a measured curve.

2. **The quadratic-attention wall got broken three separate ways in one conference cycle.** VGGT and π³ cost O(N²) in the number of input views. At CVPR 2026, ZipMap and VGG-T³ both replace the growing key-value cache with a *fixed-size* state learned by test-time training (linear time, 700+ frames in under 10 s / 1k images in 54 s), while AMB3R replaces the 2D-grid backend with a sparse voxel representation and gets O(1) per-frame visual odometry. Independently, MERG3R, S-VGGT, FastVGGT, TurboVGGT, and AVGGT attack the same wall training-free.

3. **The COLMAP question got a real answer, and it is "no, but the hybrid wins."** GLUEMAP (Pan, Schönberger, Pollefeys — CVPR 2026, released as `github.com/colmap/gluemap`) is the definitive measurement. On ETH3D, π³ scores AUC@1 of **13.2** against SIFT+GLOMAP's **45.6**; a classical pipeline is more than 3× more accurate where classical pipelines work at all. On CO3Dv2 the ordering flips completely (π³ 47.3 vs SIFT 37.0 at AUC@3). On LaMAR every feed-forward method runs out of memory. Details and full tables in §4.

4. **Dynamic scenes stopped being a separate research area.** VGGT-Ω handles motion with no motion detector at all (per-frame depth + camera instead of one rigid world pointmap), and D4RT (CVPR 2026 Best Paper) reframes the whole problem as querying a video encoding for "where is pixel (u,v) from frame t_src, at time t_tgt, in camera frame t_cam?"

**A note on the field's own taxonomy.** There is a survey, *Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT* (arXiv [2507.08448](https://arxiv.org/abs/2507.08448), Jul 2025), useful as a map of the 2024–mid-2025 period but published before every important 2026 result. I lean on it only for taxonomy, not for numbers. ⚠ It appears to be a non-peer-reviewed preprint that has also been posted to a journal called JAICS; treat its authority accordingly.

**Jargon defined once, used throughout.**

| Term | Meaning |
|---|---|
| **Pointmap** | A per-pixel 3D coordinate map: for image of size H×W, an H×W×3 array where entry (i,j) is the 3D point that pixel (i,j) sees. Same shape as an RGB image, so a standard dense-prediction network can output it. This is DUSt3R's central invention. |
| **Epipolar geometry** | The classical constraint that a point seen in one camera must lie along a specific *line* in a second camera, determined entirely by the relative pose of the two cameras. Classical stereo is built on it; feed-forward pointmap models never write it down. |
| **Bundle adjustment (BA)** | The large nonlinear least-squares problem at the end of every classical pipeline: jointly nudge all camera poses and all 3D points to minimize *reprojection error* (pixel distance between where a 3D point projects and where it was actually observed). Accurate but slow, and it needs many points visible in many views to be well-conditioned. |
| **Scale ambiguity** | From images alone you cannot tell a dollhouse from a house. Monocular and two-view reconstruction recover geometry only up to an unknown global scale factor; "metric" methods claim to fix that factor in real-world meters. |
| **Affine-invariant / scale-invariant** | Prediction targets deliberately defined only up to an unknown affine or scale transform, so the loss does not punish the network for an ambiguity it cannot resolve. |
| **ATE (Absolute Trajectory Error)** | The standard SLAM metric: align your estimated camera trajectory to ground truth (usually a rigid or similarity transform), then report RMSE of position error, in cm or m. Lower is better. |
| **AUC@X°** | Area under the pose-accuracy recall curve up to an angular error threshold of X degrees, where error is the max of relative rotation and relative translation error over all image pairs. Higher is better. **Tight thresholds (AUC@1, AUC@3) measure accuracy; loose thresholds (AUC@30) mostly measure whether you got the scene roughly right at all.** Conflating these is the single most common way this literature's numbers get misread. |
| **δ1.25 / δ1.03** | Fraction of pixels whose predicted depth is within 25% (or 3%) of ground truth. Higher is better. δ1.03 is a much harsher test and is where multi-view methods separate from monocular ones. |
| **AbsRel / rel** | Mean absolute relative depth error, \|d_pred − d_gt\|/d_gt. Lower is better. |
| **Chamfer distance** | Symmetric average nearest-neighbor distance between two point clouds. The usual reconstruction-quality number. |
| **Sim(3) / SL(4)** | Groups of transformations. Sim(3) = rotation + translation + uniform scale (7 DoF). SL(4) = general projective transformations of 3D space (15 DoF), which is what you actually need when camera intrinsics are unknown. |
| **Permutation equivariance** | If you shuffle the input images, the outputs shuffle identically and nothing else changes. VGGT does *not* have this (it pins image 1 as the world frame); π³ does. |

---

## 1. Table of the ~35 most important papers

### 1A. The DUSt3R lineage — pointmap regression and its descendants

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 1 | **DUSt3R** (Wang, Leroy, Cabon, Chidlovskii, Revaud) | **CVPR 2024**, pp. 20697–20709; arXiv [2312.14132](https://arxiv.org/abs/2312.14132), Dec 2023 | Founding paper: recast two-view reconstruction as *pointmap regression* from uncalibrated, unposed image pairs, plus a global alignment optimization for >2 views. Unifies monocular and binocular cases. | [arXiv](https://arxiv.org/abs/2312.14132) · [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf) |
| 2 | **MASt3R** (Leroy, Cabon, Revaud) | **ECCV 2024**, LNCS pp. 71–91; arXiv [2406.09756](https://arxiv.org/abs/2406.09756), 14 Jun 2024 | Adds a dense local-feature head trained with an InfoNCE matching loss, plus a *fast reciprocal matching* scheme; outputs metric pointmaps. +30% absolute VCRE AUC on Map-free localization. | [arXiv](https://arxiv.org/abs/2406.09756) · [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09080.pdf) · [code](https://github.com/naver/mast3r) |
| 3 | **Spann3R** (H. Wang, Agapito) | arXiv [2408.16061](https://arxiv.org/abs/2408.16061), Aug 2024 (⚠ 3DV 2025 venue not confirmed this session) | First to remove global alignment: an *external spatial memory* lets the network emit pointmaps already in a global frame, one frame at a time, in real time. | [arXiv](https://arxiv.org/abs/2408.16061) · [project](https://hengyiwang.github.io/projects/spanner) |
| 4 | **MASt3R-SfM** (Duisterhof, Zust, Weinzaepfel, Leroy, Cabon, Revaud) | **3DV 2025 Oral**; arXiv [2409.19152](https://arxiv.org/abs/2409.19152), Sep 2024 | Training-free SfM on frozen MASt3R: hijacks the encoder for image retrieval → quasi-linear complexity; two gradient-descent stages (3D matching loss, then 2D reprojection); **no RANSAC**, and works with *zero* camera motion. | [arXiv](https://arxiv.org/abs/2409.19152) · [OpenReview](https://openreview.net/forum?id=5uw1GRBFoT) |
| 5 | **MASt3R-SLAM** (Murai, Dexheimer, Davison) | **CVPR 2025**, pp. 16695–16705; arXiv [2412.12392](https://arxiv.org/abs/2412.12392), 16 Dec 2024 | First real-time SLAM built bottom-up on a two-view reconstruction prior. Iterative projective matching, ray-error tracking, pointmap fusion, second-order Sim(3) backend. **15 FPS**, no calibration needed. | [arXiv](https://arxiv.org/abs/2412.12392) · [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html) |
| 6 | **CUT3R** (Q. Wang, Zhang, Holynski, Efros, Kanazawa) | **CVPR 2025**, pp. 10510–10522; arXiv [2501.12387](https://arxiv.org/abs/2501.12387) ⚠ (ID via EmergentMind, not arXiv directly) | Stateful recurrent transformer with a *persistent state* updated per frame; metric-scale online pointmaps; can also probe *virtual, unobserved* views to hallucinate unseen geometry. ~17 FPS vs <1 FPS for DUSt3R+global alignment. | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html) |
| 7 | **Fast3R** (Yang, Sax, Liang, Henaff, Tang, Cao, Chai, Meier, Feiszli) | **CVPR 2025**, pp. 21924–21935 (⚠ arXiv ID not verified) | All-to-all attention over *all* frames simultaneously — no image ordering, no sequential dependency, 1000+ images in one forward pass, parallelizable across devices. 251 FPS in their throughput table. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Fast3R_Towards_3D_Reconstruction_of_1000_Images_in_One_Forward_CVPR_2025_paper.pdf) |
| 8 | **MUSt3R** (Cabon, Stoffl, Antsfeld, Csurka, Chidlovskii, Revaud, Leroy) | **CVPR 2025 Highlight** (⚠ arXiv ID not verified) | Makes DUSt3R *symmetric* and multi-view with a multi-layer memory; offline and online modes from one model. ~5× lighter and an order of magnitude faster than DUSt3R at comparable quality. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Cabon_MUSt3R_Multi-view_Network_for_Stereo_3D_Reconstruction_CVPR_2025_paper.pdf) |
| 9 | **Light3R-SfM** (Elflein et al.) | **CVPR 2025** (⚠ arXiv ID not verified) | Replaces global optimization with a *learnable latent alignment module*. 200 images in **33 s** vs MASt3R-SfM's ~27 min — a >49× speedup — at better pose accuracy than Spann3R. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Elflein_Light3R-SfM_Towards_Feed-forward_Structure-from-Motion_CVPR_2025_paper.pdf) |
| 10 | **VGGT** (J. Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny) | **CVPR 2025 — Best Paper Award**, pp. 5294–5306; arXiv [2503.11651](https://arxiv.org/abs/2503.11651), 14 Mar 2025 | One transformer, 24 layers alternating frame-wise and global attention, predicts cameras + depth + pointmaps + 3D tracks in <1 s with **no optimization at all**. 1.26B params. RealEstate10K AUC@30 85.3 (93.5 with optional BA). | [arXiv](https://arxiv.org/abs/2503.11651) · [code](https://github.com/facebookresearch/vggt) · [project](https://vgg-t.github.io/) |
| 11 | **VGGT-SLAM** (Maggio, Lim, Carlone) | **NeurIPS 2025**; arXiv [2505.12549](https://arxiv.org/abs/2505.12549), May 2025 | Aligns VGGT submaps on the **SL(4)** manifold (15-DoF projective), arguing that Sim(3) is provably insufficient for uncalibrated cameras. SL(4) optimization upstreamed into GTSAM Aug 2025. | [arXiv](https://arxiv.org/abs/2505.12549) · [code](https://github.com/MIT-SPARK/VGGT-SLAM) |
| 12 | **StreamVGGT** (Zhuo, Zheng, Guo, Wu, Zhou, Lu) | arXiv [2507.11539](https://arxiv.org/abs/2507.11539), 15 Jul 2025 | Causal/autoregressive VGGT: temporal causal attention + cached historical KV as implicit memory, distilled from bidirectional VGGT. Lets you import FlashAttention and other LLM kernels. | [arXiv](https://arxiv.org/abs/2507.11539) · [code](https://github.com/wzzheng/StreamVGGT) |
| 13 | **π³ (Pi3)** (Y. Wang, Zhou, Zhu, Chang, Zhou, Li, Chen, Pang, Shen, He) | **ICLR 2026**; arXiv [2507.13347](https://arxiv.org/abs/2507.13347), 17 Jul 2025 (v3 7 Mar 2026) | Removes the reference-view inductive bias entirely: fully permutation-equivariant, predicting affine-invariant poses + scale-invariant *local* pointmaps. Smaller (959M vs 1.26B), faster (57.4 vs 43.2 FPS on KITTI), and more accurate than VGGT. | [arXiv](https://arxiv.org/abs/2507.13347) · [code](https://github.com/yyfz/Pi3) |
| 14 | **VGGT-Long** (Deng, Ti, Xu, Yang, Xie) | arXiv [2507.16443](https://arxiv.org/abs/2507.16443), Jul 2025 | "Chunk it, loop it, align it": overlapping chunks + robust chunk alignment + lightweight loop closure pushes VGGT to **kilometer-scale** KITTI/Waymo sequences with no retraining, no calibration, no BA backend. | [arXiv](https://arxiv.org/abs/2507.16443) · [code](https://github.com/DengKaiCQ/VGGT-Long) |
| 15 | **MapAnything** (Keetha, Müller, Schönberger, Porzi, et al. — Meta + CMU) | **3DV 2026**; arXiv [2509.13414](https://arxiv.org/abs/2509.13414), Sep 2025 | *Factored* geometry: separate ray maps, depth, poses, and one global metric scale factor, with any of intrinsics/poses/depth/partial reconstructions optionally provided as input. One model, 12+ tasks, Apache 2.0. | [arXiv](https://arxiv.org/abs/2509.13414) · [project](https://map-anything.github.io/) |
| 16 | **TTT3R** (X. Chen, Y. Chen, Xiu, Geiger, A. Chen) | **ICLR 2026**; arXiv [2509.26645](https://arxiv.org/abs/2509.26645), Sep 2025 | Reinterprets CUT3R's state update as test-time training, then derives a closed-form *per-token adaptive learning rate* from alignment confidence. **Training-free**, zero new parameters, 2× better global pose on long sequences at 20 FPS / 6 GB. | [arXiv](https://arxiv.org/abs/2509.26645) · [project](https://rover-xingyu.github.io/TTT3R/) |
| 17 | **Depth Anything 3 (DA3)** (Lin, S. Chen, Liew, D. Chen, Z. Li, Shi, Feng, Kang — ByteDance Seed) | arXiv [2511.10647](https://arxiv.org/abs/2511.10647), 13 Nov 2025; **ICLR 2026** (academic version) | Radical minimalism: one *plain* DINOv2 transformer, no architectural specialization, one depth-ray prediction target instead of multi-task heads. **+35.7% camera pose accuracy and +23.6% geometric accuracy over VGGT** on their DA3-BENCH. Public data only. | [arXiv](https://arxiv.org/abs/2511.10647) · [project](https://depth-anything-3.github.io/) |
| 18 | **AMB3R** (H. Wang, Agapito) | **CVPR 2026 Highlight**, pp. 14612–14625; arXiv [2511.20343](https://arxiv.org/abs/2511.20343), Nov 2025 | Bolts a *sparse voxel backend* (space-filling-curve serialization + transformer, injected back via zero-convolution) onto a frozen VGGT. First feed-forward VO to beat optimization-based SLAM. Trained in **~80 H100 GPU-hours**. | [arXiv](https://arxiv.org/abs/2511.20343) · [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_AMB3R_Accurate_Feed-forward_Metric-scale_3D_Reconstruction_with_Backend_CVPR_2026_paper.pdf) · [code](https://github.com/HengyiWang/amb3r) |
| 19 | **D4RT** (C. Zhang, Le Moing, Koppula, Rocco, et al. — Google DeepMind/Oxford/UCL) | **CVPR 2026 — Best Paper Award**, pp. 7382–7392; arXiv [2512.08924](https://arxiv.org/abs/2512.08924), Dec 2025 | Encode the video once into a Global Scene Representation; then *query* a lightweight decoder for the 3D position of any pixel, at any time, in any camera frame. One interface yields depth, point clouds, cameras, and dense 4D tracks. **200+ FPS** pose (9× VGGT, 100× MegaSaM). | [arXiv](https://arxiv.org/abs/2512.08924) |
| 20 | **VGGT-SLAM 2.0** (Maggio, Carlone) | **RSS 2026**; arXiv [2601.19887](https://arxiv.org/abs/2601.19887), Jan 2026 | Fixes VGGT-SLAM's own 15-DoF drift and planar degeneracy with a constrained factor graph; discovers that one of VGGT's attention layers works as a free loop-closure verifier. 8.4 FPS on RTX 3090, real-time on a Jetson Thor aboard a ground robot, **23% less pose error than v1** on TUM. | [arXiv](https://arxiv.org/abs/2601.19887) · [code](https://github.com/MIT-SPARK/VGGT-SLAM) |
| 21 | **VGGT-Ω** (J. Wang, Chen, S. Zhang, Karaev, Schönberger, Labatut, Bojanowski, Novotny, Vedaldi, Rupprecht) | **CVPR 2026 Best Paper Finalist**, pp. 21486–21499; arXiv [2605.15195](https://arxiv.org/abs/2605.15195) | The scaling paper. *Register attention* + lighter dense head + fewer heads cut training memory to ~30% of VGGT, enabling 15× more supervised data (4M sequences) and 18M unlabeled videos. Clean power laws to 10B params. Sintel camera AUC@3° **22.5 → 40.0**. | [arXiv](https://arxiv.org/abs/2605.15195) · [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VGGT-ohm_CVPR_2026_paper.html) · [project](http://vggt-omega.github.io/) |
| 22 | **ZipMap** (Jin, Wu, T. Zhang, Gao, Barron, Snavely, Hołyński) | **CVPR 2026**, pp. 21748–21759 (⚠ appeared in the Paper Digest listing under the title *FRM: Linear-Time 3D Reconstruction Via Test-Time Training*) | "Zips" an entire image collection into a compact hidden scene state using test-time-training layers — **linear time, bidirectional**, 700+ frames in <10 s on one H100, >20× faster than VGGT, at equal or better accuracy. | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Jin_ZipMap_Linear-Time_Stateful_3D_Reconstruction_via_Test-Time_Training_CVPR_2026_paper.html) |
| 23 | **VGG-T³** (Elflein, R. Li, Agostinho, Gojcic, Leal-Taixé, Q. Zhou, Ošep — NVIDIA) | **CVPR 2026**, pp. 36464–36474 | Independently reaches the same conclusion as ZipMap: distill the *varying-length* KV representation of scene geometry into a *fixed-size MLP* via test-time training. 1k images in **54 s**, 11.6× speedup, linear scaling, and the MLP doubles as a queryable localization index. | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Elflein_VGG-T3_Offline_Feed-Forward_3D_Reconstruction_at_Scale_CVPR_2026_paper.html) |
| 24 | **Any4D** (Karhade, Keetha, Y. Zhang, Gupta, Sharma, Scherer, Ramanan) | **CVPR 2026** | The MapAnything design carried into 4D: scalable multi-view transformer for metric-scale dense feed-forward 4D reconstruction. ⚠ Numbers not read this session. | — |
| 25 | **OmniVGGT** (Peng et al.) | **CVPR 2026** | *GeoAdapter*: zero-initialized convolutions inject depth and camera intrinsics/extrinsics into a frozen spatial foundation model, plus stochastic multimodal fusion so any subset of modalities works at inference — at VGGT-comparable speed. | [CVPR virtual](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026) |

### 1B. Efficiency and scaling work on the VGGT/π³ backbone (2026)

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 26 | **FastVGGT** (Shen, Z. Zhang, Qu, Cao) | arXiv 2509.02560, Sep 2025 (⚠ ID via GLUEMAP's reference list, not verified directly) | Training-free acceleration of VGGT via token merging. | — |
| 27 | **SAIL-Recon** (Deng, H. Li, Xie, Ren, Q. Zhang, Tan, Guo) | arXiv 2508.17972, Aug 2025 (⚠ same caveat) | Large-scale SfM by augmenting scene regression with localization. | — |
| 28 | **MERG3R** | arXiv [2603.02351](https://arxiv.org/abs/2603.02351), Mar 2026 | Training-free, model-agnostic divide-and-conquer: reorder/partition unordered images into overlapping geometrically diverse subsets, reconstruct independently, merge by global alignment + confidence-weighted BA. Works with VGGT or π³. | [alphaXiv](https://www.alphaxiv.org/abs/2603.02351) |
| 29 | **S-VGGT** | arXiv [2603.17625](https://arxiv.org/abs/2603.17625), 18 Mar 2026 | Attacks *structural* rather than token-level redundancy: build a dense scene graph, softly assign frames to subscenes that share a common reference frame, giving a "parallel geometric bridge" so subscenes need no explicit alignment. Orthogonal to token merging. | [arXiv](https://arxiv.org/abs/2603.17625) |
| 30 | **TurboVGGT** | arXiv [2605.14315](https://arxiv.org/abs/2605.14315), 2026 | Adaptive alternating attention: learns *per-frame, per-layer* sparsity ratios and representative tokens rather than one fixed sparsity ratio. | [arXiv](https://arxiv.org/abs/2605.14315) |
| 31 | **AVGGT** | **CVPR 2026** | The most useful diagnostic result in this cluster: a layer-role analysis showing that in VGGT and π³, *early global layers form no meaningful correspondences, middle layers do the cross-view alignment, and the last layers contribute only minor refinements* — which licenses a training-free two-step acceleration. | [CVPR virtual](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026) |

### 1C. Monocular and video depth / geometry

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 32 | **Depth Anything V1** (Yang, Kang, Huang, Xu, Feng, Zhao) | **CVPR 2024**, pp. 10371–10381 | Scaled relative monocular depth by pseudo-labelling ~62M unlabeled images with a teacher; the paper that made "depth foundation model" a normal phrase. | — |
| 33 | **Depth Anything V2** (same team) | **NeurIPS 2024**; arXiv 2406.09414 (⚠ ID from the official repo's BibTeX, not the arXiv page) | Synthetic-only teacher → pseudo-labels for millions of real images → student. Much sharper fine detail than V1, far faster than diffusion depth. 25M–1.3B param family. | [code](https://github.com/DepthAnything/Depth-Anything-V2) |
| 34 | **Depth Pro** (Bochkovskii, Delaunoy, Germain, Santos, et al. — Apple) | **ICLR 2025**; arXiv [2410.02073](https://arxiv.org/abs/2410.02073), Oct 2024 | Zero-shot *metric* depth with jointly predicted focal length, 2.25 MP in 0.3 s on a V100. Its real contribution is boundary accuracy: it dominates all baselines on thin structures (hair, fur) by a large margin, and introduces boundary-recall metrics to measure it. | [arXiv](https://arxiv.org/abs/2410.02073) · [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/bc8b2058fd96978a4146f18298cb2d39-Paper-Conference.pdf) |
| 35 | **Video Depth Anything (VDA)** (S. Chen, Guo, Z. Li, et al. — ByteDance) | **CVPR 2025 Highlight**; arXiv [2501.12375](https://arxiv.org/abs/2501.12375), Jan 2025 | Replaces DAv2's head with a spatio-temporal head + a *temporal depth gradient* consistency loss, plus keyframe-based long-video inference. Arbitrarily long videos; KITTI δ1 **0.944** vs DAv2-L 0.815 and DepthCrafter 0.753, at 67 ms vs DepthCrafter's 910 ms. Smallest variant runs 30 FPS. | [arXiv](https://arxiv.org/abs/2501.12375) · [code](https://github.com/DepthAnything/Video-Depth-Anything) |
| 36 | **UniDepth V2** (Piccinelli et al.) | 2025 journal version of UniDepth (CVPR 2024) (⚠ arXiv ID and exact journal not verified) | Explicitly models camera geometry by predicting ray directions in spherical coordinates, giving consistent metric depth across domains. Their own paper concedes camera-parameter estimation (ρA) improves only marginally, "indicating that the limited diversity of training cameras remains a challenge." | [MPG PDF](https://pure.mpg.de/rest/items/item_3643939_2/component/file_3643940/content) |
| 37 | **UniK3D** (Piccinelli et al.) | 2025 (⚠ venue/ID not verified) | Camera-agnostic representation via spherical harmonics — works on non-pinhole cameras including fisheye, which no other model in this table handles. | — |
| 38 | **MoGe-2** (R. Wang, S. Xu, Dong, Deng, Xiang, Lv, Sun, Tong, Yang — Microsoft) | **NeurIPS 2025**; arXiv [2507.02546](https://arxiv.org/abs/2507.02546), 3 Jul 2025 | Cleanest treatment of the metric-vs-relative question: keep MoGe's affine-invariant pointmap for shape, add a *separate* CLS-token-conditioned MLP scale head for metric scale, so metric supervision cannot corrupt relative geometry. Plus a data-refinement pipeline using sharp synthetic labels to fix blurred real data. 35M–331M params. | [arXiv](https://arxiv.org/abs/2507.02546) · [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/336572db3e99930814d6b328d4220cb6-Paper-Conference.pdf) · [code](https://github.com/microsoft/MoGe) |
| 39 | **MetricAnything** (Ma, J. Yang, Di, X. Zhang, Cui, H. Li, Yan, W. Chen) | arXiv [2601.22054](https://arxiv.org/abs/2601.22054), Jan 2026 | The DA3-of-metric-depth. *Sparse Metric Prompt* (randomly masked depth maps) as a universal interface that decouples spatial reasoning from sensor/camera bias; ~20M image–depth pairs across **10,000+ camera models**; then distilled to a prompt-free student. Claims the **first clear scaling trend for metric depth**. | [arXiv](https://arxiv.org/abs/2601.22054) · [project](https://metric-anything.github.io/metric-anything-io/) · [code](https://github.com/metric-anything/metric-anything) |
| 40 | **Stabilizing Streaming Video Geometry via Dynamic Feature Normalization (DyFN)** (Lyu, M. Liu, X. Wu, R. Wang, Y. Huang, Sun, Shi, Qi) | **CVPR 2026** | Diagnoses streaming-geometry flicker down to its cause — "fluctuations in latent feature statistics, whose mean and variance directly determine the predicted depth's scale and shift" — and fixes it with a lightweight causal recurrent normalization module. | — |
| 41 | **Unlocking the Power of Critical Factors for 3D Visual Geometry Estimation** (G. Xu, Geng, H. Zheng, Yin, Y. Sun, H. Chen, Shen) | **CVPR 2026** | Reports a finding worth internalizing before choosing a model: "per-frame visual geometry estimation approaches typically exhibit weaker multi-frame consistency but demonstrate **superior per-frame accuracy** compared to multi-frame algorithms." | — |

### 1D. Structure-from-Motion, pose, and localization

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 42 | **COLMAP / SfM Revisited** (Schönberger, Frahm) | **CVPR 2016**, pp. 4104–4113 | The incremental SfM system that is still the accuracy reference and the source of most of this field's training labels. | — |
| 43 | **VGGSfM** (J. Wang, Karaev, Rupprecht, Novotny) | **CVPR 2024**, pp. 21686–21697 | End-to-end differentiable SfM with learned tracks fed into BA. Limited to tens of images. | — |
| 44 | **Detector-Free SfM** (He, Sun, Y. Wang, Peng, Huang, Bao, et al.) | **CVPR 2024**, pp. 21594–21603 | Removes keypoint detection from the SfM front end entirely. | — |
| 45 | **ACE-Zero / ACE0** (Brachmann, Wynn, S. Chen, Cavallari, Monszpart, Turmukhambetov, Prisacariu — Niantic) | **ECCV 2024 Oral** | Reinterprets incremental SfM as *iterated application of a visual relocalizer*, using scene-coordinate regression to build an implicit neural scene representation from unposed images. No pose priors, no sequential input, thousands of images. | [project](https://nianticlabs.github.io/acezero/) · [code](https://github.com/nianticlabs/acezero) |
| 46 | **GLOMAP / Global SfM Revisited** (Pan, Baráth, Pollefeys, Schönberger) | **ECCV 2024**; arXiv [2407.20219](https://arxiv.org/abs/2407.20219), Jul 2024 | Replaces ill-posed translation averaging with *joint camera-and-point global positioning*. On-par or superior to COLMAP at **1–2 orders of magnitude** less runtime. On LaMAR LIN (36k images): 90% recall @1m in 5.5 h, vs COLMAP's ~50% recall in >7 days. Now merged into COLMAP as the `global` mapper. | [arXiv](https://arxiv.org/abs/2407.20219) · [project](https://lpanaf.github.io/eccv24_glomap/) |
| 47 | **Reloc3r** (Dong, S. Wang, S. Liu, Cai, Fan, Kannala, Y. Yang) | **CVPR 2025**, pp. 16739–16752; arXiv [2412.08376](https://arxiv.org/abs/2412.08376) | Relative-pose regression network trained on ~8M posed image pairs + a minimalist motion-averaging module. Real-time, generalizes to novel scenes, no per-scene training. | [arXiv](https://arxiv.org/abs/2412.08376) · [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Dong_Reloc3r_Large-Scale_Training_of_Relative_Camera_Pose_Regression_for_Generalizable_CVPR_2025_paper.html) |
| 48 | **MP-SfM** (Pataki, Sarlin, Schönberger, Pollefeys) | **CVPR 2025**, pp. 21891–21901 | Injects *monocular surface priors* (depth + normals) into an incremental SfM pipeline, specifically targeting low-overlap scenes. Still the accuracy leader on the hardest low-overlap benchmark (SMERF) at the tightest thresholds. | — |
| 49 | **Doppelgangers++** (Xiangli, Cai, H. Chen, Byrne, Snavely) | **CVPR 2025**, pp. 27166–27175 | Learned visual disambiguation using geometric 3D features — detects symmetric/repeated structures that collapse reconstructions. A load-bearing component of GLUEMAP. | — |
| 50 | **GLUEMAP / Global SfM Meets Feedforward Reconstruction** (Pan, Schönberger, Pollefeys) | **CVPR 2026**, pp. 21880–21890 | The definitive classical-vs-learned analysis *and* the winning hybrid: retrieval + Doppelganger filtering → local star-graph π³ inference → rotation and similarity averaging → BA augmented with "virtual tracks" from feed-forward depth. Scales to tens of thousands of images; fits on a 24 GB RTX 4090. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Global_Structure-from-Motion_Meets_Feedforward_Reconstruction_CVPR_2026_paper.pdf) · [code](https://github.com/colmap/gluemap) |

### 1E. Matching / correspondence

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 51 | **LoFTR** | **CVPR 2021** | Detector-free semi-dense matching via self/cross attention on learned features. Made matching work in low-texture regions. | — |
| 52 | **DKM** | **CVPR 2023**; arXiv [2202.00667](https://arxiv.org/abs/2202.00667) | First *pixel-dense* matcher to beat sparse and semi-dense methods on pose estimation, via a feature pyramid + kernelized regression + dense certainty estimation. | [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00667) |
| 53 | **RoMa** (Edstedt, Q. Sun, Bökman, Wadenbäck, Felsberg) | **CVPR 2024** | Frozen DINOv2 for coarse matching + a specialized ConvNet for fine features; regression-by-classification coarse loss, robust regression fine loss. +36% on the hard WxBS benchmark. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf) |
| 54 | **UFM** | **NeurIPS 2025** (⚠ ID not verified) | Unifies wide-baseline dense matching with optical flow; much faster than RoMa, but RoMa v2 reports it is worse under extreme appearance change and worse at sub-pixel precision. | — |
| 55 | **RoMa v2** (Edstedt, Nordström, Y. Zhang, Bökman, Astermark, Larsson, Heyden, Kahl, Wadenbäck, Felsberg) | arXiv [2511.15706](https://arxiv.org/abs/2511.15706), Nov 2025 (v3 Jul 2026) ⚠ venue unconfirmed | Frozen **DINOv3** coarse matcher predicting at stride 4, decoupled two-stage matching-then-refinement, per-pixel 2×2 precision matrices, custom CUDA local-correlation kernel, and a far more diverse 10-dataset training mix (aerial + small-baseline). Also the source of the cleanest evidence that dense matchers still beat feed-forward reconstructors at two-view pose (§6). | [arXiv](https://arxiv.org/abs/2511.15706) · [code](https://github.com/Parskatt/RoMaV2) |
| 56 | **MV-RoMa** (JongMin Lee, Seungyeop Kang, Sungjoo Yoo) | **CVPR 2026 Highlight** | Fixes the fragmented-track problem: a source image matched jointly against *multiple* co-visible targets, using pairwise results as a geometric prior, then fed as high-quality tracks into SfM. The natural bridge between §1E and §1D. | [CVPR virtual](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026) |

### 1F. SLAM and streaming reconstruction (non-DUSt3R lineage)

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 57 | **DROID-SLAM** (Teed, Deng) | **NeurIPS 2021** (⚠ ID not verified) | Learned dense bundle adjustment via recurrent iterative updates of camera pose and pixelwise depth. Still the robustness reference that every 2024–2026 SLAM paper compares against. | — |
| 58 | **SplaTAM** (Keetha, Karhade, Jatavallabhula, et al.) | **CVPR 2024** | 3D-Gaussian-splatting SLAM: explicit, differentiable, silhouette-aware map. Replica ATE 0.36 cm (vs Point-SLAM 0.52), TUM-RGBD 5.48 cm (vs 8.92), ScanNet++ 1.2 cm where ORB-SLAM3 and Point-SLAM fail entirely. | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf) |
| 59 | **MonoGS / Photo-SLAM / GS-SLAM** | CVPR 2024 (⚠ IDs and exact venues not individually verified) | The 2024 Gaussian-splatting SLAM cohort. Useful chiefly as the baseline set every later paper reports against. | — |
| 60 | **DROID-Splat** (Homeyer et al.) | **ICCV 2025 Workshop (NeuSLAM)**; arXiv 2411.17660 (⚠ ID via ar5iv mirror) | Combines DROID-SLAM's end-to-end tracker with 3DGS rendering and a monocular depth prior. Its honest headline: "traditional and end-to-end tracking systems are still the best once perfect supervision is missing." | [ICCVW PDF](https://openaccess.thecvf.com/content/ICCV2025W/NeuSLAM/papers/Homeyer_DROID-Splat_Combining_end-to-end_SLAM_with_3D_Gaussian_Splatting_ICCVW_2025_paper.pdf) |
| 61 | **SEGS-SLAM** | arXiv [2501.05242](https://arxiv.org/abs/2501.05242), Jan 2025 | Structure-enhanced 3DGS SLAM with appearance embedding; monocular/stereo/RGB-D. Its comparison table is the most useful single ATE reference across the 2024 GS-SLAM cohort. | [arXiv](https://arxiv.org/pdf/2501.05242) |
| 62 | **FOUND-IT** | arXiv [2605.25371](https://arxiv.org/abs/2605.25371), May 2026 ⚠ (ID from the VGGT-SLAM repo, abstract not read) | Builds a 3D scene graph on top of VGGT-SLAM 2.0 — an early example of semantic/symbolic structure layered on a geometry foundation model. | — |
| 63 | **Flow4DGS-SLAM** (Yunsong Wang, Gim Hee Lee) | **CVPR 2026** | Optical-flow-guided 4D Gaussian splatting SLAM for dynamic scenes. | — |
| 64 | **STAC** | **CVPR 2026** | Spatio-temporally aware KV-cache compression for streaming causal 3D reconstruction — the specific fix for StreamVGGT's linear cache growth, motivated by measured spatio-temporal sparsity in the attention maps. | — |

### 1G. Evaluation infrastructure and independent audits (mostly 2026)

| Name | Venue + date | What it measures | Link |
|---|---|---|---|
| **DA3-BENCH** | With Depth Anything 3, Nov 2025 | Camera pose + any-view geometry + visual rendering, over 5 datasets. The first benchmark designed for the *any-view* setting rather than a fixed task. | [HF](https://huggingface.co/datasets/depth-anything/DA3-BENCH) |
| **UAVFF3D** | arXiv [2605.17942](https://arxiv.org/abs/2605.17942), 2026 | Geometry-aware UAV benchmark: 170k+ real and 370k+ synthetic images, with a controlled HFOV-vs-height subset to isolate projection-geometry ambiguity. Reports domain adaptation cutting Ray Error up to 84.2%, Pose ATE up to 76.0%, Chamfer up to 41.1%, and the oblique-nadir rotation gap up to 90.7%. | [HF papers](https://huggingface.co/papers/2605.17942) · [code](https://github.com/yanxian-ll/UAVFF3D) |
| **DUSt3R/MASt3R/VGGT on photogrammetric aerial blocks** | arXiv [2507.14798](https://arxiv.org/abs/2507.14798), Jul 2025 | Independent overlap sweep (10%–70%) on aerial blocks. Numbers in §4. | [alphaXiv](https://www.alphaxiv.org/overview/2507.14798) |
| **Uncertainty Quality of VGGT on DTU** | arXiv [2606.16479](https://arxiv.org/abs/2606.16479), Jun 2026 | Audits VGGT's *confidence* outputs, and flags a reproducibility problem: they cannot reproduce VGGT's reported 0.38 mm DTU Chamfer and suspect an undisclosed BA post-processing step. | [arXiv](https://arxiv.org/html/2606.16479v1) |
| **GeoT2V-Bench** | arXiv [2606.24829](https://arxiv.org/abs/2606.24829), Jun 2026 | Uses VGGT + Gaussian splatting as a *measuring instrument* on video generators. Notable for the methodological warning that "single-score evaluation is misleading." | [arXiv](https://arxiv.org/abs/2606.24829v1) |
| **HERI3D** | 2026 (⚠ venue not verified) | UAV cultural-heritage comparison of COLMAP vs NeuS vs 3DGS vs VGGT. Conclusion: "traditional photogrammetry implemented in COLMAP remains the most reliable method for geometric accuracy." | [PDF](https://run.unl.pt/server/api/core/bitstreams/be3b1790-7d43-45b3-b234-ec726718f17e/content) |
| **Understanding multi-view transformers** | arXiv [2510.24907](https://arxiv.org/abs/2510.24907), Oct 2025 | Probes and visualizes 3D representations from residual connections in a DUSt3R variant — one of very few mechanistic-interpretability efforts here. | [arXiv](https://arxiv.org/html/2510.24907v1) · [code](https://github.com/JulienGaubil/und3rstand) |

---

## 2. Plain-language explanations of the 13 most important papers

### 2.1 DUSt3R — the idea everything else is built on

**Problem.** To reconstruct 3D from photos, classical pipelines first need the camera parameters — where each camera was and what its lens was like. Getting those is a fragile multi-stage process (detect keypoints, match them, estimate two-view geometry, triangulate, bundle-adjust), and every stage can fail.

**Trick.** Skip all of it. Feed two images to a transformer and have it directly output, for every pixel, the 3D coordinate that pixel is looking at — both images' outputs expressed in the *first* image's coordinate frame. That array is a **pointmap**. Because a pointmap has the same shape as an image, you can predict it with an ordinary dense-prediction head, and because both images' pointmaps live in one frame, the relative camera pose is implicit in the output and can be read out afterwards.

**Why it mattered.** It collapsed the pipeline into one network and, crucially, dropped the perspective-camera constraints. A classical method *cannot* recover geometry from two images with almost no overlap, because there is nothing to triangulate. A network that has seen millions of scenes can guess, because it knows what rooms and buildings look like. That is a different kind of capability, not just a faster version of the old one.

### 2.2 MASt3R — making the pointmap model good at matching too

**Problem.** DUSt3R was remarkably robust to extreme viewpoint changes but its *correspondences* were imprecise — fine for a rough reconstruction, not fine for sub-pixel geometry.

**Trick.** Add a second head that emits a dense local feature descriptor per pixel, trained with an InfoNCE contrastive loss (the same objective used in self-supervised representation learning) so that pixels seeing the same 3D point get similar descriptors. Then, because comparing every pixel to every pixel is quadratic and unusably slow, introduce a *fast reciprocal matching* scheme that converges to mutual nearest neighbors in far fewer operations, with theoretical guarantees.

**Why it mattered.** It made "matching is a 3D problem, not a 2D problem" a defensible claim rather than a slogan, beating dedicated matchers by 30 absolute points of VCRE AUC on the very hard Map-free localization benchmark. It also became the backbone that MASt3R-SfM and MASt3R-SLAM were built on, which is how the pointmap idea reached practitioners.

### 2.3 MASt3R-SLAM — putting a 3D prior inside a real-time loop

**Problem.** SLAM (simultaneous localization and mapping) means building a map and tracking your position in it *while the data arrives*. Classical SLAM needs the camera to be calibrated, which is a hardware chore, and it is brittle on casual handheld video.

**Trick.** Use MASt3R's two-view prediction as the single primitive for everything — tracking, mapping, relocalization — and then engineer around the fact that MASt3R is offline and slow. The front end does *local filtering* of pointmaps (fusing predictions to reduce noise, in the spirit of filtering-based SLAM), and the back end does second-order global optimization over Sim(3). The system assumes essentially nothing about the camera except that all light rays pass through one center — so it works with unknown and even *time-varying* lenses.

**Why it mattered.** 15 FPS, no calibration, dense globally consistent maps. It is the paper that convinced the SLAM community that geometry foundation models were not just a photogrammetry curiosity, and essentially every DUSt3R-prior SLAM system since (VGGT-SLAM, AMB3R-VO) is answering it.

### 2.4 VGGT — one network, all outputs, no optimization

**Problem.** DUSt3R and MASt3R only handle *pairs*. For N images you run O(N²) pairs and then solve a global alignment optimization, which is slow and can fail.

**Trick.** Build one transformer that takes all N images at once. Patchify each with a DINO encoder, then alternate two kinds of attention for 24 layers: **frame attention** (tokens attend only within their own image, preserving per-image detail) and **global attention** (tokens attend across all images, integrating multi-view information). Then read out everything with separate heads: cameras, depth, pointmaps, and 3D point tracks.

**The subtle finding worth remembering.** Several outputs are mathematically redundant — you can derive cameras from pointmaps, and depth from pointmaps plus cameras. Training the network to predict all of them anyway *improves* accuracy. But at inference, the most accurate 3D points come not from the pointmap head but from combining the separately-predicted depth and camera heads. The redundancy helps learning and hurts nothing, but you have to know which output to actually use.

**Why it mattered.** It won CVPR 2025's Best Paper because it removed test-time optimization entirely while *beating* optimization-based methods: RealEstate10K pose AUC@30 of 85.3 against DUSt3R's 67.7 and MASt3R's 76.4, in ~0.2 s versus ~7–9 s. It also became the field's substrate — over 1,500 citations within a year, and roughly half the 2026 papers in this survey are either accelerating it, extending it, or built on frozen VGGT weights.

### 2.5 π³ — deleting the reference frame

**Problem.** Every model so far designates one image as the coordinate origin (DUSt3R's reference view; VGGT pins image 1 to identity). This is an *inductive bias with a failure mode*: if that particular image happens to be blurry, low-texture, or an outlier, the whole reconstruction is anchored to a bad estimate. It also means the output depends on input ordering, which is philosophically ugly and practically annoying.

**Trick.** Predict quantities that don't need an origin. Each frame gets an **affine-invariant camera pose** (defined only up to a global affine transform) and a **scale-invariant local pointmap** expressed in *that frame's own* camera coordinates. Remove every order-dependent component — notably frame-index positional embeddings — so the architecture is genuinely permutation-equivariant: shuffle the inputs, the outputs shuffle identically.

**Why it mattered.** It is the rare paper where removing a design choice improves everything at once. π³ is *smaller* than VGGT (959M vs 1.26B), *faster* (57.4 vs 43.2 FPS on KITTI), and more accurate on essentially every axis: RealEstate10K AUC@30 85.90 vs 77.62, DTU accuracy 1.198 cm vs 1.338 cm, Sintel ATE 0.074 vs 0.167. Order-dependence in the output drops from a standard deviation of 0.033 to 0.003. GLUEMAP, having evaluated all the alternatives, chose π³ as its feed-forward backbone.

### 2.6 MapAnything — factoring the geometry instead of predicting one blob

**Problem.** A pointmap entangles everything: camera pose, camera intrinsics, depth, and scale are all fused into one XYZ prediction. That means you cannot easily *tell* the model something you already know. If you have calibrated cameras, or a depth sensor, or poses from an IMU, a pointmap model has no input slot for that information.

**Trick.** Factor the output into four separable pieces: (1) per-pixel **ray directions** (which encode intrinsics, including non-pinhole ones), (2) **ray depth** along those rays, (3) **camera pose** per view, and (4) a single global **metric scale factor** predicted from one dedicated scale token. Each of these can also be an *input*: during training, randomly provide subsets of intrinsics, poses, depths, and metric scales, so at inference the model handles any combination — or none.

**Why it mattered.** It converts a family of specialist models into one model with a flexible interface, covering uncalibrated SfM, calibrated MVS, monocular depth, camera localization, and depth completion. Their ablation is the important part: training one universal model for 12+ tasks with the compute of two bespoke models beats three bespoke models. And on images-only two-view reconstruction it beats VGGT outright (pointmap rel 0.12 vs 0.20, pose AUC@5 51.9 vs 34.2). Released Apache 2.0, which is why it has become a standard baseline.

### 2.7 Depth Anything 3 — the minimalism result

**Problem.** By late 2025 the architectures had accumulated a lot of machinery: multiple specialized heads, bespoke attention patterns, multi-task losses, staged training. Was any of it necessary?

**Trick.** Strip it down. One *vanilla* DINOv2 transformer with no architectural modification, one added input-adaptive cross-view self-attention mechanism, and a single prediction target: a **depth-ray** representation (per-view depth plus per-pixel rays) rather than a world-frame pointmap. Camera parameters, when known, are just extra tokens. Train with a teacher-student scheme where a teacher generates pseudo-depth for real data.

**Why it mattered.** It works better than the complicated thing. On their own any-view benchmark DA3 beats VGGT by an average 35.7% in camera pose accuracy and 23.6% in geometric accuracy, and beats Depth Anything 2 at plain monocular depth — trained exclusively on public academic datasets. Two independent lessons follow: (a) the *prediction target* matters more than the architecture, and predicting per-view depth + rays instead of a world pointmap is a better target; (b) you should be suspicious of complexity in this literature. VGGT-Ω, arriving six months later from the VGGT authors themselves, adopted the same simplifications.

### 2.8 VGGT-Ω — the scaling law, and dynamic scenes for free

**Problem.** Everyone assumed these models would scale. Nobody had measured it, because training memory was the binding constraint — the high-resolution convolutional layers in DPT heads eat activation memory that neither FSDP nor gradient checkpointing can eliminate.

**Trick.** Three changes that together cut training memory by ~70%. (1) **Register attention**: VGGT already appends per-frame register tokens; in a quarter of the global attention layers, restrict cross-frame information exchange to *only* the registers, which then redistribute locally in frame-attention layers. This makes the registers a deliberate bottleneck that aggregates whole-scene information, and — since global attention maps turn out to be very sparse — it saves ~23% FLOPs and 16% memory with no measurable accuracy loss. (2) Replace the memory-hungry high-resolution convolutions with one MLP plus a pixel-shuffle upsampler. (3) Keep multi-task *losses* but delete the redundant dense heads, retaining only one dense depth head and one sparse camera head.

Then spend that memory on data: a VLM-filtered, COLMAP-and-matching-verified annotation pipeline turns ~40M internet-style videos into 0.8M high-quality annotated sequences (a third of them dynamic), combined into 4M total sequences — 15× VGGT — plus 18M unlabeled videos used with a DINO-style momentum teacher-student protocol.

**The dynamic-scenes result is elegant and worth stating separately.** VGGT was trained on static scenes and fails when objects move. VGGT-Ω fixes this with *no motion detector*: instead of predicting one rigid world-frame pointmap (which is simply incoherent when things move), predict per-frame depth plus per-frame camera pose, and let motion be implicit. A moving object gets placed correctly within each frame without the model ever committing to a single 3D position for it across time.

**Why it mattered.** Point error falls monotonically 0.107 → 0.046 from 0.2B to 10B parameters, and 0.275 → 0.073 from 2K to 2M sequences, on power-law-looking curves. Sintel camera AUC@3° goes 22.5 → 40.0 (+77% relative over MegaSaM) while running 50× faster. And the registers turn out to encode transferable spatial information: frozen, they lift a vision-language-action robot policy's LIBERO success rate from 97.1% to 98.5% and can be aligned to natural-language scene descriptions. That last result reframes reconstruction as a *pretraining proxy task* for spatial understanding generally.

### 2.9 D4RT — reconstruction as a query, not a decode

**Problem.** Dynamic 4D reconstruction means knowing where every point is in 3D at every instant, while separating object motion from camera motion. The obvious approach — decode a dense output for every frame — is expensive, and needs a separate decoder head for each output type (depth, tracks, cameras).

**Trick.** Encode the whole video once into a fixed **Global Scene Representation**. Then use one small decoder that answers a single flexible question, specified by five numbers: given pixel (u,v) in source frame t_src, where is that point in 3D at target time t_tgt, expressed in the camera frame of t_cam? (A small embedding of the local 9×9 RGB patch around the query sharpens fine detail considerably.) Because t_tgt and t_cam are *independent*, motion in time is fully disentangled from motion of the viewpoint. Every output falls out by varying the query: sweep t_tgt for a 3D track; sweep all pixels into one frame for a point cloud; set source = target = camera for a depth map. Camera pose needs no learned head at all — decode the same grid of points in two reference frames and solve for the rigid transform between them in closed form with Umeyama's algorithm.

**Why it mattered.** Queries decode independently, so they parallelize trivially and training can sample a subset. The result is 200+ FPS pose estimation on an A100 — 9× VGGT, 100× the optimization-based MegaSaM — while being *more* accurate (Sintel point-cloud L1 of 0.768 vs π³'s 1.139 and VGGT's 1.582; RealEstate10K pose AUC 83.5 vs π³'s 78.7), and 18–300× faster on tracking workloads. It won CVPR 2026's Best Paper.

**How it differs from VGGT-Ω, since both won 2026 honors for dynamic reconstruction.** VGGT-Ω is a reconstruction *backbone* betting on scale and reusable representations; it reconstructs each frame's geometry but does not put moving points into correspondence across time — it is motion-aware but not a tracker. D4RT is a unified 4D *engine* that does exactly that correspondence. Conversely VGGT-Ω accepts unordered image collections, where D4RT needs a video. They have not been benchmarked against each other and are best read as complementary bets.

### 2.10 GLUEMAP — the honest accounting, and the synthesis

**Problem.** The literature had accumulated two mutually contradictory folk claims: "feed-forward models have replaced SfM" and "classical SfM is still much more accurate." Nobody had measured both across a spectrum of scene difficulties.

**Trick, part one: measurement.** Characterize scenes by *view-graph radius* (the minimum number of hops needed to pass information between any two images — a clean proxy for structural scene complexity) and *density*. Then two things emerge. Feed-forward accuracy degrades consistently as radius grows, because global attention over a large-radius graph makes it hard to distinguish relevant from irrelevant information, and symmetric structure makes it worse. And more perversely: for VGGT and MapAnything, **adding more images of the same scene makes pose estimation worse**, the opposite of how any optimization-based pipeline behaves.

**Trick, part two: synthesis.** Use each method where it is strong. (1) Build a sparse view graph by image retrieval plus Doppelgangers++ filtering to remove symmetric/non-overlapping pairs. (2) Decompose it into *local star graphs* centered on each image and run π³ independently on each — this bounds attention to relevant images (fixing the radius problem), fits in memory, and batches in parallel (fixing the scale problem). (3) Merge the local reconstructions with classical global motion averaging — intrinsics averaging, rotation averaging, then similarity averaging where each star needs only *one* scale, which is far better conditioned than estimating per-edge scales from noisy triangulation. (4) Run bundle adjustment augmented with **virtual tracks**: synthetic multi-view correspondences generated by reprojecting sampled pixels through the feed-forward depth maps. This is the key move — it lets BA be well-conditioned even when there are not enough real feature tracks to constrain it, which is exactly the low-texture / low-overlap case where classical SfM dies.

**Why it mattered.** It wins across the board where every individual method wins somewhere and fails somewhere (ETH3D AUC@1: 53.0 uncalibrated / 74.0 calibrated, vs SIFT's 45.6 and π³'s 13.2; on LaMAR where all feed-forward methods OOM, GLUEMAP averages AUC@3 of 26.4 vs ALIKED+LightGlue's 10.9). It scales to tens of thousands of images and fits on a 24 GB consumer GPU. And it is shipped inside the COLMAP organization, which means it is the answer the classical-SfM community has accepted.

### 2.11 AMB3R — the backend the pointmap was missing

**Problem.** AMB3R opens with the sharpest conceptual criticism of the pointmap paradigm anyone has written. Pointmaps assume a one-to-one mapping from pixels to 3D points. But *the whole point of multi-view geometry is that the mapping is many-to-one* — multiple pixels see the same 3D point, and that is what correspondence means. Classical representations (TSDF grids, voxel hashes, NeRFs) all enforce **spatial compactness**: a given 3D coordinate has a unique property, so multiple observations of one scene point are forced to fuse. A network operating on 2D grids has no such mechanism.

**Trick.** Give it one. Keep VGGT frozen as a front end predicting features and geometry, add a light scale head for metric depth, and add a *backend*: fuse the front end's features into **sparse voxels**, serialize them into a 1D sequence along a Hilbert space-filling curve (which preserves 3D spatial locality), process that sequence with a transformer, unserialize, interpolate back to pixels via KNN, and inject the result into the frozen front-end decoder through zero-convolution layers (the ControlNet trick, which lets you reuse pretrained weights and confidence functions without disturbing them). Total training cost: **~80 H100 GPU-hours.**

Then two training-free pipelines. AMB3R-VO uses keyframes as memory with active memory management, giving **O(1) per-frame** online visual odometry. AMB3R-SfM clusters images by feature distance with farthest-point sampling and solves them divide-and-conquer, incrementally.

**Why it mattered.** It is the first feed-forward method to beat optimization-based SLAM on its own turf. TUM visual-odometry ATE goes from the previous best of 7.1 cm to **3.2 cm**; ETH3D SLAM from 11.2 cm to **2.6 cm**; and it surpasses MASt3R-SLAM *even when MASt3R-SLAM is given known calibration and AMB3R is not*. On dynamic TUM sequences its raw predictions are comparable to MegaSaM, which was explicitly designed for dynamic scenes and uses global BA plus trajectory smoothing. Its ablation is the cleanest possible statement of the thesis: swapping the 3D backend for a 2D one consistently hurts, and AMB3R-VO with a plain VGGT base already beats VGGT-SLAM (3.6 vs 5.3 cm). For an academic lab, the 80-GPU-hour figure is the most important number in this survey.

### 2.12 MoGe-2 — the cleanest answer to metric-vs-relative

**Problem.** Metric depth (real meters) and relative depth (shape up to unknown scale) pull in opposite directions. Training for metric scale forces the network to commit to absolute distances it often cannot infer, which degrades the *shape* it could have gotten right. Most metric models are visibly less sharp than relative ones.

**Trick.** Decouple them architecturally. Keep MoGe's affine-invariant pointmap prediction — which absorbs the ambiguity into the representation rather than fighting it — for relative geometry, and add a *separate* small MLP head, conditioned on the CLS token, that predicts one global metric scale factor. Metric supervision flows only into the scale head; it cannot corrupt the shape branch. Separately, diagnose that real data's sensor noise is what blurs fine geometry, and fix it with a data-refinement pipeline that filters and completes real depth using sharp synthetic labels.

**Why it mattered.** It is the first model to get accurate relative geometry, correct metric scale, *and* sharp detail simultaneously — Depth Pro had the sharpest boundaries but weaker metric accuracy; Metric3D v2 and Depth Anything V2 had competitive metric numbers but, as Depth Pro's authors showed, "competitive metric accuracy does not imply sharp boundaries." MoGe-2's structural lesson — separate the ambiguous global quantity from the well-determined local one — is exactly the lesson MapAnything's scale token and DA3's depth-ray target also encode. Three independent groups converged on it.

### 2.13 RoMa v2 — the reason not to throw away dedicated matchers

**Problem.** With VGGT and π³ producing correspondences as a byproduct, one might conclude that dedicated matchers are obsolete. RoMa v2's evaluation shows the opposite, decisively.

**Trick.** Swap the frozen DINOv2 coarse encoder for frozen **DINOv3**; predict coarse matches at stride 4 rather than 14 (so refinement only needs strides 4/2/1); decouple training into two stages, freezing the matcher before training the refiners, which is much faster; write a custom CUDA kernel for the local-correlation operation whose memory footprint was the practical bottleneck; predict a full per-pixel 2×2 *precision matrix* so downstream solvers know the anisotropic uncertainty of each match; and diversify training beyond RoMa's MegaDepth-only diet to ten datasets including aerial (AerialMegaDepth, BlendedMVS) for extreme rotations and small-baseline synthetic data (FlyingThings3D) for fine detail.

**Why it mattered — the number that should change your priors.** On MegaDepth-1500 two-view pose, AUC@5°: RoMa v2 **62.8**, RoMa 62.6, DKM 60.4, LoFTR 52.8, LightGlue 51.0, Reloc3r 49.6, MASt3R 42.4, UFM 41.5, and **VGGT 33.5**. A 2021 detector-free matcher beats the CVPR 2025 Best Paper by 19 AUC points at two-view pose accuracy. Feed-forward reconstruction models "can produce coarse correspondences, [but] they struggle to yield accurate high-resolution dense matches." If your downstream task is precision two-view geometry, the geometry foundation model is the wrong tool.

---

## 3. Technical trends: what converged, and the current dominant recipe

### 3.1 The 2026 default architecture

There is now a recognizable house style, and the convergence is striking because it arrived from several independent directions.

- **Backbone**: a frozen or lightly-finetuned DINOv2/DINOv3 ViT encoder, patchifying each image independently. Not one paper in 2026 trains its image encoder from scratch. MapAnything explicitly compared DINOv2 against CroCov2, DUSt3R's encoder, RADIO, and random-init patchification, and DINOv2 won on downstream performance, convergence speed, *and* generalization.
- **Fusion**: alternating **frame attention** and **global attention** (VGGT's invention). Every serious model uses it — π³, MapAnything, DA3's cross-view attention, VGGT-Ω, D4RT's encoder, even RoMa v2's coarse matcher. This is the single most-copied design element in the field.
- **Prediction target**: **per-view depth plus cameras (or rays), not a world-frame pointmap.** This is the biggest change of the last year, and it happened three times independently. DA3 predicts depth + rays; VGGT-Ω predicts per-frame depth + cameras; MapAnything factors into rays + depth + pose + scale; π³ predicts local pointmaps in each frame's own coordinates. The reasons converge too: per-view targets are what let VGGT-Ω handle dynamic scenes without a motion detector, and they are what let MapAnything accept heterogeneous inputs.
- **Scale**: a **separate, dedicated scale prediction** — one token, one small MLP. MapAnything's scale token, MoGe-2's CLS-conditioned scale head, AMB3R's lightweight scale head. Nobody regresses metric coordinates directly anymore.
- **Auxiliary supervision**: keep multi-task *losses*, delete the redundant *heads*. VGGT showed over-complete prediction helps; VGGT-Ω showed you get the same benefit from losses alone at a fraction of the memory; DA3 showed a single well-chosen target beats multi-task heads outright.
- **Injection of priors and side information**: **zero-initialized convolutions** (the ControlNet trick). AMB3R's backend, OmniVGGT's GeoAdapter. This is how you add capability to a frozen foundation model without destabilizing it, and it is why AMB3R trained in 80 GPU-hours.
- **Pose readout**: increasingly **closed-form, not learned**. Both CVPR 2026 dynamic-reconstruction papers recover camera pose by decoding a point grid in two reference frames and solving for the rigid transform with Umeyama's algorithm. No pose head, no iterative solver.

### 3.2 The efficiency story converged on one insight

Every 2026 scalability paper is a variation on the same observation: **the growing key-value cache is the bottleneck, and the global attention it feeds is very sparse.** From there, three answers:

1. **Compress the cache into fixed-size learned weights via test-time training.** ZipMap and VGG-T³ arrived at this independently, at the same conference. VGG-T³ says it most precisely: the bottleneck "stems from the varying-length Key-Value (KV) space representation of scene geometry, which we distill into a fixed-size Multi-Layer Perceptron via test-time training." TTT3R applies the same lens to CUT3R's recurrent state and derives a closed-form update rule that needs no training at all. This is the most intellectually interesting convergence of the year, and it imports a live idea from sequence modeling into 3D vision.
2. **Bottleneck the information flow architecturally.** VGGT-Ω's register attention restricts cross-frame exchange to a handful of register tokens. Replacing 25% of global attention layers costs nothing measurable; replacing all of them reduces FLOPs to 6% of the original but degrades noticeably — so there is a real, findable frontier here.
3. **Partition the problem.** VGGT-Long's chunks, MERG3R's clusters, S-VGGT's subscenes-with-shared-reference-frames, GLUEMAP's local star graphs, AMB3R-SfM's feature-distance clusters. All training-free, all model-agnostic, and GLUEMAP's version has the extra virtue of *improving accuracy* rather than merely enabling scale, because bounding attention to relevant images fixes the view-graph-radius problem.

### 3.3 Where the field's self-image is now correct

The 2024–early-2025 framing was "feed-forward replaces optimization." The 2026 framing, stated most bluntly by GLUEMAP, is that they solve different problems: feed-forward models supply **local robustness** (they work with two views, no overlap, no texture, no parallax) and classical methods supply **global consistency, accuracy, and scalability**. Every high-performing 2026 system is a hybrid of some kind — GLUEMAP explicitly, AMB3R by adding a geometric backend to a neural front end, VGGT-SLAM 2.0 by wrapping a factor graph around submaps, VGGT-Long by adding loop closure. The pure-feed-forward-end-to-end position has essentially no adherents left among the people reporting numbers on hard datasets.

---

## 4. Have feed-forward models replaced COLMAP? Evidence both ways, with numbers

**Short answer: no — and the best 2026 systems don't try. But the question decomposes, and the decomposition is the useful part.**

### 4.1 Where classical SfM still wins, decisively

**ETH3D** (high-resolution unordered collections, millimeter ground truth) is the accuracy benchmark, and the gap is not subtle. From GLUEMAP Table 1:

| Method | AUC@1 | AUC@3 | AUC@5 |
|---|---|---|---|
| GLOMAP + SIFT | 45.6 | 62.2 | 66.7 |
| GLOMAP + ALIKED + LightGlue | 42.9 | 62.1 | 67.4 |
| MASt3R-SfM | 39.2 | 55.6 | 60.5 |
| π³ + BA | 30.6 | 55.1 | 65.1 |
| VGGT | 8.6 | 24.0 | 35.0 |
| MapAnything | 5.1 | 11.1 | 18.3 |
| CUT3R | 5.0 | 11.4 | 18.8 |
| **π³** (best pure feed-forward) | **13.2** | 36.1 | 48.9 |
| GLUEMAP (hybrid) | 53.0 | 76.9 | 83.6 |
| GLUEMAP* (with GT intrinsics) | 74.0 | 85.9 | 89.0 |
| MP-SfM* (sparse, GT intrinsics) | 74.3 | — | 88.3 |

Read the AUC@1 column. The best pure feed-forward model scores 13.2 where plain SIFT scores 45.6 — a 3.5× accuracy gap. Bundle adjustment on top of π³ closes maybe two-thirds of it (30.6) but not all. GLUEMAP notes this is "a limitation that is often underreported in prior literature," which is a polite way of saying the field has been quoting loose thresholds.

**Scale.** On LaMAR (3 large indoor-outdoor scenes, 6,587 / 7,553 / 9,319 phone images, view-graph radii of 49/61/59), **MASt3R-SfM, π³, and π³+BA all report out-of-memory on every scene.** GLOMAP+ALIKED+LightGlue averages AUC@3 of 10.9; GLUEMAP averages 26.4. Feed-forward methods are not in the competition.

**GLOMAP's own scaling number** makes the point from the classical side: on LaMAR LIN (36k+ images, 250 m extent) it reaches 90% recall at 1 m in 5.5 hours, where COLMAP reaches ~50% recall in over 7 days. The classical community's scalability problem was solved in 2024 by better classical algorithms, not by neural networks.

**Independent third-party audits agree.** The aerial-photogrammetry study (arXiv 2507.14798) found that at standard overlap (≥70%), COLMAP at high resolution achieves point-cloud accuracy down to **0.06 m** and camera position error within 0.8 m, roughly 9% better than learning-based methods, and that VGGT showed "scalability issues with very large image sets (191 images), producing inconsistent alignments." HERI3D, on UAV heritage imagery, concluded flatly that "traditional photogrammetry implemented in COLMAP remains the most reliable method for geometric accuracy."

**The counterintuitive finding that should worry practitioners.** GLUEMAP measured accuracy as a function of input density and found that for VGGT and MapAnything, *sparser* input gives *better* pose estimation. Their IMC2021 table shows it directly for π³: AUC@3 of 46.2 on 5-image bags, 39.7 on 10-image bags, 36.6 on 25-image bags, 35.2 on the full collection. SIFT+GLOMAP goes the other way — 39.6 → 50.1 → 64.4 → 76.9 — because optimization exploits observational redundancy. A feed-forward model given more evidence can get worse. No optimization-based pipeline has this property, and it is a fundamental scaling concern, not an engineering detail.

### 4.2 Where feed-forward models win, decisively

**Object-centric and sparse-view.** CO3Dv2 (GLUEMAP Table 3, average over 10/20/40-image settings):

| Method | AUC@3 | AUC@10 | AUC@30 |
|---|---|---|---|
| GLOMAP + SIFT | 37.0 | 49.9 | 56.4 |
| GLOMAP + ALIKED + LightGlue | 38.1 | 52.2 | 59.9 |
| **π³** | **47.3** | **76.6** | **89.5** |
| π³ + BA | 58.4 | 80.3 | 90.7 |

At AUC@30 the gap is 89.5 vs 56.4. Low texture and low overlap are where the learned prior does work that geometry cannot.

**Extremely sparse input.** The aerial study found that at 1–2 images with 10% overlap, MASt3R and VGGT produce usable point clouds at up to 0.4 m accuracy with ~50% higher completeness, while "COLMAP frequently failed to produce any reconstruction or generated sparse, low-quality models with errors exceeding 2.3 m and completeness as low as 8%." This is a capability difference, not a quality difference.

**Speed and density.** VGGT reconstructs in ~0.2 s where DUSt3R-with-alignment takes ~7 s and VGGSfM ~10 s; on dense pointmaps VGGT cuts DUSt3R's Chamfer distance from 1.005 to 0.677 while running ~35× faster. Feed-forward outputs are *dense* by construction — millions of points where COLMAP's sparse stage gives thousands.

**Degenerate configurations classical methods cannot represent at all.** MASt3R-SfM works with *zero* camera motion (purely rotational), which is formally impossible for triangulation-based SfM. GLUEMAP inherits a version of this weakness and flags it as a limitation: "the formulation currently does [not] handle purely rotational motion due to our augmented bundle adjustment formulation."

**Low-overlap indoor scenes are the case where both pure approaches fail.** On SMERF, GLUEMAP reports that classical methods "largely fail with low overlap" (SIFT AUC@1 of 1.4–13.0) while π³ "also does not achieve high scores because the view-graph radius and symmetry is high, leading to multiple rooms collapsing in the reconstructions" (AUC@1 of 0.9–3.2). Both fail, differently. Doppelganger filtering plus motion averaging plus virtual tracks is what rescues it.

### 4.3 The SLAM/VO sub-answer is different, and it flipped in late 2025

For **online, sequential** reconstruction the answer changed. AMB3R reports what it calls the first demonstration that feed-forward visual odometry can surpass its optimization-based counterparts:

- TUM RGB visual odometry ATE RMSE: previous SOTA **7.1 cm → 3.2 cm**
- ETH3D SLAM: **11.2 cm → 2.6 cm**
- Surpasses MASt3R-SLAM *with known calibration*, while itself running uncalibrated
- On 7-Scenes it "surpasses [the] pseudo GT prior works used," validated via novel-view-synthesis PSNR
- AMB3R-VO with a plain VGGT front end already beats VGGT-SLAM: 3.6 vs 5.3 cm

For context on how far this has come: on TUM RGB-D monocular ATE (SEGS-SLAM's comparison table), ORB-SLAM3 reports 46.0 cm, DROID-SLAM 1.69 cm, MonoGS 4.01 cm, Photo-SLAM 1.54 cm. The DUSt3R-prior systems are now competing at the top of that table without calibration, which none of those baselines can do.

**So: has feed-forward replaced COLMAP?** For offline high-accuracy reconstruction on well-textured, well-overlapped, large collections: no, not remotely, and the 2026 state of the art (GLUEMAP) uses classical global SfM as its skeleton with a feed-forward model as a local subroutine. For sparse-view and object-centric reconstruction: yes, and it has for over a year. For online uncalibrated visual odometry: as of CVPR 2026, yes. For two-view pose accuracy: no, and a 2021 matcher still beats it (§6).

---

## 5. SLAM and streaming reconstruction: the 2025–2026 shift

The trend is unambiguous. VGGT-SLAM 2.0 states it as fact in its introduction: the field "has seen a paradigm shift from using classical multi-view geometry and optimization techniques towards building on top of feed-forward geometric foundation models to develop SLAM systems," and the motivation is as much engineering as accuracy — "these new hybrid SLAM systems … produce a much simpler SLAM system that is both easier to use and maintain," while yielding dense maps without requiring calibration.

**Four distinct architectural strategies emerged, and they are worth distinguishing.**

1. **Two-view prior + classical SLAM machinery** (MASt3R-SLAM). Use the pairwise model as the primitive for tracking, mapping, and relocalization; add iterative projective matching, ray-error tracking, local pointmap fusion, loop closure, and a second-order Sim(3) backend. 15 FPS, calibration-free.
2. **Submap alignment on the right manifold** (VGGT-SLAM → VGGT-SLAM 2.0). Run a multi-view model on windows of frames, then align submaps in a factor graph. VGGT-SLAM's contribution was the observation that Sim(3) is theoretically insufficient for uncalibrated cameras — the scene is determined only up to a 15-DoF projective transform — so it optimizes on SL(4). Version 2.0 then found the *cost* of that generality: 15-DoF drift and planar degeneracy. Its fix constrains the homography between overlapping frames to the form $H_{ij} = K_i^{-1} K_j \begin{pmatrix} I & 0 \\ 0^T & s\end{pmatrix}$, aligning only calibration and one scale factor while enforcing zero relative motion for frames that are physically identical — still a subset of SL(4), so the same solver works. Result: 23% less pose error on TUM, 8.4 FPS on an RTX 3090, real-time onboard a Jetson Thor. This is the cleanest example in the survey of a research group publishing the correction to its own paper.
3. **Chunk-and-align without a BA backend** (VGGT-Long). Overlapping chunks, robust chunk alignment, lightweight loop closure. The paper's own framing is a thesis worth quoting: "a sufficiently powerful base model may not necessarily require a complex [graph-based optimization backend]." Kilometer-scale on KITTI/Waymo without calibration, depth supervision, or retraining.
4. **Recurrent state with a fixed memory budget** (Spann3R → CUT3R → TTT3R → StreamVGGT/STAC). Theoretically the most attractive — linear time, constant memory — and practically the most fragile. GLUEMAP's assessment is that recurrent models "maintain a fixed state or they suffer from the same limitation as incremental classical approaches, where a single bad decision can lead to an unrecoverable failure," and additionally that they "depend on a specific, typically sequential, input order," failing entirely if an image lacks overlap with what came before. TTT3R is the direct response: framing the state update as online learning and gating it by alignment confidence gives 2× better long-sequence pose, training-free.

**The Gaussian-splatting SLAM line (SplaTAM, MonoGS, Photo-SLAM, GS-SLAM, DROID-Splat, SEGS-SLAM, Flow4DGS-SLAM) is solving a different problem** and should not be compared head-to-head on ATE alone. Its objective is a *renderable* map, so PSNR/SSIM/LPIPS matter as much as trajectory error, and the tracking front end is often borrowed (Photo-SLAM uses ORB-SLAM3; DROID-Splat uses DROID-SLAM). DROID-Splat's own conclusion is worth carrying: "traditional and end-to-end tracking systems are still the best once perfect supervision is missing." The two lines are beginning to merge — Flow4DGS-SLAM and SLARM at CVPR 2026 both combine feed-forward geometry with splatting and semantics — but as of August 2026 the geometry-foundation-model line owns trajectory accuracy and the splatting line owns rendering.

---

## 6. Matching and correspondence: the sub-field that did *not* get absorbed

This deserves emphasis because it is the clearest counterexample to the "one model for everything" narrative.

**MegaDepth-1500, two-view relative pose (from RoMa v2, Table 4):**

| Method | Venue | AUC@5° | AUC@10° | AUC@20° |
|---|---|---|---|---|
| RoMa v2 | 2025/26 | **62.8** | **77.0** | **86.6** |
| RoMa | CVPR'24 | 62.6 | 76.7 | 86.3 |
| DKM | CVPR'23 | 60.4 | 74.9 | 85.1 |
| LoFTR | CVPR'21 | 52.8 | 69.2 | 81.2 |
| LightGlue | ICCV'23 | 51.0 | 68.1 | 80.7 |
| Reloc3r | CVPR'25 | 49.6 | 67.9 | 81.2 |
| MASt3R | ECCV'24 | 42.4 | 61.5 | 76.9 |
| UFM | NeurIPS'25 | 41.5 | 57.9 | 72.4 |
| **VGGT** | **CVPR'25 (Best Paper)** | **33.5** | 52.9 | 70.0 |

LoFTR, from 2021, beats VGGT by 19 AUC points. MASt3R — the pointmap model explicitly designed for matching — is 20 points behind RoMa. RoMa v2's own diagnosis: feed-forward reconstruction models "can produce coarse correspondences, [but] they struggle to yield accurate high-resolution dense matches."

**The reason is architectural, not incidental.** Reconstruction models patchify at stride 14 (DINOv2's patch size) and predict geometry at low spatial resolution before upsampling. Dense matchers refine at strides 4, 2, and 1. Sub-pixel precision requires operating at pixel resolution, and no current geometry foundation model does.

**Which is why the two lines are being wired together rather than one replacing the other.** VGGT and VGGSfM emit tracks that get fed into BA; GLUEMAP snaps π³'s tracks to SIFT keypoints within a 1-pixel radius and merges tracks that snap to the same keypoint; MV-RoMa (CVPR 2026 Highlight) goes the other way, extending RoMa to multi-view so its correspondences become usable SfM tracks directly, using pairwise matching as a geometric prior and a pixel-wise attention refiner. The 2026 answer is: use a matcher for correspondences, use a geometry model for priors, and feed both into one optimization.

Two other matching developments worth tracking. **RoMa v2's per-pixel 2×2 precision matrices** give downstream robust estimators anisotropic uncertainty rather than a scalar confidence — a small change with real consequences for how solvers weight evidence. And **RoMa v2's data-mixture ablations** show that adding aerial datasets (AerialMegaDepth, BlendedMVS) buys robustness to large rotations and air-to-ground viewpoint change, while adding small-baseline synthetic data (FlyingThings3D) buys fine-grained detail, and that a *tiny* amount of Virtual KITTI 2 (weight 0.01, 5 scenes) makes texture-poor road surfaces work. That last finding — a 1%-weight dataset unlocking a whole domain — is the kind of thing an academic lab can discover and industry labs rarely bother to report.

---

## 7. Explicitly stated open problems and failure modes, with attribution

### On the accuracy gap being underreported

> "Transformer-based models achieve the best accuracy across all feedforward approaches. Yet, in scenes where classical methods work well, transformer-based models still lag significantly behind in terms of camera pose accuracy — **a limitation that is often underreported in prior literature.**"
> — GLUEMAP, CVPR 2026, §2.2

### On more data making things worse

> "For lower densities, counterintuitively, the accuracy of pose estimation improves. This difference is especially observable for VGGT and MapAnything. … performance of feedforward methods degrades with more input views of the same scene. Inversely, the proposed method behaves like other optimization-based pipelines, and the performance increases with the density of input."
> — GLUEMAP, CVPR 2026, §4.2

### On why global attention breaks in large scenes

> "In complex, large-scale scenes — characterized by a large view graph radius — attending information globally becomes problematic. The resulting quadratic increase in possible connections makes it difficult to distinguish relevant from irrelevant information, leading to significant performance drops. This problem is exacerbated in the presence of symmetric scene structure."
> — GLUEMAP, CVPR 2026, §2.2

### On recurrent models' unrecoverable failures

> "For recurrent networks, though theoretically scalable, they maintain a fixed state or they suffer from the same limitation as incremental classical approaches, where a single bad decision can lead to an unrecoverable failure. … If images are provided in random order, or if an image lacks visual overlap with previously processed images, pose estimation can fail entirely."
> — GLUEMAP, CVPR 2026, §2.2

### On no method being robust to the things that actually break reconstructions

> "In terms of robustness, none of the existing methods can reliably handle multiple connected components, resolve symmetric structures, or systematically reject outliers such as irrelevant input images."
> — GLUEMAP, CVPR 2026, §1

### On the pointmap parameterization being conceptually wrong

> "Yet, this raises a fundamental question: Is the mapping from 2D pixels to 3D scene points truly one-to-one? In practice, this is not the case. Due to visual overlap, multiple pixels often correspond to the same 3D point. … the network itself operates on 2D grids and lacks explicit geometric reasoning or spatial compactness."
> — AMB3R, CVPR 2026, §1

### On humans being absorbed into walls — the best-documented systematic failure mode in the literature

> "We observe a recurring failure case in near-static street-view videos with pedestrians moving through the scene: **almost all existing feed-forward reconstruction models often absorb humans into the static background**, estimating them as part of nearby walls or buildings. This failure appears to be related to ambiguous boundary pixels in existing training datasets. For example, the most widely used multiview dataset Megadepth was annotated with COLMAP on phototourism images that often contain people. At human boundaries, the patch match stereo algorithm may assign some pixels to the surrounding static architecture, causing supervision to treat parts of people as background. Excluding MegaDepth or re-estimating its depths can substantially reduce this artifact."
> — VGGT-Ω, CVPR 2026, Appendix B

### On benchmark scores hiding memorized label noise

> "We find that different types of errors in the annotations translate into specific failure modes at inference time. **What is worse, these failure modes do not affect most of the images in standard benchmarks and may thus not be detected in the quantitative results.** Instead, they emerge only when the model is tested on a new sample that resembles an incorrectly labeled sample seen during training. This behavior indicates that, while the model does learn the general principles of 3D reconstruction, it can nevertheless memorize idiosyncratic noise as well."
> — VGGT-Ω, CVPR 2026, Appendix B

### On specific remaining failure cases at 10B parameters

> "The model's performance drops significantly in the presence of strong motion blur. Meanwhile, reconstruction quality often degrades if the field of view changes abruptly (e.g., shifting from 10° to 160° in a few seconds) or the camera is highly distorted. Additionally, because the model was exposed to some noisy data (e.g., ScanNet++) during the early stage of training, its predictions are sometimes unstable in the cases like office scenes with many monitors."
> — VGGT-Ω, CVPR 2026, Appendix C

### On self-supervision not working, after a serious attempt

> "So far, we have found it useful for improving model generalization, especially for out-of-distribution data, however, **it has had little impact on most benchmarks. It is non-trivial to do better.** … We spent considerable time exploring other self-supervised protocols such as new view synthesis, including variants similar to RayZer and E-RayZer, generating tokens instead of pixels, NeRF representations or Gaussian Splats. We also tried masking image tokens, distinguishing objects across frames, incorporating temporal order, and related variants. **Only the student-teacher approach helped**, in our implementation. For example, we found methods like E-RayZer, which may work well with static scenes, struggle with dynamic ones."
> — VGGT-Ω, CVPR 2026, §4 discussion

This is an unusually valuable negative result: 18M unlabeled videos and a serious search over protocols yielded generalization gains but almost no benchmark movement. Anyone planning a self-supervised 3D pretraining project should read it before starting.

### On the monocular-depth detail ceiling

> "Our method struggles with capturing extremely fine structures, such as thin lines and hair, and with maintaining straight and aligned structures under a significant scale difference between the foreground and background. **The ambiguity in real-world metric scale can also lead to deviations in out-of-distribution scenarios.**"
> — MoGe-2, NeurIPS 2025, Limitations

### On metric accuracy not implying sharp geometry

> "The competitive metric accuracy of Metric3D v2 and Depth Anything v2 does not imply sharp boundaries."
> — Depth Pro, ICLR 2025, §4

### On camera diversity being the bottleneck for metric depth

> "The camera parameter estimation (ρA) sees only marginal improvements, indicating that the limited diversity of training cameras remains a challenge that could be addressed with additional camera-only training."
> — UniDepthV2, §"The State of The Art"

### On what MapAnything can't do

> "(a) MapAnything does not explicitly account for the noise or uncertainty in geometric inputs. … (c) While the design of MapAnything supports iterative inference, **it is yet to be explored how effective scaling of test-time compute would be for 3D reconstruction** (this ties into effectively handling noise in the inputs). (d) Multimodal features are currently fused before input; exploring ways for efficient direct input of different modalities to the transformer could be interesting."
> — MapAnything, 3DV 2026, §5

### On the layer-role structure of alternating attention

> "Our analysis reveals a clear division of roles in the alternating global-frame architecture: **early global layers do not form meaningful correspondences, middle layers perform cross-view alignment, and last layers provide only minor refinements.**"
> — AVGGT, CVPR 2026

### On the cause of streaming geometry flicker

> "Through targeted empirical analysis, we trace this instability to its root cause: **fluctuations in latent feature statistics, whose mean and variance directly determine the predicted depth's scale and shift.**"
> — DyFN, CVPR 2026

### On per-frame accuracy vs multi-frame consistency being a real tradeoff

> "Per-frame visual geometry estimation approaches typically exhibit **weaker multi-frame consistency but demonstrate superior per-frame accuracy** compared to multi-frame algorithms."
> — *Unlocking the Power of Critical Factors for 3D Visual Geometry Estimation*, CVPR 2026

### On reproducibility of headline reconstruction numbers

> "We suspect that Wang et al. (2025) perform bundle adjustment in post-processing, thereby achieving a lower Chamfer Distance. However, **this and other details of their experiment are not explicitly mentioned in the paper, so the experiments cannot be reproduced exactly.**"
> — *Uncertainty Quality of VGGT*, arXiv 2606.16479, on being unable to reproduce VGGT's 0.38 mm DTU result

### On GLUEMAP's own remaining limits

> "Our method's performance critically depends on the quality of local reconstructions by feedforward methods. For example, we can currently not handle fisheye images since the feedforward models are only trained on images taken by pinhole camera models… Furthermore, the formulation currently does [not] handle purely rotational motion due to our augmented bundle adjustment formulation. … our method currently requires the combination of different feedforward methods. An interesting direction for future work will be in developing a network that can solve the problem with a shared feedforward architecture."
> — GLUEMAP, CVPR 2026, §4.4

### On domain shift from camera geometry, not appearance

> "We argue that this failure is caused not only by appearance-domain shift, but also by **UAV-specific camera-geometry variations, especially oblique views and HFOV-height ambiguity.** Existing UAV datasets mainly emphasize scene diversity and provide limited coverage of camera configurations."
> — UAVFF3D, arXiv 2605.17942

---

## 8. Verification caveats

Items I could not confirm from a primary source in this session. None of the IDs below were invented; each is either absent (marked "not given") or traced to a secondary source that I name.

**arXiv IDs I did not verify directly, and therefore do not assert:**
- **Fast3R** — I read the CVF Open Access PDF (CVPR 2025, pp. 21924–21935) but never saw its arXiv ID from a primary source. Not given.
- **MUSt3R**, **Light3R-SfM**, **MP-SfM**, **Doppelgangers++**, **VGGSfM**, **Detector-Free SfM**, **ACE0**, **RoMa**, **DKM (venue only)**, **LoFTR**, **SplaTAM**, **MonoGS**, **Photo-SLAM**, **GS-SLAM**, **DROID-SLAM**, **MiDaS**, **Marigold**, **Metric3D v1/v2**, **UniDepth v1/v2**, **UniK3D**, **MoGe (v1)**, **DepthCrafter**, **ChronoDepth**, **DepthAnyVideo**, **NVDS**, **RollingDepth**, **MegaSaM**, **SpatialTrackerV2**, **UFM**, **SLAM3R**, **Align3R**, **Pow3R**, **MonST3R**, **MV-DUSt3R+**, **Driv3R** — all appear in this report by name and venue where I saw a venue, but **I did not verify their arXiv IDs and have not supplied any.**
- **CUT3R (2501.12387)** — ID seen on EmergentMind's page, not on arXiv itself. I read the CVF Open Access entry for venue and page numbers, which are solid.
- **Depth Anything V2 (2406.09414)** — ID read from the official GitHub repo's BibTeX, not from arXiv.
- **FastVGGT (2509.02560)** and **SAIL-Recon (2508.17972)** — IDs read from GLUEMAP's reference list only. Plausible but second-hand.
- **DROID-Splat (2411.17660)** — ID from an ar5iv mirror URL, not arXiv proper.
- **FOUND-IT (2605.25371)** — ID from the VGGT-SLAM GitHub news log. I did not read the paper or confirm it exists on arXiv.

**Venue uncertainties:**
- **Spann3R** — my search query asserted 3DV 2025 and the returned pages did not confirm it. arXiv ID 2408.16061 *is* verified; the venue is not.
- **RoMa v2** — arXiv 2511.15706 verified, with a v3 dated 6 Jul (2026). Formatting (⋆-footnoted keywords, Springer-style layout) suggests an ECCV-family submission, and a web search explicitly said it is *not* CVPR 2026. **Venue unknown.** Note also that RoMa v2 appears in a third-party benchmark's method list, so it is being used by others.
- **UniDepthV2** — I read a PDF hosted at pure.mpg.de. The journal is almost certainly a TPAMI-style venue but I did not confirm it. **UniK3D**'s venue is entirely unconfirmed.
- **Any4D** — CVPR 2026 confirmed via the Paper Digest listing and author names; I did not read the paper, so its numbers are absent here rather than approximated.
- **HERI3D** — read as an institutional-repository PDF; peer-review venue unknown.
- **Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT (2507.08448)** — arXiv ID verified. It appears simultaneously on a journal site (coscipress.com/JAICS); I would not treat it as peer-reviewed.

**A title discrepancy worth knowing about.** The CVPR 2026 paper by Jin, Wu, Zhang, Gao, Barron, Snavely, and Hołyński appears in the CVF Open Access repository as **"ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training"** (pp. 21748–21759) and in Paper Digest's CVPR 2026 listing as **"FRM: Linear-Time 3D Reconstruction Via Test-Time Training"** with an identical author list and abstract framing ("Fast Reconstruction Model"). Almost certainly a submission-vs-camera-ready rename. **Cite the CVF title.**

**Numbers I am reporting at one remove.** The VGGT camera-pose comparison table in §2.4 (DUSt3R 67.7 / MASt3R 76.4 / VGGSfM v2 78.9 / Fast3R 72.7 / VGGT 85.3 / VGGT+BA 93.5 on RealEstate10K AUC@30) and the VGGT-Ω and D4RT summary tables in §2.8 and §2.9 came from a well-sourced third-party technical blog (digitalsense.ai) that cites the papers and reproduces their figures. I independently verified VGGT-Ω's Sintel numbers (40.0 / 79.1 / 93.5 vs 22.5 / 58.3 / 74.1) and its 50×-faster-than-MegaSaM claim against the arXiv abstract and body, and D4RT's 200+ FPS / 9× / 100× claims against its arXiv Figure 3 caption. **The individual per-baseline entries in those two tables I did not cross-check against the papers themselves.** The π³-vs-VGGT comparison table in §2.5 (params, FPS, AUC@30, DTU accuracy, ATE, order-dependence std) came from a third-party paper review blog; I verified the qualitative claims and the Sintel 0.16→0.074 direction against the arXiv abstract and body, but not each cell.

**GLUEMAP, AMB3R, MapAnything, RoMa v2, VGGT-Ω, DA3, Light3R-SfM, GLOMAP, TTT3R, and VGGT-SLAM 2.0 numbers were all read directly from the papers' own text or tables** and are the most reliable figures in this report.

**Things I looked for and did not find**, so that absence is not mistaken for oversight:
- No "Depth Anything 4." MetricAnything (2601.22054) is the most recent scaling result in the depth track.
- No head-to-head benchmark comparing VGGT-Ω and D4RT. The blog covering both says explicitly that they "have not been benchmarked against each other."
- No 2026 survey of this specific area that postdates CVPR 2026. The only survey I found (2507.08448) predates every 2026 result here.
- **Pi3X** appears in the Pi3 repository as an "enhanced" model with Hugging Face weights at `yyfz233/Pi3X`. I found no paper, no arXiv ID, and no venue. Treat it as an unpublished checkpoint.
- I did not find a public benchmark that isolates *two-view* pose accuracy across both matching and reconstruction models on the same protocol beyond RoMa v2's own MegaDepth-1500 table. That table is doing a lot of work in §6 and would benefit from independent replication.

---

## 9. Under-explored gaps: what an academic lab with 8–32 GPUs should actually do

My analysis, not the papers'. Every item below is chosen because it is bottlenecked by ideas or careful experimentation rather than by compute, and because at least one 2025–2026 primary source explicitly names it as open or leaves it visibly unaddressed. **The single most encouraging data point for this section: AMB3R won a CVPR 2026 Highlight, beat optimization-based SLAM for the first time, and trained in ~80 H100 GPU-hours** — roughly 2.5 hours on 32 GPUs. The frozen-foundation-model-plus-cheap-adapter recipe has made this field newly accessible, and AMB3R is the proof.

### 9.1 The highest-leverage gap: nobody has characterized *when* feed-forward geometry fails

GLUEMAP made the first real attempt — view-graph radius and density — and found two effects nobody had documented (accuracy falls with radius; accuracy falls with *more* images). But that was a means to an end for them, occupying one figure. Nobody has built the diagnostic suite.

**Concretely buildable.** Construct a controlled benchmark that sweeps the axes these models are actually sensitive to, holding scene content fixed: view-graph radius, density, overlap fraction, parallax angle, texture energy, symmetry/Doppelganger density, field-of-view change rate, motion blur magnitude, and dynamic-content fraction. LaMAR, ScanNet++ v2, ETH3D, SMERF, and TartanAir v2 give you real scenes with ground-truth meshes from which you can *render* depth to compute true view graphs, exactly as GLUEMAP did. Then evaluate the open checkpoints — VGGT, π³, MapAnything, DA3, CUT3R, MUSt3R, AMB3R, MoGe-2 — and produce a *failure-prediction model*: given cheap statistics of an input image set, predict whether a feed-forward reconstruction will be trustworthy.

**Why this is the right project.** It costs inference, not training. VGGT-Ω's own Appendix B says the quiet part out loud: annotation-noise-induced failure modes "do not affect most of the images in standard benchmarks and may thus not be detected in the quantitative results." The field is ranking models on benchmarks that structurally cannot see their failure modes. A diagnostic suite is a benchmark contribution that everyone downstream needs, it is citation-durable, and it requires no 10B-parameter run. UAVFF3D is the closest existing work and it covers only UAV geometry — which incidentally validates the premise, since its domain adaptation cut Ray Error by up to 84.2% and Pose ATE by up to 76.0%, meaning the failures were domain-specific and fixable all along.

### 9.2 Test-time compute scaling for 3D reconstruction — named as open, attempted by nobody

MapAnything says it directly: "While the design of MapAnything supports iterative inference, **it is yet to be explored how effective scaling of test-time compute would be for 3D reconstruction.**" This is a striking gap given how much of the rest of machine learning spent 2025–2026 on exactly this question.

The ingredients are all sitting there and nobody has assembled them. MapAnything can consume its own predictions as geometric inputs (that is what the depth/pose/intrinsics input slots are *for*), so iterative refinement is a two-line change. TTT3R showed that a well-chosen inference-time update rule buys 2× on pose for free. ZipMap and VGG-T³ maintain an explicit queryable scene state, which is precisely what you would want to refine. VGGT's own paper shows optional BA lifts RealEstate10K AUC@30 from 85.3 to 93.5 — so there *is* an 8-point gap that test-time compute can close, and the question is whether a learned iterative procedure can close it faster than BA.

**Concretely:** does N rounds of MapAnything-refines-its-own-output converge, and to what? Is there a scaling curve in refinement steps analogous to VGGT-Ω's parameter curve? Does confidence-weighted rejection between rounds help (as TTT3R's confidence gating suggests)? Can you match π³+BA's accuracy without BA? This is inference-only, uses released Apache-2.0 weights, and answers a question a Meta/CMU team explicitly flagged. It is the highest expected-value item on this list.

### 9.3 Fix the annotation noise, publish the corrected labels

VGGT-Ω's Appendix B is, read correctly, a to-do list handed to the community. It names specific, reproducible data pathologies: ScanNet++ foreground-background depth leakage around chair backs and light fixtures; the doming effect; thin-structure errors; and above all the MegaDepth pedestrian problem, where COLMAP patch-match stereo assigns human boundary pixels to background architecture, teaching **"almost all existing feed-forward reconstruction models"** to absorb people into walls. Their fix was to exclude or re-estimate MegaDepth — inside a proprietary pipeline, on data nobody else has.

**MegaDepth is public.** Re-annotating it with 2026 tools (RoMa v2 for correspondences, MoGe-2 or DA3 for monocular priors, a modern segmentation model to mask people, GLOMAP instead of COLMAP for global consistency) and releasing MegaDepth-Clean is a pure data contribution with an unusually clear success metric: does the humans-in-walls artifact go away, measured on street-view sequences with pedestrians? Do the *other* named artifacts go away with matched treatment of ScanNet++ and Hypersim? This is well within 8–32 GPUs, it is verifiable, and every group training these models would use it. The failure mode is documented by the strongest group in the field, in print, with a diagnosis and no public fix.

### 9.4 Adapters on frozen foundation models — the recipe is proven and the design space is empty

Three papers in the last year added major capability to a frozen model through zero-initialized convolutions: AMB3R (sparse voxel backend, ~80 H100-hours), OmniVGGT (GeoAdapter for depth/intrinsics/extrinsics), and DA3's camera tokens. This is the ControlNet playbook arriving in 3D vision, and almost nothing has been tried yet.

Open slots, in rough order of how surprised I am that they are empty:
- **A dense-matcher adapter.** §6 shows a 29-AUC-point gap between VGGT and RoMa v2 at two-view pose, caused by VGGT predicting at stride 14. Adapt a frozen VGGT/π³ with RoMa v2-style stride-4/2/1 refiners conditioned on the coarse geometry. GLUEMAP already hacks around this by snapping π³'s tracks to SIFT keypoints within 1 pixel — that hack is a screaming signal that the refinement should be learned. Small, well-posed, obviously useful, and the evaluation protocol already exists.
- **A camera-model adapter.** GLUEMAP's stated blocker is that it "cannot handle fisheye images since the feedforward models are only trained on images taken by pinhole camera models," and UniDepthV2 blames its residual weakness on "limited diversity of training cameras." UniK3D solved this for monocular depth with spherical harmonics. Porting that to a frozen multi-view model via an adapter would unblock a named limitation in a CVPR 2026 paper.
- **An uncertainty adapter.** MapAnything limitation (a) is that it "does not explicitly account for the noise or uncertainty in geometric inputs," and arXiv 2606.16479 exists specifically because VGGT's confidence outputs are of unaudited quality. RoMa v2 shows the target: full per-pixel 2×2 precision matrices, not scalar confidences. A calibrated-uncertainty adapter plus an honest calibration study is two papers.

### 9.5 Settle the per-frame vs multi-frame tradeoff, and the pointmap-vs-factored question, at matched compute

Two structural questions are currently answered by assertion rather than controlled experiment.

**First**, a CVPR 2026 paper reports that "per-frame visual geometry estimation approaches typically exhibit weaker multi-frame consistency but demonstrate superior per-frame accuracy compared to multi-frame algorithms." If that is robust, it is important — it means the field's whole direction (bigger multi-view context) is trading per-frame accuracy for consistency, and nobody is measuring the exchange rate. Quantify it: at matched compute, sweep context length from 1 to N frames and plot per-frame accuracy against multi-frame consistency. Then ask the actual engineering question: can DyFN-style feature normalization, or a per-frame model plus a cheap consistency post-process, dominate both?

**Second**, the field switched from world-frame pointmaps to per-view depth-plus-rays three times independently (DA3, VGGT-Ω, MapAnything, and π³'s local pointmaps), and AMB3R argued the pointmap parameterization is conceptually wrong. But **nobody has ablated the representations against each other at matched compute and data.** MapAnything's Table 5a is the closest thing — Local PM+Pose vs RDP vs LPMP+Scale vs RDP+Scale — and it is one small table inside one paper, with results that are not uniform across input configurations (Local PM + Pose actually wins on the metric-scale column with images-only input). A careful representation ablation at 300M–1B parameters on public data, covering pointmap / local pointmap / depth+rays / factored+scale, and reporting separately for static vs dynamic and for tight vs loose thresholds, is a 32-GPU paper that the whole field would cite. The MapAnything codebase is Apache 2.0 and explicitly designed as a "modular framework … to facilitate future research," which makes this unusually cheap to execute.

### 9.6 The test-time-training frontier is two months old and wide open

ZipMap and VGG-T³ independently discovered, at the same conference, that the way past quadratic attention is to distill a growing KV cache into fixed-size fast weights via test-time training. TTT3R found a closed-form version for recurrent state. That is three papers converging on one idea, which in my experience means the idea has more in it than three papers.

Unanswered: What is the capacity-accuracy curve of the fixed-size state — how much scene can an MLP of size M hold before it saturates? ZipMap does 700+ frames in <10 s and VGG-T³ does 1k images in 54 s, but neither reports where quality starts degrading with state size held fixed. Does TTT3R's confidence-derived learning rate transfer from CUT3R to ZipMap's TTT layers? Can the state be *edited* — updated for a changed scene, merged across two captures, or queried for change detection? VGG-T³ already showed the state supports visual localization by querying with unseen images; that is one query type out of many. And can the state be made persistent across sessions, which is what any real robot needs and what FOUND-IT's scene-graph-on-VGGT-SLAM gestures at? These are architecture-and-analysis questions on released models, not scaling questions.

### 9.7 Do *not* do these

- **Do not train a new foundation-scale reconstruction model.** VGGT-Ω used 4M annotated sequences distilled from 40M internet videos plus 18M unlabeled videos at up to 10B parameters, and D4RT trained a billion-parameter encoder on TPUs. You will lose, and both papers' own authors note neither is lightweight to reproduce.
- **Do not report AUC@30 as your headline.** It is the loose threshold that mostly measures "did you get the scene roughly right." GLUEMAP's ETH3D table shows why: at AUC@5 the feed-forward/classical gap looks like 48.9 vs 66.7, and at AUC@1 it is 13.2 vs 45.6. Report tight thresholds, or report both and say which is which.
- **Do not evaluate only on CO3D, Replica, or 7-Scenes.** CO3D is object-centric and is where feed-forward methods look best. AMB3R explicitly removed Replica from its evaluation because it is inside DUSt3R's and VGGT's training data, and removed NRGBD for potential synthetic overlap. Contamination in this field is real and mostly unpoliced.
- **Do not bet on a self-supervised 3D pretraining objective without reading VGGT-Ω §4 first.** They tried new-view synthesis, RayZer/E-RayZer variants, token generation, NeRF and Gaussian targets, token masking, cross-frame object discrimination, and temporal ordering. Only momentum student-teacher helped, and even that "had little impact on most benchmarks." If you attempt this, know that you are attempting something a well-resourced team already failed at, and have a reason why your angle differs.
- **Do not assume a monocular-depth win transfers to multi-view geometry, or vice versa.** DA3 beats DAv2 at monocular depth *and* VGGT at pose; MoGe-2 beats everything at monocular metric geometry and is not a multi-view model at all; VGGT is a Best Paper and is 29 AUC points behind RoMa v2 at two-view pose. These are different tasks with different winners, and the "foundation model" framing obscures that more than it clarifies it.
