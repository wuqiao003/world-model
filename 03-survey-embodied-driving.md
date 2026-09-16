# World Models for Embodied AI / Robotics and Autonomous Driving
### Literature survey, 2024 – August 2026 (emphasis on Aug 2025 – Aug 2026)
*Compiled 4 August 2026. All claims traced to a primary source where possible; uncertain items are flagged inline with ⚠.*

---

## 0. How to read this report

Three things changed between August 2025 and August 2026, and they organize everything below:

1. **"World model" stopped meaning "video generator" and started meaning "action-conditioned predictor you can close a loop around."** The two 2026 surveys of the area both make this the organizing axis, and both say the same thing: visual realism is a weak proxy for control utility.
2. **World-Action Models (WAMs) became a distinct architecture class.** A pretrained video-diffusion backbone *is* the policy; actions are just another modality denoised inside the same sequence. Cosmos Policy, DreamZero, LingBot-VA, GigaWorld-Policy, and (per NVIDIA) GR00T N2 all sit here. This is the single biggest architectural shift of the period.
3. **Real-time became the gating constraint, and several systems crossed it.** DreamZero runs a 14B autoregressive video diffusion model at 7 Hz closed-loop; DreamDojo distills to ~10.8 FPS; NVIDIA OmniDreams hits 68 FPS single-camera / 105 FPS-per-camera four-view. Before mid-2025, essentially nothing in this literature ran fast enough to be in a control loop.

**Two anchor surveys** (useful as maps, and I lean on them for taxonomy and for a few 2026 arXiv IDs I could not independently re-verify):
- *World Model for Robot Learning: A Comprehensive Survey* — arXiv [2605.00080](https://arxiv.org/abs/2605.00080), May 2026.
- *World Models for Robotic Manipulation: A Survey* — arXiv [2606.00113](https://arxiv.org/abs/2606.00113), June 2026.
- For 3D/4D specifically: *3D and 4D World Modeling: A Survey* — <https://worldbench.github.io/survey> (VideoGen / OccGen / LiDARGen taxonomy).

---

## 1. Table of key papers and systems

### 1A. Robotics / embodied

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 1 | **UniPi** (Du et al.) | NeurIPS 2023 | Founding "video-prediction-as-policy": text-conditioned video generation + separate inverse dynamics model to recover actions. ⚠ arXiv ID not re-verified this session. | [NeurIPS page](https://papers.nips.cc/paper_files/paper/2023) |
| 2 | **UniSim** (Yang et al.) | ICLR 2024 (Outstanding Paper) | Learning an interactive real-world simulator from heterogeneous internet + robot data; the "one simulator for everything" framing. ⚠ arXiv ID not re-verified. | — |
| 3 | **RoboDreamer** (Zhou et al.) | ICML 2024 | Compositional world modeling: decompose instructions into reusable primitives so generation generalizes to unseen object/action combinations. ⚠ ID not re-verified. | — |
| 4 | **iVideoGPT** (Wu et al.) | arXiv 2405.15223, May 2024 (⚠ NeurIPS 2024 venue not re-verified) | Compressive tokenization + autoregressive transformer over (observation, action, reward) tokens — scalable *interactive* world model for MBRL. | [arXiv](https://arxiv.org/abs/2405.15223) |
| 5 | **EnerVerse** | arXiv 2501.01895, Jan 2025; **NeurIPS 2025** | Chunk-wise autoregressive video diffusion + sparse context memory + Free Anchor Views; EnerVerse-D pairs it with 4D Gaussian Splatting as a sim2real data flywheel. ~280 ms / 8-step action chunk on one RTX 4090. | [arXiv](https://arxiv.org/abs/2501.01895) · [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2025/file/360052c2c6d0c8ec24c476d43236ab25-Paper-Conference.pdf) |
| 6 | **GR00T N1** | arXiv 2503.14734, 17–18 Mar 2025 | Open dual-system humanoid VLA (Eagle-2 VLM at 10 Hz + DiT flow-matching action head at 120 Hz); trains on real trajectories + human video + synthetic "neural trajectories". | [arXiv](https://arxiv.org/abs/2503.14734) |
| 7 | **DreamGen** / GR00T-Dreams | arXiv 2505.12705, May 2025 | 4-stage data engine: fine-tune a video world model on the target embodiment → generate videos → label with IDM or latent actions → train policy. The canonical "dreamed data" result. | [arXiv](https://arxiv.org/abs/2505.12705) · [project](https://research.nvidia.com/labs/gear/dreamgen/) |
| 8 | **EnerVerse-AC (EVAC)** | arXiv 2505.09723, May 2025 | Action-conditional multi-view world model used explicitly as *both* data engine and policy evaluator; trained with deliberately-included failure trajectories. | [arXiv](https://arxiv.org/abs/2505.09723) |
| 9 | **V-JEPA 2 / V-JEPA 2-AC** | arXiv 2506.09985, 11 Jun 2025 (Meta) | Latent (non-generative) action-conditioned world model post-trained on <62 h of DROID; zero-shot MPC/CEM planning to image goals on Franka arms in two unseen labs. | [arXiv](https://arxiv.org/abs/2506.09985) · [code](https://github.com/facebookresearch/vjepa2) |
| 10 | **WorldVLA** (DAMO) | arXiv 2506.21539, 26 Jun 2025 | Single autoregressive model unifying action and image understanding/generation; introduces action attention masking to stop autoregressive error propagation in action chunks. | [arXiv](https://arxiv.org/abs/2506.21539) |
| 11 | **RoboScape** | arXiv 2506.23135, Jun 2025; **NeurIPS 2025 spotlight** | Physics-informed embodied world model: joint RGB generation + temporal depth prediction + keypoint-dynamics learning. Trained on 50k AgiBot-World-Beta clips with 32×A800 in ~24 h. | [arXiv](https://arxiv.org/abs/2506.23135) · [OpenReview](https://openreview.net/forum?id=wbZCBBrq3W) |
| 12 | **Genie 3** (Google DeepMind) | Blog, 5 Aug 2025 | First real-time general-purpose interactive world model: 720p, 24 fps, minutes of consistency, ~1 min visual memory, promptable world events. No paper; limited research access. | [DeepMind](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) |
| 13 | **Genie Envisioner** (GE-Base / GE-Act / GE-Sim + EWMBench) | arXiv 2508.05635, 7 Aug 2025; **ICLR 2026** | Unified platform: one instruction-conditioned video-diffusion latent space serving as policy backbone (flow-matching action decoder), neural simulator, and benchmark. | [arXiv](https://arxiv.org/abs/2508.05635) · [project](https://genie-envisioner.github.io/) |
| 14 | **WoW** (World-Omniscient World Model) | arXiv 2509.22642, 26 Sep 2025 | 14B model trained on **2M real robot interaction trajectories** (5,275 tasks, 12 robots); SOPHIA VLM critic iteratively constrains hallucination; co-trained IDM closes imagination→action. WoWBench. | [arXiv](https://arxiv.org/abs/2509.22642) |
| 15 | **Ctrl-World** | arXiv 2510.10125, Oct 2025; **ICLR 2026** | Multi-view (incl. wrist) action-conditioned world model on DROID with pose-conditioned memory retrieval; ranks policies in imagination and lifts π₀.₅-DROID from 38.7% → 83.4% via SFT on synthetic successes. | [arXiv](https://arxiv.org/abs/2510.10125) · [OpenReview](https://openreview.net/forum?id=748bHL2BAv) |
| 16 | **World-in-World** | arXiv 2510.18135, Oct 2025 | First closed-loop platform benchmarking heterogeneous world models by *task success*. Headline: visual quality ≠ task success; controllability does the work. | [arXiv](https://arxiv.org/abs/2510.18135) · [project](https://world-in-world.github.io/) |
| 17 | **GigaBrain-0** + **GigaWorld-0** | arXiv 2510.19430 (Oct 2025) / 2511.19861 (Nov 2025) | Explicit "world model as data engine" stack: video branch + 3D branch (3DGS + differentiable sysID) generating training data; VLA trained on it works on real G1/PiPER robots. | [GigaBrain-0](https://arxiv.org/abs/2510.19430) · [GigaWorld-0](https://arxiv.org/abs/2511.19861) |
| 18 | **WMPO** | arXiv 2511.09515; **ICLR 2026** | On-policy GRPO for VLAs entirely inside a *pixel-space* video world model (deliberately not latent, to match VLA pretraining). Real 5 mm insertion on Mobile ALOHA: 53% base → 60% DPO → **70%**. | [arXiv](https://arxiv.org/abs/2511.09515) · [project](https://wm-po.github.io/) |
| 19 | **RynnVLA-002** | arXiv 2511.17502, 21 Nov 2025 (DAMO) | Successor to WorldVLA; shared vocabulary over image/text/state/action + continuous Action Transformer head. LIBERO 97.4% *without* pretraining; real LeRobot SO-100 +50 points from adding the world model. | [arXiv](https://arxiv.org/abs/2511.17502) |
| 20 | **Cosmos Policy** | arXiv 2601.16163, Jan 2026 (NVIDIA) | "Latent frame injection": encode actions, future states, and *values* as extra latent frames in Cosmos-Predict2 — no architecture change, one post-training stage. LIBERO 98.5%, RoboCasa 67.1%, real ALOHA 93.6%. | [arXiv](https://arxiv.org/abs/2601.16163) |
| 21 | **DreamDojo** | arXiv 2602.06949, 9 Feb 2026; **ICML 2026** (NVIDIA) | Foundation world model pretrained on **44k hours of egocentric human video** using continuous latent actions as proxy labels; distilled to ~10.8 FPS real-time for teleop, policy eval, and MPC. | [arXiv](https://arxiv.org/abs/2602.06949) · [code](https://github.com/NVIDIA/DreamDojo) |
| 22 | **DreamZero** ("World Action Models are Zero-shot Policies") | arXiv 2602.15922, 17 Feb 2026 (NVIDIA) | 14B Wan-2.1-I2V backbone jointly denoising video + action; **7 Hz closed-loop**, >2× generalization over SOTA VLAs on real robots, +42% relative from 10–20 min of *video-only* cross-embodiment demos, new embodiment from 30 min of play. | [arXiv](https://arxiv.org/abs/2602.15922) |
| 23 | **GR00T N2** | NVIDIA GTC keynote, 2026 (⚠ conflicting reports: GTC March 2026 vs GTC Taipei ~26 Jun 2026) | Announced next-gen humanoid foundation model built on the DreamZero WAM architecture; claimed >2× success vs leading VLAs on novel tasks in novel environments; #1 on MolmoSpaces and RoboArena; GA targeted end of 2026. **No paper — vendor claims only.** | [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) |
| 24 | **Fast-WAM** | arXiv 2603.16666, 2026 (⚠ ID via survey 2605.00080, not independently verified) | Contrarian result: the gain from world-action models may come mostly from **video co-training during training**, not from test-time imagination — imagination can be dropped at inference. | — |
| 25 | **Aether** | **ICCV 2025** (arXiv 2503.18945) | Geometry-grounded unified world model: 4D reconstruction + action-conditioned prediction + goal-conditioned planning, with **camera trajectories as the action space**; trained purely on synthetic data, zero-shot to real. | [arXiv](https://arxiv.org/abs/2503.18945) · [code](https://github.com/InternRobotics/Aether) |
| 26 | **TesserAct** | arXiv 2504.20995, Apr 2025 | 4D embodied world model predicting RGB + depth + normals jointly, improving spatial consistency and downstream inverse dynamics. | [arXiv](https://arxiv.org/abs/2504.20995) |
| 27 | **WorldGrow** | arXiv 2510.21682, Oct 2025; **AAAI 2026 Oral** | Unbounded *explicit* 3D scene generation via block-wise 3D inpainting + coarse-to-fine refinement; walkable persistent worlds as an alternative substrate to autoregressive video. | [arXiv](https://arxiv.org/abs/2510.21682) · [code](https://github.com/world-grow/WorldGrow) |

### 1B. Autonomous driving

| # | Name | Venue + date | One-line contribution | Link |
|---|---|---|---|---|
| 28 | **GAIA-1** (Wayve) | arXiv 2309.17080, Sep 2023 | Driving world modeling as multimodal next-token prediction over video/text/action tokens; showed extrapolation to ego behaviors absent from expert data. Authors note it "does not yet run at real time." | [arXiv](https://arxiv.org/abs/2309.17080) |
| 29 | **Copilot4D** (Waabi) | ICLR 2024 (⚠ venue from Waabi's page; arXiv ID not re-verified) | VQ-VAE tokenization of LiDAR + discrete diffusion for 4D point-cloud forecasting; >65% Chamfer reduction at 1 s across nuScenes/KITTI-Odometry/Argoverse2. | [Waabi](https://waabi.ai/research/copilot-4d) |
| 30 | **Vista** | **NeurIPS 2024** (arXiv 2405.17398) | High-fidelity generalizable driving world model (10 Hz, 576×1024) with a unified control interface from high-level commands to trajectory/angle/speed; first use of the WM itself as a *reward* for action evaluation without ground-truth actions. | [arXiv](https://arxiv.org/abs/2405.17398) |
| 31 | **OccSora** | arXiv 2405.20337, May 2024 | Diffusion transformer over a 4D scene tokenizer generating 16 s trajectory-conditioned 4D occupancy — the occupancy-space counterpart to video world models. | [arXiv](https://arxiv.org/abs/2405.20337) |
| 32 | **Doe-1** | arXiv 2412.09627, 13 Dec 2024 | Unifies perception, prediction, and planning as one next-token generation problem over observation/description/action tokens, explicitly framed as *closed-loop* driving. | [arXiv](https://arxiv.org/abs/2412.09627) |
| 33 | **DriveDreamer-2** | **AAAI 2025**, 39(10):10412–10420 | LLM-driven generation of user-specified driving scenarios (trajectory library → HDMap generator → UniMVM multi-view video). FID 11.2 / FVD 55.7; generated data improves 3D detection and tracking. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33130) |
| 34 | **GAIA-2** (Wayve) | arXiv 2503.20523, 26 Mar 2025 | Latent-diffusion surround-view world model, up to 5 consistent cameras at 448×960, structured conditioning on ego dynamics, agents, weather, road semantics, across UK/US/Germany. | [arXiv](https://arxiv.org/abs/2503.20523) · [Wayve](https://wayve.ai/thinking/gaia-2/) |
| 35 | **DrivingSphere** | **CVPR 2025**, pp. 27531–27541 | Closed-loop simulation from a 4D occupancy world (OccDreamer) rendered to multi-view video (VideoDreamer) — occupancy as the controllable substrate, video as the sensor model. | [CVPR OA](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_DrivingSphere_Building_a_High-fidelity_4D_World_for_Closed-loop_Simulation_CVPR_2025_paper.html) |
| 36 | **Cosmos-Drive-Dreams** (NVIDIA) | arXiv 2506.09042, Jun 2025 | Cosmos-1 specialized to driving; open pipeline + weights + **81,802 synthetic clips**. Demonstrates downstream gains in 3D lane detection, 3D object detection, and policy learning, *including on top of large real datasets*. | [arXiv](https://arxiv.org/abs/2506.09042) · [code](https://github.com/nv-tlabs/Cosmos-Drive-Dreams) |
| 37 | **DriveVLA-W0** | arXiv 2510.12796; **ICLR 2026** | Future-image prediction as dense self-supervision fixes the "supervision deficit" of action-only VLA training and **amplifies the data scaling law**: at 70M frames, +28.8% ADE (VQ variant) and −15.9% collision (ViT variant) vs action-only. | [arXiv](https://arxiv.org/abs/2510.12796) · [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d70423f59c5fdd24f0dd3fa52e34623-Abstract-Conference.html) |
| 38 | **WorldLens** | arXiv 2512.10958; **CVPR 2026 Oral** | Five-axis benchmark (Generation / Reconstruction / Action-Following / Downstream / Human Preference, 24 dimensions) + WorldLens-26K human annotations + a distilled VLM critic. Finding: **"almost all existing world models trigger collisions or off-road drifts."** | [arXiv](https://arxiv.org/abs/2512.10958) · [project](https://worldbench.github.io/worldlens) |
| 39 | **Waymo World Model** | Waymo announcement, Feb 2026 | Genie 3 post-trained for driving, jointly emitting temporally consistent **camera + lidar (4D point clouds)** aligned to Waymo's sensor rig; can convert ordinary dashcam video into full multimodal sim logs. **No paper.** | [Ars Technica](https://arstechnica.com/google/2026/02/waymo-leverages-genie-3-to-create-a-world-model-for-self-driving-cars/) |
| 40 | **DriveDreamer-Policy** | arXiv 2604.01765, 2026 | Geometry-grounded driving world-action model (depth + future video + planning in one modular stack). **89.2 PDMS on NAVSIM v1, 88.7 EPDMS on NAVSIM v2**; ablation shows explicit depth complements video imagination. | [arXiv](https://arxiv.org/abs/2604.01765) |
| 41 | **NVIDIA OmniDreams** | arXiv 2606.03159, Jun 2026 | The strongest closed-loop driving WM result to date: Cosmos-distilled causal AR model on 21k h of driving, **68 FPS single-camera 720p on one GB300 / 105 FPS-per-camera four-view on 16 GPUs**, wired into AlpaSim with the Alpamayo 1 policy. Preserves policy *ranking* vs. reconstruction-based NuRec. | [arXiv](https://arxiv.org/abs/2606.03159) · [PDF](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/paper.pdf) |
| 42 | **World Engine** (OpenDriveLab) | arXiv 2606.19836, 19 Jun 2026 | Post-training loop: mine failures from logs → reconstruct as 3DGS interactive scenes → augment with a behavior world model → RL post-train. On a production stack (80k h logs, >10k sim scenarios): **collision rates −45.5%**, and 200 km of real driving with no disengagement. Fully open-sourced on nuPlan. | [arXiv](https://arxiv.org/abs/2606.19836) · [code](https://github.com/OpenDriveLab/WorldEngine) |
| 43 | **Tesla neural world simulator** | Talks (ICCV 2025; ScaledML 2026), Ashok Elluswamy | Generates **all 8 camera feeds at 36 fps, 5 MP, >1 min** from current video + control actions; used for closed-loop eval, adversarial scenario injection, and failure replay. Same network generalizes to indoor Optimus scenes. **No paper, no numbers.** | [Talk](https://www.youtube.com/watch?v=LFh9GAzHg1c) |

### 1C. Evaluation infrastructure worth knowing

| Name | Venue + date | What it measures | Link |
|---|---|---|---|
| **NAVSIM** / **NAVSIM v2** | NeurIPS 2024 D&B; v2.0 released 28 Feb 2025 | Non-reactive/pseudo-closed-loop AV planning via PDMS → EPDMS; added two-stage pseudo-simulation and reactive traffic agents | [GitHub](https://github.com/autonomousvision/navsim) |
| **Bench2Drive** | NeurIPS 2024 D&B | CARLA closed-loop, 220 routes / 44 scenarios, DS + SR + multi-ability | [GitHub](https://github.com/Thinklab-SJTU/Bench2Drive) |
| NAVSIM↔Bench2Drive correlation study | arXiv 2605.00066, 2026 | PDMS↔DS Spearman ρ=0.90 (n=8, non-monotonic, with rank inversions); Ego Progress is the strongest single predictor (ρ=0.83), collision metric NC far weaker (ρ=0.45) | [arXiv](https://arxiv.org/abs/2605.00066) |
| **EWMBench** | arXiv 2505.09694, May 2025 | Scene / motion / semantic quality of embodied world models | [arXiv](https://arxiv.org/abs/2505.09694) |
| **WorldEval** | arXiv 2505.19017, May 2025 | Can a WM rank real robot policies *and checkpoints*? | [arXiv](https://arxiv.org/abs/2505.19017) |
| **WorldGym** | arXiv 2506.00613, Jun 2025 | WM as a gym environment; policy-rank correlation and value fidelity | [arXiv](https://arxiv.org/abs/2506.00613) |
| **WorldSimBench** | **ICML 2025** | Perceptual + "manipulative" evaluation — does the video support IDM recovery? | — |
| Gemini Robotics in a **Veo** world simulator | arXiv 2512.10675, Dec 2025 | Large-scale offline policy evaluation, OOD testing, safety probing inside a video world simulator | [arXiv](https://arxiv.org/abs/2512.10675) |
| **WorldArena** | arXiv 2602.08971, 2026 (⚠ via survey) | Perception *and* functional utility (data gen, policy eval, planning) in one benchmark | — |
| CVPRW 2026 latent-centric AD taxonomy | CVPR 2026 Workshops, pp. 4510–4519 | Proposes **Closed-loop Safety Gap (CSG)** and **Deliberation Cost (DC)** as diagnostics | [CVPR OA](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Zeng_A_Latent-Centric_Perspective_on_World_Models_for_Autonomous_Driving_Taxonomy_CVPRW_2026_paper.html) |

---

## 2. What actually works on real robots / real driving stacks vs. demo-only

### 2.1 Robotics — solid, reproducible, real-hardware evidence

**Generated data really does train real policies, and the effect sizes are large but not magical.**
DreamGen is the cleanest measurement: across 9 real tasks on Fourier GR-1, Franka, and SO-100, using only 10–13 real trajectories per task, success went **37% → 46.4%** (GR-1, 4 tasks), **23% → 37%** (Franka, 3 tasks), **21% → 45.5%** (SO-100, 2 tasks). The more striking numbers are the zero-to-one ones: a GR00T N1 trained only on pick-and-place scores **0%** on most novel-verb/novel-environment tests, while DreamGen-trained policies reach **43.2%** (new behaviors, seen environment) and **28.5%** (new behaviors, unseen environment). In RoboCasa, scaling neural trajectories 333× produces a **log-linear** improvement, and training *only* on neural trajectories reaches 20.6% average over 24 tasks. NVIDIA carried this into a product: GR00T N1.5 hits 38.3% vs GR00T N1's 13.1% on 12 DreamGen tasks, and the blog is candid that these are not truly zero-shot — the model is explicitly trained on those verbs via DreamGen trajectories.

**Policy improvement by fine-tuning on world-model-generated successes works, at least for instruction following.** Ctrl-World lifts π₀.₅-DROID from **38.7% → 83.4%** on four novel-instruction task categories. The authors are explicit about the scope: this improves *instruction following*, and they "expect that our model is not accurate enough to improve performance in other aspects such as the low-level success rate on previously seen instructions."

**RL inside a world model works on a real robot, on a hard task.** WMPO on a Cobot Mobile ALOHA with 5 mm clearance insertion: base OpenVLA-OFT 53%, offline DPO 60%, WMPO **70%** (30 trials each). Small absolute numbers, small n, but it is a genuine real-robot closed-loop result rather than a video reel.

**Latent-space planning works zero-shot, on easy tasks.** V-JEPA 2-AC, post-trained on <62 h of DROID with no data from the target labs, no rewards, no task-specific training: Franka reach 100%, grasp cup 60%, grasp box 20%, pick-and-place cup 80%, box 50%. Compare Octo (10%/0%/10%/10% on the grasp/place tasks) and a Cosmos baseline (0%/20%/0%/0%). This is the strongest evidence that pixel generation is *not* required — but the task difficulty is very low and it needs image goals, not language.

**WAMs beat VLAs on real hardware, at real-time rates.** DreamZero reports >2× generalization to new tasks/environments over SOTA VLAs in real-robot experiments at 7 Hz closed-loop with a 14B model, plus a genuinely useful practical result: 10–20 minutes of *video-only* demonstrations from another robot or a human buys +42% relative on unseen tasks. Cosmos Policy reports the highest average score on real bimanual ALOHA tasks (93.6%) against diffusion policies, other video-model policies, and fine-tuned VLAs on the same demos.

**One under-appreciated negative result on real hardware.** RynnVLA-002 reports that its *discrete* action variant — the one that performs well in LIBERO — **fails entirely on the real SO-100 arm** due to overfitting and trajectory discontinuity, which is why they added a continuous Action Transformer head. This is a concrete instance of the sim-benchmark/real-robot decoupling that LIBERO scores hide.

### 2.2 Robotics — demo-only or unverified

- **Genie 3** is a research preview. No robot control results, no public access, no paper. DeepMind itself lists "limited interaction duration" (minutes, not hours) and ~1 minute of visual memory. Treat as a capability demonstration, not a tool.
- **GR00T N2** has no paper and no third-party evaluation. The "2× success vs leading VLAs" and "#1 on MolmoSpaces and RoboArena" claims come from an NVIDIA keynote. The underlying DreamZero paper *is* real and peer-reviewable; the N2 product claims are not yet.
- **LIBERO is saturated and should not be used to rank world models.** Cosmos Policy 98.5, LingBot-VA 98.5, Say-Dream-ACT 98.1, Motus 97.7, RynnVLA-002 97.4, VLA-JEPA 97.2. These are within noise of each other and of methods with completely different architectures. The 2605.00080 survey's own conclusion is that "high performance can emerge from multiple design paradigms," and that cross-benchmark transfer is poor — strong RoboTwin numbers do not predict strong CALVIN or SIMPLER numbers.
- **Most "physics-aware" claims are unvalidated in closed loop.** WoW's own framing is that the model's physics understanding is "a probabilistic distribution of plausible outcomes, leading to stochastic instabilities and physical hallucinations" — they need a VLM critic (SOPHIA) at inference to constrain it.

### 2.3 Driving — what is actually in a stack

**Three companies have world models genuinely wired into production or near-production loops, with very different evidence quality:**

- **NVIDIA (OmniDreams + AlpaSim + Alpamayo)** is the best-documented. It is deployed in a closed loop with a real policy and orchestrator, meets a real-time latency budget, and — the key result — **preserves policy ranking**: comparing OmniDreams WAM > Alpamayo 1.5 (4 cam) > (2 cam) > (1 cam) gives the same ordering under OmniDreams as under reconstruction-based NuRec, which is itself a strong proxy for logged reality. Crucially, OmniDreams *degrades much more gracefully than 3DGS reconstruction as the policy deviates from the recorded trajectory* — NuRec's FVD "rises rapidly" off the capture path while OmniDreams stays stable. That is the whole argument for generative over reconstructive simulation, and it is measured rather than asserted. They also show a WAM post-trained from OmniDreams cutting collisions 6.9% → 4.2% with ~1/5 the parameters of the VLA-based Alpamayo 1.5. The paper is honest about the cost: "a video-generation based simulator like OmniDreams natively requires much more compute than a reconstruction-based simulator."
- **Waabi** trains and validates the Waabi Driver almost entirely in Waabi World, with Copilot4D as the world model, and runs real trucks in Texas. Their published safety argument is a digital-twin divergence measurement over ~20-second snippets. Peer-reviewed detail on the *closed-loop validation* claim is thin relative to the marketing.
- **Tesla** has the most impressive raw generation demo (8 cameras, 36 fps, 5 MP, minute-long, real-time enough to "drive" inside) and says it is used for closed-loop eval, adversarial injection, and failure replay. But there is no paper, no benchmark, and no quantitative claim. Treat as existence proof of scale, not as evidence about accuracy.
- **Waymo** shipped a Genie-3-derived model producing joint camera + lidar in Feb 2026. Announcement only; no numbers.
- **OpenDriveLab's World Engine** is the most useful entry for academics because it is the only one with *both* production-scale evidence (−45.5% collisions on >10k industry scenarios; 200 km disengagement-free) *and* a fully open nuPlan-based reproduction with released code, assets, and scenarios.

**Where the driving field's honest self-assessment is harshest:** WorldLens (CVPR 2026 Oral) put a pretrained planner inside generated worlds and found that "high open-loop realism does not guarantee safe closed-loop control; **almost all existing world models trigger collisions or off-road drifts**." And "no existing world model excels universally: those with strong textures often violate physics, while geometry-stable ones lack behavioral fidelity."

**Synthetic driving data as a data engine is the least controversial win.** Cosmos-Drive-Dreams shows gains in 3D lane detection, 3D object detection, and policy learning, and — importantly — "continues to provide measurable gains even when augmenting a large-scale real-world dataset." DriveDreamer-2 similarly improved 3D detection/tracking. This is the closest thing to a settled result in Part B.

**Driving world models as a training signal, not a simulator, is the sleeper result.** DriveVLA-W0's finding that future-image prediction *amplifies the data scaling law* (gains accelerate with data, while action-only supervision saturates) is arguably more consequential than any simulator result, because it says world modeling pays off exactly where the industry is spending money.

---

## 3. Explicitly stated open problems (attributed)

**On evaluation being the central unsolved problem.**
> "Evaluation is the central unresolved problem for world models in robotic manipulation. A predictive model can be visually accurate yet physically wrong, physically plausible yet useless for action…"
> — *World Models for Robotic Manipulation: A Survey*, arXiv 2606.00113, §VIII

**On the missing metric — action alignment.**
> "The missing fidelity metric is action alignment. A manipulation world model should not only predict a plausible next state, but also predict the consequences of the specified robot action. This requires metrics that compare predicted and realized object displacement, contact onset, affordance change, progress toward goals, and executability under an inverse dynamics or controller."
> — arXiv 2606.00113, §VIII-A

**On attribution — you often cannot tell *why* a world model helped.**
> "If a world-model-augmented policy outperforms a baseline, the gain may come from better representations, additional imagined data, auxiliary prediction losses, test-time search, reward shaping, or a combination of these factors. Fast-WAM illustrates this issue by suggesting that predictive representation learning can explain much of the benefit in some systems, even when the architecture appears to support explicit test-time imagination."
> — arXiv 2606.00113, §VIII-B

**On closed-loop exploitation of learned simulators.**
> "A policy trained or searched within a world model actively optimizes against its own predictions… can a policy exploit visual artifacts, reward loopholes, contact hallucinations, or termination mistakes? If so, the model may still be useful for short-horizon ranking or data augmentation, but unsafe as an autonomous post-training environment."
> — arXiv 2606.00113, §VIII-C

**On the causal conditioning gap.**
> "The technical bottleneck is weak action conditioning: many predictive world-model objectives are trained mainly from observation history and task intent, so their futures can be plausible without being causally tied to the robot action to be executed."
> — *World Model for Robot Learning: A Comprehensive Survey*, arXiv 2605.00080, §8.1

**On whether video backbones are actually better than VLM backbones — stated as unresolved.**
> "…whether video-pretrained backbones are consistently superior to matched-scale VLM backbones for robotic control remains an open empirical question; current results should be viewed as suggestive evidence for a promising inductive bias rather than a definitive architectural conclusion."
> — arXiv 2605.00080, §3.3

**On data: failures and contact are missing.**
> "…failure recovery, decision-sensitive variation, and dense physically grounded supervision remain much scarcer than large-scale successful demonstrations." And: "Million-trajectory datasets may still contain too few insertions, slips, jams, deformable interactions, or recovery behaviors to train reliable predictive models."
> — arXiv 2605.00080 §7.2 and arXiv 2606.00113 §VII-G

**On visual quality being the wrong target (measured, not asserted).**
World-in-World's three headline findings: "(1) visual quality alone does not guarantee task success—controllability matters more; (2) scaling post-training with action-observation data is more effective than upgrading the pretrained video generators; and (3) allocating more inference-time compute allows WMs to substantially improve closed-loop performance." They also report that world-model gains are **least** pronounced in manipulation, "due to the challenge of precisely simulating contact-rich interactions and physical dynamics."
> — arXiv 2510.18135

**On driving world models failing closed-loop despite looking good.**
> "High open-loop realism does not guarantee safe closed-loop control; almost all existing world models trigger collisions or off-road drifts, underscoring that photometric realism alone cannot yield functional fidelity."
> — WorldLens, arXiv 2512.10958 / CVPR 2026

**On what a driving world model still can't do (Ctrl-World's manipulation analogue).**
> "Our model can fail on tasks involving precise interactions or long-horizon reasoning, and performance is sensitive to initial observations."
> — Ctrl-World, arXiv 2510.10125

**On the precision ceiling of WAMs.**
> "…it inherits limitations common to behavior cloning on tasks requiring sub-centimeter precision, such as key insertion or fine assembly. Our diverse pretraining strategy prioritizes breadth, which may underrepresent the dense demonstrations needed for these high-precision manipulation."
> — DreamZero, arXiv 2602.15922

**On V-JEPA 2's horizon and goal-specification limits.**
> Future work is needed because the model plans "up to roughly 16 seconds into the future," longer-horizon tasks "without requiring sub-goals will require further innovations in modeling," and V-JEPA 2-AC "currently relies upon tasks specified as image goals" rather than language.
> — V-JEPA 2, arXiv 2506.09985

**On DreamGen's remaining gap.**
> "Enabling zero-shot generalization to novel behaviors and novel environments with robot embodiments with zero ground-truth data still remains an open research question." Also: "Our tasks are relatively simple and cover a limited portion of the robot's full kinematic capabilities."
> — DreamGen, arXiv 2505.12705

**On open-loop metrics as closed-loop proxies in driving.**
The NAVSIM↔Bench2Drive study finds ρ=0.90 but "non-monotonic, with clear ranking inversions," and that "methods that maximize safety at the expense of progress rank highly in NAVSIM but underperform in closed-loop due to timeout and slow-driving penalties." A 3-metric proxy (NC×DAC×EP) matches the full 5-metric PDMS. Note n=8 paired methods — this is a small sample.
> — arXiv 2605.00066

**On the compute tradeoff nobody escapes.**
> "…a video-generation based simulator like OmniDreams natively requires much more compute than a reconstruction-based simulator, exposing a tradeoff between quality and compute."
> — OmniDreams, arXiv 2606.03159

---

## 4. Under-explored gaps: what an 8–32 GPU academic lab can do (possibly with no robot)

The dominant labs have three advantages you cannot beat: hardware fleets, proprietary data at the 10⁴–10⁵ hour scale, and 14B-parameter training runs. So do not try to build a better foundation world model. Every direction below is chosen because it is *bottlenecked by ideas and careful experimentation rather than by scale*, and because at least one 2025–2026 primary source explicitly names it as open.

### 4.1 The highest-leverage gap: action-faithfulness metrics and counterfactual benchmarks

The manipulation survey names this outright as "the missing fidelity metric," and the causal-conditioning gap in the robot-learning survey is the same problem from the modeling side. Nobody has a standard way to ask: *if I change the action, does the predicted future change in the right way?*

**Concretely buildable with public data and ~8 GPUs.** DROID, Bridge V2, RT-1, and Open-X all contain (observation, action, next-observation) tuples. You can construct a **counterfactual action benchmark**: take a held-out trajectory prefix, feed the ground-truth action chunk and a set of systematically perturbed chunks (mirrored, scaled, time-reversed, gripper-inverted, orthogonal-displacement), and score whether the model's predicted end-effector displacement, contact onset, and object motion track the perturbation. Metrics like *action-response gain* (∂predicted-displacement / ∂commanded-displacement) and *action-insensitivity rate* (fraction of perturbations producing indistinguishable rollouts) do not exist in the literature and require no robot. This directly implements the survey's request.

**Why it's tractable and citable:** it is a benchmark + diagnostic paper. Evaluating existing open checkpoints (Cosmos-Predict2.5-2B, DreamDojo-2B, Ctrl-World, iVideoGPT, IRASim, GE-Base, RoboScape) costs inference, not training. The field currently has EWMBench, WorldSimBench, RBench, WorldArena, and WorldLens — all of which measure something adjacent, none of which isolates action responsiveness as a scalar with a controlled intervention design.

### 4.2 Settle the "video backbone vs. VLM backbone" question at matched compute

Survey 2605.00080 explicitly calls this "an open empirical question." Nobody has run the controlled experiment because industry labs compare their 14B video model to somebody else's 3B VLM. At 2B scale on LIBERO + RoboCasa + SIMPLER, with identical data, identical action heads, identical training budget, and a Cosmos-Predict2.5-2B vs. a matched-parameter VLM, this is a 32-GPU experiment. It is the kind of paper that gets cited by everyone in the area for three years.

**Pair it with the Fast-WAM hypothesis** (that the benefit comes from video co-training during training, not test-time imagination). Fast-WAM asserts it for one architecture. A systematic ablation — video-co-trained-but-imagination-disabled vs. full imagination vs. no video — across three architecture families would either confirm a very important simplification or falsify it. Either outcome is publishable, and both save the field compute.

### 4.3 Failure and recovery data — the acknowledged hole in every dataset

Both surveys say it in almost the same words: recovery behaviors, slips, jams, and contact-rich transitions are undersampled. EnerVerse-AC's one distinctive design choice was to *deliberately* include failure trajectories in training. This is a data-contribution opportunity that needs simulators, not robots.

Build a public **failure/recovery corpus in RoboCasa / ManiSkill3 / LIBERO** with: (a) scripted and policy-induced failures, (b) paired counterfactual branches from the same state (one action leading to failure, one to success), (c) annotated failure onset. Then show that world models trained with it produce better *hallucination rates* and better *policy-rank correlation*, not just better FVD. The counterfactual-branch structure is the novel part — it gives you supervised signal for exactly the causal-conditioning gap in §4.1.

### 4.4 Closed-loop exploitation: does policy optimization break the learned simulator?

Survey 2606.00113 calls this "the most stringent reliability test" and notes that nobody runs it. WoVR reportedly identifies simulator reliability as the central bottleneck and proposes world-model/policy co-evolution. But there is no systematic study of **reward hacking inside learned world models**.

This is a small-compute, high-insight project: take WMPO's or VLA-RFT's setup in simulation (where you have ground truth), run GRPO for many more steps than the papers do, and characterize the failure curve — at what point does imagined return decouple from real return? Which artifacts get exploited (object teleportation, gripper penetration, premature termination)? Does world-model refinement on policy rollouts actually fix it, or just move the exploit? You need a simulator for ground truth, not a robot.

### 4.5 Latent / JEPA world models are the compute-efficient frontier and are under-populated

V-JEPA 2-AC, VLA-JEPA, JEPA-VLA, and LeWorldModel all point the same way, and the efficiency argument is decisive at your scale: no pixel decoder, no iterative denoising, no 14B backbone. V-JEPA 2-AC's stated limitations are a research agenda handed to you:
- **Hierarchical prediction across timescales** — explicitly named as needed for long-horizon tasks without hand-specified subgoals.
- **Language-conditioned goals** in the JEPA latent space — explicitly named as an open direction, with V-JEPA 2's LLM alignment offered as a starting point.

Both are doable with public DROID/Open-X data at 8–32 GPUs. The second one in particular is a well-defined, unsolved, high-visibility problem with an obvious evaluation protocol.

### 4.6 Driving: everything you need is public, and the field's own correlation evidence is thin

- **Expand the NAVSIM↔Bench2Drive correlation study.** The existing one has **n=8 paired methods**. That is a weak basis for the claim that open-loop metrics predict closed-loop driving, and the study itself flags ranking inversions. Running more planners through both benchmarks is pure engineering with public code and would substantially firm up (or overturn) a load-bearing methodological assumption.
- **Instantiate the CSG / DC diagnostics.** The CVPR 2026 workshop taxonomy *proposes* Closed-loop Safety Gap and Deliberation Cost but does not standardize them. Implementing them over NAVSIM v2 + Bench2Drive + WorldEngine's open nuPlan scenarios is a benchmark contribution.
- **Occupancy-space world models are ~an order of magnitude cheaper than multi-view video** and are the underexplored branch. The OccWorld/OccSora/DrivingSphere line fits comfortably in 8–32 GPUs on nuScenes/OpenOccupancy. The genuinely open idea: use an occupancy world model as a **verifier / reward model** on top of somebody else's video generator, rather than as a generator itself. WorldLens found that texture-strong models violate physics while geometry-stable models lack behavioral fidelity — a cheap geometric verifier gating an expensive video model is the obvious synthesis and I have not found it done.
- **World Engine gives you a production-validated recipe with open code and assets on nuPlan.** Reproducing its failure-mining → 3DGS reconstruction → behavior augmentation → RL loop with a different policy class, or ablating which of the four stages actually carries the gain, is a well-scoped paper on public data.

### 4.7 Real-time distillation at small scale

OmniDreams (68–105 FPS), DreamDojo (~10.8 FPS), and DreamZero (7 Hz control) all establish that latency is now a first-class metric — and all three achieved it through distillation (Self-Forcing, DMD, progressive long-context teachers) plus streaming KV caches and attention sinks. These techniques are published and the base models (Cosmos-Predict2.5-2B, Wan-2.1, DreamDojo-2B) are open. **Distilling an open 2B world model to interactive rates on 8 GPUs, and measuring what the distillation costs in action fidelity, is directly tractable** and answers a question the big labs report only for their own stacks. OmniDreams explicitly notes long-rollout artifacts appear "when the rolling KV cache extends beyond the short teacher's training context" — the quality/latency/horizon tradeoff surface is unmapped.

### 4.8 Memory and persistence over long horizons

Genie 3 tops out at ~1 minute of visual memory. Ctrl-World's key mechanism is pose-conditioned memory retrieval; EnerVerse uses sparse context memory; OmniDreams uses bounded local attention plus attention-sink tokens. Nobody has compared these mechanisms head-to-head. A controlled study — same 2B backbone, same DROID data, four memory mechanisms, measured on *long-horizon action fidelity* rather than FVD — is a clean 32-GPU contribution against a real, universally-acknowledged bottleneck.

### 4.9 Cross-embodiment latent actions from human video

DreamDojo's central trick (continuous latent actions as a unified proxy over 44k h of unlabeled human video) and DreamZero's result (10–20 min of video-only cross-embodiment demos → +42% relative) both suggest latent action codebooks are doing a lot of work and are poorly understood. At small scale on Ego4D/EgoDex + Open-X, you can ask questions the big labs skipped: what does the latent action space actually encode? How much of the gain survives if you replace latent actions with an IDM? How does codebook size trade against embodiment transfer? DreamGen itself notes that latent actions and IDM actions "have similar effects" — that near-equivalence is unexplained and worth a paper.

### 4.10 Things to avoid

- Training a new foundation-scale video world model. You will lose.
- Reporting LIBERO as your headline number. It is saturated (§2.2) and the surveys say so.
- Optimizing FVD/LPIPS/PSNR as a primary objective. Four independent 2025–2026 sources (World-in-World, WorldLens, WorldSimBench, both surveys) say these do not predict control utility.
- Assuming your sim result transfers. RynnVLA-002's discrete variant scored well on LIBERO and failed *entirely* on the real arm.

---

## 5. Uncertainty register

Items I could not independently verify in this session, listed so they can be checked before citation:

- **arXiv IDs taken from the reference list of survey 2605.00080 rather than from the papers themselves**: Fast-WAM (2603.16666), WoVR (2602.13977), WorldArena (2602.08971), LeWorldModel (2603.19312), UniDrive-WM (2601.04453), GigaWorld-Policy (2603.17240), Motus (2512.13030), V-JEPA 2.1 (2603.14482), DrivingGen (2601.01528), GigaBrain-0.5M* (2602.12099), JEPA-VLA (2602.11832), VLA-JEPA (2602.10098).
- **arXiv IDs and/or venues not re-verified**: UniPi, UniSim, RoboDreamer, DriveDreamer (v1), OccWorld, Copilot4D, Vid2World. iVideoGPT's NeurIPS 2024 venue.
- **GR00T N2 date conflict.** NVIDIA's newsroom places the preview at "GTC"; secondary tech coverage variously says GTC March 2026 and GTC Taipei / Computex, 26 June 2026. All GR00T N2 performance claims are vendor-reported with no paper and no third-party replication.
- **Tesla and Waymo world-model claims** rest on talks and blog posts. No quantitative results, no reproducible protocol.
- **Waabi's closed-loop safety validation claim** is documented in press coverage (MIT Technology Review, March 2025) more thoroughly than in peer-reviewed publication.
- **DreamDojo's real-time figure** is reported as 10.81 FPS in the arXiv HTML and 10.93 FPS in the OpenReview abstract; the GitHub says "10 FPS." Minor, but they disagree.
- **The n=8 sample** in the NAVSIM↔Bench2Drive correlation study is small enough that the ρ=0.90 headline should be treated as suggestive.
- I have **not** verified GR00T N1.6 / N1.7 details beyond the NVIDIA newsroom mention that N1.7 entered early access with commercial licensing.
