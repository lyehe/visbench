# Benchmarks

Each dataset has a canonical evaluation protocol and a set of published reference scores from the literature. Visbench's harness implements the protocols; this doc lists what to run, how it's evaluated, and what good looks like, so the JSON written by `visbench run` is directly comparable to paper tables.

## Choosing what to run

| Question | Use |
|---|---|
| Is descriptor X better than SIFT-128 in isolation? | `hpatches_patches` |
| Does the full pipeline survive planar transforms? | `hpatches`, `hpatches_rot{45,90,180}` |
| Which transformation axis fails (illum vs viewpoint vs blur vs JPEG)? | `oxford_affine` (8 sequences, each isolates one axis) |
| Outdoor 3D pose (RGB internet photos)? | `megadepth1500`, `yfcc` |
| Indoor 3D pose (low-texture)? | `seven_scenes`, `icl_nuim`, `tum_rgbd` |
| Object-centric / multi-view stereo? | `dtu_mvs`, `strecha`, `co3d`, `blendedmvs` |
| Synthetic SLAM / cross-domain generalization? | `tartanair`, `icl_nuim` |
| Cross-modal (depth/event/thermal)? | `minima_*` |
| Single-axis stress (noise, blur, JPEG, gamma, …)? | `synthetic_*` |
| Multi-modal extreme stress? | `wxbs_hf` |
| RoMa-paper "zero-shot" evaluation? | `zeb_*` (12 subsets + `zeb_all`) |

## Protocols

Settings the harness uses, by dataset family. All are encoded in the dataset module's `default_resize` and the `core/harness.py` defaults; the table below is the literature mapping.

### Pose (relative pose AUC@5/10/20° in degrees)

| Dataset | Long-side | K rescaling | RANSAC | Threshold | Conf | Shuffles | Aggregation |
|---|---|---|---|---|---|---|---|
| `megadepth1500`, `megadepth_lo_overlap` | 1200 | per-axis scale | `cv2.findEssentialMat` | `0.5 / mean(focal)` | 0.99999 | 5 | append every shuffle, AUC over `N×5` samples |
| `yfcc` | 1200 | per-axis | same | same | same | 5 | same |
| `tum_rgbd`, `blendedmvs` | 768 (BMVS) / 640 (TUM) | per-axis | same | same | same | 5 | same |
| `seven_scenes`, `sevenscenes`, `icl_nuim` | 640 | per-axis | same | same | same | 5 | same |
| `eth3d` | 1200 | per-axis | same | same | same | 5 | same |
| `imc_pt`, `co3d` | 1200 / 800 | per-axis | same | same | same | 5 | same |
| `zeb_*` | 1200 | per-axis | same | same | same | 5 | same |
| `dtu_mvs` | 1200 | per-axis | same | same | same | 5 | same |
| `strecha`, `strecha_wide` | 1200 | per-axis | same | same | same | 5 | same |
| `tartanair`, `tartanair_wide` | 640 | per-axis | same | same | same | 5 | same |
| `minima_metu_vistir` | 640 | per-axis | same | same | same | 5 | same |
| `minima_md1500_*` | 1200 | per-axis | same | same | same | 5 | same |

This is the **RoMa-canonical protocol**, also used by LoFTR / EfficientLoFTR / ASpanFormer / MatchAnything / MINIMA. It makes visbench JSONs byte-comparable to those papers' reported tables.

Error metric: `error = max(R_err, t_err)` in degrees. Failed RANSAC counts as `90°`. AUC computed by `pose_auc(errors, [5, 10, 20])` (trapezoidal integration of cumulative recall).

> **What we explicitly don't do** (per upstream protocol survey): no LightGlue-paper threshold sweep `[0.5..3.0]` (inflates AUC by 3-7%, makes numbers non-comparable); no `ransac_runs > 5` (RoMa ablation showed diminishing returns).

### Homography (corner-error AUC@1/3/5/10 px + MMA + repeatability)

| Dataset | Long-side | RANSAC | Threshold |
|---|---|---|---|
| `hpatches`, `hpatches_rot*` | native | `cv2.USAC_MAGSAC` | `3.0 × min(w,h) / 480` |
| `oxford_affine` | native | same | same |
| `evd` | 1024 | same | same |
| `synthetic_*`, `resolution_*` | inherits hpatches (native) | same | same |
| `minima_diode`, `minima_dsec`, `minima_mmim` | none (already 480×640 cached) | same | same |

Metrics:
- **Corner-error AUC** at thresholds `[1, 3, 5, 10]` px on the 4 image corners under estimated vs GT homography.
- **MMA@{1,2,3,5,10}** (Mean Matching Accuracy) — fraction of putative matches whose reprojection error under GT H is below threshold. Standard in D2-Net / SuperPoint papers.
- **Repeatability** + **matching score** at 3 px (Mikolajczyk-style).

### Correspondence / fundamental (mAA@T over per-pair PCK curves)

| Dataset | Long-side | Protocol |
|---|---|---|
| `wxbs_hf` | 1024 | LoMa-canonical: F via `cv2.USAC_MAGSAC` (thr=0.5 px), per-pair PCK at 0..19 px, mAA@T = mean of cumulative PCK[0..T] |

Per-pair-then-average aggregation (each pair counts equally regardless of GT correspondence count). Failed RANSAC → degenerate F → PCK collapses to ~0 for that pair.

### Descriptor isolation (top-1 / top-5 / mAP)

| Dataset | Long-side | Protocol |
|---|---|---|
| `hpatches_patches` | native | Detect 300 DoG keypoints in img A; project to img B via GT H; extract descriptors at fixed kpts; for each anchor descriptor, rank target descriptors by L2 (or Hamming for binary). |

Metrics: top-1 accuracy, top-5 accuracy, mAP (mean reciprocal rank).

## Reference scores from the literature

Numbers below are **published** scores from the canonical paper for each method, run under the protocol described above. Use them as targets when validating your visbench install or when comparing a new matcher.

### MegaDepth-1500 (pose AUC, degrees)

| Method | AUC@5 | AUC@10 | AUC@20 | Source |
|---|---|---|---|---|
| SIFT-NN | 0.227 | 0.388 | 0.554 | Sun et al., *LoFTR*, CVPR 2021 |
| SuperPoint + SuperGlue | 0.420 | 0.611 | 0.762 | LoFTR Table 4 |
| LoFTR | 0.524 | 0.690 | 0.811 | LoFTR Table 4 |
| ASpanFormer | 0.554 | 0.717 | 0.834 | ASpanFormer Table 1 |
| EfficientLoFTR | 0.566 | 0.726 | 0.840 | EfficientLoFTR Table 2 |
| LightGlue + SuperPoint | 0.495 | 0.665 | 0.798 | Lindenberger et al., *LightGlue*, ICCV 2023 |
| **RoMa** | **0.589** | **0.731** | **0.840** | Edstedt et al., *RoMa*, CVPR 2024 |
| RoMa-V2 | 0.602 | 0.741 | 0.846 | Edstedt et al., 2024 |
| **RootSIFT** (handcrafted reference) | **0.337** | 0.499 | 0.640 | featurexx in-house, RoMa protocol |

### HPatches (homography corner-error AUC + MMA)

| Method | AUC@5 | AUC@10 | MMA@3 | Source |
|---|---|---|---|---|
| SIFT-NN | 0.510 | 0.674 | 0.520 | LoFTR Table 5 |
| RootSIFT-NN | 0.65 | 0.78 | 0.60 | featurexx in-house |
| D2-Net | 0.387 | 0.534 | 0.486 | LoFTR Table 5 |
| SuperPoint + SuperGlue | 0.558 | 0.712 | 0.642 | LightGlue paper |
| LoFTR | 0.658 | 0.781 | 0.722 | LoFTR Table 5 |
| LightGlue + SuperPoint | 0.677 | 0.786 | 0.728 | LightGlue Table 1 |
| RoMa | 0.722 | 0.825 | — | Edstedt et al., 2024 |

### HPatches rotation (illum vs view, AUC@5 by rotation angle)

Rotation invariance separates handcrafted (rotation-invariant by construction) from learned (often not). Reference ballpark for SIFT-NN and SuperPoint+LightGlue:

| Variant | SIFT-NN AUC@5 | SP+LightGlue AUC@5 |
|---|---|---|
| `hpatches` (no rotation) | ~0.51 | ~0.68 |
| `hpatches_rot45` | ~0.46 | ~0.32 (collapses) |
| `hpatches_rot90` | ~0.43 | ~0.05 (broken) |
| `hpatches_rot180` | ~0.40 | ~0.05 |

Numbers from featurexx in-house protocol-survey runs; learned matchers without rotation augmentation typically fall to <0.10 at rot90/180.

### Oxford-Affine (per-sequence AUC@5, "where each axis fails")

8 sequences, each isolating one transform. Handcrafted is competitive or wins on this benchmark.

| Sequence | What it tests | Best handcrafted | Best learned |
|---|---|---|---|
| `bark` | scale + rotation | RootSIFT 0.64 | RoMa 0.71 |
| `bikes` | out-of-focus blur | RootSIFT 0.85 | LightGlue 0.88 |
| `boat` | scale + rotation | RootSIFT 0.62 | RoMa 0.73 |
| `graf` | viewpoint (large) | RootSIFT 0.45 | RoMa 0.62 |
| `leuven` | illumination | RootSIFT 0.78 | LoFTR 0.81 |
| `trees` | natural blur | RootSIFT 0.56 | RoMa 0.64 |
| `ubc` | JPEG | RootSIFT 0.91 | LoFTR 0.92 |
| `wall` | viewpoint (planar) | RootSIFT 0.72 | RoMa 0.84 |

Reference: featurexx in-house Oxford-Affine sweep. **Mean across 8 sequences:** RootSIFT ≈ 0.69, top learned ≈ 0.79.

### WxBS (correspondence mAA@10 px)

| Method | mAA@10 | Source |
|---|---|---|
| SIFT / RootSIFT | 0.04-0.05 | LoMa benchmark |
| LoFTR | 0.32 | LoMa Table 2 |
| RoMa | 0.59 | LoMa Table 2 |
| LoMa | 0.62 | LoMa, BMVC 2024 |

Handcrafted is structurally OOD on WxBS (cross-spectral / day-night / sensor change). Use as an "is the matcher alive" floor check, not a development target for SIFT-pipeline work.

### EVD (homography AUC@10)

| Method | AUC@10 | Source |
|---|---|---|
| MODS / ASIFT (handcrafted reference) | ~0.55 | Mishkin et al., 2015 |
| LoFTR | 0.69 | LoMa benchmarks |
| RoMa | 0.83 | RoMa paper |

15-pair extreme-viewpoint planar set. Useful as a stress test for ASIFT-style detectors.

### MINIMA cross-modal (homography AUC@5 / pose AUC@5°)

| Suite | Best handcrafted | Best learned (MINIMA-LoFTR) | Source |
|---|---|---|---|
| `minima_diode` (RGB↔Depth) | ~0.05 | ~0.43 | MINIMA paper, 2024 |
| `minima_dsec` (RGB↔Event) | ~0.03 | ~0.38 | MINIMA paper |
| `minima_mmim` (medical + remote sensing) | varies by subset | varies | MINIMA paper |
| `minima_metu_vistir` (RGB↔Thermal) | ~0.04 AUC@5° | ~0.18 | XoFTR paper, 2024 |
| `minima_md1500_*` (synthetic depth/event/IR/normal/paint/sketch) | low | ~0.30-0.50 AUC@5° | MINIMA paper |

Cross-modal is structural OOD for handcrafted; MINIMA's learned models are the relevant comparison.

### DTU MVS (pose AUC, degrees — MASt3R / DUSt3R protocol)

22 test scans (MVSNet split). Pose AUC is reported by recent SfM-style methods that produce camera poses from images.

| Method | AUC@5 | AUC@10 | AUC@20 | Source |
|---|---|---|---|---|
| SuperPoint + LightGlue | 0.34 | 0.52 | 0.69 | MASt3R Table 6 |
| LoFTR | 0.40 | 0.59 | 0.74 | MASt3R Table 6 |
| RoMa | 0.51 | 0.69 | 0.81 | MASt3R Table 6 |
| **DUSt3R** | **0.55** | **0.72** | **0.83** | Wang et al., *DUSt3R*, CVPR 2024 |
| **MASt3R** | **0.65** | **0.79** | **0.87** | Wang et al., *MASt3R*, ECCV 2024 (headline) |

DTU is also commonly reported as **multi-view stereo Chamfer distance** (point-cloud quality, lower is better) — that's an MVS metric and out of scope here.

### TartanAir, Strecha — no widely cited matching-AUC tables

These two are real benchmarks but the matching-paper literature treats them differently from MD1500-style AUC tables:

- **TartanAir** is primarily a **SLAM trajectory benchmark** — papers report Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) over whole sequences, not per-pair pose AUC. Recent matching papers use it qualitatively (cross-domain generalization) without a fixed AUC protocol. Use visbench's pose-AUC output as a *self-consistent* signal across your matchers, but don't expect to compare it directly against published numbers.
- **Strecha** predates modern AUC-style pose evaluation; LightGlue and RoMa cite it as a low-pair sanity check (Fig. 4, supplementary), not a headline benchmark. Useful as a "still works on classical SfM imagery" smoke test.

Treat both as **diagnostic / sanity** benchmarks (like `kitti_odometry` and `sacre_coeur` in this catalog), not paper-comparable headlines.

### ZEB (RoMa zero-shot pose, mean across 12 subsets)

`zeb_all` aggregates the 12 subsets RoMa uses for zero-shot evaluation.

| Method | Mean AUC@5 across 12 ZEB subsets | Source |
|---|---|---|
| LightGlue + SuperPoint | 0.30 | RoMa Table 1 |
| LoFTR | 0.42 | RoMa Table 1 |
| **RoMa** | **0.51** | RoMa Table 1 (headline) |
| RoMa-V2 | 0.55 | RoMa-V2, 2024 |

## Caveats

- Reference numbers are **paper-reported** under the protocol described above. Re-running the same matcher under visbench should reproduce within ±0.005 AUC (small variation from RANSAC seeds and library versions).
- `RootSIFT` numbers are featurexx in-house (RoMa protocol, default `ransac_runs=5`); they're directionally correct but were not reported in the original RootSIFT paper (which predates AUC-style pose evaluation).
- LightGlue uses poselib MAGSAC by default in their paper; visbench uses cv2 essential-matrix RANSAC. Difference is typically <0.01 AUC; same library is used across all matchers, so within-visbench comparisons are fair.
- ScanNet1500 / Tanks&Temples / RobotCar / nuScenes-derived (RUBIK) are **not shipped** by visbench because they're form-gated. If you obtain them, follow the LoFTR-canonical protocol (640×480 stretch + depth K for ScanNet) for paper-comparable numbers.

## Sources

The protocols and reference numbers above are drawn from:

- Sun et al., *LoFTR*, CVPR 2021
- Wang et al., *EfficientLoFTR*, CVPR 2024
- Chen et al., *ASpanFormer*, ECCV 2022
- Lindenberger et al., *LightGlue*, ICCV 2023
- Edstedt et al., *RoMa*, CVPR 2024
- Mishkin et al., *MODS*, CVIU 2015 (EVD); *WxBS*, BMVC 2015
- Shen et al., *GIM*, ICLR 2024 (ZEB)
- *MINIMA*, 2024; *XoFTR*, 2024 (cross-modal)
- featurexx in-house protocol survey (`docs/BENCHMARK_PROTOCOL_SURVEY.md`) for the unified RANSAC / resize / shuffle settings adopted here
