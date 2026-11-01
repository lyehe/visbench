# Dataset catalog

Every registered dataset, grouped as in `visbench/datasets/`. All are publicly
downloadable without a registration form; `visbench download <name>` fetches
them or prints the exact manual step. `visbench check <name>` verifies a local
copy's ground truth before you trust its numbers.

Data lives under `$VISBENCH_DATA_ROOT/<root>` (default `./datasets/<root>`).
Each module exposes `download(data_root)` and `iter_pairs(data_root, max_pairs=None)`.

## Pose — relative pose AUC@5/10/20° (`datasets/pose/`)

| Name | Root | Pairs | Source | License |
|---|---|---:|---|---|
| `megadepth1500` | `megadepth` | 1500 | [LoFTR test split](https://github.com/zju3dv/LoFTR) | Apache 2.0 (split); MegaDepth images per source |
| `megadepth1500_rot90` | `megadepth` | 1500 | same pairs, image B rotated 90° CCW with K and pose transformed | same |
| `megadepth_lo_overlap` | `megadepth` | 600 | lowest-overlap bins of the same scenes | same |
| `eth3d` | `eth3d` | 2561 | [ETH3D high-res multi-view](https://www.eth3d.net/datasets), top-5 co-visible partners per image | CC BY-NC-SA 4.0 |
| `aachen_v1_1`, `aachen_v1_1_hard` | `aachen_v1.1` | model-dependent | [Aachen Day-Night v1.1](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/) database SfM pairs; needs `pycolmap` (`pip install visbench[aachen]`) | Academic |

## Correspondence — WxBS protocol, mAA@10px (`datasets/correspondence/`)

| Name | Root | Pairs | Source | License |
|---|---|---:|---|---|
| `hardmatch` | `hardmatch` | 830 scored (900 test) | [HardMatch](https://github.com/davnords/HardMatch) release v1.0.0 | Wikimedia Commons per image |
| `hardmatch_val` | `hardmatch` | 73 scored | validation categories of the same release | same |
| `hardmatch_rot90` | `hardmatch` | 830 | test split, image B rotated 90° CCW | same |
| `wxbs` | `wxbs` | 32 | [vrg-prague/wxbs](https://huggingface.co/datasets/vrg-prague/wxbs) (ungated) | Academic |
| `wxbs_rot90` | `wxbs` | 32 | image B rotated 90° CCW | Academic |

## Homography (`datasets/homography/`)

| Name | Root | Pairs | Metric | Source |
|---|---|---:|---|---|
| `hpatches` | `hpatches` | 580 | corner-error AUC@3/5/10px, MMA | [HPatches](https://github.com/hpatches/hpatches-dataset) (CC BY-NC-SA 4.0) |
| `hpatches_rot45`, `_rot90`, `_rot180` | `hpatches` | 580 each | same | in-plane rotation of image B |
| `hpatches_patches` | `hpatches` | — | descriptor top-1 / mAP | descriptor isolation on fixed keypoints |
| `satast` | `satast` | 39 | pooled reprojection AUC@10px | [SatAst](https://github.com/georg-bn/satast); images served over rsync (`download_images.sh`) |

## Cross-modal — MINIMA suite (`datasets/crossmodal/`)

| Name | Root | Task | Source |
|---|---|---|---|
| `minima_metu_vistir` | `METU_VisTIR` | pose | [XoFTR METU-VisTIR](https://github.com/OnderT/XoFTR) RGB-thermal |
| `minima_diode` | `diode` | homography | [DIODE](https://diode-dataset.org/) RGB-depth (MIT) |
| `minima_dsec` | `DSEC` | homography | [DSEC](https://dsec.ifi.uzh.ch/) RGB-event |
| `minima_mmim` | `mmim` | homography | [MMIM](https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods) medical / remote sensing |
| `minima_md1500_{depth,event,infrared,normal,paint,sketch}` | `megadepth_syn` | pose | [MINIMA](https://github.com/LSXI7/MINIMA) renders of MegaDepth-1500 |

## Stress — derived from HPatches (`datasets/stress/`)

`synthetic_jpeg_q10`, `synthetic_noise_30`, `synthetic_motion_blur_15`,
`synthetic_low_light`, `synthetic_fog`, `synthetic_scale_0p5`,
`synthetic_perspective_strong`, `synthetic_rot90_low_light`: one strong
setting per degradation family, generated on first use and cached next to the
HPatches data.

## Not included

- **Form-gated or license-agreement data**, by policy: ScanNet-1500 / ScanNet++,
  Map-free (its Hugging Face copies are unofficial re-uploads), RUBIK (needs
  nuScenes), IMC (Kaggle), Oxford RobotCar, Tanks and Temples, uCO3D, Extended
  CMU Seasons. A dataset behind a plain Hugging Face click-through would
  qualify; none of these has an official one.
- **MegaDepth-X**: a reconstruction dataset with no published matching split.
- **Removed in 2026-11 as rarely or never reported by recent matching papers**:
  YFCC, Oxford affine, Strecha, Sacre Coeur, SILDA, ICL-NUIM, TUM-RGBD,
  7-Scenes, KITTI, DTU, CO3D, BlendedMVS, TartanAir, EVD, ZEB, IMC
  PhotoTourism, the resolution sweep, and the speed-only Aachen / InLoc pair
  lists. They remain in git history (`git log -- visbench/datasets/`).
