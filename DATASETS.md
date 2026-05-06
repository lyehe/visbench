# Dataset catalog

Every dataset module in `visbench/datasets/` exposes:
- `download(data_root)` — fetch and prepare data on disk.
- `iter_pairs(data_root, max_pairs=None)` — yield evaluation pairs.

`Auto` = downloads end-to-end with no manual steps. `Scripted` = public source but needs custom URL handling (gdown bypass, multi-file fetch). Datasets requiring a signed data-use agreement, registration form, or sign-up are **not shipped** — see "Excluded" below.

## Pose evaluation (relative pose AUC@5/10/20°)

| Dataset | Source | Paper | License | Mode |
|---|---|---|---|---|
| `megadepth1500` | https://github.com/zju3dv/LoFTR | Sun et al., *LoFTR*, CVPR 2021 | Apache 2.0 (code); MegaDepth: see https://www.cs.cornell.edu/projects/megadepth/ | Scripted |
| `yfcc` | https://github.com/zju3dv/LoFTR | LoFTR test split of YFCC100M-OANet | YFCC100M CC-BY (varies per image) | Scripted |
| `imc_pt` | https://www.cs.ubc.ca/~kmyi/imw2020/data.html | Jin et al., *Image Matching Challenge*, IJCV 2021 | Academic (per CVL UBC) | Scripted |
| `eth3d` | https://www.eth3d.net/datasets | Schöps et al., CVPR 2017 | CC BY-NC-SA 4.0 | Scripted |
| `tum_rgbd` | https://vision.in.tum.de/data/datasets/rgbd-dataset | Sturm et al., IROS 2012 | CC BY 4.0 | Auto (HF Hub) |
| `kitti_odometry` | https://www.cvlibs.net/datasets/kitti/eval_odometry.php | Geiger et al., CVPR 2012 | CC BY-NC-SA 3.0 | Auto (HF Hub) |
| `seven_scenes` | https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/ | Shotton et al., CVPR 2013 | Microsoft Research License | Scripted |
| `co3d` | https://github.com/facebookresearch/co3d | Reizenstein et al., ICCV 2021 | CC BY-NC 4.0 | Scripted |
| `blendedmvs` | https://github.com/YoYo000/BlendedMVS | Yao et al., CVPR 2020 | CC BY 4.0 | Auto (HF Hub) |
| `icl_nuim` | https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html | Handa et al., ICRA 2014 | CC BY-NC | Scripted |
| `zeb_*` (12 subsets + `zeb_all`) | https://github.com/xuelunshen/gim | Shen et al., *GIM*, ICLR 2024 | Per-subset (academic) | Scripted |
| `dtu_mvs` | https://roboimagedata.compute.dtu.dk/?page_id=36 | Aanaes et al., *Large-Scale Data for MVS*, IJCV 2016 | Open (DTU release) | Scripted (HF mirror `jzhangbs/mvsdf_dtu`) |
| `tartanair`, `tartanair_wide` | https://huggingface.co/datasets/theairlabcmu/tartanair | Wang et al., *TartanAir*, IROS 2020 | BSD-3-Clause | Scripted (HF, multi-TB; pick env subsets) |
| `strecha`, `strecha_wide` | https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/ | Strecha et al., CVPR 2008 | Academic | Scripted |

## Homography evaluation (corner-error AUC@3/5/10 px, MMA)

| Dataset | Source | Paper | License | Mode |
|---|---|---|---|---|
| `hpatches` | https://huggingface.co/datasets/vbalnt/hpatches | Balntas et al., CVPR 2017 | CC BY-NC-SA 4.0 | Auto (HF Hub) |
| `hpatches_rotated` | derived from `hpatches` | — | inherits HPatches | Synthetic |
| `hpatches_patches` | derived from `hpatches` | — | inherits HPatches | Synthetic |
| `oxford_affine` | https://www.robots.ox.ac.uk/~vgg/research/affine/ | Mikolajczyk & Schmid, PAMI 2005 | Academic (free for research) | Auto |
| `evd` | https://huggingface.co/datasets/vrg-prague/evd | Mishkin et al., 2015 (Extreme View Dataset) | CC BY 4.0 | Auto (HF Hub) |
| `wxbs` / `wxbs_hf` | https://cmp.felk.cvut.cz/wbs/ + HF mirror | Mishkin et al., BMVC 2015 | Academic | Scripted / Auto |

## Cross-modal (MINIMA suite)

| Dataset | Source | Paper | License | Mode |
|---|---|---|---|---|
| `minima_diode` | https://diode-dataset.org/ + MINIMA | DIODE: Vasiljevic et al., 2019; MINIMA: 2024 | MIT (DIODE) | Scripted |
| `minima_dsec` | https://dsec.ifi.uzh.ch/ | Gehrig et al., RAL 2021 | CC BY-NC-SA 4.0 | Scripted (gdown bypass) |
| `minima_mmim` | https://github.com/StaRainJ/RGB-Multimodal-Image-Matching-Database | Jiang et al., 2021 | Academic | Scripted |
| `minima_metu_vistir` | https://github.com/OnderT/XoFTR (METU-VisTIR release) | Tuzcuoglu et al., *XoFTR*, 2024 | Academic | Scripted |
| `minima_md1500_*` | MegaDepth-1500 + MINIMA syn modalities | MINIMA, 2024 | inherits MegaDepth + MINIMA | Scripted |

## Localization (qualitative)

| Dataset | Source | Paper | License | Mode |
|---|---|---|---|---|
| `aachen_pairs` / `aachen_hloc` | https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/ | Sattler et al., CVPR 2018 | Academic | Scripted |
| `inloc` / `inloc_queries` | https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/ | Taira et al., CVPR 2018 | Academic | Scripted |
| `silda` | https://github.com/abmmusa/silda | Musa et al. | Academic | Scripted |
| `sacre_coeur` | bundled with [hloc](https://github.com/cvg/Hierarchical-Localization) | — | Apache 2.0 (hloc) | Auto |

## Synthetic stressors (no external download — derived from `hpatches`)

`synthetic`, `synthetic_advanced`, `synthetic_geometric`, `synthetic_compound`, `resolution_sweep` — generated on-the-fly from `hpatches`, cached under `datasets/hpatches/_synthetic_*/`.

## Excluded (form-gated or signed-agreement — not shipped)

These benchmarks gate downloads behind a registration form, signed data-use agreement, or terms-of-use click-through. Visbench omits modules for them. To use one, sign the relevant agreement and write a small custom dataset module that points at your local copy.

| Dataset | Why excluded | Source |
|---|---|---|
| **ScanNet / ScanNet1500** | ScanNet Terms-of-Use form | https://kaldir.vc.in.tum.de/scannet/ |
| **ScanNet++** | Signed data-use agreement | https://kaldir.vc.in.tum.de/scannetpp/ |
| **Map-free Relocalization** | Signed DUA | https://research.nianticlabs.com/mapfree-reloc-benchmark/ |
| **Extended CMU-Seasons** | Registration form | https://visuallocalization.net/datasets/ |
| **Oxford RobotCar** | Registration form | https://robotcar-dataset.robots.ox.ac.uk/ |
| **Tanks & Temples** | Google Form for download links | https://www.tanksandtemples.org/ |
| **nuScenes (incl. RUBIK)** | Sign-up form | https://www.nuscenes.org/sign-up |
| **MegaDepth-X** | Multi-step preprocessing on a 1865-scene training corpus (not a clean test set) | upstream MegaDepth + MASt3R |
