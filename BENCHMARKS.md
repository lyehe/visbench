# Benchmarks

What to run, how each benchmark is scored, and the published numbers it
should be compared with. The harness implements each reference protocol, so a
`visbench run` JSON is comparable to the paper tables listed here.

## Choosing what to run

| Question | Run |
|---|---|
| Standard comparison, as in recent papers | `visbench standard --panel core` (MegaDepth-1500, HardMatch, WxBS, HPatches, SatAst) |
| Hard appearance change (sketch↔photo, day↔night, decades apart, aerial↔ground) | `hardmatch` |
| Wide baseline + modality change | `wxbs` |
| Rotation robustness | `megadepth1500_rot90`, `hardmatch_rot90`, `wxbs_rot90`, `hpatches_rot{45,90,180}` |
| Extreme scale + rotation | `satast` |
| Cross-modal (thermal, depth, event, medical) | `--panel crossmodal` |
| Which degradation breaks a method | `--panel stress` |

## Protocols

### Pose — `megadepth1500*`, `eth3d`, MINIMA pose sets

LoFTR / RoMa / LoMa protocol: long side resized to `default_resize` (1200 for
MegaDepth and ETH3D) with K scaled per axis; essential matrix via
`cv2.findEssentialMat`, threshold 0.5 px / mean focal, confidence 0.99999;
5 shuffled RANSAC runs per pair; error = max(rotation, translation-direction)
angle; AUC@5/10/20°.

### Correspondence — `hardmatch*`, `wxbs*`

WxBS / HardMatch reference evaluator (`hardmatch` v1.0.0, LoMa supplement C.1):
matches mapped back to original-image pixels; F via `cv2.USAC_MAGSAC`, 0.25 px,
confidence 0.999, 100k iterations; per pair, PCK of the annotated
correspondences under the fitted F (Hartley-Zisserman symmetric epipolar
distance) at 0..19 px; mAA@10px = mean PCK over 0..10 px. Pairs with fewer
than 8 annotations are skipped, and HardMatch also skips `dynamic` pairs. Fed
the same matches, the visbench implementation reproduces the reference
evaluator exactly.

### Homography — `hpatches*`, stress sets, MINIMA homography sets

Corner-error AUC@1/3/5/10 px with `cv2.findHomography` (threshold 3 px scaled
by image size / 480), plus MMA@1..10 px, repeatability and matching score.

### Point homography — `satast`

RoMa v2 reference (`romav2/benchmarks/satast.py`): homography fitted to the
matches in normalized coordinates (`USAC_DEFAULT`, 0.001, confidence
0.99999999, 100k iterations); every annotated point of A is mapped into B and
the reprojection errors of all points are pooled; a failed fit counts each of
its points as infinite error; AUC@5/10/20/30 px, headline AUC@10px.

### Pair caps and result reuse

`--max-pairs` (or `--max-pairs-pose` / `--max-pairs-homography`) draws a
seeded, scene-balanced sample instead of the first N pairs; the cap and seed
are stored in the result. `batch` and `standard` reuse an existing result only
when its schema version, cap and seed match and it scored at least one pair.
A dataset that yields no pairs fails the cell instead of scoring 0.

## Reference scores

Stored as fractions in `visbench/reference_scores.json`; `visbench report`
prints them as `ref:` rows. LoMa and RoMa v2 values were checked against the
arXiv HTML on 2026-11-01.

### LoMa Table 2 (arXiv 2604.04931)

| Method | MD-1500 AUC@5/10/20 | WxBS mAA@10 | HardMatch mAA@10 |
|---|---|---:|---:|
| SP+LightGlue | 43.8 / 61.8 / 76.4 | 40.4 | 34.8 |
| ALIKED+LightGlue | 48.1 / 65.7 / 79.3 | 43.9 | 35.7 |
| LoFTR | 52.8 / 69.2 / 81.2 | 50.7 | 33.1 |
| MASt3R | 42.4 / 61.5 / 76.9 | 34.5 | 33.6 |
| UFM | 41.5 / 57.9 / 72.4 | 53.3 | 33.9 |
| RoMa | 62.6 / 76.7 / 86.3 | 72.6 | 48.1 |
| RoMa v2 | 62.8 / 77.0 / 86.6 | 64.8 | 46.5 |
| LoMa-B128 | 55.1 / 71.2 / 83.2 | 61.5 | 48.2 |
| LoMa-B | 55.7 / 71.8 / 83.6 | 68.7 | 51.1 |
| LoMa-L | 56.5 / 72.7 / 84.3 | 70.6 | 53.5 |
| LoMa-G | 56.1 / 72.2 / 84.0 | 73.4 | 54.3 |

The JSON holds every row of the table (also DISK, ALIKED, DeDoDe-G, LoMa
descriptor, SP+SG, DISK+LG, VGGT). HardMatch paper numbers use the paper split
scored at 640×640; the released v1.0.0 data changed about 10 pairs for
licensing and scores at the released 1024-px images, so expect small
differences (the package README quotes LoMa at 50.6).

### SatAst AUC@10px

| Method | AUC@10px | Source |
|---|---:|---|
| RoMa | 23.5 | RoMa v2 Table 9 |
| UFM | 1.8 | RoMa v2 Table 9 |
| RoMa v2 | 37.0 | RoMa v2 Table 9 |
| SP+SuperGlue | 19.8 | LoMa supplement Table 9 |
| ALIKED+LightGlue | 12.1 | LoMa supplement Table 9 |
| LoMa-G | 22.9 | LoMa supplement Table 9 |

### HPatches AUC@3/5/10px (LoMa supplement Table 8)

| Method | AUC@3 / 5 / 10 |
|---|---|
| SP+LightGlue | 64.2 / 75.5 / 85.5 |
| ALIKED+LightGlue | 66.2 / 76.9 / 86.3 |
| LoMa-B128 | 67.2 / 78.1 / 87.8 |
| LoMa-G | 66.4 / 77.5 / 87.3 |

Older HPatches, MegaDepth and MINIMA rows (LoFTR, LightGlue, ASpanFormer,
EfficientLoFTR, MINIMA and XoFTR papers) remain in the JSON with their sources.

## Sanity values measured with visbench

Use these to confirm a local setup reproduces the protocol before trusting new
numbers (OpenCV SIFT via `--method sift`, CPU, 2026-11-01):

| Benchmark | Pairs | SIFT | Note |
|---|---:|---:|---|
| `megadepth1500` AUC@5 | 60 (scene-balanced) | 0.286 | LoFTR's published SIFT-NN: 0.227 |
| `hardmatch` mAA@10px | 60 | 0.132 | |
| `wxbs` mAA@10px | 32 | 0.046 | published SIFT ≈ 0.05 |
| `hpatches` AUC@3px | 116 | 0.547 | |
| `satast` AUC@10px | 39 | 0.585 | see below |

On SatAst, a matcher that returns the annotations' own homography scores 0.870
AUC@10px (the annotation-noise ceiling) and a random matcher 0. SIFT's high
score is real: SatAst is dominated by in-plane rotation and scale change,
which SIFT is invariant to while most learned matchers are not (LoMa reports
that it struggles with rotation there).

LoMaTRT-G through `romaxx:lomatrt:loma_G` (512×512 engines, 1024 keypoints)
gave 0.474 AUC@5 on 300 MegaDepth-1500 pairs and 0.480 mAA@10px on HardMatch,
below the paper's LoMa-G (784×784, 4096 keypoints) as expected for the
smaller setting.

## Caveats

- WxBS mAA differs between papers for the same method (RoMa: 72.6 in LoMa,
  60.8 in RoMa v2 Table 9). The JSON uses LoMa's numbers because they share
  HardMatch's protocol.
- Rotated and stress variants have no published tables; compare against your
  own run on the parent benchmark.
- `vismatch:loma-{b128,b,l,g}` use 4096 keypoints, the paper setting; LoMa's
  own evaluation additionally runs at a 784×784 internal resolution.
  `romaxx:lomatrt:*` runs at its engines' fixed size and keypoint count, so
  compare it with the paper only as a lower-resolution variant.
- Dense `romaxx:` matchers are scored on 5000 sampled matches, the RoMa
  pose-benchmark convention (`VISBENCH_ROMAXX_NUM_CORRESP`).
- Three released HardMatch test pairs contain a few annotations outside the
  image; the reference evaluator keeps them and so does visbench
  (`visbench check hardmatch` lists them).
