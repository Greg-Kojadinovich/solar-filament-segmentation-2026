# V4: training changes, rationale, and possible improvements

## Purpose and status

V4 tests whether giving the YOLO detector and crop refiner more opportunity
to learn improves solar-filament instance segmentation. It retains the model
family rather than introducing a larger architecture at the same time.

The [V4 notebook](../Notebooks/solar_filament_CV_V4.ipynb) was launched as
[Kaggle version 1](https://www.kaggle.com/code/gregkoja/solar-filament-cv-v4-longer-training)
with explicit user authorization. This document describes the configured
experiment, not a completed result. **No V4 leaderboard score is recorded here,
and training does not automatically submit predictions.**

| Historical submitted run | Public PQ |
|---|---:|
| V1: semantic U-Net | 0.29 |
| Focused V2: higher-resolution U-Net fine-tuning | 0.32 |
| V3: YOLO + crop refiner, offline disjoint-mask repair | 0.35 |
| V4 | Not yet measured |

The rejected, overlapping-mask V3 submission is not a scored baseline.
The valid V3 score belongs to submission 58316663.

## 1. Exact changes

| Setting | Original V3 run | V4 | Reason |
|---|---|---|---|
| Detector epoch ceiling | 60 | **120** | Allow learning beyond the previous budget stop |
| Detector wall-time allowance | 75 minutes | **None in application code** | Avoid ending because the training allowance expires |
| Detector early-stopping patience | 8 | **15** | Allow more time through noisy validation fluctuations |
| Refiner epoch ceiling | 20 | **40** | Give local boundary refinement more training opportunity |
| Refiner wall-time allowance | 30 minutes | **None in application code** | V3 completed only four full refiner epochs |
| Refiner early-stopping patience | 5 | **8** | Allow additional validation opportunities without unlimited training |
| Detector optimizer selection | `auto`, which selected AdamW | **Explicit AdamW** | Avoid a hidden optimizer switch when the schedule gets longer |
| Detector initial LR | Auto-selected 0.002 | **0.002** | Preserve the observed V3 optimizer setting |
| Detector momentum / Adam first-moment coefficient | Auto-selected 0.9 | **0.9** | Preserve the observed V3 setting |
| Detector weight decay | 0.0005 | **0.0005** | Keep regularization unchanged |
| Detector bias warmup LR | Auto branch sets 0 | **Explicit 0** | Preserve warmup behavior when leaving `auto` |
| Refiner optimizer | AdamW, LR 0.0003, weight decay 0.0001 | **Unchanged** | Avoid changing optimizer and training duration together |
| Refiner cosine schedule | Spans 20 epochs | **Spans 40 epochs** | Match the longer refiner ceiling |
| Ultralytics installation | Version range; actual run used 8.4.211 | **Pinned to 8.4.211** | Reduce training-library drift |
| Mask validity during selection | Original calibration allowed overlaps; CSV repaired afterward | **Disjoint masks during calibration and export** | Evaluate the mask representation that can actually be submitted |
| Initialization | Pretrained YOLO and ImageNet refiner encoder | **Same pretrained sources, new run** | Preserve V3 outputs and avoid claiming unsupported resume |

**Epoch ceilings are not promises of completed epochs.** Early stopping or
Kaggle platform limits can still end training. Removing our wall-time checks
does not remove platform restrictions, storage constraints, or GPU failures.

### What did not change

- YOLO11s-seg detector at 1024-pixel input, batch size four.
- ResNet18/U-Net refiner at 256-pixel crops, batch size 16.
- Square crop context of 1.8 times the instance extent, minimum crop side 96
  original-resolution pixels.
- Seed 2026 and the fixed observation-day-grouped train/tuning/holdout split.
- Separate annotator target sets, kept in the same partition for each physical
  image; one foreground class rather than chirality prediction.
- Detector augmentation settings; no CLAHE added to this run.
- Crop augmentation, Dice/BCE refiner loss, and float32 loss reductions.
- Calibration on the same seeded subset of up to 32 tuning images.
- Detector confidence candidates 0.15 and 0.25, refiner thresholds 0.5 and 0.65,
  and original-resolution minimum areas 20 and 64 pixels.
- Comparison of YOLO-only and refined masks before selecting an export mode.
- Full-resolution 2048-by-2048 masks and manual competition submission.

## 2. Why longer training is reasonable, but not sufficient

### The refiner has the strongest evidence of unfinished learning

V3 refiner tuning loss decreased at every completed epoch:

| Epoch | Tuning Dice/BCE loss |
|---|---:|
| 1 | 0.163618 |
| 2 | 0.144484 |
| 3 | 0.136369 |
| 4 | 0.131818 |

That is approximately a **19.4% reduction** between epochs 1 and 4.
The saved runtime reports `budget_truncated`, not convergence or early
stopping. V4 removes this particular restriction.

However, better crop loss does not guarantee better instance PQ. A crop can
contain a neighboring filament, and a well-segmented union of two filaments
can still be the wrong instance prediction.

### The detector's late improvement was less decisive

V3's trainer reported 48 epochs, but progress records show **47 full epochs
plus an interrupted epoch 48 at 18/207 batches**. The highest observed
validation mask mAP50 / mAP50-95 was **0.620 / 0.206 at epoch 46**.
Later results fluctuated near this level rather than improving monotonically.

V4 permits a longer schedule, but continued validation gains must justify it.
Detection/segmentation mAP is not the competition's PQ metric.

### V2 illustrates the risk of simply adding epochs

Focused V2 selected its best tuning-loss checkpoint at additional epoch 13,
then stopped after 21 additional epochs. Training loss kept falling without
a better tuning checkpoint. V4 therefore retains early stopping and best-weight
selection rather than exporting the final weights unconditionally.

See the [three-run EDA report](V1_V2_V3_MASK_EDA_V4.md) for the full training
curves, image overlays, and all-image measurements.

## 3. Why explicitly choose AdamW?

There is **no verified universally best optimizer** for this dataset.
AdamW is the best-supported choice for this particular comparison because
V3 actually used it, and retaining it avoids another major experimental change.

AdamW combines adaptive parameter updates with decoupled weight decay.
Learning rate controls update scale; weight decay regularizes the weights.
Neither mechanism compensates for incorrect instance targets or postprocessing.

Ultralytics `auto` is a heuristic choice, not a hyperparameter optimization
experiment. In the inspected 8.4.211 implementation it selects according to
estimated iterations and may choose MuSGD instead of AdamW for a longer run.
It can also override user-supplied initial learning-rate settings.
V4 pins AdamW and the observed V3 detector settings to avoid that hidden switch.

The detector and refiner do **not** use the same learning rate:

- Detector: **0.002**, through Ultralytics' optimizer parameter groups.
- Refiner: **0.0003**, through PyTorch AdamW.

The refiner's cosine learning-rate schedule now spans 40 rather than 20 epochs.
Consequently, V4 changes the optimization trajectory, not just the number of
updates. It is not a perfectly isolated test of elapsed training time.

## 4. Mask validity is part of the prediction pipeline

The original V3 CSV had overlapping masks in **99 test images**. Kaggle rejected
it because a pixel cannot belong to multiple submitted instances in one image.
The offline repair removed shared pixel claims from later masks and discarded
empty or undersized remnants, producing the valid 0.35 submission.

V4 applies the same deterministic ownership policy **before scoring calibration
candidates**, rather than waiting until after selecting thresholds:

1. Construct each coarse or refined instance mask.
2. Retain the proposal order.
3. Give contested pixels to the earliest retained mask.
4. Remove already-owned pixels from later masks.
5. Drop masks below the selected area cutoff after trimming.
6. Independently validate nonempty masks, IDs, RLE roundtrips, and zero overlap
   before announcing that the CSV is prepared.

This prevents the known format failure. It does **not** prove that the first
proposal is the correct owner of each contested pixel.

V4 also rejects calibration whose policy or checkpoint provenance does not
match the current experiment. V3's original pre-repair tuning scores cannot
be presented as scores for the repaired or V4 predictions.

## 5. Initialization, stopping, and reproducibility

### Fresh pretrained training, not V3 continuation

V4 starts from YOLO11s pretrained segmentation weights and an ImageNet
ResNet18 refiner encoder. It does **not** load V3's trained weights or restore
its optimizer state. Calling this a resumed V3 run would be inaccurate.

The separate run name/output directory preserves V3 artifacts. Matching settings,
annotation hashes, and checkpoint hashes guard against accidental reuse.

### Checkpoint selection

- YOLO selects its best checkpoint using Ultralytics validation fitness, not
  directly competition PQ.
- The refiner saves its lowest tuning-loss checkpoint.
- Refiner early stopping uses a minimum significant loss improvement of
  0.0001; checkpoint saving still retains any strictly lower tuning loss.
- The selected checkpoints are subsequently compared/calibrated using local
  instance PQ with disjoint masks.

These are different selection criteria. A lower training loss or higher detector
mAP alone is not sufficient evidence to submit a new candidate.

### Validation already performed before launch

- Python syntax checks and notebook configuration checks.
- Tiny real-data detector and crop-refiner training with the V4 code.
- Verification of saved explicit AdamW settings and disabled deadlines.
- Checkpoint reload, calibration policy checks, and smoke-export blocking.
- Twelve regression tests covering mask geometry/agreement, disjoint-mask
  behavior, export checks, and V4 notebook settings.

These checks establish pipeline behavior, not full-resolution convergence or
leaderboard improvement.

## 6. Other possible improvements, ranked by the error they address

These are **future experiments**, not changes silently included in the running V4.
Keep the current run stable and compare completed evidence before adding them.

| Priority | Observed risk or diagnostic | Candidate improvement | Validation needed |
|---|---|---|---|
| 1 | Threshold selection is based on 32 tuning images | Recalibrate on the full tuning partition after the initial duration comparison | Stable parameter ranking and PQ, with selection confined to tuning data |
| 2 | Disjoint trimming can give contested pixels to the wrong instance | Use detector confidence and per-pixel refinement probabilities for ownership; evaluate duplicate suppression separately | Lower FP/merges without losing matched instances; still zero overlap |
| 3 | Refiner trains on true bounds but predicts on detector bounds | Jitter training crop position/scale; evaluate proposal-derived training crops | Better full-frame disjoint PQ, not just lower crop loss |
| 4 | A crop may include several filaments | Add the coarse detector mask as an input channel | Refiner retraining and a controlled comparison showing better instance isolation |
| 5 | Repaired V3 contains 253 disconnected masks | Evaluate small-island cleanup while preserving instance identity | Better matched-instance IoU/PQ without deleting valid thin branches |
| 6 | Detector misses small or faint instances | Broader confidence calibration, then higher-resolution inference or training | Recall gains at acceptable FP and runtime cost |
| 7 | Performance varies by observing conditions | Stratify labeled validation by instance area, contrast, limb distance, date, and station where metadata supports it | Identify a repeatable failure group before changing augmentation/preprocessing |
| 8 | One model's predictions are unstable | Consider grouped-fold or multi-seed models and mask-aware ensembling | Measured gain after duplicate/ownership handling, justified against compute cost |

### Do not automatically split disconnected masks

V3's 253 disconnected masks may include erroneous islands, overlap-trimmed
fragments, or genuinely separated visible parts of one structure.
Splitting every component creates extra instances and may increase FP.
Keeping only the largest component may discard valid filament branches.
Use labeled tuning examples to decide.

### Do not interpret additional foreground as better segmentation

Across the 180 test images:

- V2: **1,470 instances**, **2,547,244 foreground pixels**.
- Repaired V3: **1,370 instances**, **2,694,933 foreground pixels**.

V3 predicts fewer but larger masks overall. This could mean better continuity,
wrong merges, or wider boundaries. Test predictions alone cannot establish
which explanation is correct.

### Higher resolution and larger models are not automatic next steps

Higher resolution may preserve thin structures but costs memory and throughput.
A larger YOLO model may help capacity but increases the cost of each trial.
First determine whether V4 is limited by missed objects, object grouping,
or boundary quality. The refiner cannot recover an object the detector never
proposes.

## 7. How to decide whether V4 is better

The local instance metric has the form

$$
PQ=\frac{\sum_{(p,g)\in TP} IoU(p,g)}
{|TP|+\tfrac12|FP|+\tfrac12|FN|}.
$$

The local matcher requires IoU strictly greater than 0.5. Both separation and
pixel boundaries matter. A mask that merges two filaments can have reasonable
foreground overlap but poor instance PQ.

Before deciding to submit:

1. Check both stopping reasons, completed epoch counts, and validation curves.
2. Compare YOLO-only and refined PQ on the same tuning images under the same
   disjoint policy.
3. Inspect FP, FN, matched-instance IoU, merge/split examples, and results by
   target size. Do not select solely on visual appeal.
4. For a fair training comparison, recalibrate the original V3 weights under
   the same updated policy; the offline-repaired 0.35 remains a historical,
   not perfectly controlled, comparator.
5. Assess the chosen configuration on a separate held-out partition, noting
   that the project's existing V1 holdout has already been inspected.
6. Validate the final serialized CSV and retain hashes, calibration, weights,
   actual package versions, and prediction audits.
7. Submit only after reviewing the candidate. A higher public PQ is the
   observed competition result, not a guaranteed consequence of longer training.

The comparison also changes patience, learning-rate schedules, and the timing
of disjoint-mask calibration. Do not attribute any future improvement solely
to epochs, AdamW, or YOLO.

## 8. Relevant files and references

- [V4 notebook](../Notebooks/solar_filament_CV_V4.ipynb)
- [V4 Kaggle metadata](../kaggle-v4/kernel-metadata.json)
- [V1/V2/V3 mask EDA and figures](V1_V2_V3_MASK_EDA_V4.md)
- [Shared mask validation and offline repair](../Python%20Files/submission_masks.py)
- [V4 configuration regression test](../pythontests/test_v4_notebook.py)
- [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/)
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

The notebook configuration and saved V3 log are the source of the numeric
settings above. Installed implementation behavior is version-specific;
general documentation is not a substitute for recording actual run versions.
