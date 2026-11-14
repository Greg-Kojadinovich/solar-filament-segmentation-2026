# Five-fold day-grouped YOLO + refiner: prepared, not launched

> **Historical protocol, superseded:** V5 now trains all five short folds by
> default (5 detector / 2 ResNet34-refiner epochs per fold), targets two hours
> total with cooperative budgets, and exports a validated five-model ensemble competition
> CSV. Read [the current V5 instructions](V5_FAST_FIVE_FOLD_SUBMISSION.md).
> The longer schedules and absence of CSV export described below are historical.

The [standalone grouped notebook](../Notebooks/solar_filament_CV_V5_grouped.ipynb)
is a separate evaluation experiment. It does not modify the active V4 notebook,
launch a Kaggle job, submit predictions, or create an ensemble.

## What five folds do

The 707 labeled physical images are divided into five outer folds, keeping
whole observation days together. Each image appears in outer evaluation exactly
once. Every annotator's labels for the same photograph stay with that photograph.
Fold assignment balances physical-image counts, not annotation counts or scores.

For each outer fold:

1. Hold out approximately 20% of labeled images for outer evaluation.
2. Reserve 10% of the remaining days for **inner tuning**.
3. Train the detector and refiner only on inner-training data.
4. Use only inner tuning for best checkpoints, detector confidence, refinement
   threshold, area filter, and YOLO-only versus refined-mode selection.
5. Freeze that selection and evaluate it on the outer images.

This is five-fold outer CV with an inner holdout, **not an additional inner
five-fold search**. It uses approximately **72% training / 8% inner tuning /
20% outer evaluation**, not 80% training. Inner tuning is necessary because
using outer evaluation for early stopping and threshold selection would bias
the reported outer scores.

All labeled data have already been involved in project development, and the
old holdout has been inspected. This is not a new untouched final benchmark.
Day grouping reduces same-day leakage but does not prevent the same solar
structure from appearing on neighboring days.

## Exact partitions for the supplied dataset

Counts are unique physical images, followed by annotator-specific target sets
in parentheses. Crop counts are instance annotations, not necessarily distinct
physical filaments; different annotators may label the same structure.

| Fold index | Training images (sets) | Inner tuning images (sets) | Outer images (sets) | Training instance crops |
|---|---:|---:|---:|---:|
| 0 | 510 (820) | 55 (97) | 142 (237) | 5,917 |
| 1 | 510 (835) | 55 (93) | 142 (226) | 5,977 |
| 2 | 508 (824) | 58 (93) | 141 (237) | 5,886 |
| 3 | 511 (837) | 55 (90) | 141 (227) | 5,987 |
| 4 | 512 (837) | 54 (90) | 141 (227) | 5,825 |

Fixed-split V4 trains on 499 physical images, 827 target sets, and 5,842
instance crops. This CV design adds only **9–13 physical training images per
model**, and one fold even has fewer annotation crops. Its principal benefit
is evaluating split robustness, not dramatically expanding each model's data.
Across folds, the training partition rotates; that is not new independent data.

If the goal is substantially more training data in one final model, a separate
final-fit protocol could use more labeled images after selecting settings and
training duration. It must not be presented as an independently evaluated
model on labels it trained on.

## Model and mask settings

The notebook inherits V4's:

- Fresh pretrained YOLO11s-seg and ImageNet ResNet18/U-Net initialization in
  **every fold**, never weights trained on that fold's outer labels.
- 1024-pixel detector input, batch four, up to 120 epochs, patience 15.
- 256-pixel context crops, batch 16, up to 40 refiner epochs, patience eight.
- Explicit detector AdamW at 0.002; refiner AdamW at 0.0003.
- No custom wall-time cutoffs, with platform limits and early stopping retained.
- Same augmentation and 32-inner-image calibration limit/grid.
- Disjoint masks during both calibration and outer evaluation.
- Pinned Ultralytics 8.4.211 and recorded actual package versions.

Five folds train ten networks: five detectors plus five refiners. At the earlier
7–9-hour ceiling estimate for one V4-sized job, five runs could cost roughly
35–45 GPU-session hours. Actual runtime depends on each partition, hardware,
early stopping, and platform constraints. This is not a guaranteed duration.

## How to run later, without changing V4

1. Import [the grouped notebook](../Notebooks/solar_filament_CV_V5_grouped.ipynb) as a
   **new Kaggle notebook**, attach the competition, and enable Internet and GPU.
2. Inspect the printed plan and counts before executing the training cell.
3. Leave `FOLDS_TO_RUN = [0]` for the first job. In separate saved sessions use
   `[1]`, `[2]`, `[3]`, and `[4]`. Do not accidentally launch duplicate folds.
4. Keep identical seed, data, settings, helper code, and run name across fold
   sessions so the shared plan hash matches. Each fold writes a separate
   `fold_N` directory.
5. Download each full output, including `cv_plan.json`, `requirements-actual.txt`,
   checkpoints, provenance, calibration, runtime/history files, and
   `fold_N/fold_result.json`.
6. To collect, attach or download those result directories, put their paths in
   `COLLECT_ROOTS`, execute setup and plan cells, then the **summary cell only**.
   Do not execute the training cell just to collect results.

The default is **one selected fold per execution**, not all five in one session.
Changing `FOLDS_TO_RUN` does not change the shared plan. Changing model settings,
annotation data, split rules, or training source requires a new experiment
directory and a new compatible set of fold results.

Completed matching fold results are reused. Existing completed checkpoint
evidence can be reloaded and calibrated without retraining. Missing pieces of
a checkpoint/runtime pair cause an explicit error. This is **not optimizer-state
resume**: an interrupted refiner with saved best weights but no completed runtime
record requires an explicit recovery decision, not a silently successful fold.

## Scores and evidence

`fold_result.json` records:

- Frozen inner-selected parameters and checkpoint hashes.
- Inner selected scores, explicitly distinct from outer evaluation.
- One outer row per physical image, first averaging available annotator scores.
- Outer means and shared-plan hash.

`cv_summary.json` lists completed and missing folds. **Five-fold mean, sample
standard deviation, and pooled out-of-fold mean are omitted until all five
folds complete.**

On completion:

- Fold mean PQ gives each fold equal weight.
- Pooled out-of-fold mean PQ gives each physical image equal weight.
- Fold sample SD is descriptive variability, not a confidence interval.

The collector rejects wrong plan hashes, missing/duplicate outer images,
nonfinite metrics, and conflicting copies of the same fold result. It never
chooses the highest-scoring copy.

Do not select a “best fold” for reporting and call that cross-validation.
Do not tune architecture or thresholds using outer scores and then describe
the same outer results as unbiased final evaluation.

## Why there is no ensemble or competition CSV

Five independently predicted mask lists cannot simply be concatenated:
duplicates and overlapping masks would violate submission requirements and
can harm PQ. A future ensemble needs validation of cross-model instance
matching, confidence handling, pixel ownership, and final disjoint-mask export.

This notebook's deliverable is **evaluation evidence**, not a submission-ready
ensemble. Nothing is submitted automatically.

## Probability masks, Dice, and possible later work

The network outputs probabilities; thresholding produces binary masks. Dice/BCE
is the crop training loss, while IoU/PQ evaluate overlap and instance matching.
Dice and IoU do not generate probabilities and do not guarantee calibrated
confidence. “Multimodal masking” is not the relevant change for this single
grayscale imaging task; semantic versus instance masks is the key distinction.

The current CV experiment intentionally does not stack new mask losses or
architectures onto the split change. Later controlled candidates include:

- Coarse detector-mask conditioning to tell the refiner which local object
  to segment.
- Jittered or proposal-derived training crops to reduce training/inference
  crop mismatch.
- Confidence/probability-aware contested-pixel ownership.
- Boundary-aware losses, evaluated against full-frame PQ rather than only
  foreground Dice.
- Calibrating on all inner tuning images, with no use of outer labels.

See [V4 changes and next steps](V4_CHANGES_AND_NEXT_STEPS.md) and
[V1/V2/V3 mask EDA](V1_V2_V3_MASK_EDA_V4.md) for the supporting observations.

## Validation performed

Eight new grouped-CV regression tests cover day separation, complete outer
coverage, annotator grouping, inner/outer orchestration, per-image averaging,
incomplete summary behavior, plan checks, and result reuse. The full focused
suite contains 20 tests.

A tiny **local** real-data fold trained both models, calibrated on inner data,
evaluated outer images, saved an explicitly incomplete five-fold summary,
reused its completed result without training, and produced no submission CSV.
Its metrics are smoke-test outputs, not full-data CV results. No remote fold
has been launched.

The [shared CV helper](../Python%20Files/grouped_yolo_cv.py) is embedded byte-for-byte in the
standalone notebook; regression tests check this correspondence.
