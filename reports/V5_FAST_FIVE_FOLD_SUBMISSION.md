# V5: short five-fold training and a competition submission CSV

## Verified result: public PQ 0.36

Kaggle submission **58327117** is **Complete**, with public PQ **0.36**,
verified through the API on November 13, 2026 (local date).
This is **+0.01** over repaired V3 and exceeds the organizers' stated **>0.35**
threshold. It is a whole-pipeline comparison, not a controlled encoder ablation.
The [accepted CSV](../V5_SubmissionFile/submission.csv) contains 1127 masks across
177 images; local decoding found zero overlaps. SHA-256:
`8c53ec5fe034defafec8066eb2a2c0f8b7332adf02b9c5c3e0c787e794ec4769`.

The original budget expired before fold 4 completed a refiner epoch.
Folds 0–3 were preserved and fold 4's refiner was retried without a deadline,
reusing its detector. The completed run did **not** meet the two-hour target.
Final per-fold epoch counts and full runtime require the preserved execution logs.
An earlier rejected upload, **58326839**, selected stale saved-version V3 output,
not the five-fold ensemble. Use the live export and verify its hash against
`ensemble_audit.json`; a notebook title alone does not establish output provenance.

## Current defaults

The [V5 notebook](../solar_filament_CV_V5_grouped.ipynb) now keeps **five folds
grouped by observation day** and includes a real competition export.
The evaluation-only, long-training default has been replaced. V4 is unchanged.

| Setting | Current V5 |
|---|---|
| Folds trained by Run All | 0, 1, 2, 3, 4 |
| Detector | YOLO11s-seg, 1024px, batch 4 |
| Detector epochs | **5 per fold**, 25 total |
| Refiner | SMP U-Net with **ImageNet-pretrained ResNet34**, 256px crops, batch 16 |
| Refiner epochs | **2 per fold**, 10 total |
| Optimizers | Explicit detector AdamW 0.002; refiner AdamW 0.0003 |
| Calibration | Up to 8 inner-tuning images per fold; disjoint masks |
| Runtime target | **Two hours for all five folds plus export**, not per fold |
| Competition prediction model | **All five folds**, equally weighted, each using its inner-selected mode |
| Output | `/kaggle/working/submission.csv` plus evidence under the run directory |
| Automatic upload | **None** |

Epoch counts were reduced after the request to retain five folds within the
short turnaround. These replace the earlier 25-detector/20-refiner proposal
for one model. Five folds at 25/20 epochs each would not be a realistic
two-hour job at the observed throughput.

## Runtime controls and limits

The timer starts before package installation. The last **20 minutes** are
reserved for checkpoint reload and final test export using all five models.
This reserve is not a measured five-model inference time. Remaining time before
minute 100 is divided equally across the remaining selected folds:

- Detector gets at most 40% of its fold's remaining allowance.
- Refiner uses the remaining fold training time.
- Both epoch ceilings and existing early-stopping logic remain active.
- Budget-truncated epochs do not count as complete refiner epochs.
- If a model cannot produce a complete validated checkpoint, the notebook
  fails explicitly; it does not fabricate a CSV or claim readiness.

These are **cooperative checks**, not hard process interruption. Installation,
downloads, validation, one GPU operation, outer evaluation, and export can
exceed the allotted times. The scored run required deadline-free recovery
and exceeded the target. The two refiner epochs may be truncated.
ResNet34 has more capacity and cost than ResNet18; short training does not
guarantee convergence or a higher score.

`turnaround.json` records actual elapsed seconds and whether the target was met.
It is written after successful export processing. A failed stage means no
successful completion record, even if some intermediate artifacts exist.

## The folds are retained

For each fold, the outer 20% is evaluation-only. The other 80% is split into
approximately 72% training and 8% inner tuning. Checkpoint and threshold
selection never use outer labels.

| Fold | Training physical images | Inner tuning | Outer evaluation |
|---|---:|---:|---:|
| 0 | 510 | 55 | 142 |
| 1 | 510 | 55 | 142 |
| 2 | 508 | 58 | 141 |
| 3 | 511 | 55 | 141 |
| 4 | 512 | 54 | 141 |

All annotator target sets for a physical photograph remain together. These
are new pretrained initializations per fold, not V4 checkpoint continuation.
All-five-fold mean, sample SD, and pooled out-of-fold mean require all five
complete matching results. The project's labeled data have already been
involved in development; this is not a new untouched final benchmark.

## Exactly what produces the submission

After all five folds finish, the notebook:

1. Reloads **each of the five folds'** best detector and ResNet34 refiner
   checkpoints sequentially, keeping only one model pair in GPU memory.
2. Reloads each matching inner calibration and verifies checkpoint hashes and
   selected parameters against the completed fold result.
3. Predicts all 180 competition test images with every fold, using that fold's
   inner-selected YOLO-only or refined mode. Stores per-fold RLE predictions.
4. Matches instances one-to-one across folds at IoU >= 0.5 using maximum-cardinality,
   maximum-IoU assignment against running cluster representative masks. Fold order
   is fixed at 0–4; at most one instance from each fold can enter a cluster.
5. Keeps clusters supported by at least **3 of 5 folds**. Each final foreground
   pixel also needs **3 of 5 fold votes**, including missing predictions as zero.
   Removes masks below 20 pixels, orders by fold support then area, and applies
   deterministic disjoint ownership with another 20-pixel post-trim filter.
6. Checks nonempty masks, original-resolution shape, RLE roundtrips, known
   image IDs, unique instance IDs, CSV readback, and zero within-image overlaps.
7. Writes the ensemble submission and copies it to `/kaggle/working/submission.csv`.
   Records the policy, all checkpoint hashes, per-image fold/final instance counts,
   zero-detection IDs, and CSV hash in `ensemble_audit.json`.

This replaces the erroneous single-fold export: **every fold contributes test
predictions**, rather than simply concatenating five overlapping instance lists.
No outer scores select folds or voting weights. Voting settings are fixed,
not claimed to be tuned or optimal. Majority voting can reduce isolated false
positives, but can also reject real low-agreement filaments or erode boundaries.
No improvement over V3's public PQ is guaranteed.

The ensemble uses a new run directory and includes its voting policy in the
immutable CV plan hash, preventing accidental reuse of the old fold-0 export plan.

The per-fold OOF metrics evaluate the individual held-out models, **not this
five-model ensemble**. Applying all five models to those labeled OOF images
would mix in models trained on those images, so it is not an unbiased ensemble
validation. Competition labels remain unavailable.

Only a completed `FINAL CSV PREPARED` indicates readiness in this workflow.
An incomplete fold collection is explicitly blocked from final export.
Smoke runs cannot produce competition exports.

Zero-detection images are audited and omitted, as in the accepted earlier
submissions. No dummy filament is invented. Final Kaggle acceptance and score
remain external checks; local validity does not guarantee a high PQ.

## Running and submitting

1. Import the updated [V5 notebook](../solar_filament_CV_V5_grouped.ipynb)
   into a **separate Kaggle notebook**. Attach competition data and enable
   Internet and GPU.
2. Leave the default five folds selected and inspect the short epoch settings.
3. Run All once. Do not start a duplicate of the existing V4 experiment.
4. Inspect the final readiness message, runtime evidence, and submission audit.
5. Submit the completed `/kaggle/working/submission.csv` manually.

No remote V5 run or competition submission was launched during this update.
V4 was not stopped, edited, or restarted.

## Validation and future improvements

Local regression tests cover budgets, day isolation, all-image outer coverage,
checkpoint provenance, inner selection, all-five-model reload without training,
incomplete-fold blocking, instance matching with reordered detections, one-vote-
per-fold enforcement, pixel voting, actual CSV readback, zero-detection audits,
ResNet34 configuration, and disjoint CSV behavior.
A tiny local real-data run verifies all five actual YOLO and ResNet34 training
folds, calibration, outer evaluation, and budget integration. The positive
ensemble-export test uses controlled synthetic predictions to verify exactly
which folds and pixels contribute; it is not a full-data GPU inference benchmark.
Tiny runs do not establish full-data runtime or score.

The shared [CV helper](../grouped_yolo_cv.py) and
[budget helper](../v5_budget.py) are embedded in the standalone notebook.
The older [fold protocol document](FIVE_FOLD_YOLO_REFINER.md) describes the
superseded evaluation-only setup, not these new execution defaults.

The five-model ensemble now includes explicit instance matching and ownership
validation. Boundary losses, proposal-conditioned refinement, and confidence-aware ownership remain
separate experiments rather than untested extra changes.
