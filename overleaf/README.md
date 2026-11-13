# First-draft package

[main.tex](./main.tex) and [references.bib](./references.bib) form a portable IEEE-style first draft. Import both into Overleaf. The linked private project was inaccessible during preparation, so this is not a verified match to its template. No LaTeX compiler was available locally; page count and compilation remain unverified. Use the organizer's exact class/template and enforce its four-page limit before submission.

The draft intentionally marks missing holdout metrics, selected epoch, hardware and leaderboard results. The 48-minute run duration is user-reported, not a local measurement. Do not replace missing results with a competitor's 0.7 score.

## Local video reproduction

The images are present, but no trained checkpoint was found locally. Download the Kaggle experiment folder with `best_model.pt`, `calibration.json`, and `split_manifest.json`; retain your baseline configuration. In the notebook set `CHECKPOINT_INPUT_DIR` to that folder, `RUN_TRAINING=False`, and `RUN_CALIBRATION=False`. Run configuration/data/helpers/checkpoint-loading/calibration and visualization cells; do not start fresh training to make a substitute video.

Kaggle CUDA accelerates neural inference. This Mac reports Apple MPS available and CUDA unavailable. For local inference, after loading the saved model:

```python
DEVICE = torch.device("mps")
model = model.to(DEVICE)
```

MPS execution of the full model/video workflow remains unverified here. CPU encoding and mask conversion still contribute runtime. Model weights and videos have not been fabricated.

The MP4 cell follows the animation helpers and is opt-in. It streams H.264 frames through FFmpeg. Use `MP4_LIMIT=10` first; use `None` for all 707 labeled and 180 test images. Labeled frames display the partition and one annotator's target. Training imagery is not independent evaluation.

## GitHub-ready clip preparation (not published)

No destination repository was provided. After checking competition sharing permissions and attribution, create a Release in the chosen repository and attach the reviewed MP4 and its audit. A README can link to it:

```markdown
[Test prediction video](https://github.com/OWNER/REPOSITORY/releases/download/TAG/test_filament_review.mp4)
```

Replace OWNER/REPOSITORY/TAG only after actual upload. This is a placeholder, not a live hosted link. A GitHub comment/issue video attachment is another option; use GitHub's upload UI to obtain its real asset URL rather than inventing one. Keep large datasets and credentials out of Git.

## Training-time expectation

If the reported 48 minutes covered exactly 20 epochs of training plus per-epoch validation, the average is 2.4 minutes per epoch. If it also included installation, calibration, inference or video, that estimate overstates training-only time. Check `training_runtime.json` and epoch logs.

On the same accelerator/configuration, a fresh 30-epoch run would therefore be roughly 72 minutes and ten more epochs roughly 24 minutes, before early stopping and additional calibration/video overhead. These are conditional estimates, not measured promises. The comparison schedules up to 40 extra epochs total (roughly 96 minutes under that assumption). Higher image resolution, GPU changes and memory pressure invalidate simple extrapolation.
