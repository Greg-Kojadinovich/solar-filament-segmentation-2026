# Solar filament CV notebook: a practical tutorial

This guide explains the implementation in [solar_filament_CV.ipynb](../../Kaggle%20Competition/Notebooks/solar_filament_CV.ipynb), its mathematics, its outputs, and its differences from two comparison notebooks. It describes the local source as reviewed on November 12, 2026. A Kaggle copy can differ if you change its cells. The third-party comparison notebooks are not distributed in this repository.

**Short answer:** your baseline trains for **20 epochs**, learns which grayscale pixels belong to filaments, and predicts masks for the unlabeled test images. An optional comparison now adds a fresh run of up to **30 epochs** and fine-tuning of up to **10 additional epochs**, both with early stopping. It reports local **Panoptic Quality (PQ), Dice, and IoU**. It does not currently print a separate F1 field. Only a Kaggle submission can obtain the hidden-test leaderboard score.

## 1. The Hugging Face warning

```text
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

This concerns **downloading public pretrained model weights**, not training accuracy or the Kaggle competition API. The segmentation library can retrieve the ImageNet-pretrained encoder through Hugging Face.

- If the download succeeds and epoch progress starts, you can continue. Authentication is not needed for this public download.
- It does not mean your GPU is disabled. Check that the notebook prints `Device: cuda`.
- A token does not improve the trained model's quality or GPU computation speed; it can improve download access/rate limits.
- If an actual download error follows, check Kaggle Internet access and the error message. Do not ignore a failed download.

### Optional authentication on Kaggle

Create a Hugging Face token with read permission, add it as a Kaggle Secret named `HF_TOKEN`, and enable that secret for this notebook. Put this optional cell **before the library-import/configuration cell**, so the environment variable exists when the Hub library loads:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Do not print, paste into source, or publish the token. This setup is optional and is not included in the notebook's default execution.

For inference-only reproduction, the notebook rebuilds the network without pretrained downloads and loads your saved checkpoint. See the [official Hub authentication documentation](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

## 2. What your implementation does, in plain language

Think of the model as learning to color a tracing sheet:

1. A solar image is the photograph.
2. Human filament outlines are the answer sheet.
3. During training, the model predicts an answer sheet and adjusts its weights when it disagrees with the human outlines.
4. During testing, it draws an answer sheet without seeing human answers.
5. Post-processing separates patches of foreground into proposed individual filaments.

The pipeline is:

```text
Grayscale JPEG + human polygon annotations
    -> separate observing days into train / tuning / holdout
    -> decode polygons into binary masks
    -> resize images and masks to 512 x 512
    -> U-Net with pretrained ResNet18 encoder
    -> a filament probability for every pixel
    -> restore probabilities to 2048 x 2048
    -> threshold + connected components + minimum area
    -> one compressed COCO RLE string per proposed filament
    -> submission.csv
```

**No human test labels are used for learning or calibration.** Running inference on the test set is not the same as measuring test accuracy.

### Data and annotation handling

The reviewed data contains:

| Item                                      | Count |
| ----------------------------------------- | ----: |
| Physical training JPEGs                   |   707 |
| Physical test JPEGs                       |   180 |
| Annotator-specific training image records | 1,154 |
| Filament annotations                      | 8,199 |

Some images have been annotated by two or three people. These are multiple interpretations of the **same photograph**, not additional independent photographs.

The notebook:

- Keeps each annotator's target as a separate training example.
- Combines individual filament masks within that person's target into one foreground mask.
- Does **not** merge different annotators' answers into a consensus target.
- Puts every image from the same observing day in the same split.
- Merges all chirality categories into foreground; it does not train a chirality classifier.

For the supplied annotation file and default seed, the observed split is:

| Partition | Physical images | Annotator-specific examples | Purpose                                             |
| --------- | --------------: | --------------------------: | --------------------------------------------------- |
| Training  |             499 |                         827 | Update neural-network weights                       |
| Tuning    |             103 |                         161 | Choose checkpoint and post-processing parameters    |
| Holdout   |             105 |                         166 | Evaluate the selected method without further tuning |

Fractions are assigned by **day**, so image counts need not equal exactly 70/15/15 percent. Different data or seeds can change them. Same-day grouping reduces leakage but does not prevent the same long-lived filament appearing on adjacent days.

Training on separate targets also gives multiply annotated photographs more training exposure. Evaluation compensates differently: it averages annotators within each image, then averages physical images. This distinction belongs in the paper.

## 3. How long does it train?

The configuration contains:

```python
epochs=20
batch_size=8
learning_rate=3e-4
image_size=512
```

An **epoch** is one pass through the training examples. A **batch** is the group of examples processed before an optimizer update.

With 827 training examples and batch size 8, the default loader produces:

$$
\left\lceil \frac{827}{8} \right\rceil = 104
$$

training batches per epoch, including a final batch of three. Across 20 epochs, that is **2,080 optimizer steps**. The loader only drops a final training batch when it would contain exactly one example.

Each epoch also checks the 161 tuning examples, using 21 batches, **without updating weights**.

Important details:

- The original baseline has no early stopping or learning-rate scheduler. The new optional comparison has early stopping, described below.
- All 20 epochs run unless interrupted or an error occurs.
- The saved checkpoint is the epoch with the **lowest tuning loss**, not necessarily epoch 20.
- `train_loss` and `tune_loss` are losses, not accuracy or F1.
- Training time depends on the selected GPU, software environment, and workload. Do not extrapolate timings from another notebook or a local smoke test.
- More epochs are not automatically better; inspect whether training loss improves while tuning loss worsens.

For a rerun with existing weights, set `RUN_TRAINING=False`. To conduct a new training experiment, choose a new `RUN_NAME`; the notebook refuses to overwrite an existing checkpoint.

## 4. Which models are actually used?

### New optional comparison: fresh training versus continuation

The **Optional controlled comparison** section in the notebook preserves your baseline and runs two candidates only when you enable `RUN_EXTRA_EXPERIMENTS=True`.

| Candidate | Starting weights | Maximum new epochs | Learning rate | Optimizer |
|---|---|---:|---:|---|
| `fresh_30` | ImageNet encoder initialization | 30 | 0.0003 | New AdamW |
| `finetune_10` | Baseline's best saved checkpoint | 10 | 0.0001 | New AdamW |

Both use **patience 5** and **minimum tuning-loss improvement 0.0001**. Five consecutive epochs without a qualifying improvement stop training. A strict new best loss is still saved even if the change is smaller than the patience-reset threshold. Fine-tuning also evaluates and saves its starting weights as epoch zero, so it can keep them if further training does not help.

These are explicit starting choices, not documented universal best parameters. A 30-epoch maximum does not force 30 epochs. Fine-tuning from the best baseline checkpoint is not necessarily “epochs 21 through 30”: the baseline best checkpoint might be epoch 17, and the original optimizer state was not saved.

The comparison uses identical day-grouped data and the same threshold/area grid on the same tuning images. It does not evaluate holdout/test data or auto-select a submission model. Existing inference/export cells keep using the baseline until you explicitly set `USE_EXTRA_CANDIDATE` in the candidate-selection cell. That cell switches the model, configuration, selected parameters, checkpoint hash, and output directory together. Run downstream visualization/export/evidence after switching; do not rerun baseline training.

Candidate directories contain their own weights, training histories, early-stopping summaries, split manifests and calibration parameters. `comparison_results.json` records tuning metrics and runtime; `convergence_comparison.png` compares loss curves. The baseline weights and random-generator states are preserved. Use a new `EXTRA_RUN_TAG` to repeat an experiment without overwriting results.

**How to run on the completed Kaggle session:** copy the Optional controlled comparison section after calibration. Run its settings and helper-definition cells, enable the flag, then run the comparison and curve cells. These schedule up to 40 extra epochs total. No training time or score improvement is guaranteed.

The [scikit-learn parameter-search guide](https://scikit-learn.org/stable/modules/grid_search.html) recommends budgeted randomized searches and log-scale distributions for wider continuous learning-rate searches. `RandomizedSearchCV` requires an estimator interface; it does not automatically fit this bare PyTorch model. Here we implement two transparent controlled candidates, not a misleading claim of scikit-learn cross-validation.

[Ultralytics training](https://docs.ultralytics.com/modes/train/) and [tuning guidance](https://docs.ultralytics.com/guides/hyperparameter-tuning/) identify epochs, image size, batch size, learning rate and patience as important controls. Their example YOLO epoch counts/image sizes are not transferable evidence of optimal U-Net settings. Wider batch/resolution searches remain separate work; changing all factors at once would obscure whether extra epochs or fine-tuning helped.

### Reading the equations

The new notebook section includes LaTeX display equations rendered by Kaggle's Markdown math support. In this guide, use a math-capable Markdown preview (such as VS Code's built-in Markdown preview) rather than the raw text editor. No system LaTeX installation is needed for notebook Markdown.

| Component                         | Used by default?          | Role                                                       |
| --------------------------------- | ------------------------- | ---------------------------------------------------------- |
| U-Net                             | Yes                       | Pixel-level filament segmentation                          |
| ResNet18                          | Yes                       | Pretrained feature-extraction encoder inside U-Net         |
| Local intensity/contrast baseline | Yes                       | Classical CV comparison; no learned weights                |
| FPN                               | Optional configuration    | Alternative segmentation architecture                      |
| ResNet34                          | In both supplied examples | Deeper encoder, not your default                           |
| ConvNeXt                          | No                        | Downstream chirality classification in the ESAIR poster    |
| SVM / one-class SVM               | No                        | Image-quality comparison methods in the anomaly papers     |
| YOLOX                             | No                        | Standard implementation predicts bounding boxes, not masks |
| Mask R-CNN                        | No                        | Possible future instance-segmentation alternative          |

### U-Net and ResNet18

U-Net is a **convolutional neural network (CNN)**. Its encoder extracts features at progressively coarser spatial scales; its decoder reconstructs a pixel-level prediction. Skip connections pass fine spatial information from encoder to decoder.

ResNet18 is the encoder, not a second model voting against U-Net. Residual blocks learn transformations with a shortcut, schematically:

$$
h_{l+1}=h_l+F(h_l;\theta_l).
$$

ImageNet pretraining provides an initial feature extractor. The model still needs supervised training on solar masks; generic pretrained weights do not already know these annotations. SMP adapts the encoder to one input channel.

ResNet34 has more layers and potentially greater feature capacity, but also more computation. Greater depth does not establish better segmentation on this dataset. Compare on identical splits before claiming an improvement.

### What “CNN,” “ConvNeXt,” and “IoU” mean

These are not interchangeable:

- **CNN:** a family of neural-network architectures using convolutions.
- **ConvNeXt:** a particular modern convolutional architecture. The poster uses it to classify filament cutouts after segmentation/grouping. It is not a vision transformer and is not used in your default notebook.
- **IoU:** an overlap metric, not a neural-network architecture.
- **MIoU:** in the ESAIR poster, **Multiscale IoU**. Do not confuse it with the common abbreviation **mIoU**, meaning mean IoU across classes.

If “Cnet” means a different named architecture, it is not present as a model named Cnet in the reviewed notebook.

## 5. Is this regression or classification?

Your implementation is **supervised binary semantic segmentation**: each pixel is classified as filament or background.

The network outputs a real number called a logit, which becomes a probability. A continuous output does not make this ordinary regression:

| Task                     | Example target                            | Typical approach                                                  |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------- |
| Regression               | Filament length or eruption time          | Predict a continuous quantity, often using squared/absolute error |
| Binary classification    | Corrupt versus normal image               | Predict a class probability or decision                           |
| Semantic segmentation    | Filament versus background at every pixel | Per-pixel classification with a spatial model                     |
| Instance segmentation    | Which individual filament owns each pixel | Separate object masks or post-processing                          |
| Chirality classification | Left/right/other label for a filament     | Classifier on an extracted filament region                        |

Your U-Net performs semantic segmentation. Connected components add a simple instance-extraction step. If two filaments touch, this step may merge them; if a filament is broken, it may split it. That is why a good pixel score can coexist with poor instance-level PQ.

Logistic regression is also a classification method despite its name. SVM classifiers and one-class SVM should not be confused with **support vector regression (SVR)**.

### Why not just classify pixel brightness?

The images have one grayscale intensity, not independent red/green/blue information. Repeating grayscale into three channels does not create physical color measurements.

A brightness threshold can confuse sunspots, shadows, clouds, and the solar limb with filaments. The neural model can also use neighboring texture, shape, and larger context.

Your classical baseline estimates a smooth background:

$$
B=G_\sigma * I,\qquad
M_{\text{CV}}(x)=D(x)\land[B(x)-I(x)>\tau_c],
$$

where $G_\sigma$ is Gaussian smoothing, $D$ is a heuristic eroded disk mask, and $\tau_c$ is a contrast threshold. In the code, the disk mask begins with $I>0.08$, and $\sigma=\text{image size}/32$.

This is a fixed algorithm with tuning parameters, **not a fitted regression or SVM model**. It is a useful baseline because it asks whether learned shape/context helps beyond local darkness.

## 6. The training mathematics

### Normalization and probability

An 8-bit grayscale pixel $v$ is converted to:

$$
I=v/255,\qquad x=(I-0.5)/0.5.
$$

The normalized neural-network input lies in $[-1,1]$. If the network outputs logit $z_i$, its foreground probability is:

$$
p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
$$

### Binary cross-entropy (BCE)

For $N$ pixels and binary labels $y_i$:

$$
L_{\mathrm{BCE}}=
-\frac{1}{N}\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

BCE penalizes incorrect confident predictions. The implementation uses `binary_cross_entropy_with_logits` for numerical stability.

### Soft Dice loss

For each image $b$, your code computes:

$$
D_b=
\frac{2\sum_i p_{bi}y_{bi}+1}
{\sum_i p_{bi}+\sum_i y_{bi}+1},
\qquad
L_{\mathrm{Dice}}=1-\frac{1}{B}\sum_bD_b.
$$

The added 1 is the implementation's smoothing constant. This is a soft, differentiable training quantity, not the thresholded holdout Dice score.

Your combined loss is:

$$
L=0.5L_{\mathrm{BCE}}+0.5L_{\mathrm{Dice}}.
$$

BCE supplies pixelwise supervision; Dice emphasizes overlap when foreground occupies only a small portion of the image. A model predicting background everywhere can achieve high raw pixel accuracy because most pixels are background, which is why accuracy alone is misleading.

AdamW optimizes the network with learning rate $3\times10^{-4}$. Schematically, it uses adaptive gradient moments plus decoupled weight decay:

$$
\theta_{t+1}
=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_t.
$$

The notebook does not explicitly override AdamW's weight-decay default; preserve the recorded PyTorch version when reproducing an experiment.

### Other objectives: IoU, MIoU and boundary losses

A possible soft-IoU training objective is:

$$
J_{\mathrm{soft}}=
\frac{\sum_i p_i y_i}
{\sum_i p_i+\sum_i y_i-\sum_i p_i y_i}.
$$

The notebook does **not** optimize this objective. Its IoU is an evaluation metric.

The poster's multiscale concept can be summarized schematically as:

$$
\operatorname{MIoU}(P,Y)
=\frac{1}{|S|}\sum_{s\in S}
J_{\mathrm{soft}}\!\left(T_s(P),T_s(Y)\right),
$$

where $T_s$ represents a scale-dependent area or boundary transformation. This is explanatory notation, **not a complete reproduction** of the poster: precise transformations, window sizes and boundary definitions require the authors' implementation.

Boundary F1 compares predicted and true boundary pixels, usually with a spatial tolerance. Its tolerance and boundary-extraction rules matter. The poster discusses BF1 loss; your notebook does not implement it. Thin barbs can change a boundary score substantially while changing an area-overlap score only slightly.

## 7. F1, Dice, IoU, and PQ: what each score answers

### Pixel precision, recall and F1

Using thresholded foreground masks:

- **TP:** a true filament pixel predicted as foreground.
- **FP:** a background pixel predicted as foreground.
- **FN:** a filament pixel missed by the model.

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},
$$

$$
F1=\frac{2TP}{2TP+FP+FN}.
$$

For the same binary masks and empty-mask convention, **foreground pixel Dice equals foreground pixel F1**:

$$
\mathrm{Dice}=\frac{2|P\cap Y|}{|P|+|Y|}
=\frac{2TP}{2TP+FP+FN}.
$$

Your holdout code reports this as `dice`. It first unions the individual predicted masks and ground-truth masks for the pixel calculation.

### Intersection over Union

$$
\mathrm{IoU}=\frac{|P\cap Y|}{|P\cup Y|}
=\frac{TP}{TP+FP+FN}.
$$

For one pair of binary masks:

$$
\mathrm{Dice}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}.
$$

This relationship need not hold when applied to separately averaged dataset scores: nonlinear transformations and averaging do not commute.

### Panoptic Quality: finding the right individual filaments

For PQ, TP/FP/FN refer to **objects**, not pixels:

1. Compute IoU between each predicted filament and each annotated filament.
2. Permit a match only when IoU is **strictly greater than 0.5**.
3. Find one-to-one matches, maximizing the number of matches first, then overlap.
4. Penalize unmatched predicted and annotated filaments.

$$
\mathrm{PQ}=
\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}
{|TP|+\tfrac12|FP|+\tfrac12|FN|}.
$$

Equivalently, for a nonempty matched set:

$$
\mathrm{PQ}=\underbrace{
\frac{\sum_{TP}\mathrm{IoU}(p,g)}{|TP|}
}_{\mathrm{Segmentation\ Quality\ (SQ)}}
\times
\underbrace{
\frac{|TP|}{|TP|+\tfrac12|FP|+\tfrac12|FN|}
}_{\mathrm{Recognition\ Quality\ (RQ),\ object\ F1}}.
$$

Your code returns PQ but does not separately print SQ or RQ. It defines both-empty prediction/truth as a score of 1, and one-empty as 0.

**Example, not a measured result:** three matches with IoUs 0.8, 0.7 and 0.6, plus one unmatched prediction and two missed filaments:

$$
RQ=\frac{3}{3+0.5+1}=\frac23,\quad
SQ=\frac{2.1}{3}=0.7,\quad
PQ=\frac{2.1}{4.5}\approx0.467.
$$

A union pixel mask can look good even if two neighboring filaments are merged into one prediction. PQ exposes that instance error more directly.

### Four different meanings of “F1”

Do not mix these in a paper:

1. **Image anomaly F1:** did a quality-control model flag corrupt images?
2. **Pixel F1 / Dice:** did segmentation label the correct foreground pixels?
3. **Boundary F1:** did predicted contours align with true boundaries?
4. **Object F1 / RQ:** did the detector find the correct individual filaments?

The published Anomalyzer F1 values are the first kind, not the segmentation accuracy of your model.

The live competition overview reviewed during notebook development names **PQ** as the leaderboard metric, despite some remaining historical Dice wording. Its final rubric includes qualitative judgment. An F1 formula itself is not subjective; human reference annotations and the broader judging rubric are separate issues.

## 8. What will your output look like?

### During training

The notebook prints dictionaries with this structure:

```text
{'epoch': 1, 'train_loss': ..., 'tune_loss': ...}
...
{'epoch': 20, 'train_loss': ..., 'tune_loss': ...}
Loaded best epoch: ... tuning loss: ...
```

The dots are placeholders here, not runnable output or claimed measurements. Loss does not have to decrease monotonically.

### During calibration

It evaluates:

- U-Net probability thresholds: 0.35, 0.50 and 0.65.
- Minimum component areas: 64 and 256 **original-resolution pixels**.
- Classical contrast thresholds: 0.025, 0.04 and 0.06, with the same area choices.

That is six candidates per method, twelve total. At most 32 physical tuning images are used for calibration by default. Checkpoint selection already used the full tuning set's loss during training.

### During holdout evaluation

The summary contains two methods, `unet` and `contrast`. Each has:

```text
scores:
    pq
    dice
    iou
    tp
    fp
    fn
    split_relations
    merge_relations
seconds
image_count
annotation_set_count
```

- Scores range from 0 to 1; larger is better for PQ/Dice/IoU.
- `dice` is the current notebook's foreground pixel F1 equivalent.
- `tp`, `fp`, and `fn` are averaged **object counts**, so they can be fractional.
- Split/merge relations use overlap greater than 0.1 as a diagnostic, not an official competition rule.
- Results average annotation sets within each physical image, then images.

Do **not** compute dataset object F1 from the already averaged TP/FP/FN and claim it is the average per-image F1. Those are different aggregation procedures. A separate F1 implementation would need a stated definition and consistent aggregation.

This documentation does not add an F1 field or change the notebook's implementation.

### Test-set output

There is **no local test F1/PQ**, because test answers are hidden. The notebook instead saves probability-map previews, predicted mask outlines, per-image counts and the submission CSV.

If any test image has no retained instances, export deliberately stops instead of fabricating a mask. Inspect the audit and verify the competition's zero-detection policy. Do not invent foreground merely to pass validation.

`submission.csv` contains:

```csv
filament_id,segmentation_rle
<image_stem>_1,<compressed COCO counts>
<image_stem>_2,<compressed COCO counts>
```

The strings encode individual 2048 x 2048 masks; they are not scores or image filenames. Submit the real CSV, not this illustrative snippet.

## 9. Where the SVM research fits

The supplied SoftwareX and ICDMW papers concern **quality control of entire solar observations**. They do not establish that one-class SVM is the filament segmentation model.

### Supervised SVM

A binary SVM learns a boundary from examples labeled normal and corrupt. A schematic soft-margin formulation is:

$$
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_j\xi_j,
\qquad
y_j(w^\top\phi(x_j)+b)\ge1-\xi_j,\quad \xi_j\ge0.
$$

An RBF kernel allows a nonlinear boundary:

$$
K(x,x')=\exp(-\gamma\|x-x'\|^2).
$$

Here $C$ trades margin size against training errors; $\gamma$ controls how localized the similarity is.

### One-class SVM

One-class SVM typically learns a normal-data region using normal examples, rather than a supervised normal-versus-corrupt boundary:

$$
\min_{w,\rho,\xi}
\frac12\|w\|^2+\frac{1}{\nu n}\sum_j\xi_j-\rho,
\qquad
w^\top\phi(x_j)\ge\rho-\xi_j,\quad \xi_j\ge0.
$$

It is a different task and objective from your supervised pixelwise U-Net. The Anomalyzer itself is a statistical, explainable grid/intensity method; it is not simply another name for SVM.

**Why it matters:** a strong image-quality detector can help choose or inspect training inputs, but cannot directly supply filament masks. Quality filtering would need its own controlled ablation, and must not remove required test images from inference.

## 10. Comparison with the supplied “top score” notebooks

The reviewed files are:

- `0-9-solar-filament-baseline-pipeline-u-net.ipynb` (third-party reference)
- `solar-filament-segmentation.ipynb` (third-party reference)

Their names and saved losses are **not independently verified leaderboard scores**. The following comparison concerns their saved source code, not an unseen corrected version or a claimed live ranking.

| Choice                  | Your CV notebook                                        | First supplied example                | Second supplied example               |
| ----------------------- | ------------------------------------------------------- | ------------------------------------- | ------------------------------------- |
| Model                   | U-Net, ResNet18 encoder                                 | U-Net, ResNet34 encoder               | U-Net, ResNet34 encoder               |
| ImageNet initialization | Yes                                                     | Yes                                   | Yes                                   |
| Epochs                  | 20                                                      | 5                                     | 20                                    |
| Learning rate           | 0.0003                                                  | 0.0003                                | 0.0002                                |
| Input channels          | One grayscale channel                                   | Three-channel RGB conversion          | Three-channel RGB conversion          |
| Model input size        | 512 x 512                                               | 512 x 512                             | 512 x 512                             |
| Loss family             | Equal Dice+BCE; explicit smoothed per-image Dice        | Equal SMP Dice+BCE                    | Equal SMP Dice+BCE                    |
| Optimizer               | AdamW                                                   | AdamW                                 | AdamW                                 |
| Data partitions         | Day-grouped train/tune/holdout                          | Random 80/20 filename split           | Random 80/20 filename split           |
| Annotation lookup       | COCO image ID -> annotations                            | Top-level JSON filename lookup        | Top-level JSON filename lookup        |
| Multiple annotators     | Separate targets; explicit evaluation averaging         | No COCO mapping implemented           | No COCO mapping implemented           |
| Calibration             | Local PQ threshold/area grid                            | Fixed 0.5 threshold, area 30          | Fixed 0.5 threshold, area 30          |
| Holdout comparison      | Neural versus classical contrast                        | No independent holdout PQ comparison  | No independent holdout PQ comparison  |
| Export resolution       | Restore probabilities to 2048 before masks              | Encode 512-pixel predictions directly | Encode 512-pixel predictions directly |
| Empty detections        | Stop with audit, no invented filament                   | Insert empty-mask row                 | Insert empty-mask row                 |
| Reproducibility         | Config/split/checkpoint/calibration hashes and evidence | Model weights and CSV                 | Model weights and CSV                 |

### Verified annotation mismatch in both examples

Both use this pattern:

```python
ann_entry = self.annotations.get(filename, [])
```

But the actual annotation JSON has top-level keys:

```text
info, licenses, categories, images, annotations
```

It is not a filename-to-mask dictionary. A direct lookup of a real training filename was checked against the supplied JSON and returned `[]`.

Consequently, **as saved and used with this dataset**, both example loaders produce empty target masks rather than decoding the actual filament annotations. Their rasterizer also catches parsing errors without surfacing them. The saved training losses fall, but falling loss against empty targets does not demonstrate learning real filament shapes.

Your loader explicitly maps filenames to image records and image IDs to annotations, then uses COCO mask conversion. Invalid ground-truth data raises an error instead of being silently replaced by background.

### Export mismatch in both examples

Their transforms resize to 512 x 512 and their export code encodes masks at that size without restoring the required 2048 x 2048 canvas. They also filter components with area 30 at the smaller resolution.

Your notebook restores probabilities first, then thresholds and measures area in original-resolution pixels. It validates RLE decoding and round trips. Upsampling gives the required canvas but **does not restore thin detail already lost by downsampling**.

### What is genuinely similar?

All three pipelines use a pretrained U-Net encoder, AdamW, Dice+BCE and connected components. Your version is not a completely new neural architecture. Its principal differences are correct annotation wiring, explicit split/annotation policies, calibration/holdout evaluation, a classical comparison, valid export resolution and reproducibility checks.

These are important engineering and experimental improvements, but they do **not** prove a higher competition score. That requires completed experiments and real submissions.

## 11. Limitations and defensible paper claims

Current limitations:

- The model trains at 512 pixels, potentially losing narrow structures.
- Connected components do not reproduce the poster's geometry/spine grouping.
- Binary targets discard individual filament identities during training.
- Predictions can merge touching filaments or fragment one object.
- Separate annotator targets can disagree; no consensus/noise model is fitted.
- Same-day grouping does not fully isolate long-lived events across days.
- A single seed/split does not establish statistical significance.
- Calibration and checkpoint selection reuse the tuning partition.
- Local PQ does not reproduce every private evaluator edge-case policy.

Reasonable next experiments, **not current results**:

1. Compare 512-pixel training with higher resolution or overlapping tiles.
2. Compare U-Net and FPN on identical grouped folds.
3. Implement and document boundary-sensitive loss versus Dice+BCE.
4. Compare connected components with geometry-based grouping.
5. Measure quality-aware filtering/weighting rather than assuming it helps.
6. Repeat with multiple seeds and grouped uncertainty estimates.

A defensible initial paper describes an established-model baseline and a reproducible evaluation, reports actual measured strengths and failures, and avoids claiming a novel architecture or state-of-the-art performance without evidence.

## 12. Files to keep from your Kaggle run

### Animation and bounding-box inspection during your current run

The notebook now includes a **Visual inspection** section after holdout evaluation and before submission export:

1. Let your current run finish its 20 epochs. Do not restart the session.
2. Let the existing checkpoint-loading, prediction-helper and calibration cells finish.
3. Copy the new Visual inspection markdown and its **two code cells** into the running Kaggle notebook. Run only these new cells. Reimporting the whole notebook can disrupt your session; rerunning the training cell is unnecessary.
4. The first cell applies the loaded **best saved checkpoint** to all test images and displays an animated GIF. It is not necessarily the epoch-20 checkpoint.
5. The second cell shows probability maps and mask-derived bounding boxes for the first, middle and last selected images. Edit `BOX_IMAGE_INDICES` to inspect other images.

Controls in the animation cell:

```python
VIS_SPLIT = 'test'          # Use 'tune' for labeled calibration-set inspection.
VIS_FRAME_LIMIT = None     # All images; use 10 for a short preview.
VIS_FRAME_DURATION_MS = 500 # Half a second per frame.
VIS_DISPLAY_SIZE = 512
```

Changing the frame limit or playback speed does not train the model. The same fixed weights predict every frame. Images span different dates and stations, so playback is a prediction gallery, **not an animation of one filament evolving**.

Colored masks show individual connected components; rectangles are the smallest axis-aligned boxes enclosing each predicted component. Their coordinates are measured on the original 2048-pixel canvas. There is no separate box detector, box-regression objective, or claim that each component is a physically correct filament. Check for giant boxes, merged nearby filaments, fragments, false sunspot detections and missed faint structures.

The GIF requires Pillow, not FFmpeg. Frames are reduced to a 256-color palette for memory efficiency; 180 frames at 512 pixels require roughly 45 MiB for stored pixel data, plus intermediate arrays/model memory. Visualization reruns inference and post-processing but never calls the optimizer.

For `VIS_SPLIT='test'`, outputs are `test_predictions.gif`, `test_animation_audit.json`, `test_filament_boxes.png`, and `test_filament_boxes.csv`. The audit records checkpoint identity, thresholds, image names and original-coordinate boxes. Choosing `tune` produces correspondingly named files.

Use the labeled tuning set and quantitative calibration when selecting threshold/model changes. Test-set appearance can help identify implementation failures, but provides no measured accuracy. Keep the independent holdout out of the tuning loop.

Under `/kaggle/working/unet_resnet18_gray_512_seed2026/`, the notebook writes:

| Generated filename                                     | Use                                                         |
| ------------------------------------------------------ | ----------------------------------------------------------- |
| `best_model.pt`                                      | Best checkpoint and saved experiment configuration          |
| `split_manifest.json`                                | Physical-image split policy and annotation hash             |
| `training_history.json`                              | Per-epoch train/tuning losses                               |
| `training_runtime.json`                              | Training duration and device                                |
| `calibration.json`                                   | Candidate grid, selected parameters and checkpoint identity |
| `holdout_summary.json`                               | Method-level metrics and evaluation runtime                 |
| `holdout_unet.csv`, `holdout_contrast.csv`         | Per-annotation-set holdout results                          |
| `training_annotation.png`                            | Training annotation sanity check                            |
| `holdout_distributions.png`, `holdout_overlay.png` | Holdout evidence                                            |
| `test_preview.png`, `test_prediction_audit.json`   | Test prediction inspection                                  |
| `submission.csv`, `submission_manifest.json`       | Validated competition export and identity checks            |
| `requirements.txt`, `experiment.json`              | Installed versions and experiment metadata                  |

These are runtime output names, not claims that a full Kaggle run has already generated them. If a cell fails, later artifacts may not exist. The final evidence cell runs after export; resolve an export failure before expecting a complete package.

Save a completed Kaggle notebook version and download the outputs. The notebook deliberately leaves `leaderboard_score` unset until you record the actual submission response. Its API upload is disabled by default.

**Keep separate:** a prediction CSV submission, the public reproducible code/weights, and the organizer's four-page report/form submission are different deliverables.

## References and implementation sources

- [Your CV implementation](../../Kaggle%20Competition/Notebooks/solar_filament_CV.ipynb)
- [Research/data companion notebook](./solar_filament_data_and_research.ipynb)
- ESAIR 2026 poster (external reference, not distributed in this repository)
- SoftwareX paper, [DOI](https://doi.org/10.1016/j.softx.2025.102120)
- ICDMW paper, [DOI](https://doi.org/10.1109/ICDMW69685.2025.00235)
- [SMP model documentation](https://smp.readthedocs.io/en/latest/models.html)
- [U-Net paper](https://arxiv.org/abs/1505.04597)
- [Panoptic segmentation / PQ paper](https://arxiv.org/abs/1801.00868)
- [Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
- [Competition overview and evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview)
