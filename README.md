# Spine MRI Model Benchmark

**Weights:** [Hugging Face — spine-mri-vision-model-training](https://huggingface.co/klasinlapaprechar/spine-mri-vision-model-training)

## Problem

Clinical MRI data often arrives misclassifed. The fields that is most often misclassified are **contrast** (which sequence the scan is), **VOI** (which part of the spine it covers), and **acq** (which plane it was acquired in). When those labels are off, a researcher cannot tell what they are looking at without opening every scan by hand. That turns a dataset into a sorting problem before any real analysis can start.

## Solution

The first approach was to train vision models that ignore the labels and read the scan itself. We trained a range of architectures on the same task: look at the scan and predict contrast, VOI, and acq. 

## Constraints

Two limits shaped what this benchmark can honestly claim.

- **Not enough labels across MRI types.** Almost all of the labelled data covers a small set of contrasts (Only t2w and t2*). We did not have comparable labels for the rest of the sequences researchers  use, including T1, FLAIR, and DWI. 
- **Severe class imbalance.** In our labeled dataset, some classes had far more scans than others. A model can post a high accuracy by mostly learning the common class and still fail on the rare ones.

## Protocol


| Item             | Setting                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------- |
| Tasks            | `type` ∈ {t2w, t2star}, `voi` ∈ {cervical, thoracic, lumbar}, `acq` ∈ {axial, sagittal} |
| n (in-domain)    | 2,646 scans / ~228 subjects                                                             |
| Splits           | locked subject-safe 80/10/10; 5-fold CV only inside train                               |
| Input (2D track) | 1.5 mm resample → percentile clip + z-score → 3 mid-slices → 224²                       |
| Input (3D track) | 1.5 mm resample → percentile clip + z-score → full volume resized to 96×128×128         |
| Optim            | AdamW, class-weighted CE, early stop on val balanced accuracy                           |
| Heads            | separate per-task networks + multi-head shared backbone                                 |
| External         | spine-generic type evaluation (site-stratified subject sample)                          |
| Composite        | `0.7 * in_domain_mean_bal_acc + 0.3 * ood_type_mean_bal_acc`                            |


## Model roster

**Track A — fully evaluated (n=13):** AlexNet, GoogLeNet, ResNet-18, DenseNet-121, EfficientNet-B0, EfficientNet-V2-S, MobileNetV3-Small, VGG11, ConvNeXt-Tiny, CoAtNet-0, MaxViT-Tiny, Swin-Tiny, frozen brain-sequence linear probe.

**Track B — same protocol, 3D + foundation extension:** full-volume 3D ResNet-18/50, 3D DenseNet-121; frozen mid-slice probes (DINOv2-B, SigLIP2-base, BiomedCLIP). Implemented under a shared registry; smoke-validated locally; full-scale server runs tracked separately from the published Track A ranking.

## Results


| Rank | Model             | Composite bal-acc |
| ---- | ----------------- | ----------------- |
| 1    | ResNet-18         | **0.9919**        |
| 2    | ConvNeXt-Tiny     | 0.9910            |
| 3    | EfficientNet-V2-S | 0.9907            |


ResNet-18: 99.2% composite balanced accuracy; **100%** T2w/T2* balanced accuracy on the external type set (134 scans, separate head). VOI remains the hardest head (DenseNet-121 separate best in-domain VOI).

Full ranking: `[results/aggregate_model_ranking.csv](results/aggregate_model_ranking.csv)`

## Interpretation

The models doing well is a good sign, as separating T2w from T2* is difficult even for seasoned MRI researchers. 

On the other hand, label coverage is still incomplete for other contrasts, including T1 and FLAIR. This is a proof that the approach is worth expanding with more annotation.

## Smoke

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_smoke.py
```

Synthetic volumes only — validates I/O + metric wiring, not published accuracies.

## Layout

```text
configs/   example experiment YAML
src/       sanitized data / metrics stubs
scripts/   smoke entrypoint
results/   aggregate ranking + protocol notes
docs/      design notes
```

## Intentionally omitted

Clinical NIfTI/DICOM, subject IDs, absolute paths, per-scan prediction tables, training host configuration.