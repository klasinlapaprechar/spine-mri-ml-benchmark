# Spine MRI Model Benchmark

**Weights:** [Hugging Face — spine-mri-vision-model-training](https://huggingface.co/klasinlapaprechar/spine-mri-vision-model-training)

## Problem

Clinical spine MRI arrives as heterogeneous DICOM with unreliable metadata. Before scans can be organized into research-ready BIDS, each volume needs scan-level labels for **type** (contrast), **acq** (orientation), and **VOI** (spine level). Expert labeling does not scale; automated typing is a prerequisite for dataset growth and downstream modeling.

When sidecars are incomplete or wrong, **images** must carry the label — which motivates the supervised bake-off below.

## Constraints

These limits shaped every design choice in this benchmark (splits, metrics, and what we claim):

- **Limited labelled categories** — We mainly have expert labels for a narrow contrast set (e.g. t2w / t2star). We do **not** have large labelled corpora for T1w, FLAIR, DWI, and other sequences, so a universal sequence classifier across all MRI types is not yet feasible.
- **PHI / clinical data** — Protected health information slows iteration: secure access, due diligence, and careful handling before every experiment or export. Public artifacts here are aggregate metrics only.
- **Class imbalance** — Some classes dominate (e.g. far more t2w than t2star). We report balanced accuracy and use class-weighted training; naive accuracy is misleading.

## Protocol

| Item | Setting |
|------|---------|
| Tasks | `type` ∈ {t2w, t2star}, `voi` ∈ {cervical, thoracic, lumbar}, `acq` ∈ {axial, sagittal} |
| n (in-domain) | 2,646 scans / ~228 subjects |
| Splits | locked subject-safe 80/10/10; 5-fold CV only inside train |
| Input (2D track) | 1.5 mm resample → percentile clip + z-score → 3 mid-slices → 224² |
| Optim | AdamW, class-weighted CE, early stop on val balanced accuracy |
| Heads | separate per-task networks + multi-head shared backbone |
| External | spine-generic type evaluation (site-stratified subject sample) |
| Composite | `0.7 * in_domain_mean_bal_acc + 0.3 * ood_type_mean_bal_acc` |

## Model roster

**Track A — fully evaluated (n=13):** AlexNet, GoogLeNet, ResNet-18, DenseNet-121, EfficientNet-B0, EfficientNet-V2-S, MobileNetV3-Small, VGG11, ConvNeXt-Tiny, CoAtNet-0, MaxViT-Tiny, Swin-Tiny, frozen brain-sequence linear probe.

**Track B — same protocol, 3D + foundation extension:** full-volume 3D ResNet-18/50, 3D DenseNet-121; frozen mid-slice probes (DINOv2-B, SigLIP2-base, BiomedCLIP). Implemented under a shared registry; smoke-validated locally; full-scale server runs tracked separately from the published Track A ranking.

## Results (Track A)

| Rank | Model | Composite bal-acc | OOD type (sep) |
|-----:|-------|------------------:|---------------:|
| 1 | ResNet-18 | **0.9919** | **1.000** |
| 2 | ConvNeXt-Tiny | 0.9910 | 0.985 |
| 3 | EfficientNet-V2-S | 0.9907 | 0.993 |

ResNet-18: 99.2% composite balanced accuracy; **100%** T2w/T2* balanced accuracy on the external type set (134 scans, separate head). VOI remains the hardest head (DenseNet-121 separate best in-domain VOI).

Full ranking: [`results/aggregate_model_ranking.csv`](results/aggregate_model_ranking.csv)

## Interpretation

T2w vs T2* is operationally hard for metadata heuristics and for human triage at scale; the bake-off shows a dedicated mid-slice CNN closes that gap under subject-safe + OOD evaluation. Label coverage is still incomplete for other contrasts (e.g. T1/FLAIR), so this is a **proof of concept for investment in expanded annotation**, not a claim of a universal sequence classifier.

Before committing to full supervised training, we also asked whether frozen TotalSegmentator features already separate contrasts — see the **alternative** probe study: [totalsegmentator-probe-study](https://github.com/klasinlapaprechar/totalsegmentator-probe-study). Production integration of learned weights is described in [clinical-dicom2bids-demo](https://github.com/klasinlapaprechar/clinical-dicom2bids-demo).

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
