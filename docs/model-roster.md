# Model roster

Track A (published ranking): ImageNet/timm 2D mid-slice backbones + one frozen linear-probe baseline.  
Track B: MONAI-style 3D CNNs on full volumes; frozen foundation encoders on the same mid-slice tensor as Track A. Registry dispatch is lazy so a missing optional weight does not abort the bake-off (`skip_failed_models`).

## Track B detail

Both families train the same three tasks (`type`, `voi`, `acq`) as separate
per-task heads plus one multi-head variant — 24 checkpoints in total.

| Model | Input | Encoder init |
|-------|-------|--------------|
| `resnet18_3d` | volume 96×128×128 | MONAI ResNet-18 + MedicalNet |
| `resnet50_3d` | volume 96×128×128 | MONAI ResNet-50 + MedicalNet |
| `densenet121_3d` | volume 96×128×128 | random — no 3D checkpoint upstream |
| `dinov2_vitb14` | 3 mid-slices, 224² | frozen `facebook/dinov2-base` |
| `siglip2_base` | 3 mid-slices, 224² | frozen `google/siglip2-base-patch16-224` |
| `biomedclip` | 3 mid-slices, 224² | frozen `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |

The foundation probes freeze the encoder and train a linear head only, so their
checkpoints embed a copy of the public pretrained weights.

Four further encoders were dropped on provenance grounds rather than
performance: `dinov3_vits16` and `dinov3_vitb16` report `gated: manual` on the
HF API and cannot be fetched unattended; `radimagenet_resnet50` and `medclip`
have no official HF repo and distribute weights through a request form.

Weights: [`round2/`](https://huggingface.co/klasinlapaprechar/spine-mri-vision-model-training/tree/main/round2)
on the Hub. Track B used a single train/val fit rather than 5-fold CV, so it
carries `final.pt` only and its metrics are not comparable to the Track A
composite ranking.
