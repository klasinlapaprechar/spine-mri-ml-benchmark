# Model roster

Track A (published ranking): ImageNet/timm 2D mid-slice backbones + one frozen linear-probe baseline.  
Track B: MONAI-style 3D CNNs on full volumes; frozen foundation encoders on the same mid-slice tensor as Track A. Registry dispatch is lazy so a missing optional weight does not abort the bake-off (`skip_failed_models`).
