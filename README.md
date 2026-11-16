# Deepfake Face Checker

Streamlit app: upload a face image → **REAL/FAKE** prediction with class probabilities.

- **EfficientNet-B0 (CNN)** and **Swin-Tiny (Transformer)**, selectable in the app.
- Grad-CAM for both models and architecture-agnostic occlusion sensitivity.
- Full-image preprocessing: EXIF transpose → RGB → 224×224 → ImageNet normalization.
- CPU inference. EfficientNet is stored in FP32 (16.3 MB); Swin is stored in FP16 and upcast at load (55.1 MB).

## Current checkpoints

Best checkpoints from the executed notebook training pipeline, replacing the earlier v2 models.

| Model | Best epoch | Validation AUC | Internal accuracy / AUC | DF40 accuracy / AUC |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 2 | 0.9869 | 94.97% / 0.9913 | 57.47% / 0.5612 |
| Swin-Tiny | 4 | 0.9841 | 94.28% / 0.9877 | 45.45% / 0.3463 |

DF40 fake recall is 15.85% for EfficientNet and 7.25% for Swin at threshold 0.5. These are single-seed research baselines with a large internal-to-external generalization gap.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. **New app** → select this repository (`ardifarizky/deepfake-face-checker`), branch `master`, main file `app.py`.
3. Deploy. CPU is enough.

## Research prototype — limitations
Not an authenticity certificate. Performance varies by generator and image domain. Grad-CAM
and occlusion maps indicate model sensitivity, not causal proof or verified manipulation regions.
Do not use output to accuse, defame, or make consequential decisions about a person.

Training data: 310,000 images from FFHQ/StyleGAN (140K Real and Fake Faces), DFDC, and
DeepFakeFace. DF40 is used only as a development-external evaluation set and is not included
in training. DF40 original: Yan et al., 2024, arXiv:2406.13495.
