# Deepfake Face Checker

Streamlit app: upload a face image → **REAL/FAKE** prediction with class probabilities.

- **EfficientNet-B0 (CNN)** and **Swin-Tiny (Transformer)**, selectable in the app.
- Full-image preprocessing: EXIF transpose → RGB → 224×224 → ImageNet normalization.
- CPU inference. Checkpoints stored as FP16 and upcast to FP32 at load (smaller repo, same predictions).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. **New app** → select this repository (`ardifarizky/deepfake-face-checker`), branch `main`, main file `app.py`.
3. Deploy. CPU is enough.

## Research prototype — limitations
Not an authenticity certificate. Performance varies by generator and image domain.
On the external DF40 benchmark the models reached ~94.6% accuracy, but mixed in-domain
accuracy was lower (80–83%) and real IMDB-WIKI images were often misclassified.
Do not use output to accuse, defame, or make consequential decisions about a person.

Data: FFHQ/StyleGAN (140k), DFDC crops, DeepFakeFace, DF40-derived mirror.
DF40: Yan et al., 2024, arXiv:2406.13495, CC BY-NC 4.0.
