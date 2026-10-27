"""Streamlit Deepfake Face Checker with persistent prediction and XAI results.
The app uses session_state so Streamlit reruns do not discard prediction outputs.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
import streamlit as st

APP = Path(__file__).resolve().parent
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]
MODEL_INFO = {
    'EfficientNet-B0 (CNN)': ('efficientnet', 'efficientnet_b0.ra_in1k', 'efficientnet_v2_best.pt'),
    'Swin-Tiny (Transformer)': ('swin', 'swin_tiny_patch4_window7_224.ms_in1k', 'swin_v2_best.pt'),
}
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@st.cache_resource
def load_model(timm_id, filename):
    import timm
    path = APP / 'checkpoints' / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {filename}')
    model = timm.create_model(timm_id, pretrained=False, num_classes=2)
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def reset_analysis():
    """Called when the selected model changes; clear only stale analysis results."""
    st.session_state.pop('prediction', None)
    st.session_state.pop('explanation', None)


def main():
    st.set_page_config(page_title='Deepfake Face Checker', page_icon='🔎', layout='centered')
    st.title('Deepfake Face Checker')

    choice = st.selectbox('Pilih model', list(MODEL_INFO), on_change=reset_analysis)
    model_key, timm_id, checkpoint = MODEL_INFO[choice]
    uploaded = st.file_uploader('Upload satu foto wajah', type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded is None:
        st.session_state.pop('prediction', None)
        st.session_state.pop('explanation', None)
        return

    raw = uploaded.getvalue()
    if len(raw) > 15 * 1024 * 1024:
        st.error('Maksimum 15 MB.')
        return

    try:
        original = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(original).convert('RGB')
        tensor = TRANSFORM(image).unsqueeze(0)
    except Exception as exc:
        st.error(f'Gambar tidak dapat dibaca: {exc}')
        return

    st.image(image.resize((224, 224), Image.Resampling.BILINEAR), caption='Gambar yang dianalisis')

    # Hash raw bytes, not a mutable UploadedFile object.
    input_id = hashlib.sha256(raw).hexdigest() + ':' + model_key
    model = load_model(timm_id, checkpoint)

    if st.button('Analisis', type='primary', width='stretch'):
        try:
            start = time.perf_counter()
            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
            elapsed = (time.perf_counter() - start) * 1000
            real, fake = float(probabilities[0]), float(probabilities[1])
            st.session_state.prediction = {
                'input_id': input_id,
                'real': real,
                'fake': fake,
                'label': 'FAKE' if fake >= .5 else 'REAL',
                'class_index': int(probabilities.argmax().item()),
                'elapsed_ms': elapsed,
                'tensor': tensor,
                'image': image,
                'model_key': model_key,
                'choice': choice,
            }
            st.session_state.pop('explanation', None)
        except Exception as exc:
            st.error(f'Inferensi gagal: {exc}')
            return

    prediction = st.session_state.get('prediction')
    if not prediction or prediction['input_id'] != input_id:
        st.info('Klik Analisis untuk melihat prediksi dan visualisasi Explainable AI.')
        return

    st.subheader(f"Prediksi {prediction['choice']}: {prediction['label']}")
    st.progress(prediction['fake'], text=f"P(fake) = {prediction['fake']:.1%}")
    st.write(f"P(real) = **{prediction['real']:.1%}** · Inferensi CPU: **{prediction['elapsed_ms']:.0f} ms**")

    st.divider()
    st.subheader('Explainable AI')
    st.caption('Visualisasi area yang memengaruhi skor kelas prediksi, bukan bukti kausal adanya artefak deepfake.')

    methods = (['Grad-CAM (khusus EfficientNet-B0)', 'Occlusion Sensitivity (semua model)']
               if model_key == 'efficientnet' else ['Occlusion Sensitivity (semua model)'])
    method = st.radio('Metode visualisasi', methods, horizontal=True)
    grid = 16
    if method.startswith('Occlusion'):
        grid = st.select_slider(
            'Resolusi grid occlusion', options=[16, 32, 64], value=16,
            help='Grid besar lebih detail tetapi lebih lambat. Mulai dari 16 untuk CPU.'
        )

    if st.button('Buat visualisasi penjelasan', width='stretch'):
        from explainability import (
            gradcam_efficientnet, occlusion_sensitivity, heatmap_image, heatmap_overlay,
        )
        try:
            start = time.perf_counter()
            # Retrieve the current prediction again: no dependency on the Analisis button state.
            current = st.session_state.prediction
            x = current['tensor']
            class_index = current['class_index']
            if method.startswith('Grad-CAM'):
                with torch.enable_grad():
                    heatmap = gradcam_efficientnet(model, x.clone().requires_grad_(True), class_index)
                method_label = 'Grad-CAM'
            else:
                heatmap = occlusion_sensitivity(model, x, class_index, grid_size=grid)
                method_label = f'Occlusion Sensitivity ({grid}×{grid})'
            elapsed = time.perf_counter() - start
            st.session_state.explanation = {
                'input_id': input_id,
                'method_label': method_label,
                'heatmap_png': heatmap_image(heatmap),
                'overlay_png': heatmap_overlay(current['image'], heatmap),
                'label': current['label'],
                'elapsed_s': elapsed,
            }
        except Exception as exc:
            st.error(f'Visualisasi gagal: {exc}')

    explanation = st.session_state.get('explanation')
    if explanation and explanation['input_id'] == input_id:
        col1, col2 = st.columns(2)
        with col1:
            st.image(explanation['heatmap_png'], caption=f"{explanation['method_label']} — area sensitivitas", width='stretch')
        with col2:
            st.image(explanation['overlay_png'], caption='Overlay pada gambar input', width='stretch')
        st.caption(
            f"Kelas yang dijelaskan: {explanation['label']} · Waktu: {explanation['elapsed_s']:.1f} detik. "
            'Area hangat menunjukkan sensitivitas lebih tinggi terhadap skor kelas tersebut.'
        )


if __name__ == '__main__':
    main()
