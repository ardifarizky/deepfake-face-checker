"""CPU Streamlit inference: full-image preprocessing + selectable v2 CNN/Swin models.
Run from hf_space_v2 or project root after checkpoints are copied.
"""
from __future__ import annotations
import io, time
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import transforms
import streamlit as st

APP=Path(__file__).resolve().parent
MEAN=[.485,.456,.406];STD=[.229,.224,.225]
MODEL_INFO={
 'EfficientNet-B0 (CNN)':('efficientnet','efficientnet_b0.ra_in1k','efficientnet_v2_best.pt'),
 'Swin-Tiny (Transformer)':('swin','swin_tiny_patch4_window7_224.ms_in1k','swin_v2_best.pt')}

@st.cache_resource
def load_model(timm_id,filename):
 import timm
 path=APP/'checkpoints'/filename
 if not path.exists():raise FileNotFoundError(f'Missing checkpoint: {filename}')
 m=timm.create_model(timm_id,pretrained=False,num_classes=2)
 state=torch.load(path,map_location='cpu',weights_only=True)
 state={k:(v.float() if v.dtype==torch.float16 else v) for k,v in state.items()}
 m.load_state_dict(state);m.eval();return m

def main():
 st.set_page_config(page_title='Deepfake Face Checker',page_icon='🔎',layout='centered')
 st.title('Deepfake Face Checker')
 choice=st.selectbox('Pilih model',list(MODEL_INFO))
 key,timm_id,file=MODEL_INFO[choice]
 uploaded=st.file_uploader('Upload satu foto wajah',type=['jpg','jpeg','png','webp'])
 if not uploaded:st.stop()
 raw=uploaded.getvalue()
 if len(raw)>15*1024*1024:st.error('Maksimum 15 MB.');st.stop()
 try:
  original=Image.open(io.BytesIO(raw))
  face=ImageOps.exif_transpose(original).convert('RGB')
 except Exception as e:st.error(str(e));st.stop()
 st.image(face.resize((224,224),Image.Resampling.BILINEAR),caption='Gambar yang dianalisis')
 if not st.button('Analisis',type='primary',use_container_width=True):st.stop()
 try:
  model=load_model(timm_id,file)
  tf=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
  x=tf(face).unsqueeze(0);start=time.perf_counter()
  with torch.inference_mode():probs=torch.softmax(model(x),dim=1)[0]
  elapsed=(time.perf_counter()-start)*1000
 except Exception as e:st.error(f'Inferensi gagal: {e}');st.stop()
 real,fake=float(probs[0]),float(probs[1]);label='FAKE' if fake>=.5 else 'REAL'
 st.subheader(f'Prediksi {choice}: {label}')
 st.progress(fake,text=f'P(fake) = {fake:.1%}')
 st.write(f'P(real) = **{real:.1%}** · Inferensi CPU: **{elapsed:.0f} ms**')

 st.divider()
 st.subheader('Explainable AI')
 st.caption('Visualisasi area yang memengaruhi skor kelas prediksi, bukan bukti kausal adanya artefak deepfake.')
 methods=['Grad-CAM (khusus EfficientNet-B0)', 'Occlusion Sensitivity (semua model)'] if key=='efficientnet' else ['Occlusion Sensitivity (semua model)']
 method=st.radio('Metode visualisasi', methods, horizontal=True)
 grid=32
 if method.startswith('Occlusion'):
  grid=st.select_slider('Resolusi grid occlusion', options=[16,32,64], value=16, help='Grid lebih besar memberi detail lebih tinggi tetapi membutuhkan lebih banyak inferensi. 16 adalah yang tercepat.')
 if st.button('Buat visualisasi penjelasan', use_container_width=True):
  from explainability import gradcam_efficientnet, occlusion_sensitivity, heatmap_overlay, heatmap_image
  try:
   start_xai=time.perf_counter()
   if method.startswith('Grad-CAM'):
    if key!='efficientnet':
     st.info('Grad-CAM pada aplikasi ini hanya tersedia untuk EfficientNet-B0. Pilih model tersebut atau gunakan Occlusion Sensitivity.')
     st.stop()
    x_grad=x.clone().requires_grad_(True)
    with torch.enable_grad():
     heatmap=gradcam_efficientnet(model,x_grad,int(probs.argmax().item()))
    xai_label='Grad-CAM'
   else:
    heatmap=occlusion_sensitivity(model,x,int(probs.argmax().item()),grid_size=grid)
    xai_label=f'Occlusion Sensitivity ({grid}×{grid})'
   elapsed_xai=(time.perf_counter()-start_xai)*1000
   col1,col2=st.columns(2)
   with col1:
    st.image(heatmap_image(heatmap),caption=f'{xai_label} — area sensitivitas',use_container_width=True)
   with col2:
    st.image(heatmap_overlay(face,heatmap),caption='Overlay pada gambar input',use_container_width=True)
   st.caption(f'Kelas yang dijelaskan: {label} · Waktu: {elapsed_xai/1000:.1f} detik. Area hangat menunjukkan sensitivitas lebih tinggi terhadap skor kelas tersebut.')
  except Exception as e:
   st.error(f'Visualisasi gagal: {e}')

if __name__=='__main__':main()
