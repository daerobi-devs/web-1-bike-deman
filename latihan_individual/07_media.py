"""
Praktik Bab 2.6: Menampilkan Media Multimedia Streamlit
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st
import os

st.title("Praktik 2.6: Menampilkan Media Multimedia pada Bike Sharing")

# 1. st.image
st.subheader("1. Menampilkan Gambar (st.image)")
st.caption("Menampilkan diagram kerangka kerja metodologi machine learning CRISP-DM Bike Sharing.")
image_path = "tugas_streamlit_bab1_bab2/assets/crisp_dm_pipeline.png"
if os.path.exists(image_path):
    st.image(image_path, caption="Kerangka Kerja CRISP-DM Bike Sharing Demand Kelompok 7", use_container_width=True)
else:
    st.info("File gambar placeholder dapat diletakkan pada folder assets.")

# 2. st.audio
st.subheader("2. Menampilkan Pemutar Audio (st.audio)")
st.caption("Memutar audio nada pemberitahuan / alarm stasiun sepeda saat dock hampir penuh.")
audio_path = "tugas_streamlit_bab1_bab2/assets/station_chime.wav"
if os.path.exists(audio_path):
    st.audio(audio_path, format="audio/wav")
else:
    st.info("Audio chime stasiun sepeda dapat dimuat via path lokal.")

# 3. st.video
st.subheader("3. Menampilkan Pemutar Video (st.video)")
st.caption("Memutar animasi simulasi pola penyewaan sepeda 24 jam (stream dari Daeroom MinIO S3).")
s3_video_url = "https://s3.daeroom.my.id/bikesharingdemand-assets/media/bike_sharing_demo.mp4"
st.video(s3_video_url)
