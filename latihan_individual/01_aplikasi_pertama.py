"""
Praktik Bab 1.3: Aplikasi Pertama Streamlit
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st

# Judul Utama Aplikasi
st.title("Aplikasi Pemantauan Bike Sharing Demand")

# Informasi Identitas Kelompok
st.header("Kelompok 7 - Machine Learning Terapan")
st.subheader("Teknik Informatika - Universitas Pamulang (UNPAM)")

# Keterangan Proyek
st.write(
    "Aplikasi web ini dibangun menggunakan framework Streamlit untuk menganalisis "
    "pola peminjaman sepeda komuter dan kasual berdasarkan parameter cuaca dan waktu operasional."
)

st.caption("Sumber Data: Capital Bikeshare Washington D.C. | Target: Total Rental (count)")
