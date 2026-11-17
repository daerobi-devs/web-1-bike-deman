"""
Praktik Bab 2.2: Elemen Text Streamlit
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st

# 1. st.title
st.title("Praktik 2.2: Eksplorasi Elemen Teks pada Bike Sharing")

# 2. st.header
st.header("1. Arsitektur Data Rental Sepeda")

# 3. st.subheader
st.subheader("1.1 Parameter Lingkungan dan Variabel Target")

# 4. st.caption
st.caption("Dokumentasi teknis fitur prediktif: Suhu, Kelembaban, Kecepatan Angin, dan Musim.")

# 5. st.markdown dengan warna dan penekanan
st.markdown(
    "Data operasional merekam pengaruh kondisi lingkungan secara langsung: "
    ":blue[Suhu Udara (°C)], :green[Kelembaban Relatif (%)], dan :red[Kecepatan Angin (km/jam)]. "
    "Kombinasi cuaca ini menentukan rasio pengguna **kasual (rekreasi)** versus **terdaftar (komuter)**."
)

# 6. st.divider
st.divider()

# 7. st.text (plain mono text)
st.subheader("1.2 Status Buffer Ingestion Sensor")
st.text("STATUS: Ingestion log sensor stasiun sepeda D.C. berjalan normal. 0 data korup terdeteksi.")

# 8. st.code (syntax highlighting code block)
st.subheader("1.3 Ekstraksi Fitur Waktu dengan Pandas")
python_code = """# Ekstraksi komponen jam dan hari kerja dari kolom datetime
import pandas as pd

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if h in [8, 9, 17, 18] else 0)
"""
st.code(python_code, language="python")

# 9. st.latex (Mathematical expressions)
st.subheader("1.4 Formula Metrik Evaluasi Regresi (RMSLE)")
st.latex(r"\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \ln(\hat{y}_i + 1) - \ln(y_i + 1) \right)^2}")
st.caption("RMSLE digunakan agar penalti kesalahan under-estimate dan over-estimate berimbang pada skala logaritmik.")
