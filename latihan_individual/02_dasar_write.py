"""
Praktik Bab 2.1: Dasar Streamlit (st.write)
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st
import pandas as pd

st.title("Praktik 2.1: Dasar Streamlit dengan st.write()")

# 1. Menampilkan teks string
st.subheader("1. Menampilkan Teks / Pesan Sistem")
st.write("Sistem Analisis Operasional Bike Sharing aktif dan siap menerima query data.")

# 2. Menampilkan angka / nilai numerik
st.subheader("2. Menampilkan Nilai Numerik / Statistik Ringkas")
st.write("Total baris data dalam sampel observasi:")
st.write(10886)

# 3. Menampilkan DataFrame sewa sepeda
st.subheader("3. Menampilkan Struktur DataFrame Cuaca & Sewa Sepeda")
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bike_sharing.csv"
df = pd.read_csv(DATA_PATH)
st.write(df.head(5))

# 4. Menampilkan dictionary parameter operasional
st.subheader("4. Menampilkan Dictionary Konfigurasi Sensor Stasiun")
st.write({
    "station_id": "ST-CAPITAL-007",
    "station_name": "Lincoln Memorial North Dock",
    "dock_capacity": 45,
    "available_bikes": 28,
    "status": "Operational",
    "battery_backup_pct": 98.5
})
