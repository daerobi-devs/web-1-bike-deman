"""
Praktik Bab 2.3: Menampilkan Data (st.dataframe, st.table, st.metric, st.json)
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st
import pandas as pd

st.title("Praktik 2.3: Menampilkan Data Operasional Sepeda")

# Memuat dataset Bike Sharing
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bike_sharing.csv"
df = pd.read_csv(DATA_PATH)

# 1. st.dataframe (Tabel interaktif dengan filter dan sorting)
st.subheader("1. Tabel Interaktif Dataset (st.dataframe)")
st.caption("Menampilkan 10 baris pertama data peminjaman sepeda.")
st.dataframe(df.head(10), use_container_width=True)

# 2. st.table (Tabel statis untuk ringkasan statistik)
st.subheader("2. Ringkasan Statistik Utama (st.table)")
summary_stats = df[['temp', 'humidity', 'windspeed', 'casual', 'registered', 'count']].describe().round(2)
st.table(summary_stats.loc[['mean', 'std', 'min', 'max']])

# 3. st.metric (Indikator Kinerja Utama / KPI Cards)
st.subheader("3. Panel Metrik Kinerja Armada (st.metric)")
col1, col2, col3, col4 = st.columns(4)

total_rentals = int(df['count'].sum())
avg_temp = round(df['temp'].mean(), 1)
total_casual = int(df['casual'].sum())
total_registered = int(df['registered'].sum())
registered_pct = round((total_registered / total_rentals) * 100, 1)

with col1:
    st.metric(label="Total Rental Sepeda", value=f"{total_rentals:,}", delta="+12.4% MoM")

with col2:
    st.metric(label="Rata-rata Suhu", value=f"{avg_temp} °C", delta="+1.2 °C")

with col3:
    st.metric(label="Rasio Komuter", value=f"{registered_pct}%", delta="+2.5%")

with col4:
    st.metric(label="Pengguna Rekreasi", value=f"{total_casual:,}", delta="-0.8%")

# 4. st.json (Metadata Stasiun IoT)
st.subheader("4. Telemetri Sensor IoT Stasiun Sepeda (st.json)")
station_metadata = {
    "station_code": "WDC-ST-042",
    "network": "Capital Bikeshare Hub",
    "location": {
        "city": "Washington, D.C.",
        "latitude": 38.8977,
        "longitude": -77.0365,
        "landmark": "Near White House Promenade"
    },
    "hardware_specs": {
        "dock_slots": 50,
        "ebike_chargers": 20,
        "firmware_version": "v3.14.2",
        "last_ping_utc": "2026-11-10T09:30:00Z"
    },
    "current_status": {
        "available_standard": 19,
        "available_ebike": 11,
        "empty_docks": 20,
        "starvation_risk": "Low"
    }
}
st.json(station_metadata)
