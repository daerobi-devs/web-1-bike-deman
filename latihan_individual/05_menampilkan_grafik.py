"""
Praktik Bab 2.4: Menampilkan Grafik Streamlit
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

st.title("Praktik 2.4: Visualisasi Grafik & Geografis Bike Sharing")

# Memuat data
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bike_sharing.csv"
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

# 1. st.line_chart: Tren Sewa Sepeda 24 Jam
st.subheader("1. Pola Fluktuasi Rental Sepeda 24 Jam (st.line_chart)")
st.caption("Pola bimodal memperlihatkan lonjakan komuter pada jam 08:00 pagi dan 17:00 sore.")
hourly_trend = df.groupby('hour')[['count', 'registered', 'casual']].mean()
st.line_chart(hourly_trend['count'])

# 2. st.area_chart: Komposisi Pengguna Komuter vs Kasual
st.subheader("2. Komposisi Rental Komuter vs Rekreasi (st.area_chart)")
st.area_chart(hourly_trend[['registered', 'casual']])

# 3. st.bar_chart: Total Rental Berdasarkan Musim
st.subheader("3. Volume Peminjaman Antar Musim (st.bar_chart)")
season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
df_season = df.copy()
df_season['season_name'] = df_season['season'].map(season_labels)
seasonal_counts = df_season.groupby('season_name')['count'].sum().reindex(["Spring", "Summer", "Fall", "Winter"])
st.bar_chart(seasonal_counts)

# 4. st.altair_chart: Scatter Plot Interaktif Suhu vs Demand
st.subheader("4. Analisis Korelasi Suhu Udara vs Rental Sepeda (st.altair_chart)")
sample_data = df.sample(300, random_state=42)
altair_chart = alt.Chart(sample_data).mark_circle(size=60).encode(
    x=alt.X('temp:Q', title='Suhu Udara (°C)'),
    y=alt.Y('count:Q', title='Total Rental (count)'),
    color=alt.Color('weather:N', title='Kondisi Cuaca', scale=alt.Scale(scheme='tableau10')),
    tooltip=['temp', 'humidity', 'windspeed', 'count']
).properties(
    height=350
).interactive()
st.altair_chart(altair_chart, use_container_width=True)

# 5. st.vega_lite_chart: Visualisasi Deklaratif Vega-Lite
st.subheader("5. Pemetaan Hubungan Kelembaban dan Kecepatan Angin (st.vega_lite_chart)")
vega_spec = {
    "mark": {"type": "point", "filled": True},
    "encoding": {
        "x": {"field": "humidity", "type": "quantitative", "title": "Kelembaban Relatif (%)"},
        "y": {"field": "windspeed", "type": "quantitative", "title": "Kecepatan Angin (km/jam)"},
        "color": {"field": "temp", "type": "quantitative", "title": "Suhu (°C)"},
        "size": {"field": "count", "type": "quantitative", "title": "Total Sewa"}
    }
}
st.vega_lite_chart(sample_data, vega_spec, use_container_width=True)

# 6. st.plotly_chart: Grafik Plotly Interaktif
st.subheader("6. Sebaran Interaktif Suhu Terasa vs Demand (st.plotly_chart)")
fig = px.scatter(
    sample_data,
    x="atemp",
    y="count",
    color="season",
    size="windspeed",
    hover_data=["temp", "humidity", "casual", "registered"],
    labels={"atemp": "Suhu Terasa (atemp °C)", "count": "Jumlah Rental Sepeda"},
    title="Sebaran Sewa Sepeda Berdasarkan Suhu Terasa (Washington D.C.)"
)
st.plotly_chart(fig, use_container_width=True)

# 7. st.map: Peta Persebaran Stasiun Sepeda Capital Bikeshare
st.subheader("7. Peta Lokasi Stasiun Sepeda Capital Bikeshare (st.map)")
st.caption("Peta interaktif titik koordinat latitude dan longitude stasiun sepeda di area Washington D.C.")
stations_data = pd.DataFrame({
    'station_name': [
        'Lincoln Memorial', 'National Mall East', 'Dupont Circle Metro',
        'Georgetown University', 'Union Station Hub', 'Smithsonian Castle',
        'Capitol South Metro', 'Adams Morgan', 'Navy Yard Station', 'Foggy Bottom GWU'
    ],
    'lat': [
        38.8893, 38.8897, 38.9096, 38.9072, 38.8978,
        38.8888, 38.8856, 38.9217, 38.8765, 38.9009
    ],
    'lon': [
        -77.0502, -77.0229, -77.0434, -77.0722, -77.0063,
        -77.0260, -77.0051, -77.0423, -77.0003, -77.0503
    ],
    'docks': [45, 35, 40, 30, 60, 42, 38, 28, 44, 36]
})
st.map(stations_data[['lat', 'lon']], zoom=12)
