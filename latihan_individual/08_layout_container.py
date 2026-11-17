"""
Praktik Bab 2.7: Tata Letak dan Wadah (Layout & Container)
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st
import pandas as pd
import time

st.title("Praktik 2.7: Layout dan Container Streamlit")

# 1. st.sidebar
st.sidebar.title("Navigasi Kontrol Armada")
sidebar_choice = st.sidebar.selectbox(
    "Pilih Mode Analisis:",
    ["Ringkasan Eksekutif", "Sensor Real-Time", "Simulasi Alokasi Truk", "Bantuan Teknis"]
)
st.sidebar.info(f"Mode Terpilih: {sidebar_choice}")
st.sidebar.caption("Kelompok 7 - Machine Learning TI UNPAM")

# 2. st.columns
st.subheader("1. Tata Letak Multi-Kolom (st.columns)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Suhu Udara Rata-rata", value="24.5 °C", delta="+2.1 °C")
    st.write("Sensor termal stasiun mencatat suhu ideal bersepeda.")

with col2:
    st.metric(label="Kelembaban Relatif", value="58 %", delta="-4.0 %")
    st.write("Tingkat kenyamanan komuter tergolong optimal.")

with col3:
    st.metric(label="Kecepatan Angin", value="12.8 km/jam", delta="+1.5 km/jam")
    st.write("Kondisi angin aman untuk mobilitas sepeda kota.")

st.divider()

# 3. st.tabs
st.subheader("2. Navigasi Multi-Tab (st.tabs)")
tab_data, tab_rebalance, tab_formula = st.tabs(["Eksplorasi Data", "Jadwal Rebalancing", "Formula Matematis"])

with tab_data:
    st.write("Tampilan cuplikan dataset Bike Sharing Demand:")
    df = pd.read_csv("tugas_streamlit_bab1_bab2/data/bike_sharing.csv")
    st.dataframe(df[['datetime', 'season', 'temp', 'humidity', 'count']].head(5), use_container_width=True)

with tab_rebalance:
    st.write("Rencana jadwal truk pemindah sepeda:")
    schedule_df = pd.DataFrame([
        {"Rute": "Rute Utara - Dupont Circle", "Truk ID": "TRK-01", "Kapasitas": 30, "Jam Dispatch": "06:30 WIB"},
        {"Rute": "Rute Pusat - National Mall", "Truk ID": "TRK-02", "Kapasitas": 25, "Jam Dispatch": "07:00 WIB"},
        {"Rute": "Rute Selatan - Navy Yard", "Truk ID": "TRK-03", "Kapasitas": 35, "Jam Dispatch": "07:15 WIB"}
    ])
    st.table(schedule_df)

with tab_formula:
    st.latex(r"\text{Target: } \text{count} = \text{casual} + \text{registered}")
    st.caption("Pemisahan pengguna kasual dan terdaftar penting untuk memprediksi karakter jam sibuk.")

st.divider()

# 4. st.expander
st.subheader("3. Wadah Buka-Tutup (st.expander)")
with st.expander("Klik untuk Membaca Metadata Kolom Dataset Bike Sharing"):
    st.markdown("""
    - **datetime**: Tanggal dan jam pencatatan data.
    - **season**: 1 = Spring (Musim Semi), 2 = Summer (Musim Panas), 3 = Fall (Musim Gugur), 4 = Winter (Musim Dingin).
    - **holiday**: Indikator hari libur nasional (0 = Bukan hari libur, 1 = Hari libur).
    - **workingday**: Indikator hari kerja (0 = Akhir pekan/libur, 1 = Hari kerja efektif).
    - **weather**: Kategori cuaca 1 (cerah) hingga 4 (hujan lebat / badai).
    - **temp**: Suhu udara terukur (°C).
    - **atemp**: Suhu yang dirasakan tubuh manusia (°C).
    - **humidity**: Tingkat kelembaban udara (persentase 0-100).
    - **windspeed**: Kecepatan hembusan angin.
    - **casual**: Jumlah sepeda yang dipinjam oleh pengguna tanpa langganan.
    - **registered**: Jumlah sepeda yang dipinjam oleh anggota terdaftar.
    - **count**: Total peminjaman sepeda (variabel target machine learning).
    """)

st.divider()

# 5. st.container
st.subheader("4. Wadah Pengelompokan Elemen (st.container)")
with st.container():
    st.write("Kontainer ini membungkus peringatan batas kapasitas dock stasiun:")
    st.warning("Perhatian: Stasiun Foggy Bottom GWU diprediksi mengalami dock starvation pada pukul 08:30!")

st.divider()

# 6. st.empty
st.subheader("5. Wadah Dinamis Placeholder (st.empty)")
st.caption("Komponen st.empty() dapat memperbarui atau menimpa konten secara real-time.")
placeholder = st.empty()

if st.button("Jalankan Simulasi Hitung Mundur Sinkronisasi"):
    for detik in range(3, 0, -1):
        placeholder.info(f"Menyinkronkan data telemetri dalam {detik} detik...")
        time.sleep(1)
    placeholder.success("Sinkronisasi data sensor stasiun sepeda selesai 100%!")
