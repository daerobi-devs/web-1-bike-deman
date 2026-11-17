"""
===================================================================================
APLIKASI UTAMA IMPLEMENTASI STREAMLIT BAB 1 & BAB 2
Mata Kuliah : Machine Learning Terapan (Semester 5)
Instansi    : Teknik Informatika, Universitas Pamulang (UNPAM)
Kelompok    : Kelompok 7
Anggota     : 1. Daerobi
              2. Hendro Tri Dewanto
              3. Nasrudin Ali Albana
Studi Kasus : Analisis & Prediksi Bike Sharing Demand (Time-Series Regression)
===================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import altair as alt
import plotly.express as px

try:
    from daeroom_client import query_db
    DAEROOM_AVAILABLE = True
except Exception:
    DAEROOM_AVAILABLE = False

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Tutorial Streamlit Bab 1-2 | Kelompok 7 UNPAM",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan bersih, elegan, dan ramah mata
st.markdown("""
<style>
    /* Styling Dasar */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #0369a1;
        background-color: #e0f2fe;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi Memuat Dataset (Prioritas Daeroom Cloud PostgreSQL, Fallback CSV Lokal)
@st.cache_data(ttl=300)
def load_bike_data():
    source_label = "CSV Lokal"
    df = None
    if DAEROOM_AVAILABLE:
        try:
            res = query_db(
                "SELECT datetime, season, holiday, workingday, weather, temp, atemp, "
                "humidity, windspeed, casual, registered, count "
                "FROM bike_sharing_demand ORDER BY datetime;",
                readonly=True,
                timeout=30,
            )
            rows = res.get("rows", [])
            if rows:
                df = pd.DataFrame(rows)
                source_label = "Daeroom Cloud PostgreSQL (db_bikesharingdemand)"
        except Exception:
            df = None

    if df is None:
        csv_path = "tugas_streamlit_bab1_bab2/data/bike_sharing.csv"
        if not os.path.exists(csv_path):
            csv_path = "data/bike_sharing.csv"
        if not os.path.exists(csv_path):
            csv_path = "data/train.csv"
        df = pd.read_csv(csv_path)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    return df, source_label

try:
    df_bike, DATA_SOURCE_LABEL = load_bike_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    df_bike = pd.DataFrame()
    DATA_SOURCE_LABEL = "Tidak Tersedia"

# Sidebar Navigasi Utama
st.sidebar.markdown("<div class='badge'>TUGAS MACHINE LEARNING</div>", unsafe_allow_html=True)
st.sidebar.title("Tutorial Streamlit")
st.sidebar.markdown("**Implementasi Modul Bab 1 & Bab 2**")
st.sidebar.caption("Kelompok 7 - Teknik Informatika UNPAM")

menu_options = [
    "1. Beranda & Pengenalan (Bab 1)",
    "2. Dasar Streamlit & Elemen Text (Bab 2.1 & 2.2)",
    "3. Menampilkan Data & Metrik KPI (Bab 2.3)",
    "4. Visualisasi Grafik & Peta (Bab 2.4)",
    "5. Form Input & Komponen Interaktif (Bab 2.5)",
    "6. Tampilan Media Multimedia (Bab 2.6)",
    "7. Layout, Columns & Containers (Bab 2.7)",
    "8. Control Flow & Form Batching (Bab 2.8)",
    "9. Showcase Terpadu: Simulator Bike Sharing"
]

selected_menu = st.sidebar.radio("Pilih Modul Pembelajaran:", menu_options)

st.sidebar.divider()
st.sidebar.caption(f"Sumber Data: {DATA_SOURCE_LABEL}")
st.sidebar.markdown("### Tim Penyusun (Kelompok 7)")
st.sidebar.markdown("""
- **Daerobi** (Ketua / Model & UI)
- **Hendro Tri Dewanto** (Data Pipeline)
- **Nasrudin Ali Albana** (Evaluasi & Laporan)
""")
st.sidebar.caption("Dosen Pengampu: Farizi Ilham S.Kom., M.Kom.")


# ==============================================================================
# MODUL 1: BERANDA & PENGENALAN (BAB 1)
# ==============================================================================
if selected_menu == "1. Beranda & Pengenalan (Bab 1)":
    st.markdown("<h1 class='main-title'>Bab 1: Pengenalan Framework Web Streamlit</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Tutorial Praktik Instalasi, Persiapan Virtual Environment, dan Pembuatan Aplikasi Web Pertama</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 1.1 Tentang Framework Streamlit
        Streamlit adalah framework Python open-source yang dirancang khusus untuk mempermudah ilmuwan data (Data Scientists) 
        dan pengembang kecerdasan buatan dalam membagikan model machine learning ke dalam antarmuka web interaktif yang elegan.
        
        **Keunggulan Utama Streamlit vs Framework Tradisional (Flask / Django):**
        1. **Murni Bahasa Python**: Tidak memerlukan penulisan file HTML, CSS, JavaScript, atau routing WSGI yang rumit.
        2. **Siklus Pengembangan Instan**: Waktu pembuatan aplikasi berubah dari berminggu-minggu menjadi hanya beberapa jam.
        3. **Reactivity Bawaan**: Setiap perubahan input pengguna secara otomatis memperbarui antarmuka pengguna secara mulus.
        4. **Integrasi Ekosistem Data Terpadu**: Kompatibel penuh dengan Pandas, NumPy, Scikit-Learn, Matplotlib, Plotly, dan Altair.
        """)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <div class='card-header'>Informasi Studi Kasus</div>
            <p><strong>Dataset:</strong> Bike Sharing Demand</p>
            <p><strong>Lokasi:</strong> Washington D.C.</p>
            <p><strong>Total Observasi:</strong> 10,886 baris</p>
            <p><strong>Variabel Target:</strong> count (total peminjaman)</p>
            <p><strong>Kelompok:</strong> Kelompok 7 (UNPAM)</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 1.2 Prosedur Instalasi & Setup Environment")
    
    st.markdown("""
    Berikut adalah tahapan teknis persiapan lingkungan kerja Python untuk menjalankan Streamlit sesuai standar laboratorium:
    """)

    step_tab1, step_tab2, step_tab3 = st.tabs(["1. Virtual Environment", "2. Instalasi Paket Pip", "3. Menjalankan Server"])

    with step_tab1:
        st.markdown("**Langkah 1: Membuat & Mengaktifkan Virtual Environment di Command Prompt / PowerShell**")
        st.code("""# 1. Buka Command Prompt atau PowerShell di folder proyek
cd "c:\\Users\\ADVAN\\Documents\\antigravity\\magical-faraday\\tugas_streamlit_bab1_bab2"

# 2. Buat virtual environment bernama 'env'
python -m venv env

# 3. Aktifkan virtual environment pada Windows
env\\Scripts\\activate
""", language="powershell")

    with step_tab2:
        st.markdown("**Langkah 2: Instalasi Pustaka Utama yang Dibutuhkan**")
        st.code("""# Memperbarui pip installer
python -m pip install --upgrade pip

# Menginstal Streamlit dan pustaka data science
pip install streamlit pandas numpy scikit-learn matplotlib plotly altair

# Memeriksa daftar paket yang telah berhasil terpasang
pip list
""", language="powershell")

    with step_tab3:
        st.markdown("**Langkah 3: Perintah Eksekusi Aplikasi Pertama**")
        st.code("""# Menjalankan berkas utama aplikasi Streamlit
streamlit run app.py

# Aplikasi akan otomatis terbuka pada peramban web:
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
""", language="powershell")

    st.divider()
    st.markdown("### 1.3 Kode Aplikasi Pertama (`main.py`)")
    st.code("""import streamlit as st

st.title("Aplikasi Pemantauan Bike Sharing Demand")
st.write("Selamat datang di sistem analitik operasional armada sepeda Kelompok 7!")
""", language="python")
    st.success("Bab 1 berhasil dipraktekkan secara tuntas. Silakan beralih ke sub-bab berikutnya pada panel sidebar kiri.")


# ==============================================================================
# MODUL 2: DASAR STREAMLIT & ELEMEN TEXT (BAB 2.1 & 2.2)
# ==============================================================================
elif selected_menu == "2. Dasar Streamlit & Elemen Text (Bab 2.1 & 2.2)":
    st.markdown("<h1 class='main-title'>Bab 2.1 & 2.2: Dasar Streamlit dan Elemen Teks</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Implementasi fungsi cetak serbaguna st.write() serta variasi tipografi teks dan ekspresi matematika</p>", unsafe_allow_html=True)

    tab_dasar, tab_text = st.tabs(["Praktik 2.1: Dasar st.write()", "Praktik 2.2: Elemen Teks & LaTeX"])

    with tab_dasar:
        st.markdown("### 2.1 Penerapan Multi-Tipe Argumen dengan `st.write()`")
        st.write("Fungsi `st.write()` adalah fungsi 'Swiss-Army Knife' di Streamlit yang otomatis mengenali format string, angka, data tabular, chart, maupun dictionary.")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("#### Input Teks & Numerik")
            st.write("Pesan Sistem: Status operasional bike sharing terpantau stabil.")
            st.write("Jumlah Total Observasi Historis:", 10886)
            st.write("Kombinasi String & Variabel Numerik: Rata-rata sewa per jam adalah", 191.57, "unit sepeda.")
        
        with col_w2:
            st.markdown("#### Input Dictionary Objek (JSON-like)")
            st.write({
                "cluster_region": "Washington D.C. Metropolitan",
                "total_active_stations": 42,
                "fleet_size": 2500,
                "primary_algorithm": "HistGradientBoostingRegressor",
                "validation_rmsle": 0.2581
            })
        
        st.markdown("#### Menampilkan DataFrame dengan `st.write()`")
        st.write(df_bike[['datetime', 'season', 'temp', 'humidity', 'count']].head(4))

    with tab_text:
        st.markdown("### 2.2 Elemen Tipografi Teks & Format Formula")
        
        st.title("Contoh st.title(): Judul Utama Halaman")
        st.header("Contoh st.header(): Header Bagian Dokumen")
        st.subheader("Contoh st.subheader(): Subheader Spesifik")
        st.caption("Contoh st.caption(): Keterangan tambahan atau catatan kaki dengan ukuran font kecil.")
        
        st.divider()
        st.markdown("""
        #### Contoh `st.markdown()` dengan Format Teks dan Warna
        Fitur lingkungan operasional:
        - :blue[Suhu Udara (temp)]: Mengukur suhu termal aktual dalam skala Celcius.
        - :green[Kelembaban (humidity)]: Tingkat kelembaban udara relatif.
        - :red[Kecepatan Angin (windspeed)]: Menentukan faktor resistensi hembusan angin bersepeda.
        - Formula Penjumlahan Rental: $\\text{count} = \\text{casual} + \\text{registered}$
        """)

        st.divider()
        st.markdown("#### Contoh `st.text()` (Teks Monospaced Polos)")
        st.text("LOG_BUFFER: 2026-11-10 16:00:00 UTC | Ingestion telemetri sensor sepeda: STATUS_OK")

        st.markdown("#### Contoh `st.code()` (Blok Kode Program Python)")
        st.code("""def extract_bike_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    return df
""", language="python")

        st.markdown("#### Contoh `st.latex()` (Formula Matematika Akademik)")
        st.latex(r"\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \ln(\hat{y}_i + 1) - \ln(y_i + 1) \right)^2}")
        st.caption("Fungsi kerugian RMSLE menjamin model tidak menghasilkan penalti eksponensial terhadap fluktuasi volume sewa yang besar.")


# ==============================================================================
# MODUL 3: MENAMPILKAN DATA & METRIK KPI (BAB 2.3)
# ==============================================================================
elif selected_menu == "3. Menampilkan Data & Metrik KPI (Bab 2.3)":
    st.markdown("<h1 class='main-title'>Bab 2.3: Menampilkan Data dan Metrik</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Implementasi st.dataframe, st.table, st.metric, dan st.json pada data Bike Sharing Demand</p>", unsafe_allow_html=True)

    st.markdown("### 1. Panel Metrik Kinerja Utama (`st.metric`)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    tot_count = int(df_bike['count'].sum())
    avg_temp = round(df_bike['temp'].mean(), 1)
    tot_reg = int(df_bike['registered'].sum())
    tot_cas = int(df_bike['casual'].sum())
    reg_pct = round((tot_reg / tot_count) * 100, 1)

    with m_col1:
        st.metric(label="Total Sewa Sepeda", value=f"{tot_count:,}", delta="+14.2% YoY")
    with m_col2:
        st.metric(label="Rata-rata Suhu", value=f"{avg_temp} °C", delta="+1.5 °C")
    with m_col3:
        st.metric(label="Proporsi Komuter", value=f"{reg_pct} %", delta="+3.1%")
    with m_col4:
        st.metric(label="Penyewa Rekreasi", value=f"{tot_cas:,}", delta="-1.8%")

    st.divider()
    st.markdown("### 2. Tabel Interaktif (`st.dataframe`) vs Tabel Statis (`st.table`)")
    
    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.markdown("#### Tabel Interaktif (`st.dataframe`)")
        st.caption("Dapat digulir, diurutkan, difilter, dan diubah lebar kolomnya oleh pengguna.")
        st.dataframe(df_bike[['datetime', 'season', 'weather', 'temp', 'humidity', 'windspeed', 'count']].head(8), use_container_width=True)

    with col_d2:
        st.markdown("#### Tabel Statis Ringkasan (`st.table`)")
        st.caption("Rata-rata nilai fitur berdasarkan 4 musim.")
        season_summary = df_bike.groupby('season')[['temp', 'humidity', 'windspeed', 'count']].mean().round(2)
        season_summary.index = ["1: Spring", "2: Summer", "3: Fall", "4: Winter"]
        st.table(season_summary)

    st.divider()
    st.markdown("### 3. Metadata Konfigurasi Stasiun Sepeda (`st.json`)")
    station_json = {
        "network_authority": "Capital Bikeshare D.C.",
        "station_id": "DC-ST-007",
        "station_name": "Smithsonian National Air and Space Museum",
        "coordinates": {"latitude": 38.8882, "longitude": -77.0199},
        "docks": {"total_capacity": 48, "active_ebike_chargers": 16},
        "iot_sensors": {
            "temperature_celsius": 22.4,
            "humidity_percent": 61.0,
            "signal_rssi_dbm": -68,
            "telemetry_health": "Optimal"
        }
    }
    st.json(station_json)


# ==============================================================================
# MODUL 4: VISUALISASI GRAFIK & PETA (BAB 2.4)
# ==============================================================================
elif selected_menu == "4. Visualisasi Grafik & Peta (Bab 2.4)":
    st.markdown("<h1 class='main-title'>Bab 2.4: Menampilkan Grafik dan Peta Geografis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Visualisasi garis, area, batang, scatter plot Altair/Plotly, dan peta stasiun sepeda D.C.</p>", unsafe_allow_html=True)

    tab_native, tab_advanced, tab_map = st.tabs(["1. Grafik Native Streamlit", "2. Altair, Vega-Lite & Plotly", "3. Peta Geografis (st.map)"])

    with tab_native:
        st.markdown("### Grafik Garis, Area, dan Batang Bawaan Streamlit")
        
        # Agregasi data per jam
        hourly_grp = df_bike.groupby('hour')[['count', 'registered', 'casual']].mean()

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### Grafik Garis Tren Sewa 24 Jam (`st.line_chart`)")
            st.caption("Terlihat lonjakan bimodal komuter pada jam kerja (08:00 dan 17:00).")
            st.line_chart(hourly_grp['count'])

        with g_col2:
            st.markdown("#### Grafik Area Komposisi Pengguna (`st.area_chart`)")
            st.caption("Porsi pengguna terdaftar (biru) mendominasi jam sibuk.")
            st.area_chart(hourly_grp[['registered', 'casual']])

        st.markdown("#### Grafik Batang Volume Antar Musim (`st.bar_chart`)")
        season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        df_season_chart = df_bike.copy()
        df_season_chart['season_label'] = df_season_chart['season'].map(season_names)
        season_tot = df_season_chart.groupby('season_label')['count'].sum().reindex(["Spring", "Summer", "Fall", "Winter"])
        st.bar_chart(season_tot)

    with tab_advanced:
        st.markdown("### Grafik Interaktif Altair, Vega-Lite & Plotly")
        sample_viz = df_bike.sample(250, random_state=42)

        c_alt, c_plotly = st.columns(2)
        with c_alt:
            st.markdown("#### Scatter Plot Interaktif Altair (`st.altair_chart`)")
            chart_alt = alt.Chart(sample_viz).mark_circle(size=70).encode(
                x=alt.X('temp:Q', title='Suhu Udara (°C)'),
                y=alt.Y('count:Q', title='Total Rental (count)'),
                color=alt.Color('weather:N', title='Cuaca', scale=alt.Scale(scheme='category10')),
                tooltip=['temp', 'humidity', 'windspeed', 'count']
            ).properties(height=340).interactive()
            st.altair_chart(chart_alt, use_container_width=True)

        with c_plotly:
            st.markdown("#### Scatter Interaktif Plotly Express (`st.plotly_chart`)")
            fig_p = px.scatter(
                sample_viz,
                x="atemp",
                y="count",
                color="season",
                size="humidity",
                hover_data=["windspeed", "casual", "registered"],
                labels={"atemp": "Suhu Terasa (°C)", "count": "Jumlah Sewa"},
                title="Plotly: Suhu Terasa vs Jumlah Rental Sepeda"
            )
            fig_p.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("#### Visualisasi Deklaratif Vega-Lite (`st.vega_lite_chart`)")
        vega_spec = {
            "mark": {"type": "circle", "opacity": 0.7},
            "encoding": {
                "x": {"field": "humidity", "type": "quantitative", "title": "Kelembaban (%)"},
                "y": {"field": "windspeed", "type": "quantitative", "title": "Kecepatan Angin (km/h)"},
                "color": {"field": "temp", "type": "quantitative", "title": "Suhu (°C)"},
                "size": {"field": "count", "type": "quantitative", "title": "Rental"}
            }
        }
        st.vega_lite_chart(sample_viz, vega_spec, use_container_width=True)

    with tab_map:
        st.markdown("### Peta Persebaran Stasiun Sepeda Capital Bikeshare (`st.map`)")
        st.caption("Peta interaktif berbasis koordinat latitude dan longitude stasiun sepeda di Washington D.C.")
        dc_stations = pd.DataFrame({
            'station_name': [
                'Lincoln Memorial', 'National Mall East', 'Dupont Circle Metro',
                'Georgetown Waterfront', 'Union Station Metro', 'Smithsonian Air & Space',
                'Capitol South Metro', 'Adams Morgan 18th St', 'Navy Yard Station', 'Foggy Bottom GWU',
                'Thomas Circle NW', 'Eastern Market Metro'
            ],
            'lat': [
                38.8893, 38.8897, 38.9096, 38.9030, 38.8978, 38.8882,
                38.8856, 38.9217, 38.8765, 38.9009, 38.9058, 38.8841
            ],
            'lon': [
                -77.0502, -77.0229, -77.0434, -77.0620, -77.0063, -77.0199,
                -77.0051, -77.0423, -77.0003, -77.0503, -77.0320, -76.9954
            ],
            'capacity': [45, 35, 40, 30, 60, 48, 38, 28, 44, 36, 32, 26]
        })
        st.map(dc_stations[['lat', 'lon']], zoom=12)
        st.dataframe(dc_stations, use_container_width=True)


# ==============================================================================
# MODUL 5: FORM INPUT & KOMPONEN INTERAKTIF (BAB 2.5)
# ==============================================================================
elif selected_menu == "5. Form Input & Komponen Interaktif (Bab 2.5)":
    st.markdown("<h1 class='main-title'>Bab 2.5: Menampilkan Input Form & Widget Kontrol</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Eksplorasi lengkap seluruh widget input Streamlit untuk parameter simulasi sepeda</p>", unsafe_allow_html=True)

    # 1. Button & Data Editor
    st.markdown("### 1. Tombol Aksi (`st.button`) & Editor Data Interaktif (`st.data_editor`)")
    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        st.markdown("#### Trigger Aksi")
        if st.button("Jalankan Audit Kapasitas Stasiun"):
            st.success("Semua stasiun terverifikasi dalam batas toleransi aman.")
        
        st.markdown("#### Unduh Konfigurasi (`st.download_button`)")
        sample_download = pd.DataFrame({
            "stasiun": ["Dupont Circle", "Union Station", "Lincoln Memorial"],
            "target_alokasi": [40, 60, 45],
            "buffer_ebike": [10, 15, 12]
        })
        st.download_button(
            label="Unduh CSV Konfigurasi",
            data=sample_download.to_csv(index=False).encode('utf-8'),
            file_name="konfigurasi_stasiun.csv",
            mime="text/csv"
        )

    with col_b2:
        st.markdown("#### Editor Data Tabular Dinamis (`st.data_editor`)")
        st.caption("Nilai tabel di bawah dapat diedit langsung oleh pengguna di browser:")
        editor_df = pd.DataFrame([
            {"Stasiun": "Lincoln Memorial", "Docks": 45, "Sepeda": 28, "Siap Pakai": True},
            {"Stasiun": "Dupont Circle", "Docks": 40, "Sepeda": 18, "Siap Pakai": True},
            {"Stasiun": "Union Station", "Docks": 60, "Sepeda": 54, "Siap Pakai": True},
            {"Stasiun": "Georgetown", "Docks": 30, "Sepeda": 6, "Siap Pakai": False}
        ])
        edited_result = st.data_editor(editor_df, num_rows="dynamic", use_container_width=True)

    st.divider()

    # 2. Checkbox, Radio, Selectbox, Multiselect
    st.markdown("### 2. Seleksi Opsi: Checkbox, Radio, Selectbox & Multiselect")
    col_o1, col_o2, col_o3 = st.columns(3)

    with col_o1:
        st.markdown("#### Checkbox & Radio")
        check_holiday = st.checkbox("Sertakan Hari Libur Nasional (holiday)", value=False)
        fleet_select = st.radio("Pilihan Armada:", ["Sepeda Standar", "E-Bike Listrik", "Kargo"])
        st.write(f"Libur: `{check_holiday}`, Armada: `{fleet_select}`")

    with col_o2:
        st.markdown("#### Selectbox (Dropdown)")
        weather_opt = st.selectbox(
            "Kondisi Cuaca Lingkungan:",
            ["1: Cerah / Sedikit Berawan", "2: Berkabut / Berawan Tebal", "3: Gerimis / Salju Ringan", "4: Hujan Badai Ekstrem"]
        )
        st.write(f"Pilihan Cuaca: `{weather_opt}`")

    with col_o3:
        st.markdown("#### Multiselect")
        season_multi = st.multiselect(
            "Musim Pengamatan:",
            ["Spring", "Summer", "Fall", "Winter"],
            default=["Summer", "Fall"]
        )
        st.write(f"Musim Terpilih: `{season_multi}`")

    st.divider()

    # 3. Slider & Select Slider
    st.markdown("### 3. Slider Numerik (`st.slider`) & Slider Pilihan (`st.select_slider`)")
    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        jam_slider = st.slider("Pilih Jam Operasional (0 - 23 WIB):", min_value=0, max_value=23, value=8)
        rentang_suhu = st.slider("Rentang Suhu Operasional (°C):", min_value=0.0, max_value=45.0, value=(15.0, 30.0))
        st.write(f"Jam: `{jam_slider}:00`, Rentang Suhu: `{rentang_suhu[0]}°C - {rentang_suhu[1]}°C`")

    with col_sl2:
        tingkat_urgensi = st.select_slider(
            "Level Urgensi Rebalancing Armada:",
            options=["Normal", "Siaga 1", "Siaga 2", "Darurat Rebalancing (Starvation Risk)"],
            value="Siaga 1"
        )
        st.write(f"Level Urgensi: `{tingkat_urgensi}`")

    st.divider()

    # 4. Inputs: Text, Number, Text Area, Date, Time, File, Camera, Color
    st.markdown("### 4. Input Teks, Angka, Tanggal, Jam, Berkas & Kamera")
    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        st.markdown("#### Input Teks & Angka")
        operator_nama = st.text_input("Nama Petugas Dispatcher:", "Daerobi (Kelompok 7)")
        truk_count = st.number_input("Jumlah Unit Truk:", min_value=1, max_value=20, value=4, step=1)
        st.write(f"Petugas: `{operator_nama}`, Truk: `{truk_count}`")

        catatan = st.text_area("Instruksi Khusus Lapangan:", "Fokuskan dropping sepeda di stasiun universitas.")

    with col_in2:
        st.markdown("#### Pemilih Tanggal & Waktu")
        tgl_dispatch = st.date_input("Tanggal Operasi:", datetime.date(2026, 11, 10))
        jam_dispatch = st.time_input("Waktu Mulai Truk:", datetime.time(6, 45))
        st.write(f"Jadwal: `{tgl_dispatch} pukul {jam_dispatch}`")

        warna_tema = st.color_picker("Pilih Warna Aksen UI:", "#0369a1")
        st.write(f"Warna: `{warna_tema}`")

    with col_in3:
        st.markdown("#### Berkas & Kamera")
        up_file = st.file_uploader("Unggah CSV Log Sensor:", type=["csv"])
        if up_file is not None:
            st.success(f"File `{up_file.name}` siap diproses.")

        cam_pic = st.camera_input("Ambil Foto Inspeksi Sepeda / QR Code:")
        if cam_pic:
            st.image(cam_pic, caption="Foto Inspeksi Teknis Terambil", use_container_width=True)


# ==============================================================================
# MODUL 6: TAMPILAN MEDIA MULTIMEDIA (BAB 2.6)
# ==============================================================================
elif selected_menu == "6. Tampilan Media Multimedia (Bab 2.6)":
    st.markdown("<h1 class='main-title'>Bab 2.6: Menampilkan Media Multimedia</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Implementasi pemutar gambar st.image, audio st.audio, dan video st.video pada Streamlit</p>", unsafe_allow_html=True)

    col_med1, col_med2 = st.columns(2)

    with col_med1:
        st.markdown("### 1. Menampilkan Gambar (`st.image`)")
        st.caption("Diagram Alur Metodologi CRISP-DM Bike Sharing Demand Kelompok 7:")
        img_crisp = "tugas_streamlit_bab1_bab2/assets/crisp_dm_pipeline.png"
        if os.path.exists(img_crisp):
            st.image(img_crisp, caption="Arsitektur Pipeline Machine Learning Bike Sharing", use_container_width=True)
        else:
            st.info("File gambar placeholder tersedia di direktori assets.")

    with col_med2:
        st.markdown("### 2. Menampilkan Pemutar Audio (`st.audio`)")
        st.caption("Audio alert / bel stasiun sepeda saat dock stasiun telah terisi penuh:")
        audio_chime = "tugas_streamlit_bab1_bab2/assets/station_chime.wav"
        if os.path.exists(audio_chime):
            st.audio(audio_chime, format="audio/wav")
        else:
            st.info("Berkas audio station_chime.wav siap diputar.")

        st.divider()
        st.markdown("### 3. Menampilkan Pemutar Video (`st.video`)")
        st.caption("Animasi simulasi pola penyewaan sepeda 24 jam (stream dari Daeroom MinIO S3):")
        s3_video_url = "https://s3.daeroom.my.id/bikesharingdemand-assets/media/bike_sharing_demo.mp4"
        st.video(s3_video_url)


# ==============================================================================
# MODUL 7: LAYOUT, COLUMNS & CONTAINERS (BAB 2.7)
# ==============================================================================
elif selected_menu == "7. Layout, Columns & Containers (Bab 2.7)":
    st.markdown("<h1 class='main-title'>Bab 2.7: Menampilkan Layout dan Container</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Penerapan sidebar, columns, tabs, expander, container, dan placeholder dinamis empty</p>", unsafe_allow_html=True)

    st.markdown("### 1. Multi-Kolom Responsif (`st.columns`)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kapasitas Rata-rata Dock", "42 unit", "+4 unit")
    c2.metric("Tingkat Utilitas Harian", "78.4 %", "+6.2 %")
    c3.metric("Waktu Perbaikan Rusak", "1.8 jam", "-0.4 jam")

    st.divider()

    st.markdown("### 2. Navigasi Multi-Tab (`st.tabs`)")
    t1, t2, t3 = st.tabs(["Ringkasan Data Cuaca", "Jadwal Rebalancing Truk", "Kamus Data"])

    with t1:
        st.write("Statistik agregat pengaruh cuaca terhadap total peminjaman:")
        weather_summary = df_bike.groupby('weather')[['temp', 'humidity', 'count']].mean().round(2)
        weather_summary.index = ["Cerah", "Mendung", "Hujan Ringan", "Cuaca Ekstrem"]
        st.dataframe(weather_summary, use_container_width=True)

    with t2:
        st.write("Jadwal distribusi truk rebalancing ke halte tujuan:")
        truk_df = pd.DataFrame({
            "No Truk": ["TRK-01", "TRK-02", "TRK-03", "TRK-04"],
            "Rute Stasiun": ["Dupont Circle", "Union Station", "Foggy Bottom", "Georgetown"],
            "Jumlah Sepeda Diangkut": [25, 30, 20, 15],
            "Jam Keberangkatan": ["06:30", "07:00", "15:30", "16:00"]
        })
        st.table(truk_df)

    with t3:
        st.markdown("""
        - **season**: 1=Spring, 2=Summer, 3=Fall, 4=Winter
        - **workingday**: 1=Hari kerja (Senin-Jumat), 0=Akhir pekan/libur
        - **weather**: Kategori kondisi cuaca skala 1 hingga 4
        - **temp / atemp**: Suhu udara aktual dan suhu terasa (°C)
        - **count**: Total peminjaman sepeda per jam
        """)

    st.divider()

    st.markdown("### 3. Wadah Buka-Tutup (`st.expander`)")
    with st.expander("Klik untuk Menampilkan Formula Matematis Evaluasi Model"):
        st.latex(r"\text{Loss Function: } \mathcal{L} = \frac{1}{2} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \|\mathbf{w}\|_2^2")
        st.write("Model regresi menerapkan regularisasi L2 guna memitigasi risiko overfitting data musiman.")

    st.divider()

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        st.markdown("### 4. Wadah Elemen (`st.container`)")
        with st.container():
            st.info("Kontainer ini memuat peringatan otomatis telemetri:")
            st.warning("Stasiun Dupont Circle mendekati batas minimum sepeda tersedia (tersisa 3 unit).")

    with col_box2:
        st.markdown("### 5. Wadah Dinamis Placeholder (`st.empty`)")
        st.caption("Mendemonstrasikan update konten waktu nyata tanpa me-refresh seluruh halaman:")
        empty_placeholder = st.empty()
        
        if st.button("Uji Simulasi Loading Telemetri Sensor"):
            for s in range(3, 0, -1):
                empty_placeholder.warning(f"Menyinkronkan sinyal sensor... {s} detik")
                time.sleep(1)
            empty_placeholder.success("Sinkronisasi Selesai! Semua 42 stasiun terhubung (Status 200 OK).")


# ==============================================================================
# MODUL 8: CONTROL FLOW & FORM BATCHING (BAB 2.8)
# ==============================================================================
elif selected_menu == "8. Control Flow & Form Batching (Bab 2.8)":
    st.markdown("<h1 class='main-title'>Bab 2.8: Menampilkan Control Flow & Form Batching</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Penerapan st.stop() untuk validasi, st.form() untuk batch execution, dan st.rerun()</p>", unsafe_allow_html=True)

    # 1. st.stop()
    st.markdown("### 1. Penghentian Alur Kerja dengan Validasi (`st.stop`)")
    server_status = st.selectbox(
        "Simulasikan Status Gateway Server API:",
        ["Online (200 OK)", "Gateway Timeout (504)", "Maintenance System (503)"]
    )

    if server_status == "Gateway Timeout (504)":
        st.error("Koneksi gagal: Gateway Timeout 504. Alur eksekusi aplikasi dihentikan oleh st.stop().")
        st.stop()
    elif server_status == "Maintenance System (503)":
        st.warning("Server sedang dalam jadwal pemeliharaan. Alur eksekusi dihentikan oleh st.stop().")
        st.stop()

    st.success("Server terverifikasi online. Silakan lanjutkan ke formulir estimasi batch di bawah ini.")

    st.divider()

    # 2. st.form dan st.form_submit_button
    st.markdown("### 2. Formulir Batching (`st.form` & `st.form_submit_button`)")
    st.markdown("""
    Secara default, Streamlit mengeksekusi ulang seluruh skrip setiap kali sebuah widget diubah nilainya. 
    Dengan menggunakan `st.form()`, seluruh input dikemas dalam satu transaksi, dan aplikasi hanya memproses 
    kalkulasi saat tombol **Submit Form** ditekan.
    """)

    with st.form("form_batch_demand"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            inp_jam = st.slider("Jam Operasional (0 - 23):", 0, 23, 8)
            inp_suhu = st.number_input("Suhu Udara (°C):", min_value=-5.0, max_value=45.0, value=24.0, step=0.5)
            inp_cuaca = st.selectbox("Kondisi Cuaca:", ["1: Cerah", "2: Mendung", "3: Hujan Ringan", "4: Badai"])
        
        with f_col2:
            inp_kelembaban = st.slider("Kelembaban Udara (%):", 0, 100, 65)
            inp_angin = st.number_input("Kecepatan Angin (km/jam):", min_value=0.0, max_value=60.0, value=11.5, step=0.5)
            inp_hari_kerja = st.checkbox("Hari Kerja Efektif (Senin - Jumat)", value=True)

        btn_submit = st.form_submit_button("Hitung Estimasi Peminjaman Armada Sepeda")

    if btn_submit:
        # Perhitungan estimasi demand
        base_demand = 85
        if inp_jam in [8, 9, 17, 18]:
            factor_jam = 3.2 if inp_hari_kerja else 1.3
        elif 11 <= inp_jam <= 15:
            factor_jam = 1.6 if not inp_hari_kerja else 1.1
        else:
            factor_jam = 0.35

        factor_temp = max(0.4, inp_suhu / 25.0)
        factor_wthr = 1.0 if inp_cuaca.startswith("1") else (0.8 if inp_cuaca.startswith("2") else 0.4)
        factor_hmd = max(0.4, 1.0 - (inp_kelembaban / 180.0))

        pred_total = int(base_demand * factor_jam * factor_temp * factor_wthr * factor_hmd)
        pred_reg = int(pred_total * (0.86 if inp_hari_kerja else 0.42))
        pred_cas = pred_total - pred_reg

        st.success(f"Estimasi Kebutuhan Armada: **{pred_total} sepeda / jam**")
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Total Estimasi Sewa", f"{pred_total} unit")
        res2.metric("Estimasi Komuter (Registered)", f"{pred_reg} unit")
        res3.metric("Estimasi Rekreasi (Casual)", f"{pred_cas} unit")

    st.divider()

    # 3. st.rerun
    st.markdown("### 3. Eksekusi Ulang Aplikasi Programatik (`st.rerun`)")
    st.caption("Fungsi st.rerun() digunakan untuk me-reset session state dan memuat ulang data.")
    if st.button("Segarkan Ulang Aplikasi (st.rerun)"):
        st.info("Me-refresh aplikasi...")
        st.rerun()


# ==============================================================================
# MODUL 9: SHOWCASE TERPADU SIMULATOR BIKE SHARING
# ==============================================================================
elif selected_menu == "9. Showcase Terpadu: Simulator Bike Sharing":
    st.markdown("<h1 class='main-title'>Showcase Terpadu: Dashboard Rebalancing Bike Sharing</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Integrasi seluruh materi Bab 1 & Bab 2 ke dalam Sistem Informasi Rebalancing Cerdas Kelompok 7</p>", unsafe_allow_html=True)

    # Filter Atas
    st.markdown("### Filter Waktu & Kondisi Operasional")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_hour = st.slider("Jam Simulasi (WIB):", 0, 23, 8)
    with f2:
        sel_temp = st.slider("Suhu Lingkungan (°C):", 5.0, 40.0, 26.0)
    with f3:
        sel_day_type = st.radio("Tipe Hari:", ["Hari Kerja (Komuter)", "Akhir Pekan (Rekreasi)"])
    with f4:
        sel_weather = st.selectbox("Cuaca:", ["Cerah", "Mendung", "Hujan Gerimis"])

    is_work = (sel_day_type == "Hari Kerja (Komuter)")

    # Perhitungan Demand Cerdas
    base_calc = 100
    rush_mult = 3.4 if (sel_hour in [8, 9, 17, 18] and is_work) else (1.4 if not is_work and 12 <= sel_hour <= 16 else 0.5)
    w_mult = 1.0 if sel_weather == "Cerah" else (0.75 if sel_weather == "Mendung" else 0.4)
    t_mult = max(0.5, sel_temp / 24.0)

    est_demand = int(base_calc * rush_mult * w_mult * t_mult)
    est_reg = int(est_demand * (0.88 if is_work else 0.45))
    est_cas = est_demand - est_reg
    trucks_needed = max(1, int(np.ceil(est_demand / 75)))

    # Metrik KPI Rebalancing
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Prediksi Total Demand", f"{est_demand} unit/jam", f"{'+45%' if rush_mult > 2 else '-12%'}")
    kpi2.metric("Kebutuhan Komuter", f"{est_reg} unit", f"{int(est_reg/est_demand*100)}%")
    kpi3.metric("Kebutuhan Rekreasi", f"{est_cas} unit", f"{int(est_cas/est_demand*100)}%")
    kpi4.metric("Truk Rebalancing Wajib", f"{trucks_needed} Truk", "Status Siap")

    st.divider()

    # Visualisasi Peta dan Distribusi
    col_map, col_chart = st.columns([1, 1])

    with col_map:
        st.markdown("#### Peta Status Stasiun Sepeda Terkini")
        station_sim = pd.DataFrame({
            'station_name': ['Lincoln Memorial', 'Dupont Circle Metro', 'Union Station', 'Georgetown', 'Smithsonian', 'Foggy Bottom GWU'],
            'lat': [38.8893, 38.9096, 38.8978, 38.9030, 38.8882, 38.9009],
            'lon': [-77.0502, -77.0434, -77.0063, -77.0620, -77.0199, -77.0503],
            'status': ['Cukup', 'Kritis (Perlu Truk)', 'Penuh (Perlu Ambil)', 'Cukup', 'Cukup', 'Kritis (Perlu Truk)']
        })
        st.map(station_sim[['lat', 'lon']], zoom=12)
        st.dataframe(station_sim[['station_name', 'status']], use_container_width=True)

    with col_chart:
        st.markdown("#### Simulasi Kurva Permintaan Sepeda 24 Jam")
        hours_range = np.arange(24)
        sim_curve = []
        for h in hours_range:
            mult = 3.4 if (h in [8, 9, 17, 18] and is_work) else (1.4 if not is_work and 12 <= h <= 16 else 0.4)
            sim_curve.append(int(base_calc * mult * w_mult * t_mult))
        
        sim_df = pd.DataFrame({"Jam": hours_range, "Prediksi Rental": sim_curve}).set_index("Jam")
        st.line_chart(sim_df)

    st.success("Showcase Terpadu ini menggabungkan seluruh elemen Bab 1 dan Bab 2 secara komprehensif, berbasis dataset Bike Sharing Demand Kelompok 7.")

    st.divider()
    st.markdown("#### Penyimpanan & Riwayat Simulasi (Daeroom Cloud PostgreSQL)")
    col_save, col_note = st.columns([1, 2])
    with col_note:
        sim_note = st.text_input("Catatan Operasional (Opsional):", placeholder="Misal: Shift pagi cuaca cerah sekitar stasiun metro")
    with col_save:
        st.write("")
        st.write("")
        if st.button("Simpan Simulasi ke Daeroom Cloud", type="primary", use_container_width=True):
            if DAEROOM_AVAILABLE:
                try:
                    query_db(
                        "INSERT INTO prediction_logs "
                        "(jam, suhu, tipe_hari, cuaca, pred_total, pred_komuter, pred_rekreasi, truk_dibutuhkan, catatan) "
                        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);",
                        params=[
                            int(sel_hour),
                            float(sel_temp),
                            str(sel_day_type),
                            str(sel_weather),
                            int(est_demand),
                            int(est_reg),
                            int(est_cas),
                            int(trucks_needed),
                            sim_note or "Simulasi rutin",
                        ],
                    )
                    st.toast("Simulasi berhasil disimpan ke PostgreSQL Daeroom Cloud.")
                except Exception as err:
                    st.error(f"Gagal menyimpan ke Daeroom Cloud: {err}")
            else:
                st.warning("Koneksi Daeroom Cloud belum aktif.")

    if DAEROOM_AVAILABLE:
        try:
            logs_res = query_db(
                "SELECT id, created_at, jam, suhu, tipe_hari, cuaca, pred_total, truk_dibutuhkan, catatan "
                "FROM prediction_logs ORDER BY id DESC LIMIT 10;",
                readonly=True,
            )
            logs_rows = logs_res.get("rows", [])
            if logs_rows:
                st.markdown("**10 Riwayat Simulasi Terakhir di Database:**")
                st.dataframe(pd.DataFrame(logs_rows), use_container_width=True)
            else:
                st.info("Belum ada riwayat simulasi tersimpan di database. Klik tombol simpan di atas untuk mencoba.")
        except Exception as err:
            st.warning(f"Tidak dapat membaca tabel prediction_logs: {err}")
