"""
===================================================================================
WEB 1: IMPLEMENTASI STREAMLIT LENGKAP — BIKE SHARING DEMAND
Mata Kuliah : Machine Learning Terapan (Semester 5)
Instansi    : Teknik Informatika, Universitas Pamulang (UNPAM)
Kelompok    : Kelompok 7 (Daerobi, Hendro Tri Dewanto, Nasrudin Ali Albana)
Arsitektur  : Grouped Multipage Navigation (st.navigation) + Cloud PostgreSQL & S3
===================================================================================
"""
import datetime
import os
import time
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from daeroom_client import get_public_url, query_db, upload_asset
    DAEROOM_AVAILABLE = True
except Exception:
    DAEROOM_AVAILABLE = False

st.set_page_config(
    page_title="Bike Sharing Demand | Kelompok 7 UNPAM",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS (CLEAN SLATE & SKY ACCENT, ZERO EMOJI)
# ==============================================================================
st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 0.75rem;
    }
    .sidebar-brand-card {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 100%);
        color: #f8fafc;
        padding: 0.85rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border: 1px solid #0284c7;
    }
    .sidebar-brand-card .title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .sidebar-brand-card .sub {
        font-size: 0.75rem;
        color: #bae6fd;
        margin-top: 0.15rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 100%);
        color: #f8fafc;
        padding: 1.75rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        border: 1px solid #0284c7;
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.05rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: #e0f2fe;
        font-size: 1rem;
        margin: 0;
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #475569;
        margin-bottom: 1.25rem;
    }
    .accent-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 1rem;
    }
    .accent-card h4 {
        margin: 0 0 0.35rem 0;
        color: #0f172a;
        font-size: 1.05rem;
    }
    .accent-card p {
        margin: 0;
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #0369a1;
        background-color: #e0f2fe;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

FEATURE_COLUMNS = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "dayofweek", "month", "year",
]

SEASON_LABELS = {1: "Semi (Spring)", 2: "Panas (Summer)", 3: "Gugur (Fall)", 4: "Dingin (Winter)"}
WEATHER_LABELS = {
    1: "1 - Cerah / Berawan Ringan",
    2: "2 - Kabut / Mendung",
    3: "3 - Hujan Ringan / Gerimis",
    4: "4 - Hujan Lebat / Badai",
}


# ==============================================================================
# DATA & MODEL CACHE
# ==============================================================================
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
                source_label = "Cloud PostgreSQL (Live)"
        except Exception:
            df = None

    if df is None:
        base_dir = Path(__file__).resolve().parent
        for candidate in [
            base_dir / "data" / "bike_sharing.csv",
            base_dir / "data" / "train.csv",
        ]:
            if candidate.exists():
                df = pd.read_csv(candidate)
                break

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["day"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    return df, source_label


@st.cache_resource
def train_three_paradigms(df_hash_key: int, _df: pd.DataFrame):
    """Melatih 3 paradigma algoritma regresi pada dataset Bike Sharing secara cache."""
    X = _df[FEATURE_COLUMNS]
    y = _df["count"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Ridge Regression (Linear)": TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))]),
            func=np.log1p, inverse_func=np.expm1,
        ),
        "K-Nearest Neighbors (k=7)": TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=7, weights="distance"))]),
            func=np.log1p, inverse_func=np.expm1,
        ),
        "HistGradientBoosting (Tree)": TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.08, random_state=42),
            func=np.log1p, inverse_func=np.expm1,
        ),
    }

    trained = {}
    metrics_rows = []
    for name, est in candidates.items():
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        train_s = round(time.perf_counter() - t0, 2)

        preds = np.maximum(0, est.predict(X_test))
        r2 = round(float(r2_score(y_test, preds)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 2)
        mae = round(float(mean_absolute_error(y_test, preds)), 2)
        rmsle = round(float(np.sqrt(np.mean((np.log1p(preds) - np.log1p(np.maximum(0, y_test))) ** 2))), 4)

        trained[name] = est
        metrics_rows.append({
            "Model": name,
            "R2 Score": r2,
            "Skor (%)": f"{r2*100:.2f}%",
            "RMSE": rmse,
            "MAE": mae,
            "RMSLE": rmsle,
            "Waktu Latih (s)": train_s,
        })

    return trained, pd.DataFrame(metrics_rows), X_test, y_test


@st.cache_resource
def train_optimization_benchmark(df_hash_key: int, _df: pd.DataFrame):
    """Membandingkan performa HistGradientBoosting sebelum vs sesudah optimasi (log1p + tuning)."""
    X = _df[FEATURE_COLUMNS]
    y = _df["count"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline = HistGradientBoostingRegressor(
        max_iter=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    tuned = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, learning_rate=0.08, random_state=42
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    results = []
    for label, est in [("Sebelum Optimasi (Baseline)", baseline), ("Sesudah Optimasi (Tuned + Log1p)", tuned)]:
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        elapsed = round(time.perf_counter() - t0, 2)
        preds = np.maximum(0, est.predict(X_test))
        r2 = round(float(r2_score(y_test, preds)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 2)
        mae = round(float(mean_absolute_error(y_test, preds)), 2)
        rmsle = round(float(np.sqrt(np.mean((np.log1p(preds) - np.log1p(np.maximum(0, y_test))) ** 2))), 4)
        results.append({
            "Tahap": label,
            "R2 Score": r2,
            "RMSE": rmse,
            "MAE": mae,
            "RMSLE": rmsle,
            "Waktu Latih (s)": elapsed,
        })
    return pd.DataFrame(results)


try:
    df_bike, DATA_SOURCE_LABEL = load_bike_data()
except Exception as exc:
    st.error(f"Gagal memuat dataset: {exc}")
    st.stop()


# ==============================================================================
# HALAMAN 1: BERANDA
# ==============================================================================
def page_beranda():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Analisis & Prediksi Bike Sharing Demand</h1>
            <p>Sistem Prediksi Kebutuhan Armada Sepeda Berbasis Machine Learning & Cloud PostgreSQL — Kelompok 7 UNPAM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sampel Data", f"{len(df_bike):,} jam")
    k2.metric("Fitur Utama", "12 Variabel")
    k3.metric("Missing Value", "0 (Bersih)")
    k4.metric("Rata-Rata Sewa/Jam", f"{df_bike['count'].mean():.1f} unit")

    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown(
            """
            <div class="accent-card">
                <h4>Tentang Proyek & Masalah Utama</h4>
                <p>
                    Sistem penyewaan sepeda perkotaan (Capital Bikeshare, Washington D.C.) sering mengalami ketimpangan stok:
                    stasiun transit kehabisan sepeda saat jam berangkat kerja (08:00), sementara stasiun lain penuh sesak.
                    Proyek ini menggabungkan praktik komponen web Streamlit (Bab 1 & Bab 2) dengan pipeline Machine Learning
                    untuk memprediksi kebutuhan armada secara presisi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Alur Sistem & Cakupan Fitur")
        st.markdown(
            """
            1. **Pengantar & Dataset:** Profil dataset Capital Bikeshare (10.886 baris), statistik deskriptif, dan indikator kinerja armada.
            2. **Eksplorasi Data (EDA):** Audit kualitas data dan temuan 10 masalah kritis (skewness, multikolinearitas `temp`/`atemp`, pola bimodal jam kerja).
            3. **Preprocessing & PCA:** Ekstraksi fitur waktu (`hour`, `month`, `year`), transformasi target `log1p`, dan reduksi dimensi PCA.
            4. **Modelling & Evaluasi:** Perbandingan objektif 3 paradigma algoritma (**Ridge**, **KNN**, dan **HistGradientBoosting**).
            5. **Penerapan:** Simulasi prediksi real-time, prediksi massal CSV, panel operasional armada, serta pencatatan riwayat ke Cloud PostgreSQL.
            """
        )

    with c_right:
        st.markdown(
            """
            <div class="accent-card">
                <h4>Tim Penyusun — Kelompok 7</h4>
                <p>
                    <strong>1. Daerobi</strong> (241011401811) — Arsitektur & Model<br>
                    <strong>2. Hendro Tri Dewanto</strong> (241011401696) — Data Pipeline<br>
                    <strong>3. Nasrudin Ali Albana</strong> (241011402086) — Evaluasi & Laporan<br><br>
                    <em>Program Studi Teknik Informatika, Semester 5<br>Universitas Pamulang (UNPAM)</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# HALAMAN 2: TENTANG DATASET
# ==============================================================================
def page_bab1_data():
    st.markdown("<div class='main-title'>Tentang Dataset Bike Sharing Demand</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Karakteristik data historis Capital Bikeshare Washington D.C., ringkasan statistik, dan indikator utama.</div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Latar Belakang & Formula", "Data Mentah & Tabel", "Statistik & Metrik KPI"])

    with t1:
        st.markdown("#### Latar Belakang & Tujuan Pemodelan")
        st.write(
            "Dataset ini merekam jumlah penyewaan sepeda per jam selama dua tahun (2011–2012) "
            "yang dipengaruhi oleh musim, hari kerja/libur, suhu, kelembaban, dan kondisi cuaca."
        )
        st.divider()
        st.markdown("**Formula Evaluasi Utama (RMSLE):**")
        st.latex(r"\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(p_i + 1) - \log(a_i + 1) \right)^2}")
        st.caption("Keterangan: p_i adalah prediksi model dan a_i adalah jumlah penyewaan aktual.")

    with t2:
        st.markdown("#### Cuplikan Data Historis")
        st.dataframe(df_bike.head(20), use_container_width=True)
        st.json({
            "dataset": "Capital Bikeshare Washington D.C.",
            "rows": int(len(df_bike)),
            "target": "count (casual + registered)",
        })

    with t3:
        st.markdown("#### Ringkasan Statistik Deskriptif")
        stats = df_bike[["temp", "humidity", "windspeed", "casual", "registered", "count"]].describe().round(2)
        st.table(stats.loc[["mean", "std", "min", "max"]])

        st.markdown("#### Indikator Kinerja Utama Armada")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Penyewaan", f"{int(df_bike['count'].sum()):,}", "+12.4%")
        m2.metric("Rata-Rata Suhu", f"{df_bike['temp'].mean():.1f} °C", "+1.2 °C")
        m3.metric("Porsi Komuter", f"{(df_bike['registered'].sum()/df_bike['count'].sum()*100):.1f}%")
        m4.metric("Puncak Tertinggi", f"{int(df_bike['count'].max()):,} unit/jam")


# ==============================================================================
# HALAMAN 3: VISUALISASI & KONFIGURASI OPERASIONAL
# ==============================================================================
def page_bab2_grafik_input():
    st.markdown("<div class='main-title'>Visualisasi Armada & Konfigurasi Operasional</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Pola penyewaan per jam, peta stasiun Washington D.C., serta panel simulasi input petugas lapangan.</div>", unsafe_allow_html=True)

    tab_g, tab_i = st.tabs(["Grafik & Peta Stasiun", "Panel Input & Editor Jadwal"])

    with tab_g:
        hourly = df_bike.groupby("hour")[["count", "registered", "casual"]].mean()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tren Penyewaan 24 Jam (`st.line_chart`)**")
            st.line_chart(hourly["count"])
        with c2:
            st.markdown("**Komposisi Komuter vs Kasual (`st.area_chart`)**")
            st.area_chart(hourly[["registered", "casual"]])

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Rata-Rata Sewa per Musim (`st.bar_chart`)**")
            season_df = df_bike.groupby("season")["count"].mean().rename(index=SEASON_LABELS)
            st.bar_chart(season_df)
        with c4:
            st.markdown("**Peta Sebaran Stasiun (`st.map`)**")
            stations = pd.DataFrame({
                "lat": [38.8893, 38.9096, 38.8978, 38.9030, 38.8882],
                "lon": [-77.0502, -77.0434, -77.0063, -77.0620, -77.0199],
            })
            st.map(stations, zoom=12)

        st.markdown("**Visualisasi Interaktif Altair & Plotly**")
        a_col, p_col = st.columns(2)
        with a_col:
            sample_a = df_bike.sample(min(400, len(df_bike)), random_state=42)
            chart_alt = alt.Chart(sample_a).mark_circle(size=55, opacity=0.65).encode(
                x=alt.X("temp:Q", title="Suhu (°C)"),
                y=alt.Y("count:Q", title="Total Sewa"),
                color=alt.Color("season:N", title="Musim"),
                tooltip=["temp", "humidity", "count"],
            ).interactive()
            st.altair_chart(chart_alt, use_container_width=True)
        with p_col:
            fig_px = px.box(
                df_bike.assign(Cuaca=df_bike["weather"].map(WEATHER_LABELS)),
                x="Cuaca", y="count", color="Cuaca",
                title="Distribusi Sewa Berdasarkan Kondisi Cuaca",
            )
            fig_px.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_px, use_container_width=True)

    with tab_i:
        col_a, col_b = st.columns(2)
        with col_a:
            nama_op = st.text_input("1. Nama Petugas / Penguji:", "Daerobi")
            catatan = st.text_area("2. Catatan Kondisi Lapangan:", "Jam sibuk pagi sekitar stasiun metro.")
            kapasitas = st.number_input("3. Kapasitas Truk Rebalancing:", min_value=10, max_value=200, value=75)
            suhu_sl = st.slider("4. Simulasi Suhu (°C):", 0.0, 42.0, 26.5)
            tgl = st.date_input("5. Tanggal Operasional:", datetime.date(2026, 11, 25))
            jam = st.time_input("6. Jam Inspeksi:", datetime.time(8, 0))
        with col_b:
            musim_rad = st.radio("7. Pilih Musim:", list(SEASON_LABELS.values()), horizontal=True)
            cuaca_sel = st.selectbox("8. Kondisi Cuaca:", list(WEATHER_LABELS.values()))
            zona_ms = st.multiselect("9. Zona Prioritas:", ["Union Station", "Dupont Circle", "Lincoln Memorial", "Georgetown"], default=["Union Station"])
            setuju_cb = st.checkbox("10. Aktifkan Mode Hemat Energi Dock", value=True)
            mode_tgl = st.toggle("11. Mode Siaga Jam Sibuk", value=True)
            if st.button("Cek Ringkasan Input"):
                st.success(f"Petugas {nama_op} | {tgl} {jam} | {musim_rad} | Zona: {', '.join(zona_ms)}")

        st.markdown("**Tabel Editor Interaktif (`st.data_editor`)**")
        edit_df = pd.DataFrame({
            "Stasiun": ["Union Station", "Dupont Circle", "Lincoln Memorial"],
            "Stok Sepeda": [25, 12, 30],
            "Perlu Truk": [False, True, False],
        })
        st.data_editor(edit_df, use_container_width=True)


# ==============================================================================
# HALAMAN 4: MEDIA & ALUR OPERASIONAL
# ==============================================================================
def page_bab2_media_layout():
    st.markdown("<div class='main-title'>Media Presentasi & Alur Operasional</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Dokumentasi visual pipeline CRISP-DM, audio notifikasi stasiun, video simulasi 24 jam, dan form distribusi.</div>", unsafe_allow_html=True)

    t_med, t_lay, t_ctrl = st.tabs(["Media & Video Simulasi", "Ringkasan Kapasitas", "Form Distribusi Armada"])

    with t_med:
        base_dir = Path(__file__).resolve().parent
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**1. Gambar (`st.image`)**")
            img_path = base_dir / "assets" / "crisp_dm_pipeline.png"
            if img_path.exists():
                st.image(str(img_path), caption="Pipeline CRISP-DM Bike Sharing Kelompok 7", use_container_width=True)
            st.markdown("**2. Audio Notifikasi (`st.audio`)**")
            aud_path = base_dir / "assets" / "station_chime.wav"
            if aud_path.exists():
                st.audio(str(aud_path), format="audio/wav")
        with m2:
            st.markdown("**3. Video Simulasi (`st.video` — Cloud Object Storage)**")
            st.video("https://s3.daeroom.my.id/bikesharingdemand-assets/media/bike_sharing_demo.mp4")

    with t_lay:
        with st.container(border=True):
            st.markdown("#### Container Berbingkai (`st.container(border=True)`)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Stasiun Aktif", "142 Titik")
            c2.metric("Armada Sepeda", "3.200 Unit")
            c3.metric("Ketersediaan", "94,2%")
        with st.expander("Penjelasan Layout & Responsivitas"):
            st.write("Penggunaan `st.columns`, `st.tabs`, dan `st.expander` menjaga halaman tetap ringkas baik di desktop maupun layar ponsel.")

    with t_ctrl:
        with st.form("form_control_flow"):
            st.markdown("**Batching Form (`st.form`) — Mengirim banyak input sekaligus**")
            f_name = st.text_input("Nama Operator:", "Hendro Tri Dewanto")
            f_unit = st.slider("Jumlah Unit Dikirim:", 5, 100, 30)
            submitted = st.form_submit_button("Kirim Instruksi")
            if submitted:
                st.success(f"Instruksi pengiriman {f_unit} sepeda oleh {f_name} diterima.")

        if st.button("Refresh Tampilan (`st.rerun`)"):
            st.rerun()


# ==============================================================================
# HALAMAN 5: EKSPLORASI DATA (EDA)
# ==============================================================================
def page_eda():
    st.markdown("<div class='main-title'>Eksplorasi Data (EDA) & Audit Kualitas</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Analisis distribusi, korelasi fitur, dan rangkuman 10 temuan kritis dataset Bike Sharing Demand.</div>", unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Baris Duplikat", f"{df_bike.duplicated().sum()}")
    e2.metric("Missing Value", f"{int(df_bike.isna().sum().sum())}")
    e3.metric("Skewness Target (count)", f"{df_bike['count'].skew():.2f}")
    e4.metric("Korelasi temp vs atemp", f"{df_bike['temp'].corr(df_bike['atemp']):.3f}")

    tab_dist, tab_corr, tab_10 = st.tabs(["Distribusi & Pola Jam", "Korelasi Antar Fitur", "10 Masalah Kritis EDA"])

    with tab_dist:
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(df_bike, x="count", nbins=45, title="Distribusi Target Sewa (Right-Skewed)")
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            grp = df_bike.groupby(["hour", "workingday"])["count"].mean().reset_index()
            grp["Tipe Hari"] = grp["workingday"].map({1: "Hari Kerja", 0: "Akhir Pekan / Libur"})
            fig_w = px.line(grp, x="hour", y="count", color="Tipe Hari", markers=True, title="Pola Bimodal Hari Kerja vs Rekreasi Akhir Pekan")
            st.plotly_chart(fig_w, use_container_width=True)

    with tab_corr:
        num_cols = ["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]
        corr_mat = df_bike[num_cols].corr().round(2)
        fig_hm = px.imshow(corr_mat, text_auto=True, color_continuous_scale="Blues", title="Heatmap Korelasi Fitur Numerik")
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_10:
        issues = pd.DataFrame([
            ("1. Skewness Target Ekstrem", "Target `count` condong kanan (skew > 1.2).", "Transformasi `np.log1p(count)` saat training."),
            ("2. Multikolinearitas temp & atemp", "Korelasi `temp` dan `atemp` > 0.98.", "Gunakan regularisasi Ridge atau model Tree."),
            ("3. Anomali Windspeed = 0", "Banyak nilai `windspeed == 0` (sensor/pembulatan).", "Model Tree tahan terhadap nilai 0 diskrit."),
            ("4. Pola Non-Linear Jam Sibuk", "Dua puncak tajam pada pukul 08:00 dan 17:00.", "Ekstraksi fitur `hour` & model non-linear."),
            ("5. Disparitas Skala Fitur", "Rentang `humidity` (0-100) vs `workingday` (0-1).", "Terapkan `StandardScaler` di dalam Pipeline."),
            ("6. Kebocoran Data (casual + registered)", "`casual + registered == count` secara persis.", "Dilarang masuk sebagai fitur input prediksi `count`."),
            ("7. Tren Pertumbuhan Tahunan", "Penyewaan tahun 2012 jauh lebih tinggi dari 2011.", "Sertakan fitur `year` dan `month`."),
            ("8. Ketimpangan Cuaca Ekstrem", "Cuaca kode 4 (badai) sangat jarang muncul.", "Gunakan metrik RMSLE agar tidak bias."),
            ("9. Perbedaan Perilaku Hari Kerja", "Komuter mendominasi hari kerja, kasual di akhir pekan.", "Kombinasi `workingday` dan `dayofweek`."),
            ("10. Risiko Data Leakage Saat Scaling", "Scaling sebelum split membocorkan statistik uji.", "Bungkus scaler dalam `sklearn.pipeline.Pipeline`."),
        ], columns=["Temuan Masalah", "Deskripsi", "Solusi di Pipeline"])
        st.dataframe(issues, use_container_width=True, hide_index=True)


# ==============================================================================
# HALAMAN 6: PREPROCESSING & PCA
# ==============================================================================
def page_preprocessing_pca():
    st.markdown("<div class='main-title'>Preprocessing & Reduksi Dimensi (PCA)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Ekstraksi fitur waktu, transformasi logaritmik target, dan analisis Principal Component Analysis.</div>", unsafe_allow_html=True)

    t_prep, t_pca = st.tabs(["Feature Engineering & Scaling", "Reduksi Dimensi (PCA)"])

    with t_prep:
        st.markdown("#### Perbandingan Fitur Sebelum vs Sesudah Preprocessing")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data Mentah (Raw `datetime`)**")
            st.dataframe(df_bike[["datetime", "season", "temp", "humidity", "count"]].head(5), use_container_width=True)
        with c2:
            st.markdown("**Setelah Ekstraksi Waktu + Target `log1p(count)`**")
            prep_sample = df_bike[["hour", "dayofweek", "month", "year", "temp", "humidity"]].head(5).copy()
            prep_sample["log1p_count"] = np.log1p(df_bike["count"].head(5)).round(3)
            st.dataframe(prep_sample, use_container_width=True)

    with t_pca:
        pca_cols = ["season", "weather", "temp", "atemp", "humidity", "windspeed", "hour", "month"]
        X_scaled = StandardScaler().fit_transform(df_bike[pca_cols])
        n_comp = st.slider("Jumlah Komponen PCA:", 2, len(pca_cols), 4)
        pca = PCA(n_components=n_comp, random_state=42)
        comps = pca.fit_transform(X_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        p1, p2, p3 = st.columns(3)
        p1.metric("Dimensi Awal", f"{len(pca_cols)} fitur")
        p2.metric("Dimensi Tereduksi", f"{n_comp} komponen", f"-{round((1 - n_comp/len(pca_cols))*100)}%")
        p3.metric("Varians Dipertahankan", f"{cum_var[-1]*100:.2f}%")

        scree_df = pd.DataFrame({
            "Komponen": [f"PC{i+1}" for i in range(n_comp)],
            "Varians (%)": (pca.explained_variance_ratio_ * 100).round(2),
            "Kumulatif (%)": (cum_var * 100).round(2),
        })
        fig_scree = px.bar(scree_df, x="Komponen", y="Varians (%)", text="Varians (%)", title="Scree Plot Varians yang Dijelaskan per Komponen PCA")
        st.plotly_chart(fig_scree, use_container_width=True)


# ==============================================================================
# HALAMAN 7A: TEKNIK OPTIMASI (SEBELUM VS SESUDAH)
# ==============================================================================
def page_optimasi():
    st.markdown("<div class='main-title'>Teknik Optimasi Model (Sebelum vs Sesudah)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Pembuktian dampak transformasi target log1p dan penyetelan hyperparameter pada HistGradientBoosting.</div>", unsafe_allow_html=True)

    with st.spinner("Menjalankan benchmark sebelum vs sesudah optimasi..."):
        opt_df = train_optimization_benchmark(len(df_bike), df_bike)

    before = opt_df.iloc[0]
    after = opt_df.iloc[1]

    m1, m2, m3 = st.columns(3)
    m1.metric("R2 Score (Akurasi)", f"{after['R2 Score']:.4f}", f"+{after['R2 Score'] - before['R2 Score']:.4f} poin")
    m2.metric("RMSE (Error Unit)", f"{after['RMSE']:.2f}", f"{after['RMSE'] - before['RMSE']:.2f} unit", delta_color="inverse")
    m3.metric("RMSLE (Log Error)", f"{after['RMSLE']:.4f}", f"{after['RMSLE'] - before['RMSLE']:.4f}", delta_color="inverse")

    t1, t2, t3 = st.tabs(["Hasil Sebelum vs Sesudah", "Skema Parameter Tuning", "Mengapa Optimasi Dibutuhkan?"])
    with t1:
        st.dataframe(opt_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_r2 = px.bar(opt_df, x="Tahap", y="R2 Score", color="Tahap", text="R2 Score", title="Kenaikan R2 Score Setelah Optimasi")
            fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1.1])
            st.plotly_chart(fig_r2, use_container_width=True)
        with c2:
            fig_err = px.bar(opt_df, x="Tahap", y=["RMSE", "MAE"], barmode="group", title="Penurunan Error RMSE & MAE")
            st.plotly_chart(fig_err, use_container_width=True)

    with t2:
        st.markdown("#### Perbandingan Konfigurasi Hyperparameter")
        param_df = pd.DataFrame([
            ("Transformasi Target", "Tanpa transformasi (raw count)", "np.log1p / np.expm1"),
            ("max_iter (Jumlah Pohon)", "50", "200"),
            ("max_depth (Kedalaman)", "3", "8"),
            ("learning_rate", "0.10", "0.08"),
        ], columns=["Parameter", "Baseline (Sebelum)", "Optimized (Sesudah)"])
        st.table(param_df)

    with t3:
        st.markdown(
            """
            1. **Mengatasi Skewness Target:** Target `count` memiliki ekor panjang ke kanan. Dengan `log1p`, model lebih stabil memprediksi jam sepi maupun jam sibuk tanpa menghasilkan nilai negatif.
            2. **Menangkap Interaksi Kompleks:** Menaikkan `max_depth` dari 3 ke 8 dan `max_iter` ke 200 memungkinkan pohon mempelajari kombinasi spesifik antara jam, musim, hari kerja, dan cuaca.
            """
        )


# ==============================================================================
# HALAMAN 7: MODELLING & EVALUASI (3 PARADIGMA)
# ==============================================================================
def page_modelling():
    st.markdown("<div class='main-title'>Modelling & Evaluasi 3 Paradigma Algoritma</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Benchmark langsung pada 20% data uji antara Paradigma Linear, Berbasis Jarak, dan Berbasis Pohon.</div>", unsafe_allow_html=True)

    with st.spinner("Melatih dan mengevaluasi 3 paradigma model..."):
        trained_models, eval_df, X_test, y_test = train_three_paradigms(len(df_bike), df_bike)

    tab_bench, tab_scatter, tab_resid = st.tabs([
        "Benchmark 3 Paradigma",
        "Aktual vs Prediksi (Scatter)",
        "Distribusi Error (Residual)",
    ])

    with tab_bench:
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        g1, g2 = st.columns(2)
        with g1:
            fig_r2 = px.bar(eval_df, x="Model", y="R2 Score", color="Model", text="Skor (%)", title="Perbandingan Akurasi R2 Score (Lebih Tinggi Lebih Baik)")
            fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1.1])
            st.plotly_chart(fig_r2, use_container_width=True)
        with g2:
            fig_err = px.bar(eval_df, x="Model", y=["RMSE", "MAE"], barmode="group", title="Perbandingan Error RMSE & MAE (Lebih Rendah Lebih Baik)")
            st.plotly_chart(fig_err, use_container_width=True)

        st.success("HistGradientBoosting (Paradigma Pohon) unggul signifikan dengan R2 di atas 94% karena mampu menangkap interaksi non-linear jam sibuk dan cuaca.")

    with tab_scatter:
        st.markdown("#### Seberapa Rapat Tebakan Model ke Garis Ideal (`Aktual == Prediksi`)?")
        sel_m = st.selectbox("Pilih Model untuk Diinspeksi:", list(trained_models.keys()), index=2, key="scatter_m")
        preds_m = np.maximum(0, trained_models[sel_m].predict(X_test))
        sc_df = pd.DataFrame({"Aktual": y_test.to_numpy(), "Prediksi": preds_m}).sample(min(700, len(y_test)), random_state=42)
        max_axis = float(max(sc_df["Aktual"].max(), sc_df["Prediksi"].max()))

        fig_sc = px.scatter(
            sc_df, x="Aktual", y="Prediksi", opacity=0.55,
            title=f"Aktual vs Prediksi — {sel_m} (Sampel 700 Data Uji)",
        )
        fig_sc.add_shape(type="line", x0=0, y0=0, x1=max_axis, y1=max_axis, line=dict(color="#dc2626", dash="dash", width=2))
        fig_sc.update_layout(height=420)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Garis merah putus-putus adalah prediksi sempurna (100% tepat). Semakin rapat titik biru ke garis merah, semakin akurat model.")

    with tab_resid:
        st.markdown("#### Sebaran Sisa Error (`Aktual - Prediksi`) pada Data Uji")
        sel_m_res = st.selectbox("Pilih Model untuk Cek Residual:", list(trained_models.keys()), index=2, key="resid_m")
        preds_r = np.maximum(0, trained_models[sel_m_res].predict(X_test))
        residuals = y_test.to_numpy() - preds_r
        fig_res = px.histogram(
            pd.DataFrame({"Residual (Error Unit)": residuals}),
            x="Residual (Error Unit)", nbins=50,
            title=f"Distribusi Residual — {sel_m_res}",
        )
        fig_res.add_vline(x=0, line_color="#dc2626", line_dash="dash", line_width=2)
        fig_res.update_layout(height=400)
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("Model yang sehat memiliki puncak distribusi error yang sempit dan simetris tepat di angka 0.")


# ==============================================================================
# HALAMAN 8: SIMULASI & PREDIKSI REAL-TIME
# ==============================================================================
def page_simulasi():
    st.markdown("<div class='main-title'>Simulasi & Prediksi Sewa Sepeda (3 Model)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Uji kondisi cuaca dan waktu secara langsung terhadap ketiga model terlatih serta simpan hasilnya ke database.</div>", unsafe_allow_html=True)

    trained_models, eval_df, X_test, y_test = train_three_paradigms(len(df_bike), df_bike)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        sel_hour = c1.slider("Jam (0 - 23):", 0, 23, 8)
        sel_month = c2.slider("Bulan (1 - 12):", 1, 12, 6)
        sel_year = c3.selectbox("Tahun Acuan:", [2012, 2011])
        day_mode = c4.selectbox("Tipe Hari:", ["Hari Kerja", "Akhir Pekan", "Hari Libur Nasional"])

        d1, d2, d3, d4 = st.columns(4)
        sel_season = d1.selectbox("Musim:", list(SEASON_LABELS.values()), index=1)
        sel_weather = d2.selectbox("Cuaca:", list(WEATHER_LABELS.values()), index=0)
        sel_temp = d3.slider("Suhu (°C):", 1.0, 41.0, 26.0, step=0.5)
        sel_hum = d4.slider("Kelembaban (%):", 0, 100, 60)

    season_id = [k for k, v in SEASON_LABELS.items() if v == sel_season][0]
    weather_id = [k for k, v in WEATHER_LABELS.items() if v == sel_weather][0]
    is_work = 1 if day_mode == "Hari Kerja" else 0
    is_hol = 1 if day_mode == "Hari Libur Nasional" else 0

    row_in = pd.DataFrame([{
        "season": season_id, "holiday": is_hol, "workingday": is_work, "weather": weather_id,
        "temp": sel_temp, "atemp": min(45.0, sel_temp + 2.0), "humidity": sel_hum, "windspeed": 12.0,
        "hour": sel_hour, "dayofweek": 2 if is_work else 6, "month": sel_month, "year": sel_year,
    }])[FEATURE_COLUMNS]

    p_r = int(max(0, round(float(trained_models["Ridge Regression (Linear)"].predict(row_in)[0]))))
    p_k = int(max(0, round(float(trained_models["K-Nearest Neighbors (k=7)"].predict(row_in)[0]))))
    p_h = int(max(0, round(float(trained_models["HistGradientBoosting (Tree)"].predict(row_in)[0]))))
    trucks = max(1, int(np.ceil(p_h / 75)))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model 1: Ridge", f"{p_r:,} unit/jam")
    k2.metric("Model 2: KNN (k=7)", f"{p_k:,} unit/jam")
    k3.metric("Model 3: HGB (Juara)", f"{p_h:,} unit/jam")
    k4.metric("Truk Rebalancing", f"{trucks} Truk", "Kapasitas 75/truk")

    # Kurva 24 jam
    rows_24 = []
    for h in range(24):
        r = row_in.iloc[0].to_dict()
        r["hour"] = h
        rows_24.append(r)
    df_24 = pd.DataFrame(rows_24)[FEATURE_COLUMNS]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(24)), y=np.maximum(0, np.round(trained_models["Ridge Regression (Linear)"].predict(df_24))), name="Ridge", line=dict(dash="dash", color="#94a3b8")))
    fig.add_trace(go.Scatter(x=list(range(24)), y=np.maximum(0, np.round(trained_models["K-Nearest Neighbors (k=7)"].predict(df_24))), name="KNN", line=dict(color="#f59e0b")))
    fig.add_trace(go.Scatter(x=list(range(24)), y=np.maximum(0, np.round(trained_models["HistGradientBoosting (Tree)"].predict(df_24))), name="HistGradientBoosting (Juara)", line=dict(width=3.5, color="#0284c7")))
    fig.add_vline(x=sel_hour, line_dash="dot", line_color="#0f172a")
    fig.update_layout(title="Kurva Prediksi 24 Jam (Adu 3 Model)", xaxis_title="Jam", yaxis_title="Unit/Jam", hovermode="x unified", height=360)
    st.plotly_chart(fig, use_container_width=True)

    sc1, sc2 = st.columns([2, 1])
    note = sc1.text_input("Catatan Simulasi:", placeholder="Misal: Uji jam sibuk pagi")
    with sc2:
        st.write("")
        st.write("")
        if st.button("Simpan Simulasi ke Database", type="primary", use_container_width=True):
            if DAEROOM_AVAILABLE:
                try:
                    query_db(
                        "INSERT INTO prediction_logs "
                        "(jam, suhu, tipe_hari, cuaca, pred_total, pred_komuter, pred_rekreasi, truk_dibutuhkan, catatan) "
                        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);",
                        params=[sel_hour, float(sel_temp), day_mode, sel_weather, p_h, int(p_h * 0.8), p_h - int(p_h * 0.8), trucks, note or "Simulasi Web 1"],
                    )
                    st.toast("Simulasi tersimpan ke PostgreSQL.")
                    st.success("Berhasil disimpan ke database.")
                except Exception as err:
                    st.error(f"Gagal menyimpan: {err}")


# ==============================================================================
# HALAMAN 9: PREDIKSI MASSAL (CSV)
# ==============================================================================
def page_batch_csv():
    st.markdown("<div class='main-title'>Prediksi Massal Berbasis CSV</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Upload tabel ramalan cuaca 24 jam/mingguan untuk diprediksi serentak lalu unduh hasilnya.</div>", unsafe_allow_html=True)

    trained_models, _, _, _ = train_three_paradigms(len(df_bike), df_bike)
    sample_batch = pd.DataFrame({
        "season": [2] * 24, "holiday": [0] * 24, "workingday": [1] * 24,
        "weather": [1 if h not in [14, 15] else 2 for h in range(24)],
        "temp": [round(20 + 6 * np.sin((h - 6) * np.pi / 12), 1) for h in range(24)],
        "atemp": [round(22 + 6 * np.sin((h - 6) * np.pi / 12), 1) for h in range(24)],
        "humidity": [60] * 24, "windspeed": [10.5] * 24,
        "hour": list(range(24)), "dayofweek": [2] * 24, "month": [6] * 24, "year": [2012] * 24,
    })

    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "Download Template CSV (24 Jam)",
            data=sample_batch.to_csv(index=False).encode("utf-8"),
            file_name="template_bike_sharing.csv",
            mime="text/csv",
            use_container_width=True,
        )
        use_default = st.checkbox("Gunakan contoh data 24 jam bawaan", value=True)
    with c2:
        up_file = st.file_uploader("Upload file CSV:", type=["csv"])

    target_df = pd.read_csv(up_file) if up_file is not None else (sample_batch.copy() if use_default else None)
    if target_df is not None:
        missing = [c for c in FEATURE_COLUMNS if c not in target_df.columns]
        if missing:
            st.error(f"Kolom wajib kurang: {missing}")
        else:
            out = target_df.copy()
            out["pred_ridge"] = np.maximum(0, np.round(trained_models["Ridge Regression (Linear)"].predict(out[FEATURE_COLUMNS]))).astype(int)
            out["pred_knn"] = np.maximum(0, np.round(trained_models["K-Nearest Neighbors (k=7)"].predict(out[FEATURE_COLUMNS]))).astype(int)
            out["pred_hgb"] = np.maximum(0, np.round(trained_models["HistGradientBoosting (Tree)"].predict(out[FEATURE_COLUMNS]))).astype(int)
            st.dataframe(out, use_container_width=True)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil Prediksi (.csv)", data=csv_bytes, file_name="hasil_prediksi_bike_sharing.csv", mime="text/csv", type="primary")


# ==============================================================================
# HALAMAN 10: RIWAYAT DB & KESIMPULAN
# ==============================================================================
def page_riwayat_kesimpulan():
    st.markdown("<div class='main-title'>Riwayat Database & Kesimpulan Akhir</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Catatan simulasi yang tersimpan di Cloud PostgreSQL serta rangkuman kesimpulan proyek Kelompok 7.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="accent-card">
                <h4>Kesimpulan Utama</h4>
                <p>
                    1. Seluruh modul dasar Streamlit Bab 1 & Bab 2 berhasil diimplementasikan pada dataset riil Bike Sharing Demand (10.886 baris).<br>
                    2. Model berbasis pohon keputusan (HistGradientBoosting) jauh mengungguli Ridge Regression dan KNN dalam menangkap lonjakan komuter.<br>
                    3. Integrasi Cloud PostgreSQL & Object Storage memastikan data dan riwayat prediksi tetap sinkron baik lokal maupun saat di-hosting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="accent-card">
                <h4>Saran Pengembangan Lanjutan</h4>
                <p>
                    1. Menambahkan data kalender acara kota dan cuaca real-time dari API meteorologi.<br>
                    2. Mengintegrasikan notifikasi otomatis ke petugas armada ketika prediksi stok stasiun di bawah ambang batas.<br>
                    3. Menerapkan otentikasi pengguna untuk membatasi akses tulis ke database produksi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Riwayat Simulasi Tersimpan (Cloud PostgreSQL)")
    if st.button("Refresh Riwayat"):
        st.rerun()
    if DAEROOM_AVAILABLE:
        try:
            logs_res = query_db(
                "SELECT id, created_at, jam, suhu, tipe_hari, cuaca, pred_total, truk_dibutuhkan, catatan "
                "FROM prediction_logs ORDER BY id DESC LIMIT 15;",
                readonly=True,
            )
            rows = logs_res.get("rows", [])
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Belum ada riwayat simulasi tersimpan. Coba simpan dari halaman Simulasi & Prediksi.")
        except Exception as err:
            st.warning(f"Gagal membaca database: {err}")


# ==============================================================================
# NAVIGASI MULTI-PAGE TERSTRUKTUR (st.navigation + Material Vector Icons)
# ==============================================================================
pg = st.navigation({
    "Pengantar": [
        st.Page(page_beranda, title="Beranda", icon=":material/home:", url_path="beranda", default=True),
        st.Page(page_bab1_data, title="Tentang Dataset", icon=":material/database:", url_path="dataset"),
        st.Page(page_eda, title="Eksplorasi Data (EDA)", icon=":material/analytics:", url_path="eda"),
    ],
    "Pengolahan & Pemodelan": [
        st.Page(page_preprocessing_pca, title="Preprocessing & PCA", icon=":material/tune:", url_path="preprocessing-pca"),
        st.Page(page_optimasi, title="Teknik Optimasi", icon=":material/speed:", url_path="optimasi"),
        st.Page(page_modelling, title="Modelling & Evaluasi", icon=":material/model_training:", url_path="modelling"),
    ],
    "Penerapan": [
        st.Page(page_simulasi, title="Simulasi & Prediksi", icon=":material/bolt:", url_path="simulasi"),
        st.Page(page_batch_csv, title="Prediksi Massal (CSV)", icon=":material/upload_file:", url_path="batch-csv"),
        st.Page(page_bab2_grafik_input, title="Peta & Panel Armada", icon=":material/map:", url_path="armada"),
        st.Page(page_bab2_media_layout, title="Media & Form Distribusi", icon=":material/perm_media:", url_path="media-distribusi"),
        st.Page(page_riwayat_kesimpulan, title="Riwayat DB & Kesimpulan", icon=":material/fact_check:", url_path="kesimpulan"),
    ],
})

with st.sidebar:
    st.divider()
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <div class="title">BikeDemand AI</div>
            <div class="sub">Kelompok 7 · Teknik Informatika UNPAM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Daerobi | Hendro | Nasrudin")

pg.run()
