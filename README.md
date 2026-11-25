# BikeDemand AI — Analisis & Prediksi Bike Sharing Demand

Aplikasi web berbasis **Streamlit** untuk analisis data, pemodelan Machine Learning 3 paradigma, dan simulasi prediksi kebutuhan armada sepeda perkotaan (**Capital Bikeshare, Washington D.C.**).

Proyek ini disusun oleh **Kelompok 7** sebagai bagian dari mata kuliah **Machine Learning Terapan (Semester 5)**, Program Studi Teknik Informatika, **Universitas Pamulang (UNPAM)**.

---

## Anggota Kelompok 7

| No | Nama | NIM | Peran |
| :--- | :--- | :--- | :--- |
| 1 | **Daerobi** | `241011401811` | Arsitektur Aplikasi, Pemodelan & UI |
| 2 | **Hendro Tri Dewanto** | `241011401696` | Data Pipeline & Preprocessing |
| 3 | **Nasrudin Ali Albana** | `241011402086` | Evaluasi Model & Dokumentasi Laporan |

---

## Cakupan Fitur & Struktur Halaman (`st.navigation`)

Aplikasi menggunakan arsitektur multipage terstruktur ke dalam 3 kelompok navigasi:

### 1. Pengantar
- **Beranda:** Ringkasan eksekutif, KPI utama dataset (`10.886` record), alur sistem, diagram CRISP-DM, dan video simulasi 24 jam.
- **Tentang Dataset:** Latar belakang masalah, formula evaluasi RMSLE, cuplikan tabel interaktif (`st.dataframe`), statistik deskriptif (`st.table`), dan metrik utama.
- **Eksplorasi Data (EDA):** Audit kualitas data (0 missing value), pola bimodal jam kerja vs akhir pekan, heatmap korelasi fitur, dan rangkuman **10 Masalah Kritis EDA**.

### 2. Pengolahan & Pemodelan
- **Preprocessing & PCA:** Ekstraksi fitur waktu (`hour`, `dayofweek`, `month`, `year`), transformasi target `np.log1p`, serta analisis reduksi dimensi menggunakan **Principal Component Analysis (PCA)** lengkap dengan Scree Plot interaktif.
- **Teknik Optimasi:** Pembuktian komparatif performa **Sebelum Optimasi (Baseline)** vs **Sesudah Optimasi (Tuned + `log1p`)** beserta perbandingan parameter hyperparameter.
- **Modelling & Evaluasi:** Benchmark langsung pada 20% data uji untuk 3 paradigma algoritma:
  1. **Paradigma Linear / Parametrik:** `Ridge Regression` (dengan `StandardScaler`)
  2. **Paradigma Berbasis Jarak / Geometri:** `K-Nearest Neighbors (k=7)`
  3. **Paradigma Berbasis Pohon / Ensemble:** `HistGradientBoostingRegressor` (Model Utama)
  Dilengkapi grafik perbandingan R2/RMSE/MAE, **Scatter Plot Aktual vs Prediksi**, dan **Distribusi Sisa Error (Residual)**.

### 3. Penerapan
- **Simulasi & Prediksi:** Simulator interaktif untuk menguji kondisi jam, musim, cuaca, suhu, dan kelembaban secara *real-time* terhadap ketiga model, lengkap dengan estimasi kebutuhan truk rebalancing dan penyimpanan log ke database PostgreSQL.
- **Prediksi Massal (CSV):** Pemrosesan batch dari file CSV ramalan cuaca (tersedia template CSV 24 jam) serta tombol unduh hasil prediksi.
- **Peta & Panel Armada:** Visualisasi grafik tren, peta lokasi stasiun Washington D.C., panel input petugas, dan `st.data_editor`.
- **Riwayat DB & Kesimpulan:** Menampilkan riwayat simulasi tersimpan dari PostgreSQL beserta kesimpulan dan saran pengembangan.

---

## Struktur Direktori Repository

```text
.
├── app.py                                           # Aplikasi utama Streamlit (st.navigation)
├── daeroom_client.py                                # Helper koneksi REST API PostgreSQL & S3
├── requirements.txt                                 # Daftar dependensi Python
├── Dockerfile                                       # Konfigurasi container siap deploy
├── .dockerignore                                    # Filter build Docker agar tetap ringan
├── .env.example                                     # Template environment variables
├── .streamlit/config.toml                           # Konfigurasi tema UI aplikasi
├── data/
│   └── bike_sharing.csv                             # Dataset utama (10.886 baris)
├── assets/                                          # Aset visual diagram & audio
├── latihan_individual/                              # 9 skrip modular latihan Bab 1 & Bab 2
├── screenshots/                                     # Tangkapan layar dokumentasi aplikasi
├── Laporan_Tutorial_Streamlit_Bab1_Bab2_Kelompok7.docx
└── PANDUAN_TUTORIAL_STREAMLIT_BAB1_BAB2.html
```

---

## Menjalankan Aplikasi di Lokal

1. Clone repository dan masuk ke direktori proyek:
   ```bash
   git clone https://github.com/daerobi-devs/web-1-bike-deman.git
   cd web-1-bike-deman
   ```

2. Buat virtual environment dan install dependensi:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Salin `.env.example` menjadi `.env.local` (opsional, untuk mengaktifkan penyimpanan riwayat ke PostgreSQL; jika tidak diisi, aplikasi otomatis memakai dataset lokal `data/bike_sharing.csv`):
   ```bash
   copy .env.example .env.local
   ```

4. Jalankan server Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## Menjalankan dengan Docker

```bash
docker build -t web-1-bike-deman .
docker run -p 8501:8501 --env-file .env.local web-1-bike-deman
```

Aplikasi akan berjalan pada `http://localhost:8501` dengan endpoint healthcheck `/_stcore/health`.
