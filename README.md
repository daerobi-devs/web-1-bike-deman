# web-1-bike-deman

Aplikasi Web Streamlit Implementasi Modul Bab 1 & Bab 2 — Studi Kasus **Bike Sharing Demand (Kelompok 7 UNPAM)**.

## Fitur Utama
- Implementasi lengkap materi Bab 1 (Pengenalan & Aplikasi Pertama) hingga Bab 2 (`st.write`, elemen teks, tabel & KPI, grafik & peta, form widget, multimedia, layout container, dan control flow).
- Terintegrasi dengan **Daeroom Cloud PostgreSQL** (`db_bikesharingdemand`) dan **MinIO S3 Storage** (`bikesharingdemand-assets`).
- Dilengkapi `Dockerfile` siap deploy (Coolify / Docker / Streamlit Community Cloud).

## Menjalankan di Lokal
```bash
cp .env.example .env.local
pip install -r requirements.txt
streamlit run app.py
```

## Menjalankan dengan Docker
```bash
docker build -t web-1-bike-deman .
docker run -p 8501:8501 --env-file .env.local web-1-bike-deman
```
