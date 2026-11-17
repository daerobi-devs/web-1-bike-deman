"""
Praktik Bab 2.5: Menampilkan Input Form & Widget Streamlit
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st
import pandas as pd
import datetime

st.title("Praktik 2.5: Eksplorasi Widget Input & Form Bike Sharing")

# 1. st.button
st.subheader("1. Tombol Aksi (st.button)")
if st.button("Hitung Status Kesiapan Armada Sepeda"):
    st.success("Armada 100% siap operasi. Truk rebalancing terjadwal pukul 06:00 dan 15:00.")

# 2. st.data_editor (Modern replacement for st.experimental_data_editor)
st.subheader("2. Editor Data Interaktif (st.data_editor)")
st.caption("Pengguna dapat mengedit konfigurasi stasiun dan kuota sepeda langsung pada tabel:")
df_editor_sample = pd.DataFrame([
    {"station": "Lincoln Memorial", "docks": 45, "bikes": 28, "is_active": True},
    {"station": "Dupont Circle", "docks": 40, "bikes": 15, "is_active": True},
    {"station": "Union Station", "docks": 60, "bikes": 52, "is_active": True},
    {"station": "Georgetown Waterfront", "docks": 30, "bikes": 8, "is_active": False}
])
edited_df = st.data_editor(df_editor_sample, num_rows="dynamic", use_container_width=True)

# 3. st.download_button
st.subheader("3. Tombol Unduh Data (st.download_button)")
csv_data = edited_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Unduh Konfigurasi Stasiun (CSV)",
    data=csv_data,
    file_name="konfigurasi_stasiun_bike_sharing.csv",
    mime="text/csv"
)

# 4. st.checkbox, st.radio, st.selectbox
st.subheader("4. Filter Kategori: Checkbox, Radio, dan Selectbox")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    filter_workingday = st.checkbox("Hanya Tampilkan Hari Kerja", value=True)
    st.write(f"Filter Hari Kerja: **{filter_workingday}**")

with col_f2:
    fleet_type = st.radio("Pilih Tipe Armada:", ["Sepeda Standar", "E-Bike Listrik", "Kargo Sepeda"])
    st.write(f"Armada Terpilih: **{fleet_type}**")

with col_f3:
    weather_choice = st.selectbox(
        "Kondisi Cuaca Saat Ini:",
        ["1: Cerah / Berawan Tipis", "2: Berkabut / Mendung", "3: Gerimis / Salju Ringan", "4: Hujan Badai"]
    )
    st.write(f"Cuaca: **{weather_choice}**")

# 5. st.multiselect
st.subheader("5. Multi Seleksi Musim (st.multiselect)")
selected_seasons = st.multiselect(
    "Pilih Musim Pengamatan Operasional:",
    options=["Spring (Musim Semi)", "Summer (Musim Panas)", "Fall (Musim Gugur)", "Winter (Musim Dingin)"],
    default=["Summer (Musim Panas)", "Fall (Musim Gugur)"]
)
st.write(f"Musim Terpilih: {selected_seasons}")

# 6. st.slider & st.select_slider
st.subheader("6. Slider Numerik dan Slider Pilihan")
col_s1, col_s2 = st.columns(2)

with col_s1:
    operational_hour = st.slider("Pilih Jam Operasional (0 - 23 WIB):", min_value=0, max_value=23, value=8)
    target_temp = st.slider("Simulasi Rentang Suhu (°C):", min_value=0.0, max_value=45.0, value=(18.0, 28.0))
    st.write(f"Jam Dipilih: **{operational_hour}:00**, Rentang Suhu: **{target_temp[0]}°C s/d {target_temp[1]}°C**")

with col_s2:
    demand_level = st.select_slider(
        "Tingkat Antisipasi Lonjakan Sewa:",
        options=["Sangat Rendah", "Normal", "Tinggi", "Puncak Komuter (Rush Hour)", "Kritis (Rebalancing Urgent)"],
        value="Normal"
    )
    st.write(f"Status Lonjakan: **{demand_level}**")

# 7. st.text_input, st.number_input, st.text_area
st.subheader("7. Input Teks, Angka, dan Area Teks")
col_i1, col_i2 = st.columns(2)

with col_i1:
    dispatcher_name = st.text_input("Nama Pengawas Lapangan / Operator:", "Daerobi (Kelompok 7)")
    allocated_trucks = st.number_input("Jumlah Truk Rebalancing yang Dialokasikan:", min_value=1, max_value=20, value=3, step=1)
    st.write(f"Operator: **{dispatcher_name}** | Truk: **{allocated_trucks} unit**")

with col_i2:
    dispatch_notes = st.text_area(
        "Catatan Instruksi Pemindahan Sepeda:",
        "Prioritaskan pengiriman sepeda ke stasiun Dupont Circle dan Foggy Bottom sebelum jam 07:30 pagi."
    )
    st.write(f"Panjang Catatan: **{len(dispatch_notes)} karakter**")

# 8. st.date_input & st.time_input
st.subheader("8. Pemilih Tanggal dan Waktu (st.date_input, st.time_input)")
col_d1, col_d2 = st.columns(2)

with col_d1:
    schedule_date = st.date_input("Tanggal Operasional Dispatch:", datetime.date(2026, 11, 10))
    st.write(f"Tanggal: **{schedule_date}**")

with col_d2:
    schedule_time = st.time_input("Waktu Mulai Patroli Rebalancing:", datetime.time(7, 30))
    st.write(f"Waktu: **{schedule_time}**")

# 9. st.file_uploader
st.subheader("9. Unggah Berkas Log Sensor Sepeda (st.file_uploader)")
uploaded_file = st.file_uploader("Pilih Berkas CSV Log Sensor Stasiun:", type=["csv"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.success(f"Berkas berhasil diunggah! Ukuran: {uploaded_df.shape[0]} baris, {uploaded_df.shape[1]} kolom.")
    st.dataframe(uploaded_df.head(3))

# 10. st.camera_input
st.subheader("10. Input Kamera Inspeksi Fisik Sepeda (st.camera_input)")
st.caption("Digunakan teknisi lapangan untuk memindai QR code stasiun atau mengecek kondisi fisik sepeda.")
camera_photo = st.camera_input("Ambil Foto Kondisi Sepeda / Dock:")
if camera_photo is not None:
    st.image(camera_photo, caption="Hasil Jepretan Inspeksi Teknis Sepeda", use_container_width=True)

# 11. st.color_picker
st.subheader("11. Pemilih Warna Tema Dashboard (st.color_picker)")
brand_color = st.color_picker("Pilih Warna Utama Indikator Armada:", "#0284c7")
st.write(f"Kode Warna Terpilih: `{brand_color}`")
