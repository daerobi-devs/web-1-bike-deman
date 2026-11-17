"""
Praktik Bab 2.8: Kontrol Alur Kerja (Control Flow & Forms)
Konteks Data: Bike Sharing Demand (Kelompok 7)
"""
import streamlit as st

st.title("Praktik 2.8: Kontrol Alur Kerja (Control Flow) Streamlit")

# 1. st.stop: Validasi Input dan Penghentian Eksekusi
st.subheader("1. Penghentian Eksekusi Alur Kerja (st.stop)")
input_sensor_status = st.selectbox(
    "Status Koneksi Server Telemetri Stasiun:",
    ["Online (Normal)", "Maintenance Mode", "Offline / Terputus"]
)

if input_sensor_status == "Offline / Terputus":
    st.error("Koneksi sensor terputus! Eksekusi dihentikan oleh st.stop().")
    st.stop()
elif input_sensor_status == "Maintenance Mode":
    st.warning("Server sedang pemeliharaan rutin! Data historis tidak dapat dimuat.")
    st.stop()

st.success("Server aktif normal. Eksekusi alur kerja dilanjutkan ke tahap formulir batching.")

st.divider()

# 2. st.form dan st.form_submit_button: Formulir Batching
st.subheader("2. Formulir Estimasi Kebutuhan Armada Sepeda (st.form)")
st.caption(
    "Dengan st.form(), kalkulasi model machine learning hanya dipicu ketika tombol submit diklik, "
    "mencegah aplikasi melakukan rerun berulang pada setiap perubahan slider."
)

with st.form("form_estimasi_demand"):
    st.write("Silakan masukkan parameter kondisi lingkungan:")
    
    col1, col2 = st.columns(2)
    with col1:
        jam = st.slider("Jam Operasional (0 - 23 WIB):", min_value=0, max_value=23, value=8)
        suhu = st.number_input("Suhu Udara (°C):", min_value=-5.0, max_value=45.0, value=25.0, step=0.5)
        cuaca = st.selectbox("Kondisi Cuaca:", ["1: Cerah", "2: Mendung/Kabut", "3: Hujan Ringan", "4: Cuaca Ekstrem"])
    
    with col2:
        kelembaban = st.slider("Kelembaban Relatif (%):", min_value=0, max_value=100, value=60)
        kecepatan_angin = st.number_input("Kecepatan Angin (km/jam):", min_value=0.0, max_value=60.0, value=12.0, step=0.5)
        is_workingday = st.checkbox("Hari Kerja Efektif (Senin - Jumat)", value=True)
    
    # Tombol submit khusus formulir
    submit_button = st.form_submit_button(label="Hitung Estimasi Peminjaman Sepeda")

# Pemrosesan hasil setelah form di-submit
if submit_button:
    # Simulasi estimasi berbasis logika regresi sederhana
    baseline = 80
    if 7 <= jam <= 9 or 17 <= jam <= 19:
        factor_jam = 2.8 if is_workingday else 1.2
    elif 11 <= jam <= 15:
        factor_jam = 1.4 if not is_workingday else 1.0
    else:
        factor_jam = 0.4
    
    factor_suhu = max(0.5, suhu / 25.0)
    factor_cuaca = 1.0 if cuaca.startswith("1") else (0.8 if cuaca.startswith("2") else 0.4)
    factor_kelembaban = 1.0 - (kelembaban / 200.0)
    
    estimasi_count = int(baseline * factor_jam * factor_suhu * factor_cuaca * factor_kelembaban)
    estimasi_komuter = int(estimasi_count * (0.85 if is_workingday else 0.45))
    estimasi_kasual = estimasi_count - estimasi_komuter
    
    st.success(f"Hasil Estimasi Total Peminjaman Sepeda: **{estimasi_count} unit/jam**")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Total Estimasi Demand", f"{estimasi_count} sepeda")
    col_res2.metric("Pengguna Komuter (Registered)", f"{estimasi_komuter} sepeda")
    col_res3.metric("Pengguna Rekreasi (Casual)", f"{estimasi_kasual} sepeda")

st.divider()

# 3. st.rerun(): Memicu Eksekusi Ulang Aplikasi
st.subheader("3. Memicu Rerun Programatik (st.rerun)")
st.caption("Fungsi st.rerun() menggantikan st.experimental_rerun() pada versi Streamlit modern.")
if st.button("Segarkan Ulang Aplikasi (Trigger Rerun)"):
    st.info("Aplikasi sedang dimuat ulang...")
    st.rerun()
