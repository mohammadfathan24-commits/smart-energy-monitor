import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")

# Judul aplikasi
st.set_page_config(page_title="Smart Energy Monitor", page_icon="💡", layout="wide")

st.title("💡 Smart Energy Monitor")
st.markdown("AI untuk memprediksi **penggunaan listrik dan saran hemat energi** di sekolah atau kantor.")

st.divider()

# Layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    ac = st.slider("Jumlah penggunaan AC", 0, 100, 50)
    computer = st.slider("Jumlah penggunaan Komputer", 0, 100, 30)

with col2:
    lighting = st.slider("Jumlah penggunaan Lampu", 0, 100, 20)
    weekday = st.selectbox(
        "Pilih Hari",
        ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
    )

month = st.slider("Bulan", 1, 12, 6)

# Konversi hari ke angka
weekday_map = {
    "Senin":0,
    "Selasa":1,
    "Rabu":2,
    "Kamis":3,
    "Jumat":4,
    "Sabtu":5,
    "Minggu":6
}

weekday_num = weekday_map[weekday]

st.divider()

# Tombol prediksi
if st.button("🔍 Prediksi Penggunaan Listrik"):

    data = np.array([[ac, computer, lighting, weekday_num, month]])
    prediksi = model.predict(data)

    st.subheader("⚡ Hasil Prediksi")
    st.success(f"Estimasi penggunaan listrik: **{prediksi[0]:.2f} kWh**")

    # Grafik penggunaan
    st.subheader("📊 Grafik Penggunaan Perangkat")

    perangkat = ["AC","Computer","Lighting"]
    nilai = [ac, computer, lighting]

    fig, ax = plt.subplots()
    ax.bar(perangkat, nilai)
    ax.set_ylabel("Penggunaan")
    ax.set_title("Distribusi Penggunaan Energi")

    st.pyplot(fig)

    st.divider()

    # Saran hemat energi
    st.subheader("💡 Saran Hemat Energi")

    if ac > 70:
        st.warning("Matikan beberapa AC jika tidak digunakan.")

    if computer > 60:
        st.warning("Kurangi komputer yang tidak dipakai.")

    if lighting > 50:
        st.warning("Matikan lampu di ruangan kosong.")

    if ac < 70 and computer < 60 and lighting < 50:
        st.success("Penggunaan energi sudah cukup efisien 👍")
