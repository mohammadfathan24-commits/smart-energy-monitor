import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Smart Energy Monitor",
    page_icon="⚡",
    layout="wide"
)

# Judul
st.title("⚡ Smart Energy Monitor")
st.write("AI Dashboard untuk memonitor dan memprediksi penggunaan listrik")

st.divider()

# Layout input
col1, col2 = st.columns(2)

with col1:
    ac = st.slider("Penggunaan AC", 0, 100, 50)
    computer = st.slider("Penggunaan Komputer", 0, 100, 30)

with col2:
    lighting = st.slider("Penggunaan Lampu", 0, 100, 20)
    weekday = st.selectbox(
        "Hari",
        ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
    )

# Hitung energi
total = ac + computer + lighting
tarif = 1500
prediksi_biaya = total * tarif

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Energi", total)
col2.metric("Tarif Listrik", "Rp1500")
col3.metric("Prediksi Tagihan", f"Rp {prediksi_biaya:,}")

st.divider()

# Data grafik
data = pd.DataFrame({
    "Perangkat": ["AC", "Komputer", "Lampu"],
    "Penggunaan": [ac, computer, lighting]
})

# Grafik
fig, ax = plt.subplots()
ax.bar(data["Perangkat"], data["Penggunaan"])
ax.set_ylabel("Penggunaan Energi")

st.subheader("📊 Grafik Penggunaan Energi")
st.pyplot(fig)

st.divider()

# AI Recommendation
st.subheader("🤖 Rekomendasi AI Hemat Energi")

if ac > 70:
    st.warning("Kurangi penggunaan AC untuk menghemat listrik.")

if computer > 60:
    st.warning("Matikan komputer yang tidak digunakan.")

if lighting > 50:
    st.warning("Gunakan pencahayaan alami pada siang hari.")

if total < 100:
    st.success("Penggunaan energi sudah efisien!")
