import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Energy Monitor", page_icon="⚡")

st.title("⚡ Smart Energy Monitor")
st.write("AI sederhana untuk memprediksi penggunaan listrik.")

st.divider()

# Input penggunaan perangkat
ac = st.slider("Penggunaan AC", 0, 100, 50)
computer = st.slider("Penggunaan Komputer", 0, 100, 30)
lighting = st.slider("Penggunaan Lampu", 0, 100, 20)

# Hitung total penggunaan
total = ac + computer + lighting

# Prediksi biaya
tarif = 1500
prediksi_biaya = total * tarif

st.subheader("🔋 Total Penggunaan Energi")
st.write(total)

st.subheader("💰 Prediksi Tagihan Listrik")
st.write(f"Rp {prediksi_biaya:,}")

# Grafik penggunaan
data = pd.DataFrame({
    "Perangkat": ["AC", "Komputer", "Lampu"],
    "Penggunaan": [ac, computer, lighting]
})

fig, ax = plt.subplots()
ax.bar(data["Perangkat"], data["Penggunaan"])
ax.set_ylabel("Penggunaan Energi")

st.pyplot(fig)

# Rekomendasi AI sederhana
st.subheader("🤖 Rekomendasi Hemat Energi")

if ac > 70:
    st.warning("Matikan AC jika ruangan kosong.")
    
if computer > 60:
    st.warning("Matikan komputer yang tidak digunakan.")
    
if lighting > 50:
    st.warning("Kurangi penggunaan lampu di siang hari.")

if total < 100:
    st.success("Penggunaan energi sudah cukup efisien!")
