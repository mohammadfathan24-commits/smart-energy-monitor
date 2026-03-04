import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ======================
# Buat Dataset Dummy
# ======================
dates = pd.date_range(start="2025-01-01", end="2025-12-31")
np.random.seed(42)

data = {
    "date": dates,
    "ac_usage": np.random.normal(60, 10, len(dates)),
    "computer_usage": np.random.normal(40, 8, len(dates)),
    "lighting_usage": np.random.normal(25, 5, len(dates))
}

df = pd.DataFrame(data)

df["total_usage"] = df["ac_usage"] + df["computer_usage"] + df["lighting_usage"]
tarif = 1500
df["monthly_bill"] = df["total_usage"] * tarif
df["weekday"] = df["date"].dt.weekday
df["month"] = df["date"].dt.month

# ======================
# Training Model
# ======================
X = df[["ac_usage","computer_usage","lighting_usage","weekday","month"]]
y = df["monthly_bill"]

model = RandomForestRegressor()
model.fit(X, y)

# ======================
# Streamlit UI
# ======================
st.title("💡 Smart Energy Monitor")
st.subheader("Prediksi Tagihan & Rekomendasi Hemat Energi")

ac = st.slider("Penggunaan AC (kWh)", 0, 120, 60)
computer = st.slider("Penggunaan Komputer (kWh)", 0, 100, 40)
lighting = st.slider("Penggunaan Lampu (kWh)", 0, 60, 25)
weekday = st.slider("Hari (0=Senin, 6=Minggu)", 0, 6, 2)
month = st.slider("Bulan", 1, 12, 6)

data_input = pd.DataFrame({
    "ac_usage": [ac],
    "computer_usage": [computer],
    "lighting_usage": [lighting],
    "weekday": [weekday],
    "month": [month]
})

prediksi = model.predict(data_input)

st.subheader("💰 Estimasi Tagihan:")
st.write("Rp", int(prediksi[0]))

st.subheader("📊 Rekomendasi Hemat Energi")

if ac > 70:
    st.warning("⚠ Kurangi penggunaan AC atau naikkan suhu ke 26°C")
if computer > 50:
    st.warning("⚠ Matikan komputer yang tidak digunakan")
if lighting > 35:
    st.warning("⚠ Gunakan cahaya alami di siang hari")

if ac <= 70 and computer <= 50 and lighting <= 35:
    st.success("✅ Penggunaan sudah efisien")