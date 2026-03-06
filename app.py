import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Smart Energy Monitor AI",
    page_icon="⚡",
    layout="wide"
)

# HEADER
st.title("⚡ Smart Energy Monitor AI")
st.write("AI Dashboard untuk memonitor dan memprediksi penggunaan energi.")

st.divider()

# SIDEBAR
st.sidebar.title("⚙️ Kontrol")

mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Upload Dataset", "Generate Dataset"]
)

# GENERATE DATASET
if mode == "Generate Dataset":

    st.subheader("⚡ Generate Dataset Energi")

    rows = st.slider("Jumlah data", 10, 200, 50)

    data = {
        "ac": np.random.randint(1,10,rows),
        "computer": np.random.randint(5,40,rows),
        "lighting": np.random.randint(10,60,rows),
        "weekday": np.random.randint(0,7,rows),
        "month": np.random.randint(1,13,rows),
    }

    df = pd.DataFrame(data)

    df["energy_usage"] = (
        df["ac"]*30 +
        df["computer"]*10 +
        df["lighting"]*5 +
        np.random.randint(20,80,rows)
    )

else:

    file = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
    else:
        st.info("Upload dataset atau gunakan Generate Dataset.")
        st.stop()

# DATA PREVIEW
st.subheader("📂 Dataset Preview")
st.dataframe(df)

# STATISTICS
st.subheader("📊 Statistik Energi")

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Data", len(df))
col2.metric("Jumlah Kolom", len(df.columns))
col3.metric("Rata Energi", round(df["energy_usage"].mean(),2))

st.divider()

# VISUALIZATION
st.subheader("📈 Grafik Energi")

column = st.selectbox("Pilih Kolom", df.columns)

fig = px.line(df, y=column)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# MACHINE LEARNING
st.subheader("🤖 AI Prediksi Energi")

X = df[["ac","computer","lighting","weekday","month"]]
y = df["energy_usage"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestRegressor()

model.fit(X_train,y_train)

pred = model.predict(X_test)

pred_df = pd.DataFrame({
    "Data Asli":y_test.values,
    "Prediksi AI":pred
})

fig2 = px.line(pred_df,title="Prediksi AI vs Data Asli")

st.plotly_chart(fig2,use_container_width=True)

st.success("Model AI berhasil dilatih!")

st.divider()

# ELECTRICITY COST
st.subheader("💰 Prediksi Tagihan Listrik")

tarif = st.number_input(
    "Tarif listrik per kWh (Rp)",
    value=1500
)

avg_energy = y.mean()

cost = avg_energy * tarif

st.metric("Estimasi Tagihan",f"Rp {cost:,.0f}")

st.divider()

# AI RECOMMENDATION
st.subheader("💡 AI Rekomendasi Hemat Energi")

if df["ac"].mean() > 6:
    st.warning("Gunakan AC seperlunya atau naikkan suhu AC.")

if df["computer"].mean() > 25:
    st.info("Matikan komputer yang tidak digunakan.")

if df["lighting"].mean() > 40:
    st.info("Gunakan lampu LED hemat energi.")

st.divider()

# DOWNLOAD
st.subheader("📥 Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "energy_data.csv",
    "text/csv"
)

st.caption("Smart Energy Monitor AI • Project AI untuk efisiensi energi")




