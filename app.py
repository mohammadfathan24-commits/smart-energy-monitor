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
    ["Upload Dataset", "Generate Dataset", "Manual Input"]
)

# ==============================
# GENERATE DATASET
# ==============================

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

# ==============================
# UPLOAD DATASET
# ==============================

elif mode == "Upload Dataset":

    file = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
    else:
        st.info("Upload dataset terlebih dahulu.")
        st.stop()

# ==============================
# MANUAL INPUT DASHBOARD
# ==============================

elif mode == "Manual Input":

    st.subheader("⚡ Smart Energy Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        ac_use = st.slider("Penggunaan AC (kWh)",0,100,50)
        computer_use = st.slider("Penggunaan Komputer (kWh)",0,100,40)

    with col2:
        lighting_use = st.slider("Penggunaan Lampu (kWh)",0,100,60)

        day = st.selectbox(
            "Hari",
            ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
        )

        month = st.selectbox(
            "Bulan",
            ["Jan","Feb","Mar","Apr","Mei","Jun",
             "Jul","Agu","Sep","Okt","Nov","Des"]
        )

    weekday_map = {
        "Senin":0,"Selasa":1,"Rabu":2,"Kamis":3,
        "Jumat":4,"Sabtu":5,"Minggu":6
    }

    month_map = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,"Mei":5,"Jun":6,
        "Jul":7,"Agu":8,"Sep":9,"Okt":10,"Nov":11,"Des":12
    }

    df = pd.DataFrame({
        "ac":[ac_use],
        "computer":[computer_use],
        "lighting":[lighting_use],
        "weekday":[weekday_map[day]],
        "month":[month_map[month]]
    })

    df["energy_usage"] = ac_use + computer_use + lighting_use

    st.divider()

    total_energy = df["energy_usage"].iloc[0]

    tarif = 1500

    bill = total_energy * tarif

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Energi (kWh)",total_energy)
    col2.metric("Tarif Listrik",f"Rp {tarif}")
    col3.metric("Prediksi Tagihan",f"Rp {bill:,.0f}")

    st.divider()

    st.subheader("🤖 AI Rekomendasi Penghematan")

    if ac_use > 70:
        st.warning("AC terlalu tinggi ⚠️ Pertimbangkan menaikkan suhu atau mematikannya.")

    if lighting_use > 70:
        st.info("Lampu cukup tinggi 💡 Gunakan lampu LED atau matikan yang tidak diperlukan.")

    if computer_use > 70:
        st.info("Komputer aktif lama 💻 Matikan jika tidak digunakan.")

    if total_energy < 150:
        st.success("Penggunaan energi cukup efisien 👍")

# ==============================
# DATA PREVIEW
# ==============================

st.subheader("📂 Dataset Preview")
st.dataframe(df)

# ==============================
# STATISTICS
# ==============================

st.subheader("📊 Statistik Energi")

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Data", len(df))
col2.metric("Jumlah Kolom", len(df.columns))
col3.metric("Rata Energi", round(df["energy_usage"].mean(),2))

st.divider()

# ==============================
# VISUALIZATION
# ==============================

st.subheader("📈 Grafik Energi")

column = st.selectbox("Pilih Kolom", df.columns)

fig = px.line(df, y=column)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==============================
# MACHINE LEARNING
# ==============================

st.subheader("🤖 AI Prediksi Energi")

X = df[["ac","computer","lighting","weekday","month"]]
y = df["energy_usage"]

if len(df) < 5:

    st.warning("Data terlalu sedikit untuk melatih AI. Gunakan Generate Dataset minimal 10 data.")

else:

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

# ==============================
# DOWNLOAD
# ==============================

st.subheader("📥 Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "energy_data.csv",
    "text/csv"
)

st.caption("Smart Energy Monitor AI • Project AI untuk efisiensi energi")
