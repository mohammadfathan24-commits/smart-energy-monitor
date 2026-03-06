import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("⚡ Smart Energy Monitor AI")

st.write("Aplikasi AI sederhana untuk memonitor penggunaan energi dan memprediksi tagihan listrik.")

# SIDEBAR MODE
mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Generate Dataset", "Upload Dataset", "Manual Input"]
)

# ==============================
# MODE 1 : GENERATE DATASET
# ==============================

if mode == "Generate Dataset":

    st.subheader("⚡ Generate Dataset Energi")

    rows = st.slider("Jumlah data", 10, 200, 50)

    data = {
        "ac": np.random.randint(10,80,rows),
        "computer": np.random.randint(5,60,rows),
        "lighting": np.random.randint(10,70,rows),
        "weekday": np.random.randint(0,7,rows),
        "month": np.random.randint(1,13,rows)
    }

    df = pd.DataFrame(data)

    df["total_energy"] = df["ac"] + df["computer"] + df["lighting"]
    df["bill"] = df["total_energy"] * 1500

    st.write("Dataset Energi:")
    st.dataframe(df)

    # TRAIN MODEL
    X = df[["ac","computer","lighting"]]
    y = df["bill"]

    model = LinearRegression()
    model.fit(X,y)

    st.success("Model AI berhasil dilatih dari dataset yang digenerate.")

# ==============================
# MODE 2 : UPLOAD DATASET
# ==============================

elif mode == "Upload Dataset":

    st.subheader("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Upload file CSV")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Preview Dataset:")
        st.dataframe(df.head())

        if {"ac","computer","lighting","bill"}.issubset(df.columns):

            X = df[["ac","computer","lighting"]]
            y = df["bill"]

            model = LinearRegression()
            model.fit(X,y)

            st.success("Model AI berhasil dilatih dari dataset yang diupload.")

        else:

            st.error("Dataset harus memiliki kolom: ac, computer, lighting, bill")

# ==============================
# MODE 3 : MANUAL INPUT
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

    st.divider()

    # HITUNG TOTAL ENERGI
    total_energy = ac_use + computer_use + lighting_use

    tarif = 1500

    bill = total_energy * tarif

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Energi (kWh)", total_energy)
    col2.metric("Tarif Listrik", f"Rp {tarif}")
    col3.metric("Prediksi Tagihan", f"Rp {bill:,.0f}")

    st.divider()

    # AI REKOMENDASI
    st.subheader("🤖 AI Rekomendasi Penghematan")

    if ac_use > 70:
        st.warning("AC terlalu tinggi ⚠️ Pertimbangkan menaikkan suhu AC atau mematikannya saat tidak digunakan.")

    if lighting_use > 70:
        st.info("Lampu cukup tinggi 💡 Gunakan lampu LED atau matikan lampu yang tidak diperlukan.")

    if computer_use > 70:
        st.info("Komputer aktif lama 💻 Matikan komputer jika tidak digunakan.")

    if total_energy < 150:
        st.success("Penggunaan energi cukup efisien 👍")
