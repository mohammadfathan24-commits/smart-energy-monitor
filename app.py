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

st.title("⚡ Smart Energy Monitor AI Dashboard")
st.write("AI untuk analisis dan prediksi penggunaan energi")

st.divider()

# Upload dataset
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("📂 Dataset Preview")
    st.dataframe(df)

    st.divider()

    # Statistik
    st.subheader("📊 Statistik Data")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", len(df))
    col2.metric("Jumlah Kolom", len(df.columns))
    col3.metric("Missing Data", df.isna().sum().sum())

    st.divider()

    # Grafik penggunaan energi
    st.subheader("⚡ Visualisasi Energi")

    column = st.selectbox("Pilih kolom untuk grafik", df.columns)

    fig = px.line(df, y=column, title="Grafik Energi")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Machine Learning
    st.subheader("🤖 AI Prediksi Energi")

    if len(df.columns) >= 2:

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor()

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        pred_df = pd.DataFrame({
            "Data Asli": y_test.values,
            "Prediksi AI": pred
        })

        fig2 = px.line(pred_df, title="Prediksi AI vs Data Asli")

        st.plotly_chart(fig2, use_container_width=True)

        st.success("Model AI berhasil dilatih!")

        st.divider()

        # Rekomendasi hemat energi
        st.subheader("💡 Rekomendasi Hemat Energi")

        avg = y.mean()

        if avg > 500:
            st.warning("Penggunaan energi tinggi. Matikan perangkat tidak terpakai.")
        elif avg > 300:
            st.info("Penggunaan energi sedang. Optimalkan penggunaan AC dan lampu.")
        else:
            st.success("Penggunaan energi sudah efisien!")

else:

    st.info("Silakan upload dataset CSV untuk memulai analisis.")

st.divider()

st.caption("Smart Energy Monitor AI | Project AI untuk efisiensi energi")
