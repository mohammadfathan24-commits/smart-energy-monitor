import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Smart Energy Monitor AI",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Energy Monitor AI – Smk Alstar")
st.caption("Dashboard AI untuk memonitor & mengoptimalkan energi listrik.")

st.divider()

# =====================
# SIDEBAR MODE
# =====================
st.sidebar.title("⚙️ Pilih Mode")
mode = st.sidebar.selectbox(
    "Mode",
    ["Generate Dataset", "Upload Dataset", "Manual Input"]
)

# =====================
# GENERATE DATASET
# =====================
if mode == "Generate Dataset":
    rows = st.sidebar.slider("Jumlah data", 10, 200, 50)
    data = {
        "ac": np.random.randint(1,10,rows),
        "computer": np.random.randint(5,40,rows),
        "lighting": np.random.randint(10,60,rows),
        "weekday": np.random.randint(0,7,rows),
        "month": np.random.randint(1,13,rows)
    }
    df = pd.DataFrame(data)
    df["energy_usage"] = df["ac"]*30 + df["computer"]*10 + df["lighting"]*5 + np.random.randint(20,80,rows)

    # ✅ Preview dataset
    st.subheader("📂 Dataset Preview")
    st.dataframe(df)

# =====================
# UPLOAD DATASET
# =====================
elif mode == "Upload Dataset":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        # ✅ Preview dataset
        st.subheader("📂 Dataset Preview")
        st.dataframe(df)
    else:
        st.info("Upload dataset dulu.")
        st.stop()

# =====================
# MANUAL INPUT
# =====================
elif mode == "Manual Input":
    st.subheader("⚡ Manual Input")
    col1, col2, col3 = st.columns(3)
    ac_use = col1.slider("AC (kWh)",0,100,40)
    computer_use = col2.slider("Komputer (kWh)",0,100,30)
    lighting_use = col3.slider("Lampu (kWh)",0,100,50)

    day = st.selectbox("Hari", ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"])
    month = st.selectbox("Bulan", ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"])

    weekday_map = {"Senin":0,"Selasa":1,"Rabu":2,"Kamis":3,"Jumat":4,"Sabtu":5,"Minggu":6}
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"Mei":5,"Jun":6,"Jul":7,"Agu":8,"Sep":9,"Okt":10,"Nov":11,"Des":12}
    df = pd.DataFrame({
        "ac":[ac_use],
        "computer":[computer_use],
        "lighting":[lighting_use],
        "weekday":[weekday_map[day]],
        "month":[month_map[month]],
        "energy_usage":[ac_use+computer_use+lighting_use]
    })

    # ✅ Preview dataset
    st.subheader("📂 Dataset Preview")
    st.dataframe(df)

# =====================
# DASHBOARD METRIK
# =====================
st.subheader("📊 Dashboard Metrik")
col1,col2,col3 = st.columns(3)
col1.metric("Jumlah Data", len(df))
col2.metric("Rata Energi", round(df["energy_usage"].mean(),2))
col3.metric("Prediksi Tagihan", f"Rp {round(df['energy_usage'].mean()*1500,0):,.0f}")

st.divider()

# =====================
# LINE CHART
# =====================
st.subheader("📈 Visualisasi Energi")
column = st.selectbox("Pilih Kolom untuk Chart", df.columns)
fig = px.line(df, y=column)
st.plotly_chart(fig,use_container_width=True)

st.divider()

# =====================
# AI PREDICTION
# =====================
st.subheader("🤖 AI Prediksi Energi")
X = df[["ac","computer","lighting","weekday","month"]]
y = df["energy_usage"]
if len(df) < 5:
    st.warning("Minimal 5 data untuk AI training.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_df = pd.DataFrame({"Actual":y_test.values,"AI Prediction":pred})
    fig2 = px.line(pred_df, title="AI Prediction vs Actual")
    st.plotly_chart(fig2,use_container_width=True)
    st.success("Model AI berhasil dilatih!")

st.divider()

# =====================
# SMART RECOMMENDATION
# =====================
st.subheader("💡 Rekomendasi Hemat Energi")
devices = {"AC": df["ac"].mean(), "Computer": df["computer"].mean(), "Lighting": df["lighting"].mean()}
highest = max(devices, key=devices.get)
st.warning(f"Perangkat paling boros energi: {highest}")

if highest=="AC":
    st.info("Naikkan suhu AC atau matikan saat ruangan kosong")
elif highest=="Computer":
    st.info("Matikan komputer jika tidak digunakan")
elif highest=="Lighting":
    st.info("Gunakan lampu LED hemat energi")

# =====================
# DOWNLOAD DATA
# =====================
st.subheader("📥 Download Dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv,"energy_data.csv","text/csv")
st.caption("Smart Energy Monitor AI • Versi Simpel Tanpa Heatmap")
