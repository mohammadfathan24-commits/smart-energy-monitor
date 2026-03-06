import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Smart Energy Monitor AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DARK MODE
st.markdown(
    """
    <style>
    .main {background-color: #0E1117; color: #ffffff;}
    .st-bf {color:white;}
    </style>
    """, unsafe_allow_html=True
)

# =====================
# HEADER
# =====================
st.title("⚡ Smart Energy Monitor AI – V11 Ultimate")
st.caption("AI Dashboard canggih untuk monitoring, analisis, dan optimasi energi listrik.")

st.divider()

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Kontrol")
mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Generate Dataset","Upload Dataset","Manual Input"]
)

# =====================
# GENERATE DATASET
# =====================
if mode == "Generate Dataset":
    rows = st.sidebar.slider("Jumlah data",10,300,50)
    data = {
        "ac": np.random.randint(1,10,rows),
        "computer": np.random.randint(5,40,rows),
        "lighting": np.random.randint(10,60,rows),
        "weekday": np.random.randint(0,7,rows),
        "month": np.random.randint(1,13,rows)
    }
    df = pd.DataFrame(data)
    df["energy_usage"] = (
        df["ac"]*30 + df["computer"]*10 + df["lighting"]*5 + np.random.randint(20,80,rows)
    )

# =====================
# UPLOAD DATASET
# =====================
elif mode == "Upload Dataset":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.info("Upload dataset terlebih dahulu.")
        st.stop()

# =====================
# MANUAL INPUT
# =====================
elif mode == "Manual Input":
    st.subheader("⚡ Energy Simulator")
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

# =====================
# DASHBOARD METRICS
# =====================
st.subheader("📊 Dashboard Metrik Energi")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Jumlah Data", len(df))
col2.metric("Rata Energi", round(df["energy_usage"].mean(),2))
col3.metric("AC Avg", round(df["ac"].mean(),2))
col4.metric("Lampu Avg", round(df["lighting"].mean(),2))

st.divider()

# =====================
# VISUALIZATION
# =====================
st.subheader("📈 Visualisasi Energi")
column = st.selectbox("Pilih Kolom", df.columns)
fig = px.line(df, y=column)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================
# MACHINE LEARNING AI PREDICTION
# =====================
st.subheader("🤖 AI Prediksi Energi")
X = df[["ac","computer","lighting","weekday","month"]]
y = df["energy_usage"]
if len(df) < 5:
    st.warning("Dataset terlalu sedikit untuk training AI.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_df = pd.DataFrame({"Actual":y_test.values,"AI Prediction":pred})
    fig2 = px.line(pred_df, title="AI Prediction vs Actual")
    st.plotly_chart(fig2,use_container_width=True)
    st.success("Model AI berhasil dilatih")

# =====================
# ELECTRICITY COST & CO2
# =====================
st.subheader("💰 Prediksi Tagihan & 🌱 CO2")
tarif = st.number_input("Tarif listrik per kWh (Rp)", value=1500)
avg_energy = df["energy_usage"].mean()
cost = avg_energy * tarif
co2 = avg_energy * 0.85
col1,col2 = st.columns(2)
col1.metric("Estimasi Tagihan", f"Rp {cost:,.0f}")
col2.metric("CO2 Emission (kg)", round(co2,2))

st.divider()

# =====================
# ENERGY EFFICIENCY SCORE
# =====================
st.subheader("⚡ Energy Efficiency Score")
max_energy = 300
score = max(0, 100-(avg_energy/max_energy*100))
st.metric("Energy Score", f"{score:.0f}/100")
if score > 80:
    st.success("Energi sangat efisien")
elif score > 60:
    st.info("Energi cukup efisien")
else:
    st.warning("Penggunaan energi tinggi ⚠️")

st.divider()

# =====================
# HEATMAP
# =====================
st.subheader("🔥 Heatmap Energi")
heatmap_data = df.groupby(["weekday","month"])["energy_usage"].mean().reset_index()
fig3 = px.density_heatmap(
    heatmap_data, x="month", y="weekday", z="energy_usage",
    color_continuous_scale="reds"
)
st.plotly_chart(fig3,use_container_width=True)

st.divider()

# =====================
# ENERGY FORECAST
# =====================
st.subheader("📉 Prediksi Konsumsi Energi")
future_days = 10
forecast = np.linspace(avg_energy, avg_energy*1.1, future_days)
forecast_df = pd.DataFrame({"Day":range(1,future_days+1),"Energy Forecast":forecast})
fig4 = px.line(forecast_df, x="Day", y="Energy Forecast")
st.plotly_chart(fig4,use_container_width=True)

st.divider()

# =====================
# AI SMART RECOMMENDATION & ANOMALY
# =====================
st.subheader("🤖 Smart Energy Recommendation & Anomaly Detection")
devices = {"AC":df["ac"].mean(), "Computer":df["computer"].mean(), "Lighting":df["lighting"].mean()}
highest = max(devices, key=devices.get)
st.warning(f"Perangkat paling boros energi: {highest}")

if highest=="AC":
    st.info("Naikkan suhu AC atau matikan saat ruangan kosong")
elif highest=="Computer":
    st.info("Matikan komputer jika tidak digunakan")
elif highest=="Lighting":
    st.info("Gunakan lampu LED hemat energi")

# Anomaly detection
threshold = avg_energy * 1.5
if df["energy_usage"].max() > threshold:
    st.error("⚠️ Terdeteksi pemborosan energi! Periksa perangkat yang aktif berlebihan.")

# =====================
# GAUGE METERS
# =====================
st.subheader("🎛️ Gauge Meter Perangkat")
for device, value in devices.items():
    fig_g = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': device + " (kWh)"},
        gauge = {'axis': {'range':[0,100]},
                 'bar': {'color':'limegreen' if value<50 else 'red'}}
    ))
    st.plotly_chart(fig_g,use_container_width=True)

st.divider()

# =====================
# DOWNLOAD DATA
# =====================
st.subheader("📥 Download Dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv,"energy_data.csv","text/csv")
st.caption("Smart Energy Monitor AI V11 • Ultimate AI Energy Dashboard")
