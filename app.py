import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(page_title="Gym Member Analytics Dashboard", layout="wide")

# Load Data untuk visualisasi dashboard
# Catatan: Dalam skenario nyata, kita memuat data dari database/CSV asli
df = pd.read_csv('gym_members_exercise_tracking.csv')
df_filtered = df[df['Age'].between(18, 34)]

# Load Model Artifacts
model = joblib.load('gym_model.pkl')
scaler = joblib.load('gym_scaler.pkl')

# Header Dashboard
st.title("📊 Gym Member Experience Analytics Dashboard")
st.markdown("Analisis data keanggotaan gym dan klasifikasi tingkat pengalaman untuk kelompok usia 18-34 tahun.")

# Row 1: Key Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Anggota (18-34)", len(df_filtered))
with col2:
    st.metric("Rata-rata BMI", round(df_filtered['BMI'].mean(), 2))
with col3:
    st.metric("Model Accuracy", "95%")
with col4:
    st.metric("Model F1-Score", "0.96")

st.divider()

# Row 2: Visualizations
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribusi Tingkat Pengalaman")
    level_counts = df_filtered['Experience_Level'].value_counts().reset_index()
    level_counts.columns = ['Level', 'Total']
    level_counts['Level Name'] = level_counts['Level'].map({1: 'Beginner', 2: 'Intermediate', 3: 'Expert'})
    fig_pie = px.pie(level_counts, values='Total', names='Level Name', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.subheader("Hubungan Frekuensi Latihan & Durasi")
    fig_scatter = px.scatter(df_filtered, x="Workout_Frequency (days/week)",
                             y="Session_Duration (hours)",
                             color=df_filtered['Experience_Level'].astype(str),
                             labels={'color': 'Exp Level'},
                             template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

# Row 3: Detail Data & Prediction Side Panel
st.divider()
st.subheader("Simulasi Prediksi & Detail Data")

tab1, tab2 = st.tabs(["Simulator Prediksi", "Dataset Preview"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Input Parameter Baru:**")
        age = st.slider("Age", 18, 34, 25)
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
        height = st.number_input("Height (m)", 1.4, 2.2, 1.7)
        session = st.slider("Session Duration (hr)", 0.5, 3.0, 1.0)
        freq = st.slider("Frequency (days/week)", 1, 7, 3)

    with c2:
        st.write("**Hasil Analisis Model:**")
        # Dummy preprocessing sederhana sesuai fitur model
        # (Untuk demonstrasi, sisa fitur diisi nilai rata-rata)
        input_features = np.array([[age, 1, weight, height, 180, 130, 65, session, 2, 20, 2.5, freq]])
        scaled_feat = scaler.transform(input_features)
        pred = model.predict(scaled_feat)[0]
        levels = {1: "Beginner", 2: "Intermediate", 3: "Expert"}

        st.success(f"Prediksi Tingkat Pengalaman: **{levels[pred]}**")
        st.info("Gunakan panel kiri untuk merubah parameter dan melihat perubahan hasil prediksi secara real-time.")

with tab2:
    st.dataframe(df_filtered.head(15), use_container_width=True)

st.caption("Dashboard dibuat dengan Streamlit & Plotly | Data Mining Project 2026")