import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Konfigurasi halaman utama
st.set_page_config(page_title="Project Data Mining - Clustering Gaji", layout="wide")

st.title("📊 Analisis Segmentasi Distribusi Gaji Pekerjaan di Indonesia Menggunakan Algoritma K-Means")
st.write("""
Sistem ini dirancang untuk mengidentifikasi pola distribusi pendapatan dan segmentasi karakteristik 
pekerjaan di Indonesia secara objektif melalui pendekatan berbasis data (*data-driven*). 
Pengelompokan dilakukan berdasarkan kemiripan nilai rata-rata gaji untuk memetakan variasi tingkat pendapatan.
""")

# 1. Load Data dan Model Pickle
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("data_profesi_clean.csv")

# Panggil data & model
try:
    kmeans, scaler = load_models()
    df = load_data()
except FileNotFoundError:
    st.error("File model (.pkl) atau dataset (.csv) tidak ditemukan di folder!")
    st.stop()

# 2. Tampilkan Statistik Ringkas Data
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sampel Pekerjaan", f"{len(df):,}")
with col2:
    st.metric("Gaji Tertinggi", f"Rp {df['Gaji_Rata2'].max():,.0f}")
with col3:
    st.metric("Rata-rata Gaji", f"Rp {df['Gaji_Rata2'].mean():,.0f}")

# 3. Tampilkan Grafik Scatter Plot seperti di Colab
st.subheader("📈 Visualisasi Cluster Gaji")
fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(x=df.index, y=df['Gaji_Rata2'], hue=df['Cluster'], palette='viridis', s=50, alpha=0.8, ax=ax)
plt.title('Visualisasi Cluster Gaji (Berdasarkan Indeks Data)')
plt.xlabel('Indeks Pekerjaan')
plt.ylabel('Gaji (Rp)')
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig, use_container_width=False)

# 4. FITUR INTERAKTIF: Prediksi Gaji Baru (Ini yang bikin dosen suka!)
st.sidebar.header("🔮 Fitur Prediksi Cluster")
st.sidebar.write("Masukkan nominal gaji untuk melihat masuk ke cluster mana pekerjaan tersebut.")
input_gaji = st.sidebar.number_input("Masukkan Gaji (Rp):", min_value=0, value=5000000, step=500000)

if st.sidebar.button("Prediksi Cluster"):
    # Lakukan scaling pada input baru
    gaji_scaled = scaler.transform([[input_gaji]])
    # Prediksi menggunakan model K-Means
    hasil_cluster = kmeans.predict(gaji_scaled)[0]
    
    st.sidebar.success(f"Gaji Rp {input_gaji:,.0f} masuk ke dalam **Cluster {hasil_cluster}**")

# 5. Tampilkan Tabel Data
st.subheader("🔍 Telusuri Data Berdasarkan Cluster")
pilih_cluster = st.selectbox("Pilih Cluster:", sorted(df['Cluster'].unique()))
filtered_df = df[df['Cluster'] == pilih_cluster][['Judul Pekerjaan', 'Perusahaan', 'Lokasi', 'Gaji_Rata2']]
st.dataframe(filtered_df, use_container_width=True)