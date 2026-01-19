# --- impor library yg dipakai ---
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta, date
import json
#---import matplotlib.pyplot as plt---

# --- Konfigurasi Halaman Aplikasi ---
st.set_page_config(page_title="Forecast Harga IDR/USD", layout="wide") # Mengatur judul tab browser dan layout halaman
st.title("Aplikasi Prediksi Harga IDR/USD") # Judul utama di halaman aplikasi

# --- Fungsi-Fungsi Pembantu ---


def create_time_series_features(df, target_col='Close'):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    lag_days = [1, 7, 14]
    for lag in lag_days:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    window_sizes = [7, 14]
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df

#Decorator @st.cache_data memberi tahu Streamlit untuk menyimpan hasil fungsi di cache model hanya akan dimuat dari disk satu kali, membuat aplikasi lebih cepat
@st.cache_data
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None #Mengembalikan None jika file tidak ditemukan

@st.cache_data
def load_metrics(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

#fungsi untuk melakukan prediksi pada satu tanggal di masa depan
def forecast_future_date(model, future_date_obj, historical_data, adjust_for_long_term=False, annual_growth_rate=0.0):
    try:
        #Ambil 30 hari data terakhir dari dataset historis untuk menghitung fitur rolling/lag.
        last_data = historical_data.tail(30).copy()
        
        #Buat DataFrame baru yang hanya berisi satu baris untuk tanggal yang dipilih
        future_df = pd.DataFrame(index=[pd.to_datetime(future_date_obj)])
        future_df['Close'] = np.nan # Kolom target diisi NaN karena ini bagian ini yang ingin prediksi
        
        #Gabungkan data historis terakhir dengan data masa depan
        combined_df = pd.concat([last_data, future_df])
        
        #Panggil fungsi create_time_series_features pada data gabungan(secara otomatis menghitung)
        combined_features = create_time_series_features(combined_df, target_col='Close')
        
        #Ambil hanya baris terakhir yaitu baris untuk tanggal masa depan yang sudah lengkap fiturnya
        future_row = combined_features.tail(1)
        
        # Daftar fitur harus sama dengan saat training data
        FEATURES = [
            'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
            'lag_1', 'lag_7', 'lag_14',
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14'
        ]
        
        #Siapkan input untuk model
        X_future = future_row[FEATURES]
        
        #melakukan prediksi
        prediction = model.predict(X_future)
        
        # Penyesuaian untuk prediksi jangka panjang
        if adjust_for_long_term:
            last_historical_date = historical_data.index.max()
            years_diff = (pd.to_datetime(future_date_obj) - last_historical_date).days / 365.25
            if years_diff > 0:
                # Apply compound growth
                prediction = prediction[0] * ((1 + annual_growth_rate) ** years_diff)
                return prediction
        
        return prediction[0] # Mengembalikan hasil prediksi tunggal
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        return None

#memuat model dan tampilan ui
MODEL_PATH = 'model_forecast_idr.joblib'
METRICS_PATH = 'evaluasi_model.json'
DATA_PATH = 'dataset.csv'

#Memuat model dan metrik saat aplikasi pertama kali dijalankan
model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

#Memeriksa apakah file model dan metrik berhasil dimuat
if model is None or metrics is None:
    st.error(f"Error: File model ('{MODEL_PATH}') atau file evaluasi ('{METRICS_PATH}') tidak ditemukan.")
    st.warning("Pastikan Anda sudah menjalankan skrip training yang baru untuk menghasilkan kedua file tersebut.")
else:
    st.success("Model prediksi dan data evaluasi berhasil dimuat!")

    # Membuat expander (area yang bisa dibuka-tutup) untuk menampilkan performa model
    with st.expander("📊 Lihat Performa & Evaluasi Model Terbaru", expanded=True):
        #Mengambil nilai metrik dari dictionary yang sudah dimuat. .get() digunakan untuk menghindari error jika kunci tidak ada
        mae = metrics.get('mae', 0); mse = metrics.get('mse', 0); rmse = metrics.get('rmse', 0)
        mape = metrics.get('mape', 0); r2 = metrics.get('r2', 0); accuracy = metrics.get('accuracy', 0)
        
        #Membuat layout 3 kolom agar tampilan lebih rapi.
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Tingkat Akurasi (100% - MAPE)", value=f"{accuracy:.2f} %")
            st.metric(label="Persentase Kesalahan (MAPE)", value=f"{mape:.2f} %")
        with col2:
            st.metric(label="Mean Absolute Error (MAE)", value=f"Rp {mae:,.2f}")
            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:,.2f}")
        with col3:
            st.metric(label="Root Mean Squared Error (RMSE)", value=f"Rp {rmse:,.2f}")
            st.metric(label="R² Score", value=f"{r2:.4f}")

    #Bagian input tanggal untuk prediksi
    st.header("Pilih Tanggal untuk Prediksi")
    try:
        #Memuat data historis
        historical_df = pd.read_csv(DATA_PATH)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        historical_df = historical_df.set_index('Date')[['Close']].dropna()

        #Membuat input tanggal untuk pengguna.
        today = date.today()
        selected_date = st.date_input(
            "Pilih tanggal untuk prediksi:",
            value=today,
            min_value=today
        )
        #tombol untuk memulai prediksi
        if st.button("🔮 Prediksi Harga"):
            # Konversi tanggal terpilih ke datetime untuk perbandingan
            selected_date_dt = pd.to_datetime(selected_date)
            last_historical_date = historical_df.index.max()

            # Cek apakah tanggal yang dipilih ada di masa lalu atau masa kini
            if selected_date_dt <= last_historical_date:
                try:
                    # Ambil nilai langsung dari dataset
                    past_value = historical_df.loc[selected_date_dt, 'Close']
                    st.subheader(f"Data Historis untuk {selected_date.strftime('%d-%m-%Y')}")
                    st.metric(label="Harga Penutupan (dari dataset)", value=f"Rp {past_value:,.2f}")
                except KeyError:
                    st.warning(f"Tidak ada data untuk tanggal {selected_date.strftime('%d-%m-%Y')}. Ini mungkin akhir pekan atau hari libur.")
            else:
                # Jika tanggal ada di masa depan, lakukan prediksi
                with st.spinner('Menghitung prediksi...'):
                    # --- Perhitungan Tingkat Pertumbuhan Tahunan ---
                    start_date = historical_df.index.min()
                    end_date = last_historical_date
                    start_price = historical_df.loc[start_date, 'Close']
                    end_price = historical_df.loc[end_date, 'Close']
                    num_years = (end_date - start_date).days / 365.25
                    
                    # Menghitung Compound Annual Growth Rate (CAGR)
                    # Rumus: (Ending Value / Beginning Value)^(1 / Number of Years) - 1
                    if num_years > 0 and start_price > 0 and end_price > 0:
                        annual_growth_rate = (end_price / start_price)**(1 / num_years) - 1
                    else:
                        annual_growth_rate = 0.0

                    # Memanggil fungsi prediksi dengan penyesuaian jangka panjang
                    hasil_prediksi = forecast_future_date(
                        model, 
                        selected_date, 
                        historical_df,
                        adjust_for_long_term=True,
                        annual_growth_rate=annual_growth_rate
                    )

                # Jika hasil prediksi berhasil didapatkan.
                if hasil_prediksi is not None:
                    st.subheader("Hasil Prediksi:")
                    st.metric(label=f"Prediksi Harga pada {selected_date.strftime('%d-%m-%Y')}", value=f"Rp {hasil_prediksi:,.2f}")
                    with st.expander("Detail Prediksi"):
                        st.info(f"Prediksi disesuaikan dengan tingkat pertumbuhan tahunan rata-rata sebesar **{annual_growth_rate:.2%}**(CAGR) yang dihitung dari data historis(01-01-2020 sampai 18-06-2025)")

    except FileNotFoundError:
        st.error(f"Dataset '{DATA_PATH}' tidak ditemukan. File ini diperlukan untuk membuat prediksi.")

#footer
st.markdown("---")
st.markdown("KELOMPOK 8 | ZIDAN | JOSHUA | SEN ARYA | ALEVIAN | RAMA | Model: XGBoost")