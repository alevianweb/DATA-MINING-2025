# --- impor library yg dipakai ---
import pandas as pd  
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np  
import joblib  
import json

#time series fitur
def create_time_series_features(df, target_col='Close'):
    """
    Fungsi utama untuk membuat fitur (variabel input) dari data time series.
    Fitur ini akan digunakan oleh model untuk belajar pola dari data.
    """
    #Membuat salinan dataframe untuk menghindari perubahan pada data asli
    df = df.copy() 
    
    #1.Fitur Kalender: Fitur ini membantu model mempelajari pola berdasarkan waktu (mingguan, bulanan, tahunan).
    df['dayofweek'] = df.index.dayofweek  #Hari dalam seminggu (Senin=0, Minggu=6)
    df['quarter'] = df.index.quarter      #Kuartal dalam setahun (1-4)
    df['month'] = df.index.month          #Bulan dalam setahun (1-12)
    df['year'] = df.index.year            #Tahun
    df['dayofyear'] = df.index.dayofyear  #Hari dalam setahun (1-366)
    df['dayofmonth'] = df.index.day       #Tanggal dalam sebulan (1-31)
    df['weekofyear'] = df.index.isocalendar().week.astype(int)  #Minggu dalam setahun (1-53)
    
    #2.Fitur Lag Memberi tahu model tentang nilai-nilai di masa lalu agar model dapat memahami tren data
    lag_days = [1, 7, 14]  #melihat data 1 hari, 7 hari (1 minggu), dan 14 hari (2 minggu) yang lalu
    for lag in lag_days:
        # .shift(lag) akan menggeser data ke bawah sebanyak 'lag' baris, sehingga baris saat ini berisi data dari 'lag' hari sebelumnya
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        
    #3.Fitur Rolling Window Memberi tahu model tentang tren dan volatilitas jangka pendek
    window_sizes = [7, 14]  #melihat jendela waktu 7 hari dan 14 hari
    for window in window_sizes:
        #Menghitung rata-rata harga dalam 'window' hari terakhir.
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        #Menghitung standar deviasi harga dalam 'window' hari terakhir (untuk mengukur volatilitas)
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        
    # Menghapus baris yang mengandung nilai NaN (Not a Number)
    # Nilai NaN ini muncul di awal data karena fitur lag dan rolling tidak bisa dihitung (misal, lag 7 hari tidak ada untuk data hari ke-3)
    df = df.dropna()
    
    return df

#eval model

def calculate_mape(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    # Mengubah input menjadi array numpy untuk perhitungan yang efisien.
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menghitung MAPE. Penambahan 1e-8 untuk menghindari error pembagian dengan nol jika ada nilai asli (y_true) yang bernilai 0.
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluate_model(model, X_test, y_test):
    """Mengevaluasi performa model pada data uji dan mengembalikan hasilnya dalam bentuk dictionary."""
    # Membuat prediksi pada data uji (data yang belum pernah dilihat model).
    predictions = model.predict(X_test)
    
    # Menghitung berbagai metrik performa.
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  # RMSE adalah akar dari MSE
    mape = calculate_mape(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Menyusun metrik ke dalam format dictionary agar mudah disimpan sebagai JSON.
    metrics = {
        "mae": mae, "mse": mse, "rmse": rmse,
        "mape": mape, "r2": r2, "accuracy": 100 - mape  # Akurasi custom berdasarkan MAPE
    }
    return metrics

#train data

def train_and_save_all(data_path, model_filename, metrics_filename):
    """Fungsi utama untuk menjalankan seluruh proses: memuat data, melatih, mengevaluasi, dan menyimpan."""
    
    #1.Memuat dan Mempersiapkan Data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])  #Mengubah kolom 'Date' menjadi format tanggal
    df = df.set_index('Date')[['Close']].dropna()  #Menjadikan 'Date' sebagai index dan hanya mengambil kolom target 'Close'
    
    #2.Membuat Fitur
    df_features = create_time_series_features(df, target_col='Close')
    
    #3.Mendefinisikan Fitur (X) dan Target (y)
    #Daftar semua nama kolom yang akan menjadi input (fitur) bagi model.
    FEATURES = [
        'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14'
    ]
    TARGET = 'Close'  #Kolom yang ingin kita prediksi
    
    X = df_features[FEATURES]  #Data fitur
    y = df_features[TARGET]   #Data target
    
    #4. Membagi Data
    #data dibagi jadi 80% untuk training dan 20% untuk testing
    #shuffle=False untuk data time series agar tidak "mengintip" data masa depan saat training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    #5.Melatih Model untuk Evaluasi
    #Mengatur parameter model XGBoos
    #early_stopping_rounds=50: proses training akan berhenti jika performa model di data validasi tidak membaik setelah 50 iterasi untuk mencegah overfitting
    eval_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50
    )
    print("Melatih model untuk evaluasi dengan fitur baru...")
    #Melatih model dengan data training dan mengevaluasinya pada data testing di setiap langkah
    eval_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    #6.Mengevaluasi dan Menyimpan Hasil
    evaluation_results = evaluate_model(eval_model, X_test, y_test)
    print(f"Menyimpan hasil evaluasi ke file: {metrics_filename}")
    with open(metrics_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4)  #Menyimpan dictionary metrik ke file JSON
    print("Hasil evaluasi berhasil disimpan.")
    
    #7.Melatih Model Final
    #Setelah tahu jumlah iterasi terbaik dari 'eval_model', maka akan melatih model baru pada SELURUH data (train+test)
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=eval_model.best_iteration,  #Menggunakan jumlah pohon terbaik
        learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    print("Melatih model final dengan seluruh data...")
    final_model.fit(X, y, verbose=False)
    
    #8.Menyimpan Model Final
    print(f"Menyimpan model ke file: {model_filename}")
    joblib.dump(final_model, model_filename)  #Menyimpan objek model yang sudah dilatih
    print("Model berhasil disimpan.")

#fungsi yg akan dijalankan saat skrip di run
if __name__ == '__main__':
    file_path = 'dataset.csv'
    model_file = 'model_forecast_idr.joblib'
    metrics_file = 'evaluasi_model.json'
    
    #Memanggil fungsi utama untuk memulai seluruh proses
    train_and_save_all(file_path, model_file, metrics_file)
    print("\nProses training dan penyimpanan selesai.")