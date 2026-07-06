import pickle
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import streamlit as st

# Load data iris untuk mengambil nama fitur
iris = load_iris()

st.title("Aplikasi Deployment Model Iris 🌸")

# Menambahkan pilihan tipe model di sidebar
tipe_model = st.sidebar.selectbox("Pilih Model", ["Numpy Array", "Pandas DataFrame"])

st.write(f"Menggunakan Model: **{tipe_model}**")

# Input field dari user untuk karakteristik bunga
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# Tombol Prediksi
if st.button("Prediksi"):
    if tipe_model == "Numpy Array":
        # 12 & 13. Load model numpy dan buat prediksi
        with open("model_numpy.pkl", "rb") as model_file:
            loaded_model_numpy = pickle.load(model_file)
        
        # Format data baru sesuai bentukan list/array [[2, 3, 4, 5]]
        new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = loaded_model_numpy.predict(new_data)
        
    else:
        # 12 & 13. Load model pandas dan buat prediksi
        with open("model_pandas.pkl", "rb") as model_file:
            loaded_model_pandas = pickle.load(model_file)
            
        # Format data baru sesuai bentukan DataFrame dengan columns asli
        new_data_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)
        prediction = loaded_model_pandas.predict(new_data_df)
        
    # Menampilkan Hasil
    target_names = iris.target_names
    st.success(f"Hasil Prediksi Kelas: {prediction[0]} (Iris-{target_names[prediction[0]]})")