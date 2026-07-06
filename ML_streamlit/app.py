import pickle
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import streamlit as st

print("IPython:", IPython.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)

# 2. Load data iris bawaan dari sklearn
iris = load_iris()

# 3. Menampilkan tipe data iris & contoh data teratas
X, y = iris.data, iris.target
print("\nTipe data X:", type(X))
print("Contoh 5 data teratas:\n", X[:5])
print("Contoh 5 label teratas:\n", y[:5])

# 4. Menampilkan data pandas dataframe nya
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
print("\n5 Data teratas DataFrame:\n", df.head())

# 5. Lakukan preprocessing (cek data null, duplicat, outlier jika diperlukan)

# 6 & 7. Membuat pipeline (StandardScaler + LinearSVC)
pipe_numpy = make_pipeline(StandardScaler(), LinearSVC())
pipe_pandas = make_pipeline(StandardScaler(), LinearSVC())

# 8. Melakukan splitting data train dengan data tes (test_size=0.2, random_state=46)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=46
)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=46)

# 9. Melakukan training & menampilkan hasil evaluasi untuk numpy array
print("\n--- Training Model Numpy ---")
pipe_numpy.fit(X_train, y_train)
y_pred = pipe_numpy.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Melakukan training & menampilkan hasil evaluasi untuk pandas dataframe
print("\n--- Training Model Pandas ---")
X_train_df, y_train_df = df_train.drop("target", axis=1), df_train["target"]
X_test_df, y_test_df = df_test.drop("target", axis=1), df_test["target"]

pipe_pandas.fit(X_train_df, y_train_df)
y_pred_df = pipe_pandas.predict(X_test_df)
print(classification_report(y_test_df, y_pred_df))

# 11. Simpan model yang sudah dibuat ke file .pkl
# Menyimpan model numpy
with open("model_numpy.pkl", "wb") as f:
    pickle.dump(pipe_numpy, f)

# Menyimpan model pandas
with open("model_pandas.pkl", "wb") as model_file:
    pickle.dump(pipe_pandas, model_file)

print("\nModel numpy dan pandas berhasil disimpan!")

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