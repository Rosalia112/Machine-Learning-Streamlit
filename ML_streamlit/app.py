from __future__ import print_function

import IPython
print('IPython:', IPython.__version__)

import numpy
print('numpy:', numpy.__version__)

import pandas
print('pandas:', pandas.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)

from sklearn.datasets import load_iris
iris = load_iris()

#mengetahui tipe data
X, y = iris.data, iris.target
type(X)

#data numpy dan contoh 5 data teratas dan labelnya
X,y = iris.data, iris.target

print (X[:5])
print(y[:5])

#data pandas
import pandas as pd

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = iris.target

df.head()

df.isnull().sum()
#buat pipeline untuk data numpy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

pipe_numpy = make_pipeline(StandardScaler(), LinearSVC())
#Splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#split dataset
#numpy (ada 4 parameter karena antara kolom dan label dibedakan)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

# pandas (kolom dan label disatukan)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=46)#training
#numpy (dilakukan scaling dan model )
pipe_numpy.fit(x_train, y_train)
y_pred=pipe_numpy.predict(x_test)
#menampilkan hasil evaluasi dari model yang sudah dibuat
print(classification_report(y_test, y_pred))
#training
#pandas
x_train_df, y_train_df = df_train.drop('target', axis=1), df_train['target']
x_test_df, y_test_df = df_test.drop('target', axis=1), df_test['target']
pipe_pandas.fit(x_train_df, y_train_df)
y_pred_df = pipe_pandas.predict(x_test_df)
print("Hasil Evaluasi Model Pandas")
print(classification_report(y_test_df, y_pred_df))
#save model dengan pickle
import pickle

#numpy
with open('model_numpy.pkl','wb') as f:
  pickle.dump(pipe_numpy, f)

#pandas
with open('model_pandas.pkl', 'wb') as f:
  pickle.dump(pipe_pandas, f)#load model
#numpy
with open('model_numpy.pkl', 'rb') as model_file:
  loaded_model_numpy = pickle.load(model_file)
    #pandas
with open('model_pandas.pkl', 'rb') as model_file:
  loaded_model_pandas = pickle.load(model_file)
