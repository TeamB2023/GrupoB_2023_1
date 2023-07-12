
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import warnings
import streamlit as st
from math import sqrt

# To plot
plt.style.use('seaborn-darkgrid')

# To ignore warnings
warnings.filterwarnings("ignore")
# sklearn
st.set_page_config(page_title="RNN", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("# RNN")
st.sidebar.header("RNN")
st.markdown(
    """
    # Redes Neuronales Recurrentes
    es un tipo de modelo de redes neuronales que procesa secuencias de datos. Las RNN tienen una estructura en la que las salidas de algunas neuronas
    se vuelven a utilizar como entradas en otras neuronas en la misma red, permitiendo que la red tenga una memoria temporal y procese secuencias de 
    datos de manera efectiva. Esto las hace adecuadas para tareas como el procesamiento del lenguaje natural, la predicción de series temporales y otras
    tareas que involucren secuencias de datos. Los modelos RNN se pueden implementar utilizando diferentes tipos de celdas recurrentes, como las celdas
    LSTM y GRU, que tienen una mayor capacidad de memoria y pueden mejorar la capacidad de la red para procesar secuencias de datos a largo plazo. 
    """
)

ticker = st.text_input('Etiqueta de cotización', 'PEN')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
hist = tic.history(period="max", auto_adjust=True)
hist
st.write("## Date time")
testdf = yf.download("PEN", start="2022-03-31",
                     end=dt.datetime.now(), progress=False)
# que se vean los 6 primeros
testdf

st.write("## Realizar la preparación de datos de RNN model entrenamiento ")
training_set = hist.iloc[:, 1:2].values
# QUE SE vean los 5 primeros
training_set[:5]
st.write('Minimos y Maximos')
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
X_train = []
y_train = []

for i in range(60, 566):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train,  y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# ponle un titulo
st.write("## Grafica de los datos de entrenamiento x")
st.line_chart(X_train[0])
# ponle un nombre a esa leyenda
st.write("## Grafica de los datos de entrenamiento y ")
st.line_chart(y_train[:100])
