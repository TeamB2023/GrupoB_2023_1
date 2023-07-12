import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# To ignore warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="SVC")

st.markdown("# SVC")
st.sidebar.header("SVC")
st.write(
    """En esta página podrás ver cómo funciona el modelo SVC en la predicción del mercado de valores"""
)
st.write('Escriba una etiqueta de cotizacion')
ticker = st.text_input('Etiqueta de cotización', 'NFLX')
st.write('La etiqueta de cotización actual es', ticker)

tic = yf.Ticker(ticker)
tic
# Opciones para el radio button
opciones = ['Usar Todo el Registo', 'Seleccionar Rango de Fechas']

# Crear el radio button para seleccionar una opción
opcion_seleccionada = st.radio('Selecciona una opción', opciones)

# Verificar la opción seleccionada
if opcion_seleccionada == 'Usar Todo el Registo':
    hist = tic.history(period="max", auto_adjust=True)
    st.write("Datos Completos:")
elif opcion_seleccionada == 'Seleccionar Rango de Fechas':
    # Apartado para enviar un dato
    from datetime import datetime
    start_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2022-12-31', '%Y-%m-%d').date()
    st.write("Primero Seleccionalos el rango de fechas de los datos a usar:")
    Finit = st.date_input("Ingrese una Fecha de Inicio:",value = start_date)
    Fend = st.date_input("Ingrese una Fecha de Final:",value = end_date)
    hist = tic.history(start=Finit, end=Fend, auto_adjust=True)
    st.write("Datos en el rango seleccionado:")

hist
df = hist
df.info()

# Crea variables predictoras
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Guarda todas las variables predictoras en una variable X
X = df[['Open-Close', 'High-Low']]
X.head()

# Variables objetivas
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
cls = SVC().fit(X_train, y_train)

df['Predicted_Signal'] = cls.predict(X)
# Calcula los retornos diarios
df['Return'] = df.Close.pct_change()
# Calcula retornos de estrategia
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
# Calcula retornos acumulativos
df['Cum_Ret'] = df['Return'].cumsum()
st.write("Dataframe con retornos acumulativos")
df
# Haz un plot de retornos de estrategia acumulativos
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
st.write("Dataframe con retornos de estrategia acumulativos")
df


st.write("Plot Strategy Returns vs Original Returns")
fig = plt.figure()
plt.plot(df['Cum_Ret'], color='red')
plt.plot(df['Cum_Strategy'], color='blue')
st.pyplot(fig)

st.write("Haz llegado hasta el final de esta sección. Gracias")