import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# To ignore warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="SVR")

st.markdown("# SVR")
st.sidebar.header("SVR")
st.write(
    """En esta página podrás ver cómo funciona el modelo SVR en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'BVN')
st.write('La etiqueta de cotización actual es', ticker)
tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
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

# Support vector Regression
cls = SVR().fit(X_train, y_train)

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

st.write("Devoluciones de la estrategia de trama frente a las devoluciones originales")
fig = plt.figure()
plt.plot(df['Cum_Ret'], color='green')
plt.plot(df['Cum_Strategy'], color='yellow')
st.pyplot(fig)

st.write("Haz llegado hasta el final de esta sección. Gracias")
