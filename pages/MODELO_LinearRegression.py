import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# To ignore warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="LR")

st.markdown("# LR")
st.sidebar.header("LR")
st.write(
    """En esta página podrás ver cómo funciona el modelo de Regresión Líneal en la predicción del mercado de valores"""
)

ticker = st.text_input('Etiqueta de cotización', 'AAPL')
st.write('La etiqueta de cotización actual es', ticker)
tic = yf.Ticker(ticker)
tic

hist = tic.history(period="max", auto_adjust=True)
hist

df = hist
df.info()

st.write("Mapa de calor de las variables")
import seaborn as sns
fig_ = plt.figure(1 , figsize = (17 , 8))
cor = sns.heatmap(df.corr(), annot = True)
st.pyplot(fig_)

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

LR = LinearRegression()
LR.fit(X_train,y_train)

df['Predicted_Signal'] = LR.predict(X)

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
plt.plot(df['Cum_Ret'], color='red')
plt.plot(df['Cum_Strategy'], color='black')
st.pyplot(fig)

st.write("Haz llegado hasta el final de esta sección. Gracias")