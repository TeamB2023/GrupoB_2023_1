from datetime import datetime, timedelta, timezone, date
from sklearn.metrics import precision_score
import os
import yfinance as yf
import warnings
import streamlit as st
import pytz

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


# To ignore warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Random Forest")

st.markdown("# Random Forest")
st.sidebar.header("Random Forest")
st.write(
    """En esta página podrás ver cómo funciona el modelo Random Forest en la predicción del mercado de valores"""
)

st.write("¿Qué es? El índice Standard & Poor's 500, también conocido como S&P 500, es uno de los índices bursátiles más importantes de Estados Unidos, y se considera el índice más representativo de la situación real del mercado.")

if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)
sp500

sp500.plot.line(y="Close", use_index=True)

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)

st.write("El objetivo es lograr precidir si el día de mañana, el precio de la acción subirá o bajará")
st.write("Creamos una columna Tomorrrow (Mañana) que refiere al precio de la acción al día siguiente. Con la función shift de Pandas, lo que haremos es correr 1 día el precio de Cierre, y lo trasladamos al día anterior")
sp500

st.write("Ahora establecemos una columna Target, donde validaremos si el precio de nuestra colunma Tomorrow es mayor al precio de cierre. La respuesta será en forma de entero. Visualizaremos nuevamente el dataframe después de este paso.")
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500

st.write("La columna de Target devuelve valores de 0 y 1, donde 1 significa que el precio de la acción de mañana es mayor al precio de cierre de la acción de hoy.")

st.write("Ahora, vamos a considerar datos a partir del año 1990, debido a que en el mercado de valores, tener data de registros muy antiguos pueden ser contraproducentes, puesto que pudieron existir cambios significantes en el mercado fundalmentalmente.")
start_date = datetime(1990, 1, 1, 0, 0, tzinfo=pytz.utc)
today = datetime.now().replace(tzinfo=pytz.UTC)
sp500 = sp500.loc[start_date:today].copy()
sp500


# elegimos como parámetro tener por lo menos 100 árboles de decisiones
model = RandomForestClassifier(
    n_estimators=100, min_samples_split=100, random_state=1)

# creamos un set de entrenamiento y un set de prueba
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Seleccionamos las variables predictoras
predictors = ["Close", "Volume", "Open", "High", "Low"]
# entrenamos nuestro modelo
model.fit(train[predictors], train["Target"])


preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
p_score1 = precision_score(test["Target"], preds)

st.write("Nuestro puntaje de precisión es", p_score1)

st.write("Realizamos una gráfica con plot para comparar nuestra predicción con el Target actual")

combined = pd.concat([test["Target"], preds], axis=1)
f = combined.plot()
graph = f.figure
st.pyplot(graph)


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


predictions = backtest(sp500, model, predictors)

st.subheader("Backtesting")
st.write("Veamos cuántos días logró predecir nuestro modelo (de que iba a subir y/O bajar el precio del SP500)")
st.table(predictions["Predictions"].value_counts())

st.write("Veamos el puntaje de precisión del modelo")
p_scorem = precision_score(predictions["Target"], predictions["Predictions"])
st.write("Nuestro puntaje de precisión del modelo es", p_scorem)

st.write("Tenemos una mejor precisión que con el anterior modelo")

st.write("Vemos el porcentaje que el modelo predijo que iba a subir y/o bajar el precio del SP500")
st.table(predictions["Target"].value_counts() / predictions.shape[0])

st.subheader("Añadiendo predictores adicionales al modelo desarrollado")
st.write("Calcularemos el precio promedio de cierre en los últimos 2 días, 5 días, 3 meses. el último año y los últimos 4 años.  Luego buscaremos el radio entre el precio de cierre de hoy con el precio de cierre de estos periodos elegidos. Estos nos ayudará a saber si el precio de mercado ha subido considerablemente o no.")
horizons = [2, 5, 60, 250, 1000]

# creamos nuevos predictores
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

# vemos nuevamente el dataset
sp500

st.write("¿Por qué tenemos bastantes datos nulos? Si el modelo no encuentra suficientes días para realizar el rolling average, simplmemente colocará valores nulos")
st.write("Así que, descartamos esos vacíos en las columnas")

sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
sp500

st.subheader("Mejorando nuestro modelo")
st.write("Vamos a cambiar algunos valores de los parámetros para el RandomForestClassifier")
model = RandomForestClassifier(
    n_estimators=200, min_samples_split=50, random_state=1)


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


predictions = backtest(sp500, model, new_predictors)
st.write("Veamos cuántos días logró predecir nuestro modelo (de que iba a subir y/O bajar el precio del SP500)")
st.table(predictions["Predictions"].value_counts())

st.write("Volvemos a ver la precisión del modelo con los cambios realizados")
p_scoremn = precision_score(predictions["Target"], predictions["Predictions"])
st.write("Nuestro puntaje de precisión del modelo es", p_scoremn)

st.write("Vemos el porcentaje que logró predecir el modelo de que el precio iba a subir y/o bajar")
st.table(predictions["Target"].value_counts() / predictions.shape[0])

st.write("Haz llegado hasta el final de esta sección. Gracias")