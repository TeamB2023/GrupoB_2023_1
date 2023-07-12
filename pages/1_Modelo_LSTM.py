import streamlit as st
# Título de la aplicación
st.title("MODELO LSTM")
# Descripción del modelo LSTM
st.markdown("""
Haremos una LSTM muy simple con Keras para predecir el precio de las acciones de una etiqueta de cotizacion
            """)
st.write('Escriba una etiqueta de cotizacion')
ticker = st.text_input('Etiqueta de cotización', 'MSFT')
st.write('La etiqueta de cotización actual es', ticker)
#Cargamos los Datos
import yfinance as yf
brk = yf.Ticker(ticker)
# Opciones para el radio button
opciones = ['Usar Todo el Registo', 'Seleccionar Rango de Fechas']

# Crear el radio button para seleccionar una opción
opcion_seleccionada = st.radio('Selecciona una opción', opciones)

# Verificar la opción seleccionada
if opcion_seleccionada == 'Usar Todo el Registo':
    dataset = brk.history(period="max", auto_adjust=True)
    st.write("Datos Completos:")
elif opcion_seleccionada == 'Seleccionar Rango de Fechas':
    # Apartado para enviar un dato
    from datetime import datetime
    start_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2022-12-31', '%Y-%m-%d').date()
    st.write("Primero Seleccionalos el rango de fechas de los datos a usar:")
    Finit = st.date_input("Ingrese una Fecha de Inicio:",value = start_date)
    Fend = st.date_input("Ingrese una Fecha de Final:",value = end_date)
    dataset = brk.history(start=Finit, end=Fend, auto_adjust=True)
    st.write("Datos en el rango seleccionado:")
#Codigo 
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import warnings

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

warnings.filterwarnings("ignore")
style.use('ggplot')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funciones auxiliares

def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()
# Mostrar Datos
dataset
dataset.describe()

"""**Dividimos los datos en train y test**"""

# Sets de entrenamiento y validación (test)
# La LSTM se entrenará con datos de 2019 hacia atrás. La validación se hará con datos de 2020 en adelante.
# En ambos casos sólo se usará el valor más Alto de la acción para cada día
#
set_entrenamiento = dataset[:'2019'].iloc[:,1:2]
set_validacion = dataset['2020':].iloc[:,1:2]
# Crear la figura y los ejes para la gráfica
fig, ax = plt.subplots()
set_entrenamiento['High'].plot(legend=True)
set_validacion['High'].plot(legend=True)
# Configurar leyendas y título
plt.legend(['Entrenamiento ( -2019)', 'Validación (2020- )'])
plt.title('División de datos en entrenamiento y validación')

# Mostrar la gráfica en Streamlit
st.pyplot(fig)

"""**Normalización del set de entrenamiento**"""

# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida un dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

"""**Red LSTM**"""

# Definir el modelo LSTM y compilarlo
dim_entrada = (X_train.shape[1], 1)
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X_train, Y_train, epochs=20, batch_size=32)

# Realizar predicciones en los datos de validación
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step, len(x_test)):
    X_test.append(x_test[i-time_step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
fig, ax = plt.subplots()
ax.plot(set_validacion.values, label='Valores reales')
ax.plot(prediccion, label='Predicciones')
ax.legend()
st.pyplot(fig)

import math
trainScore = modelo.evaluate(X_train, Y_train, verbose=0)
train_mse = trainScore[0]
train_rmse = math.sqrt(trainScore[0])

st.write(f"Train Score: {train_mse:.2f} MSE ({train_rmse:.2f} RMSE)")
