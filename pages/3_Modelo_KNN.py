from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import yfinance as yf
plt.style.use('seaborn-darkgrid')
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report,confusion_matrix
import streamlit as st


st.set_page_config(page_title="KNN")

st.markdown("# KNN")
st.sidebar.header("KNN")
st.write(
    """El contenido de la página permite visualizar resultados de predicción de precios de acciones utilizando el modelo KNN."""
)

ticker1 = st.text_input('Etiqueta de cotización', 'INTC')
st.write('La etiqueta de cotización actual es', ticker1)
intc = yf.Ticker(ticker1)
# Opciones para el radio button
opciones = ['Usar Todo el Registo', 'Seleccionar Rango de Fechas']

# Crear el radio button para seleccionar una opción
opcion_seleccionada = st.radio('Selecciona una opción', opciones)

# Verificar la opción seleccionada
if opcion_seleccionada == 'Usar Todo el Registo':
    hist = intc.history(period="max", auto_adjust=True)
    st.write("Datos Completos:")
elif opcion_seleccionada == 'Seleccionar Rango de Fechas':
    # Apartado para enviar un dato
    from datetime import datetime
    start_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2022-12-31', '%Y-%m-%d').date()
    st.write("Primero Seleccionalos el rango de fechas de los datos a usar:")
    Finit = st.date_input("Ingrese una Fecha de Inicio:",value = start_date)
    Fend = st.date_input("Ingrese una Fecha de Final:",value = end_date)
    hist = intc.history(start=Finit, end=Fend, auto_adjust=True)
    st.write("Datos en el rango seleccionado:")
    
hist.head()
df = hist

df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

X = df[['Open-Close', 'High-Low']]
X.head()

y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.7
split = int(split_percentage*len(df))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

print("Predicciones del clasificador:")
# test_data_predicted = knn.predict(X_test)
# print(test_data_predicted)
# st.write(knn.predict(X_test))
# print("Resultados esperados:")
# print(y_test)
# st.write(y_test)

# Datos predecidos 
st.write("Dataframe con los resultados predecidos")
df['Predicted_Signal'] = knn.predict(X)
df

# print(accuracy_score(knn.predict(X_test), y_test))
# Precisión del modelo
st.write("Precisión del modelo")
st.write(accuracy_score(knn.predict(X_test), y_test))


st.write("Grafica de la tasa de error vs. valor de K")
tasa_error = []
for i in range(1,40):
  knn_g = KNeighborsClassifier(n_neighbors=i)
  knn_g.fit(X_train,y_train)
  pred_i = knn_g.predict(X_test)
  tasa_error.append(np.mean(pred_i != y_test))

fig = plt.figure(figsize=(10,6),dpi=250)
plt.plot(range(1,40),tasa_error,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Tasa de Error vs. Valor de K')
plt.xlabel('K')
plt.ylabel('Tasa de Error')
st.pyplot(fig)

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('CON K=19')
print(classification_report(y_test,pred))

st.markdown(
    """
    ### Recomendaciones:
    - Conocer el concepto básico de KNN y cómo se utilizan para la clasificación de datos.
    - Elegir un conjunto de datos adecuado para el modelo y evaluar los resultados.
    - Dividir el conjunto de datos en conjuntos de entrenamiento y prueba para evaluar el desempeño.
    - Asegurarse de ajustar los parámetros del modelo para obtener el mejor rendimiento.
    - Evaluar el rendimiento del modelo utilizando métricas de clasificación.
    - Si el modelo no funciona bien, puede probar con otros parametros o un conjunto de datos diferentes.
"""
)

st.write("Haz llegado hasta el final de esta sección. Gracias")