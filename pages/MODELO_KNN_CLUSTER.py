import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Configurar página de Streamlit
st.set_page_config(page_title="KNN with Clustering")

# Título y descripción
st.markdown("# KNN with Clustering")
st.write(
    """Este es un modelo de KNN con clustering aplicado a datos históricos de acciones."""
)

# Entradas del usuario
ticker_symbols = st.text_input(
    "Símbolos de las acciones (separados por coma)", "AAPL,MSFT,GOOGL"
)
start_date = st.text_input("Fecha de inicio (formato: AAAA-MM-DD)", "2018-01-01")
end_date = st.text_input("Fecha de fin (formato: AAAA-MM-DD)", "2022-12-31")
k = st.text_input("Valor de K (número de vecinos)", "5")

# Obtener símbolos de acciones ingresados por el usuario
symbols = [symbol.strip() for symbol in ticker_symbols.split(",")]

# Obtener datos históricos de las acciones
df = pd.DataFrame()

data = []
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    ticker_history = ticker.history(start=start_date, end=end_date)
    ticker_selection = ticker_history.drop(
        ["Dividends", "Stock Splits", "Volume"], axis=1
    )
    ticker_suffix = ticker_selection.add_suffix(f"_{symbol}")
    data.append(ticker_suffix)

df = pd.concat(data, axis=1)

# Calcular los retornos diarios de los precios de cierre
returns_cols = [col for col in df.columns if col.startswith("Close_")]
df[returns_cols] = df[returns_cols].pct_change()
df.dropna(inplace=True)

# Definir las variables predictoras (X) y la variable objetivo (y)
X = df[returns_cols]
y = df[returns_cols].idxmax(axis=1)  # Clasificar según el símbolo de cierre más alto

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar el modelo KNN
k_value = int(k)  # Convertir el valor de K a entero
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test)

# Evaluar el modelo
accuracy = knn.score(X_test, y_test)

# Mostrar resultados
st.write("Exactitud del modelo:", accuracy)

##### Gráfica de tasa de error vs. valor de K #####
###################################################

tasa_error = []
for i in range(1, 40):
    knn_g = KNeighborsClassifier(n_neighbors=i)
    knn_g.fit(X_train, y_train)
    pred_i = knn_g.predict(X_test)
    tasa_error.append(np.mean(pred_i != y_test))

fig = plt.figure(figsize=(10, 6), dpi=250)
plt.plot(
    range(1, 40),
    tasa_error,
    color="blue",
    linestyle="dashed",
    marker="o",
    markerfacecolor="red",
    markersize=10,
)
plt.title("Tasa de Error vs. Valor de K")
plt.xlabel("K")
plt.ylabel("Tasa de Error")
st.pyplot(fig)

###### Gráfico de barras (histograma) de accuracy en función de k #####
#######################################################################

acc = []
ks = []

for i in range(1, 200):
    ks.append(i)
    _knn = KNeighborsClassifier(n_neighbors=i)
    _knn.fit(X_train, y_train)

    _y_pred = _knn.predict(X_test)
    acc.append(_knn.score(X_test, y_test))

data = {"K": ks, "Accuracy": acc}
df_acc = pd.DataFrame(data)

fig = plt.figure(figsize=(10, 6))
plt.bar(df_acc["K"], df_acc["Accuracy"])
plt.xlabel("Numero de vecinos (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy del modelo en función de k")

st.pyplot(fig)


###### Grafico de dispersion con vecinos cercanos #####
#######################################################

# Crear una figura y los ejes para los sub-gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Sub-gráfico 1: Puntos de datos por etiqueta
for label in set(y_pred):
    label_points = X_test[y_pred == label]
    x_points = label_points.iloc[:, 0]
    y_points = label_points.iloc[:, 1]
    axes[0].scatter(x_points, y_points, label=label)
axes[0].set_xlabel("Valor X")
axes[0].set_ylabel("Valor Y")
axes[0].set_title("Puntos de datos por etiqueta")
axes[0].legend()

# Sub-gráfico 2: Punto de prueba desconocido
latest_data = df.tail(1)
x_unknown = [latest_data["Close_AAPL"].values[0], latest_data["Close_MSFT"].values[0]]
axes[1].scatter(x_unknown[0], x_unknown[1], color="red", label="Desconocido")
axes[1].set_xlabel("Valor X")
axes[1].set_ylabel("Valor Y")
axes[1].set_title("Punto de prueba desconocido")
axes[1].legend()

# Sub-gráfico 3: Vecinos cercanos
distances = [
    (i, ((x[0] - x_unknown[0]) ** 2 + (x[1] - x_unknown[1]) ** 2) ** 0.5)
    for i, x in enumerate(X_test.values)
]
distances.sort(key=lambda x: x[1])
nearest_neighbors_indices = [idx for idx, _ in distances[:k_value]]
colors = ["red", "green", "blue", "yellow", "purple"]
color_iterator = iter(colors)
for idx in nearest_neighbors_indices:
    x_neighbor = X_test.iloc[idx, 0]
    y_neighbor = X_test.iloc[idx, 1]
    axes[2].plot(
        [x_unknown[0], x_neighbor],
        [x_unknown[1], y_neighbor],
        color="gray",
        linestyle="dashed",
    )
    color = next(color_iterator)
    axes[2].scatter(
        x_neighbor, y_neighbor, color=color, label=f"Vecino de índice {idx}"
    )
axes[2].scatter(
    x_unknown[0], x_unknown[1], color="fuchsia", label="Punto de prueba desconocido"
)
axes[2].set_xlabel("Valor X")
axes[2].set_ylabel("Valor Y")
axes[2].set_title("Vecinos cercanos")
axes[2].legend()

# Ajustar espaciado entre sub-gráficos
plt.tight_layout()

# Mostrar los sub-gráficos en Streamlit
st.pyplot(fig)

##### Recomendaciones #####
###########################

st.markdown("### Recomendaciones:")
st.write(
    "- Conoce el concepto básico de KNN y cómo se utiliza para la clasificación de datos."
)
st.write(
    "- Elige un conjunto de datos adecuado para el modelo y evalúa los resultados."
)
st.write(
    "- Divide el conjunto de datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento."
)
st.write(
    "- Asegúrate de ajustar los parámetros del modelo para obtener el mejor rendimiento."
)
st.write("- Evalúa el rendimiento del modelo utilizando métricas de clasificación.")
st.write(
    "- Si el modelo no funciona bien, puedes probar con otros parámetros o un conjunto de datos diferente."
)

st.write(
    "¡Has llegado al final de esta sección! ¡Gracias por utilizar KNN with Clustering en Streamlit!"
)
