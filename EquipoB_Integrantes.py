import streamlit as st

st.set_page_config(
    page_title="Trabajo Grupal Semana 14",
)

st.write("# Despliegue web de modelos del Grupo B 2023-1")
st.write("# Curso: Inteligencia de Negocios")
st.write("# Prof: Ernesto Cancho Rodriguez")

st.sidebar.success("Seleccione un modelo del menu")

st.markdown(
    """
    ### Integrantes:
    
    - Cordova Sandoval Rafael - 17200268  (Despliegue)
    - Jimenez Castaneda, Luis Francisco - 15200213  
    - Del Aguila Febres Brayan - 17200270  
    - Caceres Estana Juan Alfonso - 19200288   
    - Ambrocio Milla Katherine Celine - 18200324  
    - Rios Sanchez Anthony Ulises - 19200099  
    - Hidalgo Diaz Sebastian Eduardo - 18200082  

    ### Especificaciones:
    **Donde muestra las predicciones/los resultados:**
    - Graficamente. 
    - Numericamente los valores de las predicciones (print de dataframe con la prediccion o clasificacion).
    
    **Donde se muestra el EDA:**
    - Ploteo de los precios reales.
    (Ploteo de media movil los precios reales.)

    **Donde el usuario pueda indicar:**
    - El modelo ejecutar.
    - La accion o instrumento financiero que quiera analizar.
    """
)
