import streamlit as st
import numpy as np

st.title("🧠 Calculadora de Neuronas Artificiales")
st.write("Esta herramienta permite calcular la salida de una neurona con función de activación sigmoide logística.")

st.header("🔢 Entradas y Pesos")

# Número de entradas
num_inputs = st.slider("Selecciona el número de entradas (sin contar el sesgo):", 1, 10, 4)

# Crear sliders para entradas y pesos
inputs = []
weights = []

for i in range(num_inputs):
    xi = st.number_input(f"Entrada x{i+1}", value=0.0, key=f"x{i}")
    wi = st.number_input(f"Peso w{i+1}", value=0.0, key=f"w{i}")
    inputs.append(xi)
    weights.append(wi)

# Sesgo y su peso
st.subheader("⚙️ Sesgo")
bias_value = 1.0
bias_weight = st.number_input("Peso del sesgo (bias)", value=0.0, key="bias")

# Agregar el sesgo como entrada fija
inputs.append(bias_value)
weights.append(bias_weight)

# Cálculo del neto y la salida
net_input = np.dot(inputs, weights)
output = 1 / (1 + np.exp(-net_input))

st.header("📤 Resultado")
st.write(f"**Entrada neta (net):** {net_input:.4f}")
st.write(f"**Salida de la neurona (y):** {output:.4f}")
