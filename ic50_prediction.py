import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Fungsi sigmoid untuk memodelkan dosis-respons
def sigmoid(concentration, ic50, hill_slope, bottom, top):
    return bottom + (top - bottom) / (1 + (concentration / ic50)**hill_slope)

# Judul aplikasi
st.title("IC50count - Aplikasi Analisis Dosis-Respon")

# Unggah file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = data['Concentration'].values
    Y = data['Cell_Viability'].values
    
    # Perkiraan awal
    initial_guesses = [10, 1, min(Y), max(Y)]
    
    # Fit data
    params, covariance = curve_fit(sigmoid, X, Y, p0=initial_guesses)
    ic50_value = params[0]
    hill_slope = params[1]
    Y_pred = sigmoid(X, *params)
    
    # Hitung R-squared
    ss_res = np.sum((Y - Y_pred)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Tampilkan hasil IC50 dan R-squared
    st.write(f"IC50: {ic50_value:.2f} µM")
    st.write(f"R²: {r_squared:.2f}")
    
    # Buat grafik
    fig, ax = plt.subplots()
    ax.scatter(X, Y, label='Data Asli', color='blue')
    X_fit = np.linspace(min(X), max(X), 100)
    Y_fit = sigmoid(X_fit, *params)
    ax.plot(X_fit, Y_fit, label='Kurva Fit Sigmoid', color='red')
    ax.set_xscale('log')
    ax.set_xlabel('Konsentrasi (µM)')
    ax.set_ylabel('Viabilitas Sel (%)')
    ax.set_title(f'Kurva Dosis-Respons (IC50: {ic50_value:.2f} µM, R²: {r_squared:.2f})')
    ax.legend()
    
    # Tampilkan grafik di Streamlit
    st.pyplot(fig)
