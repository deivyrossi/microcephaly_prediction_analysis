# utils.py
import os
import joblib

def ensure_dirs():
    os.makedirs('resultados/graficos', exist_ok=True)
    os.makedirs('resultados/tables', exist_ok=True)

def save_model(model, path='modelo_rf_final.joblib'):
    joblib.dump(model, path)

def load_model(path='modelo_rf_final.joblib'):
    return joblib.load(path)
