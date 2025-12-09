# src/sensitivity_analysis.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score
from analysis import calculate_block_bootstrap_ci, DATA_DIR, RESULTS_DIR, MODEL_PATH, PICO_TEST_X_PATH, PICO_TEST_Y_PATH, PICO_TEST_DATAS_PATH, FORA_PICO_TEST_X_PATH, FORA_PICO_TEST_Y_PATH, FORA_PICO_TEST_DATAS_PATH, TRAIN_COLS_PATH, ensure_int_labels
import json

def main():
    print("--- Iniciando Análise de Sensibilidade do Bootstrap ---")
    
    # Carrega modelo e colunas
    model = joblib.load(MODEL_PATH)
    with open(TRAIN_COLS_PATH, 'r') as f:
        train_cols = json.load(f)

    # Carrega dados
    try:
        X_pico = pd.read_csv(PICO_TEST_X_PATH, sep=';', encoding='latin1').reindex(columns=train_cols, fill_value=0)
        y_pico = ensure_int_labels(pd.read_csv(PICO_TEST_Y_PATH, sep=';', encoding='latin1').iloc[:,0])
        datas_pico = pd.to_datetime(pd.read_csv(PICO_TEST_DATAS_PATH, sep=';', encoding='latin1').iloc[:,0])
        
        X_fora = pd.read_csv(FORA_PICO_TEST_X_PATH, sep=';', encoding='latin1').reindex(columns=train_cols, fill_value=0)
        y_fora = ensure_int_labels(pd.read_csv(FORA_PICO_TEST_Y_PATH, sep=';', encoding='latin1').iloc[:,0])
        datas_fora = pd.to_datetime(pd.read_csv(FORA_PICO_TEST_DATAS_PATH, sep=';', encoding='latin1').iloc[:,0])
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    block_sizes = [7, 14, 28, 56]
    results = []

    print(f"\nTestando tamanhos de bloco: {block_sizes}")
    
    for days in block_sizes:
        print(f"\nProcessing Block Size: {days} days...")
        
        # Pico
        ci_pico = calculate_block_bootstrap_ci(model, X_pico, y_pico, datas_pico, block_length_days=days, n_iterations=1000)
        pico_recall_lower, pico_recall_upper = ci_pico['recall']
        
        # Fora
        ci_fora = calculate_block_bootstrap_ci(model, X_fora, y_fora, datas_fora, block_length_days=days, n_iterations=1000)
        fora_recall_lower, fora_recall_upper = ci_fora['recall']
        
        results.append({
            'Bloco (Dias)': days,
            'Pico Recall IC (95%)': f"{pico_recall_lower:.3f} - {pico_recall_upper:.3f}",
            'Fora Recall IC (95%)': f"{fora_recall_lower:.3f} - {fora_recall_upper:.3f}"
        })

    df_res = pd.DataFrame(results)
    print("\n--- Resultados da Análise de Sensibilidade ---")
    print(df_res)
    df_res.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_analysis.csv'), index=False)

if __name__ == "__main__":
    main()