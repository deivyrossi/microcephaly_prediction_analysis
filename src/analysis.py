# analysis.py (MODELO V7.6 - Fixed Block Bootstrap)
"""
Script de análise final para o MODELO V7.
- Carrega o 'modelo_rf_final_v7.joblib'.
- Carrega os dados de teste JÁ PROCESSADOS (X, y, e DATAS).
- Roda SHAP, Permutation Importance, McNemar.
- Roda BLOCK BOOTSTRAP (Corrigido: Lógica estrita de MBB).
- Salva resultados em CSV/JSON e Gráficos.
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
import shap
import matplotlib
matplotlib.use('Agg') # Backend não-interativo para servidores/scripts
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
    classification_report,
    recall_score,
    precision_score,
    matthews_corrcoef,
    make_scorer
)
from sklearn.inspection import permutation_importance
from scipy.stats import binomtest

# -------------------------
# Configs / Paths
# -------------------------
DATA_DIR = 'dados/processados'
RESULTS_DIR = 'resultados_v7'
FIG_DIR = os.path.join(RESULTS_DIR, 'graficos')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')

# Paths dos dados (v7.4/v7.5 structure)
PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'PICO_test_X_v7.csv')
PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'PICO_test_y_v7.csv')
PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'PICO_test_datas_v7.csv')

FORA_PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_X_v7.csv')
FORA_PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_y_v7.csv')
FORA_PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_datas_v7.csv')

MODEL_PATH = 'modelo_rf_final_v7.joblib'
DT_BASELINE_PATH = 'modelo_dt_baseline_v7.joblib'
TRAIN_COLS_PATH = os.path.join(RESULTS_DIR, 'train_columns_v7.json')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

RANDOM_STATE = 42
POS_LABEL = 1

# -------------------------
# Utilidades
# -------------------------
def ensure_int_labels(y):
    if y.dtype == object:
        try:
            y = y.astype(int)
        except Exception:
            y = y.map(lambda v: int(str(v).strip()))
    if hasattr(y, 'values'):
        y_values = y.values
    else:
        y_values = y
    return pd.Series(y_values).astype(int)

def safe_pos_index(model, pos_label=POS_LABEL):
    if hasattr(model, 'classes_'):
        classes = list(model.classes_)
        if pos_label in classes:
            return classes.index(pos_label)
    try:
        classes_int = [int(c) for c in model.classes_]
        if pos_label in classes_int:
            return classes_int.index(pos_label)
    except Exception:
        pass 
    raise ValueError(f"Modelo não contém a classe {pos_label}")

# -------------------------
# Block Bootstrap CI (CORRIGIDO v7.6)
# -------------------------
def get_block_indices_corrected(n_samples, block_length):
    """
    Gera índices para Moving Block Bootstrap (MBB) de forma correta.
    Evita preenchimento aleatório (padding) que quebraria a estrutura temporal.
    
    Estratégia:
    1. Sorteia blocos contíguos suficientes para cobrir n_samples.
    2. Concatena os blocos.
    3. Trunca o excesso para voltar ao tamanho n_samples.
    """
    indices = np.arange(n_samples)
    
    # Quantos blocos precisamos para cobrir ou exceder n_samples?
    n_blocks_needed = int(np.ceil(n_samples / block_length))
    
    # Posições iniciais válidas (qualquer ponto onde um bloco de tamanho L caiba)
    max_start_index = n_samples - block_length + 1
    
    if max_start_index < 1:
        # Caso o dataset seja menor que o bloco (raro, mas possível em testes pequenos)
        # Nesse caso, faz resample simples com reposição (fallback)
        return np.random.choice(indices, size=n_samples, replace=True)

    # Sorteia os pontos de início dos blocos
    start_indices = np.random.randint(0, max_start_index, size=n_blocks_needed)
    
    # Constrói a lista de índices concatenando os blocos
    boot_indices = []
    for start in start_indices:
        block = indices[start : start + block_length]
        boot_indices.extend(block)
    
    # Trunca para o tamanho exato original (remove o excesso do último bloco)
    return np.array(boot_indices[:n_samples])

def calculate_block_bootstrap_ci(model, X, y, datas, block_length_days=28, n_iterations=1000, pos_label=POS_LABEL):
    print(f"Calculando CIs com BLOCK Bootstrap (v7.6 - Fixed) ({n_iterations} iterações)...")
    print(f"Janela de tempo do bloco: {block_length_days} dias.")
    
    stats = {'recall': [], 'precision': [], 'ap': [], 'mcc': []}
    
    # 1. Garante ordenação temporal (Crítico para Block Bootstrap)
    if isinstance(datas, pd.Series):
        datas = pd.to_datetime(datas)
        sort_idx = datas.argsort()
    else:
        sort_idx = np.argsort(datas)
        
    X_sorted = X.iloc[sort_idx].values
    y_sorted = y.iloc[sort_idx].values
    y_bin_sorted = (y_sorted == pos_label).astype(int)
    
    pos_idx = safe_pos_index(model)
    n_samples = len(X_sorted)
    
    # Calcula tamanho do bloco em amostras
    dias_totais = (datas.max() - datas.min()).days
    if dias_totais == 0: dias_totais = 1
    amostras_por_dia = n_samples / dias_totais
    block_length_n = int(amostras_por_dia * block_length_days)
    
    # Proteções de tamanho de bloco
    if block_length_n < 2: block_length_n = 2
    if block_length_n >= n_samples: block_length_n = n_samples // 2
        
    print(f"Tamanho do bloco calculado: {block_length_n} amostras")

    for i in range(n_iterations):
        # 1. Gera índices preservando blocos
        block_indices = get_block_indices_corrected(n_samples, block_length_n)
        
        # 2. Cria a amostra bootstrap
        X_res = X_sorted[block_indices]
        y_res = y_sorted[block_indices]
        y_bin_res = y_bin_sorted[block_indices]

        # Proteção contra amostras homogêneas (só uma classe)
        if len(np.unique(y_res)) < 2:
            continue 

        # 3. Predição e Métricas
        y_pred = model.predict(X_res)
        y_score = model.predict_proba(X_res)[:, pos_idx]

        stats['recall'].append(recall_score(y_res, y_pred, pos_label=pos_label, zero_division=0))
        stats['precision'].append(precision_score(y_res, y_pred, pos_label=pos_label, zero_division=0))
        stats['ap'].append(average_precision_score(y_bin_res, y_score))
        stats['mcc'].append(matthews_corrcoef(y_res, y_pred))

    # Calcula Percentis (IC 95%)
    cis = {}
    for key, values in stats.items():
        if not values: 
            cis[key] = (np.nan, np.nan)
        else:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            cis[key] = (lower, upper)

    print(f"CIs (95%): {cis}")
    return cis

# -------------------------
# Outras Funções de Análise (Inalteradas)
# -------------------------
def plot_pr_curve_for_dataset(model, X, y, label, savepath):
    pos_idx = safe_pos_index(model)
    y_bin = (y == POS_LABEL).astype(int)
    y_score = model.predict_proba(X)[:, pos_idx]
    precision, recall, thresholds = precision_recall_curve(y_bin, y_score)
    ap = average_precision_score(y_bin, y_score)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title(f'Precision-Recall ({label}) — AP={ap:.4f}')
    plt.grid(True); plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight'); plt.close()

def analyze_shap(model, X, label, savepath):
    print(f"Gerando SHAP para {label}...")
    explainer = shap.TreeExplainer(model)
    pos_idx = safe_pos_index(model)
    
    # Otimização: Usa apenas uma amostra do X se for muito grande (>2000) para background, 
    # mas calcula SHAP value para tudo ou subconjunto
    shap_values_output = explainer(X)
    
    shap_values_pos = None
    if len(shap_values_output.values.shape) == 3:
        shap_values_pos = shap_values_output.values[..., pos_idx]
    else:
        shap_values_pos = shap_values_output.values

    # Salva valores
    csv_path = savepath.replace('.png', '_values.csv')
    pd.DataFrame(shap_values_pos, columns=X.columns).to_csv(csv_path, index=False)
    
    # Plota
    shap.summary_plot(shap_values_pos, features=X, feature_names=X.columns, show=False)
    plt.title(f"SHAP Summary Plot ({label})")
    plt.tight_layout(); plt.savefig(savepath, dpi=300, bbox_inches='tight'); plt.close()

def permutation_importance_recall(model, X, y, n_repeats=10, savepath_png=None, savepath_csv=None):
    scorer = make_scorer(recall_score, pos_label=POS_LABEL)
    res = permutation_importance(model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    
    df = pd.DataFrame({'feature': X.columns, 'mean': res.importances_mean, 'std': res.importances_std})
    df = df.sort_values('mean', ascending=False)
    if savepath_csv: df.to_csv(savepath_csv, index=False)
    
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top['feature'], top['mean'], xerr=top['std'])
    plt.xlabel('Queda média no Recall'); plt.title('Permutation Importance (Recall)')
    plt.tight_layout()
    if savepath_png: plt.savefig(savepath_png, dpi=300, bbox_inches='tight'); plt.close()

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    b = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    c = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    n = b + c
    if n == 0: return {'p_value': 1.0, 'note': 'No disagreement'}
    p_value = binomtest(min(b, c), n=n, p=0.5, alternative='two-sided').pvalue
    return {'b': b, 'c': c, 'n': n, 'p_value': float(p_value)}

def threshold_sweep(model, X, y, savepath_csv):
    pos_idx = safe_pos_index(model)
    y_score = model.predict_proba(X)[:, pos_idx]
    y_bin = (y == POS_LABEL).astype(int)
    p, r, t = precision_recall_curve(y_bin, y_score)
    pd.DataFrame({'threshold': np.append(t, 1), 'precision': p, 'recall': r}).to_csv(savepath_csv, index=False)

# -------------------------
# MAIN
# -------------------------
def main():
    # 1. Load Model and Columns
    if not os.path.exists(MODEL_PATH):
        print("ERRO: Modelo não encontrado. Rode train.py primeiro.")
        return
    
    model = joblib.load(MODEL_PATH)
    with open(TRAIN_COLS_PATH, 'r') as f:
        train_cols = json.load(f)
        
    print("Modelo e colunas carregados.")

    # 2. Load Data
    try:
        # PICO
        X_pico = pd.read_csv(PICO_TEST_X_PATH, sep=';', encoding='latin1').reindex(columns=train_cols, fill_value=0)
        y_pico = ensure_int_labels(pd.read_csv(PICO_TEST_Y_PATH, sep=';', encoding='latin1').iloc[:,0])
        datas_pico = pd.to_datetime(pd.read_csv(PICO_TEST_DATAS_PATH, sep=';', encoding='latin1').iloc[:,0])
        
        # FORA
        X_fora = pd.read_csv(FORA_PICO_TEST_X_PATH, sep=';', encoding='latin1').reindex(columns=train_cols, fill_value=0)
        y_fora = ensure_int_labels(pd.read_csv(FORA_PICO_TEST_Y_PATH, sep=';', encoding='latin1').iloc[:,0])
        datas_fora = pd.to_datetime(pd.read_csv(FORA_PICO_TEST_DATAS_PATH, sep=';', encoding='latin1').iloc[:,0])
        
        print(f"Dados carregados: Pico={len(X_pico)}, Fora={len(X_fora)}")
    except Exception as e:
        print(f"ERRO ao carregar dados: {e}")
        return

    # 3. Metrics & Plots
    plot_pr_curve_for_dataset(model, X_pico, y_pico, "Pico (Teste v7.6)", os.path.join(FIG_DIR, 'pr_pico_v7.png'))
    plot_pr_curve_for_dataset(model, X_fora, y_fora, "Fora de Pico (v7.6)", os.path.join(FIG_DIR, 'pr_fora_v7.png'))
    
    analyze_shap(model, X_pico, "Pico (Teste v7.6)", os.path.join(FIG_DIR, 'shap_pico_v7.png'))
    analyze_shap(model, X_fora, "Fora de Pico (v7.6)", os.path.join(FIG_DIR, 'shap_fora_v7.png'))
    
    permutation_importance_recall(model, X_fora, y_fora, savepath_png=os.path.join(FIG_DIR, 'perm_importance_fora_v7.png'), savepath_csv=os.path.join(TABLE_DIR, 'perm_importance_fora_v7.csv'))
    threshold_sweep(model, X_fora, y_fora, os.path.join(TABLE_DIR, 'threshold_sweep_fora_v7.csv'))

    # 4. McNemar
    if os.path.exists(DT_BASELINE_PATH):
        dt_model = joblib.load(DT_BASELINE_PATH)
        m_res = mcnemar_test(y_fora.values, dt_model.predict(X_fora), model.predict(X_fora))
        print(f"McNemar Result: {m_res}")
        with open(os.path.join(TABLE_DIR, 'mcnemar_result_v7.json'), 'w') as f:
            json.dump(m_res, f)

    # 5. Block Bootstrap (CORRIGIDO)
    print("\n--- Iniciando Bootstrap ---")
    ci_pico = calculate_block_bootstrap_ci(model, X_pico, y_pico, datas_pico)
    ci_fora = calculate_block_bootstrap_ci(model, X_fora, y_fora, datas_fora)
    
    cis = {'pico': ci_pico, 'fora': ci_fora}
    with open(os.path.join(TABLE_DIR, 'bootstrap_cis_v7.json'), 'w') as f:
        json.dump(cis, f)
        
    print("Análise Concluída.")

if __name__ == "__main__":
    main()