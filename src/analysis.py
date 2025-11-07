# analysis.py (MODELO V7.2 - Salvando SHAP values)
"""
Script de análise final para o MODELO V7.
- Carrega o 'modelo_rf_final_v7.joblib'.
- Carrega os dados de teste JÁ PROCESSADOS (X, y, e DATAS).
- Roda SHAP, Permutation Importance, McNemar.
- Roda BLOCK BOOTSTRAP para CIs.
- v7.2: Salva os SHAP VALUES em .csv (Ponto 9)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- INÍCIO DA CORREÇÃO DO 'main thread' ---
import matplotlib
matplotlib.use('Agg')
# --- FIM DA CORREÇÃO ---

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
import os
import shap
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    matthews_corrcoef # Importado
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample # Removido para o Block Bootstrap
from scipy.stats import binomtest

# -------------------------
# Configs / paths (ATUALIZADOS PARA V7)
# -------------------------
DATA_DIR = 'dados/processados'

# --- Paths do Modelo V7 ---
PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'PICO_test_X_v7.csv')
PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'PICO_test_y_v7.csv')
PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'PICO_test_datas_v7.csv') # NOVO

FORA_PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_X_v7.csv')
FORA_PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_y_v7.csv')
FORA_PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_datas_v7.csv') # NOVO

# Paths dos modelos e resultados
MODEL_PATH = 'modelo_rf_final_v7.joblib'
DT_BASELINE_PATH = 'modelo_dt_baseline_v7.joblib'
RESULTS_DIR = 'resultados_v7' 
TRAIN_COLS_PATH = os.path.join(RESULTS_DIR, 'train_columns_v7.json')
# -------------------------

FIG_DIR = os.path.join(RESULTS_DIR, 'graficos')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

RANDOM_STATE = 42
POS_LABEL = 1  # classe positiva (Confirmado)

# -------------------------
# Utilidades
# -------------------------
def ensure_int_labels(y):
    """Garante que labels são inteiros (1,5) em vez de strings."""
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
    """Retorna o índice da coluna de predict_proba que corresponde à classe pos_label."""
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
        
    raise ValueError(f"Modelo não contém a classe {pos_label} em model.classes_ (Classes: {model.classes_})")

# -------------------------
# PR curve + AP
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
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    return precision, recall, thresholds, ap

# -------------------------
# permutation importance (scoring = recall da classe 1)
# -------------------------
def permutation_importance_recall(model, X, y, n_repeats=30, top_k=15, savepath_png=None, savepath_csv=None):
    scorer = make_scorer(recall_score, pos_label=POS_LABEL)
    res = permutation_importance(model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    imp_mean = res.importances_mean
    imp_std = res.importances_std
    features = X.columns
    df = pd.DataFrame({'feature': features, 'mean': imp_mean, 'std': imp_std})
    df = df.sort_values('mean', ascending=False).reset_index(drop=True)
    if savepath_csv:
        df.to_csv(savepath_csv, index=False)
    
    top = df.head(top_k).copy()
    top = top[::-1]  
    plt.figure(figsize=(10, max(3, top_k*0.3)))
    
    plt.barh(top['feature'], top['mean'], xerr=top['std'])
    
    plt.xlabel('Queda média no Recall (importância por permutação)')
    plt.title('Permutation Importance (top {})'.format(top_k))
    plt.tight_layout()
    if savepath_png:
        plt.savefig(savepath_png, dpi=300, bbox_inches='tight')
    plt.close()
    return df

# -------------------------
# Block Bootstrap CI (NOVA FUNÇÃO)
# -------------------------
def get_block_indices(n_samples, block_length, overlap=True):
    """Gera índices para o Moving Block Bootstrap."""
    indices = np.arange(n_samples)
    n_blocks = n_samples - block_length + 1 if overlap else n_samples // block_length
    
    # Sorteia os pontos de início dos blocos com reposição
    start_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
    
    # Cria os blocos
    all_indices = []
    for start in start_indices:
        if overlap:
            block = indices[start : start + block_length]
        else:
            block = indices[start * block_length : (start + 1) * block_length]
        all_indices.extend(block)
        
    # Trunca ou preenche para bater o n_samples original
    if len(all_indices) > n_samples:
        return all_indices[:n_samples]
    elif len(all_indices) < n_samples:
        # Preenche com sorteio aleatório
        padding = np.random.choice(indices, size=(n_samples - len(all_indices)), replace=True)
        all_indices.extend(padding)
    
    return all_indices

def calculate_block_bootstrap_ci(model, X, y, datas, block_length_days=28, n_iterations=1000, pos_label=POS_LABEL):
    """
    Calcula CIs usando Moving Block Bootstrap (MBB).
    'datas' é a Série de Timestamps usada para ordenar.
    'block_length_days' é o tamanho do bloco (ex: 28 dias / 4 semanas).
    """
    print(f"Calculando CIs com BLOCK Bootstrap ({n_iterations} iterações)... (Isso pode demorar)")
    print(f"Tamanho do bloco: {block_length_days} dias.")
    stats = {'recall': [], 'precision': [], 'ap': [], 'mcc': []}
    
    # 1. Ordena os dados por data (ESSENCIAL para o block bootstrap)
    sort_idx = datas.argsort()
    X_sorted = X.iloc[sort_idx].values
    y_sorted = y.iloc[sort_idx].values
    
    y_bin_sorted = (y_sorted == pos_label).astype(int)
    pos_idx = safe_pos_index(model)
    n_samples = len(X_sorted)
    
    # Calcula o tamanho do bloco em NÚMERO DE AMOSTRAS
    dias_totais = (datas.max() - datas.min()).days
    if dias_totais == 0: dias_totais = 1 # Evita divisão por zero
    amostras_por_dia = n_samples / dias_totais
    block_length_n = int(amostras_por_dia * block_length_days)
    if block_length_n < 2: block_length_n = 2 # Garante um bloco mínimo
    if block_length_n >= n_samples: block_length_n = n_samples // 2 # Garante que o bloco não é maior que os dados
        
    print(f"Tamanho do bloco (em amostras): {block_length_n}")

    for i in range(n_iterations):
        # 1. Pega os índices dos blocos (com reposição)
        block_indices = get_block_indices(n_samples, block_length_n, overlap=True)
        
        # 2. Cria a amostra-bloco
        X_res = X_sorted[block_indices]
        y_res = y_sorted[block_indices]
        y_bin_res = y_bin_sorted[block_indices]

        if len(np.unique(y_res)) < 2:
            continue 

        # 3. Calcula métricas
        y_pred = model.predict(X_res)
        y_score = model.predict_proba(X_res)[:, pos_idx]

        stats['recall'].append(recall_score(y_res, y_pred, pos_label=pos_label, zero_division=0))
        stats['precision'].append(precision_score(y_res, y_pred, pos_label=pos_label, zero_division=0))
        stats['ap'].append(average_precision_score(y_bin_res, y_score))
        stats['mcc'].append(matthews_corrcoef(y_res, y_pred))

    cis = {}
    for key, values in stats.items():
        if not values: 
            cis[key] = (np.nan, np.nan)
        else:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            cis[key] = (lower, upper)

    print(f"CIs (95% Block Bootstrap): {cis}")
    return cis

# -------------------------
# SHAP analysis (ATUALIZADO Ponto 9)
# -------------------------
def analyze_shap(model, X, label, savepath):
    """
    Calcula, salva (CSV) e plota o summary_plot do SHAP.
    """
    print(f"Calculando SHAP values para {label}... (Isso pode demorar)")
    
    explainer = shap.TreeExplainer(model)
    pos_idx = safe_pos_index(model)
    shap_values_output = explainer(X)
    
    shap_values_pos = None
    try:
        if isinstance(shap_values_output, list):
            shap_values_pos = shap_values_output[pos_idx]
        elif hasattr(shap_values_output, 'values'):
            if len(shap_values_output.values.shape) == 3:
                shap_values_pos = shap_values_output.values[..., pos_idx]
            elif len(shap_values_output.values.shape) == 2:
                 shap_values_pos = shap_values_output.values
            else:
                raise ValueError(f"Shape inesperado do SHAP: {shap_values_output.values.shape}")
        else:
             raise TypeError(f"Tipo inesperado do SHAP: {type(shap_values_output)}")

    except Exception as e:
        print(f"Erro ao extrair SHAP values: {e}")
        try:
            if pos_idx == 1 and isinstance(shap_values_output, list) and len(shap_values_output) == 2:
                print("Fallback: Usando índice 1 da lista SHAP.")
                shap_values_pos = shap_values_output[1]
            elif pos_idx == 0 and isinstance(shap_values_output, list) and len(shap_values_output) == 2:
                print("Fallback: Usando índice 0 da lista SHAP.")
                shap_values_pos = shap_values_output[0]
            else:
                raise e
        except Exception as fallback_e:
            print(f"Fallback falhou: {fallback_e}")
            raise e

    if shap_values_pos is None:
         raise ValueError("Não foi possível extrair os SHAP values para a classe positiva.")

    # --- INÍCIO DA ADIÇÃO (Ponto 9) ---
    try:
        shap_df = pd.DataFrame(shap_values_pos, columns=X.columns)
        # Garante que o path termina em .png, e o substitui por .csv
        if savepath.endswith('.png'):
            shap_df_path = savepath.replace('.png', '_values.csv')
        else:
            shap_df_path = savepath + '_values.csv'
            
        shap_df.to_csv(shap_df_path, index=False, encoding='utf-8-sig')
        print(f"Valores SHAP salvos em: {shap_df_path}")
    except Exception as e:
        print(f"AVISO: Falha ao salvar valores SHAP em CSV: {e}")
    # --- FIM DA ADIÇÃO ---

    print(f"Gerando SHAP summary plot para {label}...")
    
    shap.summary_plot(shap_values_pos, features=X, feature_names=X.columns, show=False)
    
    plt.title(f"SHAP Summary Plot ({label})")
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico SHAP salvo em: {savepath}")


# -------------------------
# threshold sweep
# -------------------------
def threshold_sweep(model, X, y, savepath_csv=None):
    pos_idx = safe_pos_index(model)
    y_score = model.predict_proba(X)[:, pos_idx]
    y_bin = (y == POS_LABEL).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_bin, y_score)
    df = pd.DataFrame({
        'threshold': np.append(thresholds, 1.0),
        'precision': np.append(precision[:-1], precision[-1]),
        'recall': np.append(recall[:-1], recall[-1])
    })
    if savepath_csv:
        df.to_csv(savepath_csv, index=False)
    return df

# -------------------------
# McNemar test
# -------------------------
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    b = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    c = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'n': n, 'p_value': None, 'note': 'No discordant pairs'}
    p_value = binomtest(min(b, c), n=n, p=0.5, alternative='two-sided').pvalue
    return {'b': b, 'c': c, 'n': n, 'p_value': float(p_value)}

# -------------------------
# MAIN: run everything
# -------------------------
def main():
    # 1) load RF model (v7)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}. Rode o 'train.py' (v7) primeiro.")
    model = joblib.load(MODEL_PATH)
    print("Modelo RandomForest (v7) carregado.")
    
    # Carrega as colunas de treino (v7)
    feature_names_model = None
    try:
        with open(TRAIN_COLS_PATH, 'r') as f:
            feature_names_model = json.load(f)
        print(f"Feature names (colunas v7) carregados do 'train_columns_v7.json': {len(feature_names_model)}")
    except FileNotFoundError:
             raise FileNotFoundError(f"Aviso: {TRAIN_COLS_PATH} não encontrado. "
                                     "Rode o 'train.py' (v7) primeiro.")

    
    # 2) load PICO TEST set (v7) e FORA DE PICO set (v7)
    print("\nCarregando dados de teste (v7) pré-processados...")
    
    # --- CARREGA DADOS DE TESTE 'PICO' (v7) ---
    try:
        X_pico = pd.read_csv(PICO_TEST_X_PATH, sep=';', encoding='latin1')
        y_pico_raw = pd.read_csv(PICO_TEST_Y_PATH, sep=';', encoding='latin1')
        datas_pico_raw = pd.read_csv(PICO_TEST_DATAS_PATH, sep=';', encoding='latin1') # CARREGA DATAS
        
        y_pico = y_pico_raw.squeeze()
        if isinstance(y_pico, pd.DataFrame):
             y_pico = y_pico_raw[y_pico_raw.columns[0]]
        y_pico = ensure_int_labels(y_pico)
        
        # Converte datas para datetime
        datas_pico = pd.to_datetime(datas_pico_raw.squeeze())
        
        X_pico = X_pico.reindex(columns=feature_names_model, fill_value=0)

    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivos de teste 'Pico' (v7) não encontrados. "
                                 f"Verifique {PICO_TEST_X_PATH}, {PICO_TEST_Y_PATH} e {PICO_TEST_DATAS_PATH}. "
                                 "Rode o 'train.py' (v7) primeiro.")

    # --- CARREGA DADOS DE TESTE 'FORA DE PICO' (v7) ---
    try:
        X_fora = pd.read_csv(FORA_PICO_TEST_X_PATH, sep=';', encoding='latin1')
        y_fora_raw = pd.read_csv(FORA_PICO_TEST_Y_PATH, sep=';', encoding='latin1')
        datas_fora_raw = pd.read_csv(FORA_PICO_TEST_DATAS_PATH, sep=';', encoding='latin1') # CARREGA DATAS
        
        y_fora = y_fora_raw.squeeze()
        if isinstance(y_fora, pd.DataFrame):
             y_fora = y_fora_raw[y_fora_raw.columns[0]]
        y_fora = ensure_int_labels(y_fora)
        
        # Converte datas para datetime
        datas_fora = pd.to_datetime(datas_fora_raw.squeeze())
        
        X_fora = X_fora.reindex(columns=feature_names_model, fill_value=0)

    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivos de teste 'Fora de Pico' (v7) não encontrados. "
                                 f"Verifique {FORA_PICO_TEST_X_PATH}, {FORA_PICO_TEST_Y_PATH} e {FORA_PICO_TEST_DATAS_PATH}. "
                                 "Rode o 'train.py' (v7) primeiro.")
    
    print(f"Pico (Teste v7): {len(X_pico)} amostras; Fora (v7): {len(X_fora)} amostras")
    print(f"Número de features alinhadas: {len(X_pico.columns)}")

    # 3) PR curves (Pico vs Fora)
    print("\nGerando Precision-Recall Curve (Pico v7)...")
    plot_pr_curve_for_dataset(model, X_pico, y_pico, "Pico (Teste v7)", savepath=os.path.join(FIG_DIR, 'pr_pico_v7.png'))
    print("Salvo:", os.path.join(FIG_DIR, 'pr_pico_v7.png'))

    print("Gerando Precision-Recall Curve (Fora de Pico v7)...")
    plot_pr_curve_for_dataset(model, X_fora, y_fora, "Fora de Pico (v7)", savepath=os.path.join(FIG_DIR, 'pr_fora_v7.png'))
    print("Salvo:", os.path.join(FIG_DIR, 'pr_fora_v7.png'))

    # 4) Análise SHAP (Pico vs Fora)
    print("\nGerando gráficos SHAP (Pico v7)... (Isso pode demorar)")
    analyze_shap(model, X_pico, "Pico (Teste v7)", os.path.join(FIG_DIR, 'shap_pico_v7.png'))
    
    print("\nGerando gráficos SHAP (Fora de Pico v7)... (Isso pode demorar)")
    analyze_shap(model, X_fora, "Fora de Pico (v7)", os.path.join(FIG_DIR, 'shap_fora_v7.png'))

    # 5) permutation importance on Fora de Pico
    print("\nCalculando Permutation Importance (Fora de Pico v7)...")
    perm_df = permutation_importance_recall(
        model, X_fora, y_fora, n_repeats=30,
        top_k=20,
        savepath_png=os.path.join(FIG_DIR, 'perm_importance_fora_v7.png'),
        savepath_csv=os.path.join(TABLE_DIR, 'perm_importance_fora_v7.csv')
    )
    print("Permutation importance salvo:", os.path.join(TABLE_DIR, 'perm_importance_fora_v7.csv'))

    # 6) threshold sweep (Fora de Pico)
    print("\nGerando threshold sweep (Fora de Pico v7)...")
    df_thresh = threshold_sweep(model, X_fora, y_fora, savepath_csv=os.path.join(TABLE_DIR, 'threshold_sweep_fora_v7.csv'))
    print("Threshold sweep salvo:", os.path.join(TABLE_DIR, 'threshold_sweep_fora_v7.csv'))

    # 7) McNemar: compare DecisionTree baseline vs RF on the Fora de Pico set
    print("\nRodando McNemar test (baseline v7 vs RandomForest v7) ...")
    if os.path.exists(DT_BASELINE_PATH):
        dt_model = joblib.load(DT_BASELINE_PATH)
        print("Baseline DecisionTree (v7) carregado de", DT_BASELINE_PATH)
    else:
        print(f"AVISO: Baseline {DT_BASELINE_PATH} não encontrado. Pulando McNemar test.")
        dt_model = None

    if dt_model:
        y_pred_rf = model.predict(X_fora)
        y_pred_dt = dt_model.predict(X_fora)

        print("Gerando relatórios de classificação (v7) para o conjunto 'Fora de Pico'...")
        rep_rf = classification_report(y_fora, y_pred_rf, output_dict=True)
        rep_dt = classification_report(y_fora, y_pred_dt, output_dict=True)
        pd.DataFrame(rep_rf).to_csv(os.path.join(TABLE_DIR, 'classification_report_rf_fora_v7_analysis.csv'))
        pd.DataFrame(rep_dt).to_csv(os.path.join(TABLE_DIR, 'classification_report_dt_fora_v7_analysis.csv'))
        print("Classification reports (v7) salvos em tabelas.")

        m_res = mcnemar_test(y_fora.values, y_pred_dt, y_pred_rf)
        print("McNemar test result (v7):", m_res)
        with open(os.path.join(TABLE_DIR, 'mcnemar_result_v7.json'), 'w') as f:
            json.dump(m_res, f, indent=2)

    # 8) Bootstrap CI (AGORA COM BLOCK BOOTSTRAP)
    print("\nCalculando Intervalos de Confiança (Block Bootstrap v7)...")
    ci_pico = calculate_block_bootstrap_ci(model, X_pico, y_pico, datas_pico, block_length_days=28)
    ci_fora = calculate_block_bootstrap_ci(model, X_fora, y_fora, datas_fora, block_length_days=28)
    
    # Salva CIs em um JSON
    cis = {'pico_teste_v7_block': ci_pico, 'fora_de_pico_v7_block': ci_fora}
    with open(os.path.join(TABLE_DIR, 'bootstrap_cis_v7.json'), 'w') as f:
        json.dump(cis, f, indent=2)
    print("Intervalos de Confiança (v7) salvos em:", os.path.join(TABLE_DIR, 'bootstrap_cis_v7.json'))

    print("\nAnálises (v7) concluídas. Artefatos salvos em:", RESULTS_DIR)
    print("Gráficos:", FIG_DIR)
    print("Tabelas:", TABLE_DIR)


if __name__ == "__main__":
    main()