# train.py (MODELO V7.5 - Leak-Proof / À Prova de Vazamento)
"""
Treino e salvamento de modelos.
Versão final (v7.5) - CORREÇÕES CRÍTICAS DE DATA LEAKAGE:
 1. Split Cronológico ocorre ANTES de qualquer Imputação ou Encoding.
 2. Imputer é fitado APENAS no X_train.
 3. Encoding alinhado (reindex) para garantir colunas idênticas no teste.
 4. Threshold Tuning calculado via Cross-Validation no Treino (sem espiar o Teste).
 5. Mantém correções anteriores (ISO Week, Mediana, drop_first=False).
"""
import os
import json
import argparse
from pprint import pprint

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils import check_random_state
from sklearn.impute import SimpleImputer

# Tenta importar o SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
    print("Biblioteca 'imbalanced-learn' (para SMOTE) encontrada.")
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("AVISO: 'imbalanced-learn' não encontrado. Teste de SMOTE será pulado.")

# ---------- CONFIG ----------
PICO_PATH = 'dados/processados/DADOS_PICO_EPIDEMICO.csv'
FORA_PICO_PATH = 'dados/processados/DADOS_FORA_PICO.csv' 
SINAN_ZIKA_PATH = 'dados/processados/SINAN_ZIKA_AGREGADO_SEMANAL_UF.csv'
CNES_INFRA_PATH = 'dados/processados/CNES_INFRAESTRUTURA_AGREGADO_ANO_UF.csv'
RESULTS_DIR = 'resultados_v7' 
FIG_DIR = os.path.join(RESULTS_DIR, 'graficos')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')
MODEL_RF_PATH = 'modelo_rf_final_v7.joblib'
MODEL_DT_PATH = 'modelo_dt_baseline_v7.joblib'
TRAIN_COLS_PATH = os.path.join(RESULTS_DIR, 'train_columns_v7.json')
DATA_DIR = 'dados/processados' 
PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'PICO_test_X_v7.csv')
PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'PICO_test_y_v7.csv')
FORA_PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_X_v7.csv')
FORA_PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_y_v7.csv')
IMPUTER_PATH = os.path.join(RESULTS_DIR, 'imputer_idade_v7.joblib')
PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'PICO_test_datas_v7.csv')
FORA_PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_datas_v7.csv')

RANDOM_STATE = 42
POS_LABEL = 1

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

IBGE_UF_MAP = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
    31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS',
    50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
}

def ensure_int_labels(series):
    if series.dtype == object:
        try:
            return series.astype(int)
        except Exception:
            return series.map(lambda v: int(str(v).strip()))
    return series.astype(int)

# ---------- Pipeline de Carregamento (v7.5) ----------
def load_and_merge_data(path, base_features, sinan_df_lagged, cnes_df, target, dataset_name="Pico"):
    """
    Carrega e faz o merge. Deixa NaNs na infraestrutura para imputação posterior.
    """
    cols_to_load = base_features + [target, 'DT_NOTIFIC', 'UFRES'] 
    df = pd.read_csv(path, sep=';', encoding='latin1', usecols=lambda c: c in cols_to_load)
    
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
    df.dropna(subset=['DT_NOTIFIC', 'UFRES'], inplace=True) 
    
    df['IDADEGES'] = pd.to_numeric(df['IDADEGES'], errors='coerce')
    df[target] = ensure_int_labels(df[target])
    
    df['ANO_DA_NOTIFICACAO'] = df['DT_NOTIFIC'].dt.year
    
    # ISO Week (Correção Ponto 1)
    try:
        iso_week = df['DT_NOTIFIC'].dt.isocalendar()
        df['SEMANA_EPI'] = iso_week.apply(lambda x: f"{int(x.year)}_{int(x.week):02d}", axis=1)
    except Exception as e:
        print(f"AVISO: Falha isocalendar: {e}. Usando fallback.")
        df['SEMANA_EPI'] = df['DT_NOTIFIC'].dt.strftime('%Y_%U')

    df['UFRES_CODE_STR'] = df['UFRES'].astype(str) 
    df['UFRES_SIGLA'] = df['UFRES'].map(IBGE_UF_MAP)
    
    # Merge SINAN
    df = pd.merge(df, sinan_df_lagged, left_on=['SEMANA_EPI', 'UFRES_CODE_STR'], right_on=['SEMANA_EPI', 'SG_UF_NOT'], how='left')
    
    # Merge CNES
    df['ANO_ANTERIOR'] = df['ANO_DA_NOTIFICACAO'] - 1
    df = pd.merge(df, cnes_df, left_on=['ANO_ANTERIOR', 'UFRES_SIGLA'], right_on=['ANO', 'UF'], how='left')

    # Sanity Check
    pct_missing_cnes = df['MEDICOS_SUS_TOTAL'].isnull().mean() * 100
    print(f"[{dataset_name}] Missing CNES (Infra) antes da imputação: {pct_missing_cnes:.2f}%")

    # Preenche ZIKA com 0 (sem notificação = 0 casos)
    df['CASOS_ZIKA_CUMUL_LAG1'] = df['CASOS_ZIKA_CUMUL_LAG1'].fillna(0)
    
    # NÃO preenche CNES com 0 (Deixa NaN para o imputer mediano)

    datas = df['DT_NOTIFIC']
    y = df[target]
    
    cols_to_drop_pre_encode = [
        target, 'DT_NOTIFIC', 'ANO_DA_NOTIFICACAO', 'ANO_ANTERIOR', 'SEMANA_EPI', 
        'UFRES_CODE_STR', 'UFRES_SIGLA', 'SG_UF_NOT', 'ANO', 'UF'
    ]
    X = df.drop(columns=cols_to_drop_pre_encode, errors='ignore')
    
    return X, y, datas

# ---------- RandomizedSearch helper ----------
def randomized_search_rf(X_train, y_train, datas_train, random_state=RANDOM_STATE, n_iter=30, n_jobs=-1):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs)
    tscv = TimeSeriesSplit(n_splits=5) 
    
    # Log dos folds
    print("\n--- Folds TimeSeriesSplit (CV) ---")
    if isinstance(datas_train, pd.Series):
        datas_train = datas_train.values
    for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        d_val = datas_train[val_idx]
        print(f"Fold {i+1}: Validação de {pd.to_datetime(d_val.min()).date()} até {pd.to_datetime(d_val.max()).date()} ({len(val_idx)} amostras)")
    
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, scoring='recall', cv=tscv, verbose=1, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    print(f"Melhor Recall CV: {search.best_score_:.4f}")
    print(f"Melhores Params: {search.best_params_}")
    return search.best_estimator_

# ---------- Main v7.5 ----------
def main(use_search=False, n_iter=30):
    base_features = ['EXANT_GES', 'FEBRE_GES', 'IDADEGES']
    target = 'CLASSIFIN'

    # 1. Carrega Features Externas
    print("Carregando features externas...")
    try:
        df_sinan_raw = pd.read_csv(SINAN_ZIKA_PATH, sep=';', encoding='utf-8-sig', dtype={'SEMANA_EPI': str, 'SG_UF_NOT': str})
        # ... (lógica de lag do Sinan - igual v7.4) ...
        df_sinan_raw['ANO_SEMANA_NUM'] = df_sinan_raw['SEMANA_EPI'].str.replace('_', '').astype(int)
        df_sinan_raw = df_sinan_raw.sort_values(by=['SG_UF_NOT', 'ANO_SEMANA_NUM'])
        df_sinan_raw['CASOS_ZIKA_CUMULATIVO'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CONFIRMADOS'].cumsum()
        df_sinan_raw['CASOS_ZIKA_CUMUL_LAG1'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CUMULATIVO'].shift(1).fillna(0)
        df_sinan_lagged = df_sinan_raw[['SEMANA_EPI', 'SG_UF_NOT', 'CASOS_ZIKA_CUMUL_LAG1']].copy()
    except FileNotFoundError:
        print("ERRO: SINAN não encontrado.")
        return
    
    try:
        df_cnes = pd.read_csv(CNES_INFRA_PATH, sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print("ERRO: CNES não encontrado.")
        return

    # 2. Carrega Dados Brutos (Pico e Fora)
    print("Carregando dados brutos...")
    X_pico_raw, y_pico_raw, datas_pico = load_and_merge_data(PICO_PATH, base_features, df_sinan_lagged, df_cnes, target, "Pico")
    X_fora_raw, y_fora, datas_fora = load_and_merge_data(FORA_PICO_PATH, base_features, df_sinan_lagged, df_cnes, target, "Fora de Pico")

    # 3. SPLIT CRONOLÓGICO (ANTES DE PROCESSAR) - Correção Crítica #1
    print("Realizando Split Cronológico (80/20) no PICO antes do processamento...")
    
    # Ordena Pico
    sort_idx = datas_pico.argsort()
    X_pico_sorted = X_pico_raw.iloc[sort_idx]
    y_pico_sorted = y_pico_raw.iloc[sort_idx]
    datas_pico_sorted = datas_pico.iloc[sort_idx]
    
    split_index = int(len(X_pico_sorted) * 0.8)
    
    # Define conjuntos
    X_train_raw = X_pico_sorted.iloc[:split_index].copy()
    y_train = y_pico_sorted.iloc[:split_index].copy()
    datas_train = datas_pico_sorted.iloc[:split_index].copy()
    
    X_test_pico_raw = X_pico_sorted.iloc[split_index:].copy()
    y_test_pico = y_pico_sorted.iloc[split_index:].copy()
    datas_test_pico = datas_pico_sorted.iloc[split_index:].copy()
    
    # O conjunto "Fora de Pico" é inteiramente teste de drift
    X_test_fora_raw = X_fora_raw.copy()
    # y_fora e datas_fora já definidos
    
    print(f"Treino: {len(X_train_raw)} | Teste Pico: {len(X_test_pico_raw)} | Teste Fora: {len(X_test_fora_raw)}")
    print(f"Distribuição Treino: \n{y_train.value_counts()}")

    # 4. IMPUTAÇÃO (Fit apenas no Treino) - Correção Crítica #2
    print("Aplicando Imputação (Fit no Treino, Transform no resto)...")
    
    cols_to_impute = ['IDADEGES', 'MEDICOS_SUS_TOTAL', 'LEITOS_OBSTETRICIA_SUS', 'LEITOS_UTI_NEONATAL_SUS']
    # Garante que colunas existem
    cols_to_impute = [c for c in cols_to_impute if c in X_train_raw.columns]
    
    imputer = SimpleImputer(strategy='median')
    
    # Fit apenas no treino!
    imputer.fit(X_train_raw[cols_to_impute])
    
    # Transform em todos
    X_train_raw[cols_to_impute] = imputer.transform(X_train_raw[cols_to_impute])
    X_test_pico_raw[cols_to_impute] = imputer.transform(X_test_pico_raw[cols_to_impute])
    X_test_fora_raw[cols_to_impute] = imputer.transform(X_test_fora_raw[cols_to_impute])
    
    joblib.dump(imputer, IMPUTER_PATH)

    # 5. ENCODING (One-Hot alinhado) - Correção Crítica #3
    print("Aplicando One-Hot Encoding (Alinhado ao Treino)...")
    
    cols_to_encode = ['UFRES', 'EXANT_GES', 'FEBRE_GES']
    
    # Get dummies no treino
    X_train_encoded = pd.get_dummies(X_train_raw, columns=cols_to_encode, dummy_na=True, drop_first=False)
    final_cols = X_train_encoded.columns.tolist()
    
    # Salva colunas
    with open(TRAIN_COLS_PATH, 'w') as f:
        json.dump(final_cols, f)
    
    # Aplica encoding nos testes e realinha colunas (reindex)
    # Isso garante que se o teste tiver uma UF nova, ela é ignorada. Se faltar uma UF, ela entra como 0.
    X_test_pico_encoded = pd.get_dummies(X_test_pico_raw, columns=cols_to_encode, dummy_na=True, drop_first=False)
    X_test_pico_encoded = X_test_pico_encoded.reindex(columns=final_cols, fill_value=0)
    
    X_test_fora_encoded = pd.get_dummies(X_test_fora_raw, columns=cols_to_encode, dummy_na=True, drop_first=False)
    X_test_fora_encoded = X_test_fora_encoded.reindex(columns=final_cols, fill_value=0)
    
    print(f"Features finais: {len(final_cols)}")

    # Renomeia para facilitar
    X_train = X_train_encoded
    X_test_pico = X_test_pico_encoded
    X_test_fora = X_test_fora_encoded

    # 6. TREINO E BASELINE
    print("\nTreinando DecisionTree (Baseline)...")
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    joblib.dump(dt, MODEL_DT_PATH)
    
    # Relatórios Baseline
    pd.DataFrame(classification_report(y_test_pico, dt.predict(X_test_pico), output_dict=True)).to_csv(os.path.join(TABLE_DIR, 'classification_report_dt_pico.csv'))
    pd.DataFrame(classification_report(y_fora, dt.predict(X_test_fora), output_dict=True)).to_csv(os.path.join(TABLE_DIR, 'classification_report_dt_fora_pico.csv'))

    # 7. EXPERIMENTOS ÉTICOS
    print("\n--- INICIANDO EXPERIMENTOS (v7.5 Leak-Proof) ---")
    RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'max_features': 'log2', 'min_samples_split': 10, 'random_state': RANDOM_STATE, 'n_jobs': -1}

    # Exp 1: Class Weight (Principal)
    print("[Exp 1] Class Weight Balanced (Principal)...")
    if use_search:
        modelo_cw = randomized_search_rf(X_train, y_train, datas_train, random_state=RANDOM_STATE, n_iter=n_iter)
    else:
        modelo_cw = RandomForestClassifier(class_weight='balanced', **RF_PARAMS)
        modelo_cw.fit(X_train, y_train)
    
    joblib.dump(modelo_cw, MODEL_RF_PATH)
    
    print(">>> Resultado Exp 1 (Pico):")
    print(classification_report(y_test_pico, modelo_cw.predict(X_test_pico)))
    
    # Exp 2: SMOTE
    if IMBLEARN_AVAILABLE:
        print("\n[Exp 2] SMOTE...")
        pipeline_smote = ImbPipeline(steps=[('smote', SMOTE(random_state=RANDOM_STATE)), ('model', RandomForestClassifier(**RF_PARAMS))])
        pipeline_smote.fit(X_train, y_train)
        print(">>> Resultado Exp 2 (SMOTE):")
        print(classification_report(y_test_pico, pipeline_smote.predict(X_test_pico)))

    # Exp 3: Threshold Tuning (HONESTO - via CV) - Correção Crítica #4
    print("\n[Exp 3] Threshold Tuning (Honesto via CV)...")
    modelo_tt = RandomForestClassifier(**RF_PARAMS)
    
    # Obtém probabilidades no treino via Cross-Validation (sem vazamento)
    y_probas_cv = cross_val_predict(modelo_tt, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Encontra melhor threshold no TREINO
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_probas_cv, pos_label=POS_LABEL)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Melhor limiar descoberto no CV (Treino): {best_threshold:.4f}")
    
    # Treina no X_train inteiro para aplicar no teste
    modelo_tt.fit(X_train, y_train)
    
    # Aplica no Teste
    y_probas_test = modelo_tt.predict_proba(X_test_pico)[:, 1]
    y_pred_tt = (y_probas_test >= best_threshold).astype(int)
    y_pred_tt_labels = np.where(y_pred_tt == 1, POS_LABEL, 5)
    
    print(">>> Resultado Exp 3 (Threshold Tuning):")
    print(classification_report(y_test_pico, y_pred_tt_labels))

    # 8. SALVAMENTO E FINALIZAÇÃO
    print("\nSalvando dados de teste finais...")
    # Salva com headers corretos
    X_test_pico.to_csv(PICO_TEST_X_PATH, sep=';', index=False, encoding='latin1')
    y_test_pico.to_frame().to_csv(PICO_TEST_Y_PATH, sep=';', index=False, encoding='latin1')
    datas_test_pico.to_frame().to_csv(PICO_TEST_DATAS_PATH, sep=';', index=False, encoding='latin1')
    
    X_test_fora.to_csv(FORA_PICO_TEST_X_PATH, sep=';', index=False, encoding='latin1')
    # O nome correto da variável é 'y_fora' (definido lá no início, linha 172)
    y_fora.to_frame().to_csv(FORA_PICO_TEST_Y_PATH, sep=';', index=False, encoding='latin1')
    # Usa datas_fora (provém de load_and_merge_data) em vez da variável indefinida datas_test_fora
    datas_fora.to_frame().to_csv(FORA_PICO_TEST_DATAS_PATH, sep=';', index=False, encoding='latin1')
    
    # Avaliação Final Fora de Pico (Drift)
    print("\n--- Avaliação Final DRIFT (Fora de Pico) ---")
    print(classification_report(y_fora, modelo_cw.predict(X_test_fora)))
    pd.DataFrame(classification_report(y_fora, modelo_cw.predict(X_test_fora), output_dict=True)).to_csv(os.path.join(TABLE_DIR, 'classification_report_rf_fora_pico.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true', help='Roda RandomizedSearchCV')
    parser.add_argument('--n_iter', type=int, default=50)
    args = parser.parse_args()
    main(use_search=args.search, n_iter=args.n_iter)