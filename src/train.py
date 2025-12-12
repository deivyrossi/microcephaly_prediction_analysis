# src/train.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_curve, recall_score
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

RESULTS_DIR = 'resultados' 
FIG_DIR = os.path.join(RESULTS_DIR, 'graficos')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')
MODEL_RF_PATH = 'modelo_rf_final.joblib'
MODEL_DT_PATH = 'modelo_dt_baseline.joblib'
TRAIN_COLS_PATH = os.path.join(RESULTS_DIR, 'train_columns.json')

DATA_DIR = 'dados/processados' 
PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'PICO_test_X.csv')
PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'PICO_test_y.csv')
FORA_PICO_TEST_X_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_X.csv')
FORA_PICO_TEST_Y_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_y.csv')
PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'PICO_test_datas.csv')
FORA_PICO_TEST_DATAS_PATH = os.path.join(DATA_DIR, 'FORA_PICO_test_datas.csv')
IMPUTER_PATH = os.path.join(RESULTS_DIR, 'imputer_idade.joblib')

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

def load_and_merge_data(path, base_features, sinan_df_lagged, cnes_df, target, dataset_name="Pico"):
    """
    Carrega e faz o merge. Híbrido: aceita dados brutos (DT_NOTIFIC) ou anonimizados (SEMANA_EPI).
    Se sinan_df/cnes_df forem None, assume que as colunas já existem no CSV.
    """
    df = pd.read_csv(path, sep=';', encoding='latin1')
    datas = None 

    # --- LÓGICA HÍBRIDA DE DATA ---
    if 'DT_NOTIFIC' in df.columns:
        # CENÁRIO A: DADOS BRUTOS (Com data exata)
        df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
        df.dropna(subset=['DT_NOTIFIC'], inplace=True)
        try:
            iso_week = df['DT_NOTIFIC'].dt.isocalendar()
            df['SEMANA_EPI'] = iso_week.apply(lambda x: f"{int(x.year)}_{int(x.week):02d}", axis=1)
        except Exception:
            df['SEMANA_EPI'] = df['DT_NOTIFIC'].dt.strftime('%Y_%U')
        df['ANO_DA_NOTIFICACAO'] = df['DT_NOTIFIC'].dt.year
        datas = df['DT_NOTIFIC']

    elif 'SEMANA_EPI' in df.columns:
        # CENÁRIO B: DADOS ANONIMIZADOS (Sem data exata)
        print(f"AVISO [{dataset_name}]: Usando dados anonimizados. Ordenação por Semana Aproximada.")
        try:
            df['DT_NOTIFIC_APPROX'] = pd.to_datetime(df['SEMANA_EPI'] + '_1', format='%G_%V_%u', errors='coerce')
        except:
            df['DT_NOTIFIC_APPROX'] = pd.to_datetime(df['SEMANA_EPI'].str[:4] + '-01-01') 
        datas = df['DT_NOTIFIC_APPROX']
        # Extrai ano da string "YYYY_WW"
        df['ANO_DA_NOTIFICACAO'] = df['SEMANA_EPI'].astype(str).str.split('_').str[0].astype(int)

    else:
        raise ValueError("ERRO CRÍTICO: Dataset não possui 'DT_NOTIFIC' nem 'SEMANA_EPI'.")

    # --- PROCESSAMENTO COMUM ---
    df.dropna(subset=['UFRES'], inplace=True)
    df[target] = ensure_int_labels(df[target])
    df['IDADEGES'] = pd.to_numeric(df['IDADEGES'], errors='coerce')

    df['UFRES_CODE_STR'] = df['UFRES'].astype(str) 
    df['UFRES_SIGLA'] = df['UFRES'].map(IBGE_UF_MAP)
    
    # --- MERGES CONDICIONAIS ---
    
    # Merge SINAN (se disponível)
    if sinan_df_lagged is not None:
        df = pd.merge(df, sinan_df_lagged, left_on=['SEMANA_EPI', 'UFRES_CODE_STR'], right_on=['SEMANA_EPI', 'SG_UF_NOT'], how='left')
        df['CASOS_ZIKA_CUMUL_LAG1'] = df['CASOS_ZIKA_CUMUL_LAG1'].fillna(0)
    
    # Merge CNES (se disponível)
    if cnes_df is not None:
        df['ANO_ANTERIOR'] = df['ANO_DA_NOTIFICACAO'] - 1
        df = pd.merge(df, cnes_df, left_on=['ANO_ANTERIOR', 'UFRES_SIGLA'], right_on=['ANO', 'UF'], how='left')

    if 'MEDICOS_SUS_TOTAL' in df.columns:
        pct_missing_cnes = df['MEDICOS_SUS_TOTAL'].isnull().mean() * 100
        print(f"[{dataset_name}] Missing CNES (Infra) antes da imputação: {pct_missing_cnes:.2f}%")

    y = df[target]
    
    # --- WHITELIST (Seleção Final) ---
    features_externas_esperadas = [
        'CASOS_ZIKA_CUMUL_LAG1', 
        'MEDICOS_SUS_TOTAL', 
        'LEITOS_OBSTETRICIA_SUS', 
        'LEITOS_UTI_NEONATAL_SUS'
    ]
    
    cols_permitidas = base_features + features_externas_esperadas + ['UFRES']
    
    # Filtra colunas existentes (seja via merge ou já presentes no CSV)
    final_cols = [c for c in cols_permitidas if c in df.columns]
    
    X = df[final_cols].copy()
    
    print(f"[{dataset_name}] Colunas Finais no X: {list(X.columns)}")
    return X, y, datas

def randomized_search_rf(X_train, y_train, datas_train, random_state=RANDOM_STATE, n_iter=30, n_jobs=-1):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs)
    tscv = TimeSeriesSplit(n_splits=5) 
    
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

# ---------- Main ----------
def main(use_search=False, n_iter=30):
    base_features = ['EXANT_GES', 'FEBRE_GES', 'IDADEGES']
    target = 'CLASSIFIN'

    # 1. Carrega Features Externas (Tenta carregar, se não der, segue em frente)
    print("Carregando features externas...")
    df_sinan_lagged = None
    df_cnes = None

    if os.path.exists(SINAN_ZIKA_PATH):
        try:
            df_sinan_raw = pd.read_csv(SINAN_ZIKA_PATH, sep=';', encoding='utf-8-sig', dtype={'SEMANA_EPI': str, 'SG_UF_NOT': str})
            df_sinan_raw['ANO_SEMANA_NUM'] = df_sinan_raw['SEMANA_EPI'].str.replace('_', '').astype(int)
            df_sinan_raw = df_sinan_raw.sort_values(by=['SG_UF_NOT', 'ANO_SEMANA_NUM'])
            df_sinan_raw['CASOS_ZIKA_CUMULATIVO'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CONFIRMADOS'].cumsum()
            df_sinan_raw['CASOS_ZIKA_CUMUL_LAG1'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CUMULATIVO'].shift(1).fillna(0)
            df_sinan_lagged = df_sinan_raw[['SEMANA_EPI', 'SG_UF_NOT', 'CASOS_ZIKA_CUMUL_LAG1']].copy()
        except Exception as e:
            print(f"AVISO: Falha ao ler SINAN ({e}).")
    else:
        print("AVISO: Arquivo SINAN não encontrado. Assumindo que o Dataset de entrada já possui features de Zika.")

    if os.path.exists(CNES_INFRA_PATH):
        try:
            df_cnes = pd.read_csv(CNES_INFRA_PATH, sep=';', encoding='utf-8-sig')
        except Exception as e:
            print(f"AVISO: Falha ao ler CNES ({e}).")
    else:
        print("AVISO: Arquivo CNES não encontrado. Assumindo que o Dataset de entrada já possui features de Infra.")

    # 2. Carrega Dados Principais
    print("Carregando dados principais...")
    try:
        X_pico_raw, y_pico_raw, datas_pico = load_and_merge_data(PICO_PATH, base_features, df_sinan_lagged, df_cnes, target, "Pico")
        X_fora_raw, y_fora, datas_fora = load_and_merge_data(FORA_PICO_PATH, base_features, df_sinan_lagged, df_cnes, target, "Fora de Pico")
    except FileNotFoundError as e:
        print(f"ERRO FATAL: Arquivo de dados principal não encontrado: {e}")
        return

    # 3. Split Cronológico
    print("Realizando Split Cronológico (80/20)...")
    sort_idx = datas_pico.argsort()
    X_pico_sorted = X_pico_raw.iloc[sort_idx]
    y_pico_sorted = y_pico_raw.iloc[sort_idx]
    datas_pico_sorted = datas_pico.iloc[sort_idx]
    
    split_index = int(len(X_pico_sorted) * 0.8)
    
    X_train_raw = X_pico_sorted.iloc[:split_index].copy()
    y_train = y_pico_sorted.iloc[:split_index].copy()
    datas_train = datas_pico_sorted.iloc[:split_index].copy()
    
    X_test_pico_raw = X_pico_sorted.iloc[split_index:].copy()
    y_test_pico = y_pico_sorted.iloc[split_index:].copy()
    datas_test_pico = datas_pico_sorted.iloc[split_index:].copy()
    
    X_test_fora_raw = X_fora_raw.copy()
    
    print(f"Treino: {len(X_train_raw)} | Teste Pico: {len(X_test_pico_raw)} | Teste Fora: {len(X_test_fora_raw)}")

    # 4. Imputação
    print("Aplicando Imputação (Fit no Treino)...")
    cols_to_impute = ['IDADEGES', 'MEDICOS_SUS_TOTAL', 'LEITOS_OBSTETRICIA_SUS', 'LEITOS_UTI_NEONATAL_SUS']
    cols_to_impute = [c for c in cols_to_impute if c in X_train_raw.columns]
    
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train_raw[cols_to_impute])
    
    X_train_raw[cols_to_impute] = imputer.transform(X_train_raw[cols_to_impute])
    X_test_pico_raw[cols_to_impute] = imputer.transform(X_test_pico_raw[cols_to_impute])
    X_test_fora_raw[cols_to_impute] = imputer.transform(X_test_fora_raw[cols_to_impute])
    
    joblib.dump(imputer, IMPUTER_PATH)

    # 5. Encoding
    print("Aplicando One-Hot Encoding...")
    cols_to_encode = ['UFRES', 'EXANT_GES', 'FEBRE_GES']
    cols_to_encode = [c for c in cols_to_encode if c in X_train_raw.columns] # Segurança

    X_train_encoded = pd.get_dummies(X_train_raw, columns=cols_to_encode, dummy_na=True, drop_first=False)
    final_cols = X_train_encoded.columns.tolist()
    
    with open(TRAIN_COLS_PATH, 'w') as f:
        json.dump(final_cols, f)
    
    X_test_pico_encoded = pd.get_dummies(X_test_pico_raw, columns=cols_to_encode, dummy_na=True, drop_first=False)
    X_test_pico_encoded = X_test_pico_encoded.reindex(columns=final_cols, fill_value=0)
    
    X_test_fora_encoded = pd.get_dummies(X_test_fora_raw, columns=cols_to_encode, dummy_na=True, drop_first=False)
    X_test_fora_encoded = X_test_fora_encoded.reindex(columns=final_cols, fill_value=0)
    
    X_train = X_train_encoded
    X_test_pico = X_test_pico_encoded
    X_test_fora = X_test_fora_encoded

    # 6. Baseline
    print("\nTreinando DecisionTree (Baseline)...")
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    joblib.dump(dt, MODEL_DT_PATH)

    # 7. Experimentos e Modelo Final
    print("\n--- INICIANDO EXPERIMENTOS ---")
    
    # MELHORES HIPERPARÂMETROS (Fixos para reprodutibilidade e performance alta)
    BEST_PARAMS = {
        'n_estimators': 300, 
        'max_depth': 30, 
        'max_features': 'log2', 
        'min_samples_split': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    # Exp 1: Class Weight (Principal)
    print("[Exp 1] Class Weight Balanced (Principal)...")
    if use_search:
        modelo_cw = randomized_search_rf(X_train, y_train, datas_train, random_state=RANDOM_STATE, n_iter=n_iter)
    else:
        modelo_cw = RandomForestClassifier(class_weight='balanced', **BEST_PARAMS)
        modelo_cw.fit(X_train, y_train)
    
    joblib.dump(modelo_cw, MODEL_RF_PATH)
    
    print(">>> Resultado Exp 1 (Pico):")
    print(classification_report(y_test_pico, modelo_cw.predict(X_test_pico)))
    
    # Exp 2: SMOTE (Opcional)
    if IMBLEARN_AVAILABLE:
        print("\n[Exp 2] SMOTE...")
        # Usa os mesmos params para comparação justa
        pipeline_smote = ImbPipeline(steps=[('smote', SMOTE(random_state=RANDOM_STATE)), ('model', RandomForestClassifier(class_weight=None, **BEST_PARAMS))])
        pipeline_smote.fit(X_train, y_train)
        print(">>> Resultado Exp 2 (SMOTE):")
        print(classification_report(y_test_pico, pipeline_smote.predict(X_test_pico)))

    # 8. Ablação
    print("\n--- INICIANDO ESTUDO DE ABLAÇÃO ---")
    
    cols_clinico = [c for c in final_cols if c.startswith('UFRES') or c.startswith('EXANT') or c.startswith('FEBRE') or c == 'IDADEGES']
    cols_epidemio = cols_clinico + ['CASOS_ZIKA_CUMUL_LAG1'] if 'CASOS_ZIKA_CUMUL_LAG1' in final_cols else cols_clinico
    cols_completo = final_cols 

    experiments_ablation = [
        ("Apenas Clínico", cols_clinico),
        ("Clínico + Zika", cols_epidemio),
        ("Completo (+Infra)", cols_completo)
    ]
    
    for name, cols in experiments_ablation:
        print(f"Treinando Ablação: {name} ({len(cols)} features)...")
        # Garante que as colunas existem
        valid_cols = [c for c in cols if c in X_train.columns]
        
        clf_ab = RandomForestClassifier(class_weight='balanced', **BEST_PARAMS)
        clf_ab.fit(X_train[valid_cols], y_train)
        
        rec_pico = recall_score(y_test_pico, clf_ab.predict(X_test_pico[valid_cols]), pos_label=POS_LABEL)
        rec_fora = recall_score(y_fora, clf_ab.predict(X_test_fora[valid_cols]), pos_label=POS_LABEL)
        
        print(f"   -> Recall Pico: {rec_pico:.4f} | Recall Drift: {rec_fora:.4f}")

    # 9. Salvando Test Sets
    print("\nSalvando dados de teste finais...")
    X_test_pico.to_csv(PICO_TEST_X_PATH, sep=';', index=False, encoding='latin1')
    y_test_pico.to_frame().to_csv(PICO_TEST_Y_PATH, sep=';', index=False, encoding='latin1')
    datas_test_pico.to_frame().to_csv(PICO_TEST_DATAS_PATH, sep=';', index=False, encoding='latin1')
    
    X_test_fora.to_csv(FORA_PICO_TEST_X_PATH, sep=';', index=False, encoding='latin1')
    y_fora.to_frame().to_csv(FORA_PICO_TEST_Y_PATH, sep=';', index=False, encoding='latin1')
    datas_fora.to_frame().to_csv(FORA_PICO_TEST_DATAS_PATH, sep=';', index=False, encoding='latin1')

    print("\n--- Avaliação Final DRIFT (Fora de Pico) ---")
    print(classification_report(y_fora, modelo_cw.predict(X_test_fora)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true', help='Roda RandomizedSearchCV')
    parser.add_argument('--n_iter', type=int, default=50)
    args = parser.parse_args()
    main(use_search=args.search, n_iter=args.n_iter)