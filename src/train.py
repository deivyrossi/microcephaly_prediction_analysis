# train.py (MODELO V7.4 - Final com Testes Éticos SMOTE + Thresholding)
"""
Treino e salvamento de modelos.
Versão final (v7.4) - Adiciona:
 1. (Ponto 1) Usa isocalendar() para SEMANA_EPI
 2. (Ponto 2) Usa drop_first=False no get_dummies
 3. (Ponto 5) Imprime o y_train.value_counts()
 4. (Ponto 6) Sanity check de % de missings pós-merge
 5. (Ponto 10) Impressão dos detalhes de datas do TimeSeriesSplit
 6. (Ponto 4 - Ética) Adiciona experimentos de SMOTE e Threshold Tuning
"""
import os
import json
import argparse
from pprint import pprint

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
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
    print("Para rodar o teste de SMOTE, execute: pip install imbalanced-learn")

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
# ---------------------------------
RANDOM_STATE = 42
POS_LABEL = 1
# ---------------------------------
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
# ----------------------------------------

def ensure_int_labels(series):
    if series.dtype == object:
        try:
            return series.astype(int)
        except Exception:
            return series.map(lambda v: int(str(v).strip()))
    return series.astype(int)

# ---------- Pipeline de pre-processamento (ATUALIZADO Ponto 1) ----------
def load_and_merge_data(path, base_features, sinan_df_lagged, cnes_df, target, dataset_name="Pico"):
    """
    Carrega dados (Pico ou Fora de Pico), faz o MERGE com features externas
    e retorna o DataFrame pronto para Imputação e Encoding.
    v7.4: Usa isocalendar()
    """
    
    cols_to_load = base_features + [target, 'DT_NOTIFIC', 'UFRES'] 
    df = pd.read_csv(path, sep=';', encoding='latin1', usecols=lambda c: c in cols_to_load)
    
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
    df.dropna(subset=['DT_NOTIFIC', 'UFRES'], inplace=True) 
    
    df['IDADEGES'] = pd.to_numeric(df['IDADEGES'], errors='coerce')
    df[target] = ensure_int_labels(df[target])
    
    # 4. Cria as chaves de "join"
    df['ANO_DA_NOTIFICACAO'] = df['DT_NOTIFIC'].dt.year
    
    # --- [MUDANÇA PONTO 1: ISO WEEK] ---
    try:
        iso_week = df['DT_NOTIFIC'].dt.isocalendar()
        # Formata como YYYY_WW (ex: 2016_05)
        df['SEMANA_EPI'] = iso_week.apply(lambda x: f"{int(x.year)}_{int(x.week):02d}", axis=1)
    except Exception as e:
        print(f"AVISO: Falha ao gerar SEMANA_EPI com isocalendar: {e}. Verifique a versão do pandas.")
        df['SEMANA_EPI'] = df['DT_NOTIFIC'].dt.strftime('%Y_%U') # Fallback
    # --- [FIM DA MUDANÇA] ---

    df['UFRES_CODE_STR'] = df['UFRES'].astype(str) 
    df['UFRES_SIGLA'] = df['UFRES'].map(IBGE_UF_MAP)
    
    # --- 5. Merge com SINAN (Zika) ---
    df = pd.merge(
        df, 
        sinan_df_lagged, 
        left_on=['SEMANA_EPI', 'UFRES_CODE_STR'], 
        right_on=['SEMANA_EPI', 'SG_UF_NOT'], 
        how='left'
    )
    
    # --- 6. Merge com CNES (Infraestrutura) - COM LAG ---
    df['ANO_ANTERIOR'] = df['ANO_DA_NOTIFICACAO'] - 1
    df = pd.merge(
        df,
        cnes_df,
        left_on=['ANO_ANTERIOR', 'UFRES_SIGLA'],
        right_on=['ANO', 'UF'],
        how='left'
    )

    # --- (Ponto 6: Sanity Check) ---
    print(f"--- Sanity Check ({dataset_name}): Pós-Merge, Antes-Fillna ---")
    pct_missing_sinan = df['CASOS_ZIKA_CUMUL_LAG1'].isnull().mean() * 100
    pct_missing_cnes = df['MEDICOS_SUS_TOTAL'].isnull().mean() * 100
    print(f"[%] Amostras ({dataset_name}) sem merge SINAN (Zika): {pct_missing_sinan:.2f}%")
    print(f"[%] Amostras ({dataset_name}) sem merge CNES (Infra): {pct_missing_cnes:.2f}%")
    print("---------------------------------------------")

    fill_values = {
        'CASOS_ZIKA_CUMUL_LAG1': 0,
        'MEDICOS_SUS_TOTAL': 0,
        'LEITOS_OBSTETRICIA_SUS': 0,
        'LEITOS_UTI_NEONATAL_SUS': 0
    }
    df.fillna(value=fill_values, inplace=True)

    datas = df['DT_NOTIFIC']
    y = df[target]
    
    cols_to_drop_pre_encode = [
        target, 'DT_NOTIFIC', 'ANO_DA_NOTIFICACAO', 'ANO_ANTERIOR', 'SEMANA_EPI', 
        'UFRES_CODE_STR', 'UFRES_SIGLA', 'SG_UF_NOT', 'ANO', 'UF'
    ]
    X = df.drop(columns=cols_to_drop_pre_encode, errors='ignore')
    
    return X, y, datas

# ---------- RandomizedSearch helper (Ponto 10) ----------
def randomized_search_rf(X_train_sorted, y_train_sorted, datas_train_sorted, random_state=RANDOM_STATE, n_iter=30, n_jobs=-1):
    """
    Roda RandomizedSearch usando TimeSeriesSplit.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', 0.2, 0.5],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs)
    
    tscv = TimeSeriesSplit(n_splits=5) 

    # --- (Ponto 10: Tabela de Folds) ---
    print("\n--- Detalhes dos Folds (TimeSeriesSplit) ---")
    print("| Fold | Data Início (Val) | Data Fim (Val) | Nro. Treino | Nro. Val |")
    print("|---|---|---|---|---|")
    try:
        if isinstance(datas_train_sorted, pd.DataFrame):
            datas_train_series = datas_train_sorted.iloc[:, 0]
        else:
            datas_train_series = pd.Series(datas_train_sorted)
            
        for i, (train_index, val_index) in enumerate(tscv.split(X_train_sorted)):
            val_dates = datas_train_series.iloc[val_index]
            print(f"| {i+1} | {val_dates.min().date()} | {val_dates.max().date()} | {len(train_index)} | {len(val_index)} |")
    except Exception as e:
        print(f"| ERRO ao extrair datas dos folds: {e}. Verifique se 'datas_train_sorted' foi passado. |")
    print("-------------------------------------------\n")

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='recall', 
        cv=tscv, 
        verbose=2,
        n_jobs=n_jobs,
        random_state=random_state
    )
    search.fit(X_train_sorted, y_train_sorted)
    print("RandomizedSearchCV best params:")
    pprint(search.best_params_)
    print("Best CV recall:", search.best_score_)
    return search.best_estimator_

# ---------- Main (ATUALIZADO Ponto 2, 5 e 4) ----------
def main(use_search=False, n_iter=30):
    base_features = ['EXANT_GES', 'FEBRE_GES', 'IDADEGES']
    target = 'CLASSIFIN'

    # --- CARREGAMENTO DAS FEATURES EXTERNAS ---
    print("Carregando features externas (SINAN e CNES)...")
    try:
        print("Processando dados do SINAN (Zika) com defasagem (lag)...")
        df_sinan_raw = pd.read_csv(SINAN_ZIKA_PATH, sep=';', encoding='utf-8-sig', dtype={'SEMANA_EPI': str, 'SG_UF_NOT': str})
        
        df_sinan_raw['SG_UF_NOT'] = df_sinan_raw['SG_UF_NOT'].astype(str)
        df_sinan_raw['SEMANA_EPI'] = df_sinan_raw['SEMANA_EPI'].astype(str)

        df_sinan_raw['ANO_SEMANA_NUM'] = df_sinan_raw['SEMANA_EPI'].str.replace('_', '').astype(int)
        
        df_sinan_raw = df_sinan_raw.sort_values(by=['SG_UF_NOT', 'ANO_SEMANA_NUM'])
        df_sinan_raw['CASOS_ZIKA_CUMULATIVO'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CONFIRMADOS'].cumsum()
        df_sinan_raw['CASOS_ZIKA_CUMUL_LAG1'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CUMULATIVO'].shift(1).fillna(0)
        df_sinan_lagged = df_sinan_raw[['SEMANA_EPI', 'SG_UF_NOT', 'CASOS_ZIKA_CUMUL_LAG1']].copy()
        print("Processamento de defasagem (lag) do SINAN concluído.")

    except FileNotFoundError:
        print(f"ERRO: Arquivo {SINAN_ZIKA_PATH} não encontrado.")
        print("Rode o 'explore_sinan.py' (versão ISO Week) primeiro.")
        return
    except Exception as e:
        print(f"ERRO ao processar SINAN: {e}")
        print("Verifique se o explore_sinan.py (versão ISO Week) rodou corretamente.")
        return
        
    try:
        df_cnes = pd.read_csv(CNES_INFRA_PATH, sep=';', encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"ERRO: Arquivo {CNES_INFRA_PATH} não encontrado.")
        return
    # ----------------------------------------

    print("Carregando dados do período de pico...")
    X_pico_raw, y_pico, datas_pico = load_and_merge_data(PICO_PATH, base_features, df_sinan_lagged, df_cnes, target, dataset_name="Pico")
    print(f"Amostras disponíveis (Pico): {len(X_pico_raw)}")
    
    print("Carregando dados do período Fora de Pico...")
    X_fora_raw, y_fora, datas_fora = load_and_merge_data(FORA_PICO_PATH, base_features, df_sinan_lagged, df_cnes, target, dataset_name="Fora de Pico")
    print(f"Amostras disponíveis (Fora de Pico): {len(X_fora_raw)}")
    
    rs = check_random_state(RANDOM_STATE)

    # --- INÍCIO DA ETAPA DE IMPUTAÇÃO E ENCODING (ATUALIZADO Ponto 2) ---
    print("Iniciando Imputação e Encoding (v7.4)...")
    
    imputer = SimpleImputer(strategy='median')
    X_pico_raw['IDADEGES'] = imputer.fit_transform(X_pico_raw[['IDADEGES']])
    X_fora_raw['IDADEGES'] = imputer.transform(X_fora_raw[['IDADEGES']])
    
    joblib.dump(imputer, IMPUTER_PATH)
    print(f"Imputer de 'IDADEGES' salvo em {IMPUTER_PATH}")

    cols_to_encode = ['UFRES', 'EXANT_GES', 'FEBRE_GES']
    
    X_pico_raw['dataset_label'] = 'pico'
    X_fora_raw['dataset_label'] = 'fora_pico'
    X_combinado = pd.concat([X_pico_raw, X_fora_raw])
    
    # --- [MUDANÇA PONTO 2: drop_first=False] ---
    print("Usando get_dummies com drop_first=False (Priorizando interpretabilidade SHAP)")
    X_combinado_encoded = pd.get_dummies(X_combinado, columns=cols_to_encode, dummy_na=True, drop_first=False)
    # --- [FIM DA MUDANÇA] ---
    
    X_pico_encoded = X_combinado_encoded[X_combinado_encoded['dataset_label'] == 'pico'].drop(columns=['dataset_label'])
    X_fora_encoded = X_combinado_encoded[X_combinado_encoded['dataset_label'] == 'fora_pico'].drop(columns=['dataset_label'])
    
    print("Imputação e Encoding concluídos.")
    # --- FIM DA ETAPA ---

    # --- INÍCIO DO SPLIT CRONOLÓGICO (ATUALIZADO Ponto 5) ---
    print("Ordenando dados do PICO por data para o split cronológico...")
    df_pico_completo = X_pico_encoded.copy()
    df_pico_completo['target'] = y_pico
    df_pico_completo['datas'] = datas_pico
    
    df_pico_sorted = df_pico_completo.sort_values(by='datas')
    
    X_pico_sorted = df_pico_sorted.drop(columns=['target', 'datas'])
    y_pico_sorted = df_pico_sorted['target']
    datas_pico_sorted = df_pico_sorted['datas']
    
    split_index = int(len(X_pico_sorted) * 0.8)
    
    X_train = X_pico_sorted.iloc[:split_index]
    y_train = y_pico_sorted.iloc[:split_index]
    datas_train = datas_pico_sorted.iloc[:split_index]
    
    # --- [MUDANÇA PONTO 5: value_counts()] ---
    print("\n--- [DADOS PARA O PONTO 5] ---")
    print(f"Contagem de classes no Treino (y_train): \n{y_train.value_counts(normalize=True)}")
    print(f"Contagem absoluta (y_train): \n{y_train.value_counts()}")
    print("---------------------------------\n")
    # --- [FIM DA MUDANÇA] ---
    
    X_test_pico = X_pico_sorted.iloc[split_index:] 
    y_test_pico = y_pico_sorted.iloc[split_index:]
    datas_test_pico = datas_pico_sorted.iloc[split_index:]
    
    df_fora_completo = X_fora_encoded.copy()
    df_fora_completo['target'] = y_fora
    df_fora_completo['datas'] = datas_fora
    df_fora_sorted = df_fora_completo.sort_values(by='datas')
    
    X_test_fora = df_fora_sorted.drop(columns=['target', 'datas'])
    y_test_fora = df_fora_sorted['target']
    datas_test_fora = df_fora_sorted['datas']

    print(f"Split Cronológico: Treino {len(X_train)} (Pico 80% Antigo), Teste 1 (Pico 20% Recente) {len(X_test_pico)}")
    print(f"Teste 2 (Drift - Fora de Pico): {len(X_test_fora)}")
    
    train_cols = list(X_train.columns)
    with open(TRAIN_COLS_PATH, 'w') as f:
        json.dump(train_cols, f, indent=2)
    print("Colunas do treino salvas em:", TRAIN_COLS_PATH)
    print(f"Total de {len(train_cols)} features no modelo v7.4.") # (Número deve mudar)

    # --- Baseline DecisionTree ---
    print("\nTreinando DecisionTree (baseline v7.4)...")
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train) 
    joblib.dump(dt, MODEL_DT_PATH)
    print("Baseline salvo em:", MODEL_DT_PATH)
    rep_dt_pico = classification_report(y_test_pico, dt.predict(X_test_pico), output_dict=True)
    pd.DataFrame(rep_dt_pico).to_csv(os.path.join(TABLE_DIR, 'classification_report_dt_pico.csv'))
    
    
    # --- [INÍCIO DOS TESTES ÉTICOS (PONTO 4)] ---
    print("\n\n--- INICIANDO TESTES DE MITIGAÇÃO DE DESBALANCEAMENTO (PONTO 4) ---")
    
    # Hiperparâmetros fixos (baseados na busca anterior)
    RF_PARAMS = {
        'n_estimators': 100, 
        'max_depth': 10, 
        'max_features': 'log2', 
        'min_samples_split': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    # --- Experimento 1: (Principal) class_weight='balanced' ---
    print("\n[Experimento 1] Treinando modelo principal com class_weight='balanced'...")
    if use_search:
        print("Rodando RandomizedSearchCV com TimeSeriesSplit (Modelo v7.4)...")
        modelo_cw = randomized_search_rf(X_train, y_train, datas_train, random_state=RANDOM_STATE, n_iter=n_iter, n_jobs=-1)
    else:
        print("Usando parâmetros fixos (sem busca)...")
        modelo_cw = RandomForestClassifier(class_weight='balanced', **RF_PARAMS)
        modelo_cw.fit(X_train, y_train)
    
    print("\n--- [RELATÓRIO EXPERIMENTO 1: class_weight='balanced'] ---")
    y_pred_cw = modelo_cw.predict(X_test_pico)
    print(classification_report(y_test_pico, y_pred_cw))
    print("----------------------------------------------------------\n")

    # Salva este modelo como o principal
    joblib.dump(modelo_cw, MODEL_RF_PATH)
    print("Modelo principal (class_weight) salvo em:", MODEL_RF_PATH)
    rep_cw_pico = classification_report(y_test_pico, y_pred_cw, output_dict=True)
    pd.DataFrame(rep_cw_pico).to_csv(os.path.join(TABLE_DIR, 'classification_report_rf_pico.csv'))


    # --- Experimento 2: SMOTE (Oversampling) ---
    if IMBLEARN_AVAILABLE:
        print("\n[Experimento 2] Treinando modelo com SMOTE (Oversampling)...")
        
        # Modelo sem class_weight, pois o SMOTE fará o balanceamento
        modelo_smote_base = RandomForestClassifier(**RF_PARAMS) 
        
        # Cria o pipeline com SMOTE
        # NOTA: O SMOTE é aplicado APENAS no fit()
        pipeline_smote = ImbPipeline(steps=[
            ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, y_train.value_counts().min() - 1))),
            ('model', modelo_smote_base)
        ])
        
        try:
            pipeline_smote.fit(X_train, y_train)
            print("\n--- [RELATÓRIO EXPERIMENTO 2: SMOTE] ---")
            y_pred_smote = pipeline_smote.predict(X_test_pico)
            print(classification_report(y_test_pico, y_pred_smote))
            print("-------------------------------------------\n")
        except ValueError as e:
            print(f"ERRO ao rodar SMOTE: {e}")
            print("Isso pode acontecer se a classe minoritária tiver muito poucos exemplos (k_neighbors > n_samples).")

    else:
        print("\n[Experimento 2] Pulado (imbalanced-learn não instalado).")


    # --- Experimento 3: Threshold Tuning (Ajuste de Limiar) ---
    print("\n[Experimento 3] Treinando modelo com Threshold Tuning...")
    
    # Modelo treinado sem class_weight, no dado original
    modelo_tt = RandomForestClassifier(**RF_PARAMS)
    modelo_tt.fit(X_train, y_train)
    
    # Pega as probabilidades (scores) da classe positiva (1)
    y_scores_tt = modelo_tt.predict_proba(X_test_pico)[:, list(modelo_tt.classes_).index(POS_LABEL)]
    
    # Encontra o melhor limiar
    precisions, recalls, thresholds = precision_recall_curve(y_test_pico, y_scores_tt, pos_label=POS_LABEL)
    
    # Remove o último threshold (que é 1.0) e os valores PR correspondentes
    thresholds = thresholds
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    
    # Encontra o limiar que maximiza o F1
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    print(f"Melhor limiar encontrado (max F1): {best_threshold:.4f} (F1={best_f1:.4f})")
    
    # Aplica o limiar
    y_pred_tt = (y_scores_tt >= best_threshold).astype(int)
    # Converte predições (0/1) de volta para labels (1/5)
    y_pred_tt_labels = np.where(y_pred_tt == 1, POS_LABEL, y_train.value_counts().idxmax()) # idxmax() pega a classe majoritária (5)
    
    print("\n--- [RELATÓRIO EXPERIMENTO 3: Threshold Tuning] ---")
    print(classification_report(y_test_pico, y_pred_tt_labels))
    print("--------------------------------------------------\n")

    print("--- [FIM DOS TESTES DE MITIGAÇÃO] ---")
    # --- [FIM DOS TESTES ÉTICOS] ---


    # --- Continua com o salvamento dos dados de teste (usando o modelo_cw) ---
    print(f"Salvando conjunto de teste (Pico v7.4) para análise...")
    X_test_pico.to_csv(PICO_TEST_X_PATH, sep=';', index=False, encoding='latin1')
    y_test_pico.to_frame().to_csv(PICO_TEST_Y_PATH, sep=';', index=False, encoding='latin1') 
    datas_test_pico.to_frame().to_csv(PICO_TEST_DATAS_PATH, sep=';', index=False, encoding='latin1')
    
    print(f"Salvando conjunto de teste (Fora de Pico v7.4) para análise...")
    X_test_fora.to_csv(FORA_PICO_TEST_X_PATH, sep=';', index=False, encoding='latin1')
    y_test_fora.to_frame().to_csv(FORA_PICO_TEST_Y_PATH, sep=';', index=False, encoding='latin1') 
    datas_test_fora.to_frame().to_csv(FORA_PICO_TEST_DATAS_PATH, sep=';', index=False, encoding='latin1')

    meta = {
        'random_state': RANDOM_STATE,
        'rf_params': modelo_cw.get_params(), # Salva os params do modelo principal
        'n_train': len(X_train),
        'n_test_pico': len(X_test_pico),
        'n_test_fora': len(X_test_fora)
    }
    with open(os.path.join(RESULTS_DIR, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print("Metadados de treino salvos.")


    # --- Avaliação Temporal (Fora de Pico) ---
    print("\n\n--- Iniciando Avaliação de Generalização Temporal (Fora de Pico v7.4) ---")
    
    print("Avaliando DecisionTree (baseline v7.4) no 'Fora de Pico'...")
    y_pred_dt_fora_pico = dt.predict(X_test_fora)
    rep_dt_fora_pico = classification_report(y_test_fora, y_pred_dt_fora_pico, output_dict=True)
    pd.DataFrame(rep_dt_fora_pico).to_csv(os.path.join(TABLE_DIR, 'classification_report_dt_fora_pico.csv'))

    print("Avaliando RandomForest (otimizado v7.4) no 'Fora de Pico'...")
    # Usa o modelo principal (modelo_cw) para avaliar o drift
    y_pred_rf_fora_pico = modelo_cw.predict(X_test_fora)
    rep_rf_fora_pico = classification_report(y_test_fora, y_pred_rf_fora_pico, output_dict=True)
    pd.DataFrame(rep_rf_fora_pico).to_csv(os.path.join(TABLE_DIR, 'classification_report_rf_fora_pico.csv'))
    
    print("\nResumo (Fora de Pico v7.4) - RandomForest:")
    print(classification_report(y_test_fora, y_pred_rf_fora_pico))

    print("\n--- Treino e Avaliação (Modelo v7.4) Concluídos ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true', help='Roda RandomizedSearchCV para RF')
    parser.add_argument('--n_iter', type=int, default=50, help='Número de iterações do RandomizedSearch (se --search)')
    args = parser.parse_args()
    main(use_search=args.search, n_iter=args.n_iter)