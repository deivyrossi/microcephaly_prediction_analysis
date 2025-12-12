import pandas as pd
import os

# Caminhos (Ajuste se necessário)
PICO_PATH = 'dados/processados/DADOS_PICO_EPIDEMICO.csv'
FORA_PICO_PATH = 'dados/processados/DADOS_FORA_PICO.csv'
SINAN_PATH = 'dados/processados/SINAN_ZIKA_AGREGADO_SEMANAL_UF.csv'
CNES_PATH = 'dados/processados/CNES_INFRAESTRUTURA_AGREGADO_ANO_UF.csv'
OUTPUT_DIR = 'dados/publicos'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapeamento UF (Cópia do train.py)
IBGE_UF_MAP = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
    31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS',
    50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
}

def process_and_anonimize(df_path, sinan_df, cnes_df, output_name):
    print(f"Processando {df_path}...")
    df = pd.read_csv(df_path, sep=';', encoding='latin1')
    
    # 1. Recria as colunas temporais e chaves (Igual ao train.py)
    df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
    df.dropna(subset=['DT_NOTIFIC'], inplace=True)
    
    # ISO Week
    iso_week = df['DT_NOTIFIC'].dt.isocalendar()
    df['SEMANA_EPI'] = iso_week.apply(lambda x: f"{int(x.year)}_{int(x.week):02d}", axis=1)
    df['ANO'] = df['DT_NOTIFIC'].dt.year
    df['ANO_ANTERIOR'] = df['ANO'] - 1
    
    # Chaves de Merge
    df['UFRES_CODE_STR'] = df['UFRES'].astype(str)
    df['UFRES_SIGLA'] = df['UFRES'].map(IBGE_UF_MAP)

    # 2. Faz o Merge (Para ter as colunas externas)
    # Merge SINAN
    df = pd.merge(df, sinan_df, left_on=['SEMANA_EPI', 'UFRES_CODE_STR'], right_on=['SEMANA_EPI', 'SG_UF_NOT'], how='left')
    df['CASOS_ZIKA_CUMUL_LAG1'] = df['CASOS_ZIKA_CUMUL_LAG1'].fillna(0) # Preenche Zeros do Zika

    # Merge CNES
    df = pd.merge(df, cnes_df, left_on=['ANO_ANTERIOR', 'UFRES_SIGLA'], right_on=['ANO', 'UF'], how='left')
    # Nota: NÃO imputamos a mediana aqui. Deixamos NaN para o train.py decidir a estratégia.

    # 3. Seleção de Colunas Seguras (Anonimização)
    cols_seguras = [
        'SEMANA_EPI', 'ANO', 'UFRES', # Tempo e Espaço (Genéricos)
        'IDADEGES', 'EXANT_GES', 'FEBRE_GES', # Sintomas
        'CASOS_ZIKA_CUMUL_LAG1', # Feature Externa 1
        'MEDICOS_SUS_TOTAL', 'LEITOS_OBSTETRICIA_SUS', 'LEITOS_UTI_NEONATAL_SUS', # Features Externas 2
        'CLASSIFIN' # Target
    ]
    
    # Filtra colunas que realmente existem (caso falte alguma)
    cols_existentes = [c for c in cols_seguras if c in df.columns]
    
    df_safe = df[cols_existentes].copy()
    
    # Salva
    safe_path = os.path.join(OUTPUT_DIR, output_name)
    df_safe.to_csv(safe_path, index=False, sep=';', encoding='latin1')
    print(f"Salvo: {safe_path} ({len(df_safe)} registros)")

# --- Carregamento dos Dados Externos (Pré-processamento igual ao train.py) ---
print("Carregando dados externos...")
# SINAN
df_sinan_raw = pd.read_csv(SINAN_PATH, sep=';', encoding='utf-8-sig', dtype={'SEMANA_EPI': str, 'SG_UF_NOT': str})
df_sinan_raw['ANO_SEMANA_NUM'] = df_sinan_raw['SEMANA_EPI'].str.replace('_', '').astype(int)
df_sinan_raw = df_sinan_raw.sort_values(by=['SG_UF_NOT', 'ANO_SEMANA_NUM'])
df_sinan_raw['CASOS_ZIKA_CUMULATIVO'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CONFIRMADOS'].cumsum()
df_sinan_raw['CASOS_ZIKA_CUMUL_LAG1'] = df_sinan_raw.groupby('SG_UF_NOT')['CASOS_ZIKA_CUMULATIVO'].shift(1).fillna(0)
df_sinan_lagged = df_sinan_raw[['SEMANA_EPI', 'SG_UF_NOT', 'CASOS_ZIKA_CUMUL_LAG1']].copy()

# CNES
df_cnes = pd.read_csv(CNES_PATH, sep=';', encoding='utf-8-sig')

# --- Execução ---
process_and_anonimize(PICO_PATH, df_sinan_lagged, df_cnes, 'DATASET_PICO_ANONIMIZADO.csv')
process_and_anonimize(FORA_PICO_PATH, df_sinan_lagged, df_cnes, 'DATASET_FORA_PICO_ANONIMIZADO.csv')