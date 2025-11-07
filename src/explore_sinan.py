import pandas as pd
import glob
import os
from simpledbf import Dbf5
from dbf_reader import DbfReader

# --- Configs ---
INPUT_DIR = "dados/brutos/SINAN_ZIKA/" 
OUTPUT_DIR = "dados/processados/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "SINAN_ZIKA_AGREGADO_SEMANAL_UF.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# -------------

def load_dbf_as_df(filepath):
    """
    Tenta carregar um arquivo DBF usando duas bibliotecas (simpledbf ou dbf_reader)
    como fallback, já que arquivos DBF do governo podem ser problemáticos.
    """
    try:
        dbf = Dbf5(filepath, codec='latin1')
        return dbf.to_dataframe()
    except Exception as e1:
        print(f"Falha ao ler {filepath} com simpledbf (Erro: {e1}). Tentando com dbf_reader...")
        try:
            with DbfReader(filepath, encoding='latin1') as dbf:
                return pd.DataFrame(list(dbf))
        except Exception as e2:
            print(f"ERRO: Não foi possível ler o arquivo DBF {filepath} com nenhuma biblioteca.")
            print(f"Erro dbf_reader: {e2}")
            return pd.DataFrame() # Retorna DF vazio

def process_sinan_zika_files():
    """
    Carrega todos os arquivos ZIKABR*.dbf, filtra, agrega e salva
    um CSV limpo com a contagem de casos confirmados por semana e UF.
    """
    
    files_dbf = glob.glob(os.path.join(INPUT_DIR, "ZIKABR*.dbf"))
    
    if not files_dbf:
        print(f"ERRO: Nenhum arquivo ZIKABR*.dbf encontrado em {INPUT_DIR}")
        return

    print(f"Encontrados {len(files_dbf)} arquivos DBF para processar...")
    
    lista_de_dataframes = []
    for f in files_dbf:
        print(f"Processando: {f}")
        df_ano = load_dbf_as_df(f)
        if not df_ano.empty:
            lista_de_dataframes.append(df_ano)

    if not lista_de_dataframes:
        print("ERRO: Nenhum dado foi carregado.")
        return

    df_zika_completo = pd.concat(lista_de_dataframes, ignore_index=True)
    print(f"Total de {len(df_zika_completo)} registros brutos de Zika carregados.")

    df_confirmados = df_zika_completo[df_zika_completo['CLASSI_FIN'] == '1'].copy()
    print(f"Total de {len(df_confirmados)} registros CONFIRMADOS.")

    colunas_interesse = ['DT_SIN_PRI', 'SG_UF_NOT']
    
    if not all(col in df_confirmados.columns for col in colunas_interesse):
        print("ERRO: Colunas DT_SIN_PRI ou SG_UF_NOT não encontradas.")
        print(f"Colunas disponíveis: {df_confirmados.columns.tolist()}")
        return
        
    df_limpo = df_confirmados[colunas_interesse].copy()
    df_limpo['DT_SIN_PRI'] = pd.to_datetime(df_limpo['DT_SIN_PRI'], errors='coerce')
    df_limpo.dropna(inplace=True)
    print(f"Registros confirmados com data e UF válidas: {len(df_limpo)}")

    # 4. Engenharia de Feature: Cria a Semana Epidemiológica (Ano + Semana)
    
    # --- [MUDANÇA PONTO 1: ISO WEEK] ---
    # Usa o padrão ISO 8601 (isocalendar) para semana epidemiológica
    print("Gerando semana epidemiológica (ISO 8601)...")
    try:
        iso_week = df_limpo['DT_SIN_PRI'].dt.isocalendar()
        # Formata como YYYY_WW (ex: 2016_05)
        df_limpo['SEMANA_EPI'] = iso_week.apply(lambda x: f"{int(x.year)}_{int(x.week):02d}", axis=1)
    except AttributeError:
        # Fallback para versões mais antigas do pandas (embora dt.isocalendar() seja padrão)
        print("AVISO: dt.isocalendar() falhou, tentando com apply...")
        iso_week = df_limpo['DT_SIN_PRI'].apply(lambda x: x.isocalendar() if pd.notnull(x) else (None, None, None))
        df_limpo['SEMANA_EPI'] = iso_week.apply(lambda x: f"{int(x[0])}_{int(x[1]):02d}" if x[0] is not None else None)
    # --- [FIM DA MUDANÇA] ---

    # 5. Agrega os dados: Conta os casos por Semana e por UF
    print("Agregando casos por UF e Semana Epidemiológica...")
    df_agregado = df_limpo.groupby(['SG_UF_NOT', 'SEMANA_EPI']).size().reset_index(name='CASOS_ZIKA_CONFIRMADOS')

    # 6. Salva o resultado
    df_agregado.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    
    print("\n--- Sucesso! ---")
    print(f"Dados agregados de Zika salvos em: {OUTPUT_FILE}")
    print("\nAmostra dos dados processados:")
    print(df_agregado.head())

if __name__ == "__main__":
    process_sinan_zika_files()