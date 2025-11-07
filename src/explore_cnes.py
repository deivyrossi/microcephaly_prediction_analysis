import pandas as pd
import os
import glob
from simpledbf import Dbf5
from dbf_reader import DbfReader
import re # Para extrair UF e Ano do nome do arquivo

# --- Configs ---
# Pasta onde você salvou os arquivos DBF baixados e convertidos
INPUT_DIR = "dados/brutos/CNES_TEMP/" 

# Onde salvar o arquivo processado final
OUTPUT_DIR = "dados/processados/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "CNES_INFRAESTRUTURA_AGREGADO_ANO_UF.csv")

# Códigos de interesse (baseado no dicionário de dados IT_CNES_1706.pdf)
CODIGOS_MEDICOS_CBO_PREFIX = '225' # '225...' CBO para Médicos
CODIGOS_LEITOS_OBSTETRICIA = ['03'] # '03' = LEITO OBSTETRICO
CODIGOS_LEITOS_UTI_NEONATAL = ['48', '49'] # '48' (UTI NEO II), '49' (UTI NEO III)

os.makedirs(OUTPUT_DIR, exist_ok=True)
# -------------

def load_dbf_as_df(filepath):
    """
    Tenta carregar um arquivo DBF usando duas bibliotecas (simpledbf ou dbf_reader)
    como fallback.
    """
    try:
        # Tenta primeiro com simpledbf
        dbf = Dbf5(filepath, codec='latin1')
        return dbf.to_dataframe()
    except Exception as e1:
        # print(f" DEBUG: Falha ao ler {filepath} com simpledbf. Tentando com dbf_reader...")
        try:
            # Tenta com dbf_reader como fallback
            with DbfReader(filepath, encoding='latin1') as dbf:
                return pd.DataFrame(list(dbf))
        except Exception as e2:
            print(f" ERRO: Não foi possível ler o arquivo DBF {filepath} com nenhuma biblioteca.")
            print(f"  Erro simpledbf: {e1}")
            print(f"  Erro dbf_reader: {e2}")
            return pd.DataFrame() # Retorna DF vazio

def extract_info_from_filename(filename):
    """
    Extrai o Tipo (PF, LT), UF e Ano do nome do arquivo.
    Ex: 'PFMG1612.dbf' -> ('PF', 'MG', 2016)
    """
    # Regex para capturar (PF ou LT)(UF de 2 letras)(Ano com 2 digitos)(Mês 12)
    match = re.match(r'(PF|LT)([A-Z]{2})(\d{2})(12)\.dbf', filename, re.IGNORECASE)
    if match:
        tipo, uf, ano_2_digitos, mes = match.groups()
        ano = int(f"20{ano_2_digitos}") # Converte '16' para 2016
        return tipo.upper(), uf.upper(), ano # Garante maiúsculas
    return None, None, None

def process_locais_cnes_dbfs():
    """
    Processa todos os arquivos DBF (PF* e LT*) baixados manualmente 
    da pasta INPUT_DIR.
    """
    print(f"--- Processando arquivos DBF locais de {INPUT_DIR} ---")
    
    # Encontra todos os arquivos DBF na pasta
    all_dbf_files = glob.glob(os.path.join(INPUT_DIR, "*.dbf"))
    
    if not all_dbf_files:
        print(f"ERRO: Nenhum arquivo .dbf encontrado em {INPUT_DIR}")
        print("Por favor, baixe os arquivos do DATASUS/TabWin e coloque-os nesta pasta.")
        return

    print(f"Encontrados {len(all_dbf_files)} arquivos .dbf para processar...")

    lista_dados_medicos = []
    lista_dados_leitos = []

    for filepath in all_dbf_files:
        filename = os.path.basename(filepath)
        tipo, uf, ano = extract_info_from_filename(filename)
        
        if not tipo:
            print(f"AVISO: Arquivo '{filename}' não segue o padrão de nome esperado (ex: PFMG1612.dbf) e será ignorado.")
            continue
            
        print(f"Processando: {filename} (Tipo: {tipo}, UF: {uf}, Ano: {ano})")
        df = load_dbf_as_df(filepath)
        if df.empty:
            print(f"  AVISO: DataFrame vazio para {filename}. Pulando.")
            continue
            
        # Processa Arquivos de Profissionais (PF)
        if tipo == 'PF':
            try:
                # Garante que os nomes das colunas estejam em maiúsculas (padrão DBF)
                df.columns = [col.upper() for col in df.columns]
                
                df_medicos_sus = df[
                    (df['PROF_SUS'] == '1') & 
                    (df['CBO'].astype(str).str.startswith(CODIGOS_MEDICOS_CBO_PREFIX))
                ]
                contagem_medicos = len(df_medicos_sus)
                
                lista_dados_medicos.append({
                    'UF': uf,
                    'ANO': ano,
                    'MEDICOS_SUS_TOTAL': contagem_medicos
                })
            except KeyError as e:
                print(f"  ERRO: Coluna {e} não encontrada em {filename}. Pulando arquivo.")
            except Exception as e_geral:
                print(f"  ERRO inesperado ao processar {filename}: {e_geral}")


        # Processa Arquivos de Leitos (LT)
        elif tipo == 'LT':
            try:
                # Garante que os nomes das colunas estejam em maiúsculas (padrão DBF)
                df.columns = [col.upper() for col in df.columns]
                
                colunas_leitos = ['CODLEITO', 'QT_SUS']
                df_leitos = df[colunas_leitos].copy()
                
                df_leitos['QT_SUS'] = pd.to_numeric(df_leitos['QT_SUS'], errors='coerce').fillna(0)
                df_agregado = df_leitos.groupby('CODLEITO')['QT_SUS'].sum().reset_index()
                
                leitos_obstetricia = df_agregado[df_agregado['CODLEITO'].isin(CODIGOS_LEITOS_OBSTETRICIA)]['QT_SUS'].sum()
                leitos_uti_neo = df_agregado[df_agregado['CODLEITO'].isin(CODIGOS_LEITOS_UTI_NEONATAL)]['QT_SUS'].sum()

                lista_dados_leitos.append({
                    'UF': uf,
                    'ANO': ano,
                    'LEITOS_OBSTETRICIA_SUS': leitos_obstetricia,
                    'LEITOS_UTI_NEONATAL_SUS': leitos_uti_neo
                })
            except KeyError as e:
                print(f"  ERRO: Coluna {e} não encontrada em {filename}. Pulando arquivo.")
            except Exception as e_geral:
                print(f"  ERRO inesperado ao processar {filename}: {e_geral}")

    # 3. Junta os dois DataFrames
    df_medicos = pd.DataFrame(lista_dados_medicos)
    df_leitos = pd.DataFrame(lista_dados_leitos)

    if df_medicos.empty and df_leitos.empty:
        print("\nERRO: Nenhum dado de Profissional ou Leitos foi processado.")
        return

    # Agrega por UF e Ano (caso haja duplicatas)
    if not df_medicos.empty:
        df_medicos = df_medicos.groupby(['UF', 'ANO']).sum().reset_index()
    if not df_leitos.empty:
        df_leitos = df_leitos.groupby(['UF', 'ANO']).sum().reset_index()

    if df_medicos.empty:
        print("\nAVISO: Nenhum dado de Profissional (PF) foi processado.")
        df_final = df_leitos
    elif df_leitos.empty:
        print("\nAVISO: Nenhum dado de Leitos (LT) foi processado.")
        df_final = df_medicos
    else:
        print("\nJuntando dados de Médicos e Leitos...")
        df_final = pd.merge(df_medicos, df_leitos, on=['UF', 'ANO'], how='outer')

    # Salva o arquivo final
    df_final.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"\n--- Sucesso! ---")
    print(f"Dados de infraestrutura do CNES salvos em: {OUTPUT_FILE}")
    print("\nAmostra dos dados processados:")
    print(df_final.head())


if __name__ == "__main__":
    process_locais_cnes_dbfs()