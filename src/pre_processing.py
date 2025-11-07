import pandas as pd
from simpledbf import Dbf5
import glob
import os

# --- 1. CONFIGURAÇÃO ---
print("Iniciando o script de preparação de dados...")
pasta_dados_brutos = 'dados/brutos/'
arquivo_pico_csv = os.path.join(pasta_dados_brutos, 'DADOS_PICO_EPIDEMICO.csv')
arquivo_fora_pico_csv = os.path.join(pasta_dados_brutos, 'DADOS_FORA_PICO.csv')

# --- 2. UNIFICAÇÃO DOS ARQUIVOS DBF ---
print("\n--- FASE 1: Unificando arquivos DBF ---")
caminhos_dbf = glob.glob(os.path.join(pasta_dados_brutos, '*.dbf'))

if not caminhos_dbf:
    print(f"ERRO: Nenhum arquivo .dbf encontrado na pasta '{pasta_dados_brutos}'.")
    exit()

print(f"Encontrados {len(caminhos_dbf)} arquivos DBF para processar.")
df_completo = pd.concat(
    [Dbf5(dbf, codec='latin1').to_dataframe() for dbf in caminhos_dbf],
    ignore_index=True
)
print(f"Total de registros brutos unificados: {len(df_completo)}")

# --- 3. LIMPEZA E DIVISÃO POR PERÍODO ---
print("\n--- FASE 2: Limpeza e Divisão por Período ---")

# Converte a coluna de data para o formato datetime [cite: 73, 129]
df_completo['DT_NOTIFIC'] = pd.to_datetime(df_completo['DT_NOTIFIC'], format='%d%m%Y', errors='coerce')
df_completo.dropna(subset=['DT_NOTIFIC'], inplace=True) # Remove registros com data inválida

# Define a data de corte para o fim do período epidêmico
data_corte = pd.to_datetime('2017-12-31')

# Cria os dois DataFrames baseados no período
df_pico = df_completo[df_completo['DT_NOTIFIC'] <= data_corte].copy()
df_fora_pico = df_completo[df_completo['DT_NOTIFIC'] > data_corte].copy()

print(f"Registros no período de pico (até {data_corte.date()}): {len(df_pico)}")
print(f"Registros no período fora de pico (após {data_corte.date()}): {len(df_fora_pico)}")

# --- 4. FILTRO ESTRATÉGICO FINAL ---
print("\n--- FASE 3: Filtrando por Casos com Desfecho Final ---")

def filtrar_por_classifin(df, nome_periodo):
    """Função para filtrar o DataFrame e imprimir um relatório."""
    print(f"\nProcessando período: {nome_periodo}")
    # Converte CLASSIFIN para numérico para garantir a comparação [cite: 133]
    df['CLASSIFIN'] = pd.to_numeric(df['CLASSIFIN'], errors='coerce')
    
    # Define as classes de interesse: 1 (Confirmado) e 5 (Descartado) [cite: 133]
    classes_finais = [1, 5]
    df_filtrado = df[df['CLASSIFIN'].isin(classes_finais)].copy()
    
    print("Distribuição das classes:")
    print(df_filtrado['CLASSIFIN'].value_counts())
    print(f"Total de registros válidos para o modelo: {len(df_filtrado)}")
    return df_filtrado

# Aplica o filtro em ambos os DataFrames
df_pico_final = filtrar_por_classifin(df_pico, "Pico Epidêmico")
df_fora_pico_final = filtrar_por_classifin(df_fora_pico, "Fora de Pico")

# --- 5. SALVANDO OS ARQUIVOS FINAIS ---
print("\n--- FASE 4: Salvando os arquivos processados ---")
df_pico_final.to_csv(arquivo_pico_csv, index=False, sep=';', encoding='latin1')
print(f"Arquivo do período de pico salvo em: '{arquivo_pico_csv}'")

df_fora_pico_final.to_csv(arquivo_fora_pico_csv, index=False, sep=';', encoding='latin1')
print(f"Arquivo do período fora de pico salvo em: '{arquivo_fora_pico_csv}'")

print("\nPreparação de dados concluída!")