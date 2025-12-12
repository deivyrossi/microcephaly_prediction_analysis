import pandas as pd


features = ['EXANT_GES', 'FEBRE_GES', 'UFRES', 'IDADEGES']

df_pico = pd.read_csv('dados/processados/DADOS_PICO_EPIDEMICO.csv', sep = ';', encoding= 'latin1')
df_fora = pd.read.csv('dados/processados/DADOS_FORA_PICO', sep = ';', encoding= 'latin1')

missing_pico = (df_pico[features].isnull().sum() / len (df_pico)) * 100
missing_fora = (df_pico[features].isnull().sum() / len (df_fora)) * 100

tabela_missing = pd.DataFrame({
    'Pico (%):': missing_pico,
    'Fora(%):': missing_fora })

print(tabela_missing)   
tabela_missing.to_csv('resultados/tables/tabela_missings.csv')
