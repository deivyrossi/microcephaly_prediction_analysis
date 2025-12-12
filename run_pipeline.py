import subprocess
import sys
import os

def run_step(script_path, step_name):
    """Executa um script Python e verifica se houve erro."""
    print(f"\n{'='*60}")
    print(f"INICIANDO PASSO: {step_name}")
    print(f"Executando: {script_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(script_path):
        print(f"ERRO: O arquivo '{script_path}' não foi encontrado.")
        sys.exit(1)

    try:
        # Executa o script
        subprocess.run([sys.executable, script_path], check=True)
        print(f"\nSUCESSO: {step_name} concluído.")
    except subprocess.CalledProcessError:
        print(f"\nERRO: Falha na execução de {step_name}.")
        sys.exit(1)

def main():
    print("--- PIPELINE DE REPRODUÇÃO DO ARTIGO ---\n")

    # 1. ETL (Preparação dos Dados)
    # Se você tiver os dados brutos e quiser processar do zero:
    # run_step("src/pre_processing.py", "Processamento do RESP (DBF -> CSV)")
    # run_step("src/explore_sinan.py", "Processamento do SINAN (Zika)")
    # run_step("src/explore_cnes.py", "Processamento do CNES (Infra)")
    
    # OBS: Se estiver usando os dados já prontos (CSV) do Zenodo na pasta dados/processados,
    # pode pular os passos acima comentando-os.

    # 2. Treinamento
    print("\n[INFO] Iniciando treinamento do modelo")
    run_step("src/train.py", "Treinamento e Validação")

    # 3. Análise e Gráficos
    print("\n[INFO] Gerando gráficos e tabelas")
    run_step("src/analysis.py", "Geração de Resultados")
    
    # 4. Bootstrap (Sensibilidade)
    print("\n[INFO] Rodando análise de sensibilidade")
    run_step("src/sensitivity_analysis.py", "Bootstrap Sensitivity")

    print(f"\n{'='*60}")
    print("🎉 PIPELINE FINALIZADO! Todos os resultados estão na pasta 'resultados'")

if __name__ == "__main__":
    main()