# Análise Preditiva e Generalização Temporal de Microcefalia no Brasil (2015-2024)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Este repositório contém o código-fonte oficial e os artefatos de reprodução para o artigo **"Análise Preditiva e Generalização Temporal de um Modelo de Árvore de Decisão para Classificação de Casos de Microcefalia no Brasil"**.

O projeto aplica técnicas de Machine Learning (Random Forest) para classificar casos suspeitos de microcefalia, com foco na robustez temporal e no diagnóstico de *Concept Drift* entre o período epidêmico do Zika Vírus (2015–2017) e o período pós-epidêmico.

## ⚠️ Aviso de Privacidade e Dados (LGPD)

**Os dados brutos contendo informações individuais de pacientes (RESP-Microcefalia) NÃO estão incluídos neste repositório.**

Em conformidade com a **Lei Geral de Proteção de Dados (LGPD — Lei nº 13.709/2018)**, dados que permitam reidentificação (datas de nascimento completas, municípios, etc.) foram excluídos do controle de versão.

### Como obter os dados para reprodução?
Os dados devem ser obtidos diretamente das fontes públicas oficiais:

1. **RESP-Microcefalia:** portal de dados abertos do DATASUS.  
2. **SINAN-Zika:** arquivos de notificações (`.dbf`).  
3. **CNES:** dados de infraestrutura (leitos e profissionais).

Coloque os arquivos baixados em `dados/brutos/` seguindo a estrutura abaixo.

## 📂 Estrutura do Repositório

```text
.
├── src/
│   ├── explore_sinan.py    # Processamento dos dados do SINAN (Gera features de Zika)
│   ├── train.py            # Pipeline de Treino, Validação Cruzada e Testes Éticos
│   └── analysis.py         # Avaliação de Drift, SHAP, Bootstrap e McNemar
├── dados/
│   ├── brutos/             # [Vazio] Local para colocar os arquivos baixados do DATASUS
│   └── processados/        # [Vazio] Local onde os scripts salvarão os dados limpos
├── resultados_v7/          # [GitIgnore] Artefatos gerados (Gráficos, Tabelas, Logs)
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
```

---

## 🛠️ Instalação e Ambiente

Recomenda-se usar um ambiente virtual (`venv` ou `conda`).

```bash
# clonar o repositório
git clone https://github.com/deivyrossi/microcephaly_prediction_analysis.git
cd microcephaly_prediction_analysis

# criar/ativar venv (exemplo com venv)
python3 -m venv venv
source venv/bin/activate  # Linux / macOS
# .\venv\Scripts\activate # Windows (PowerShell)

# instalar dependências
pip install -r requirements.txt
```

Principais bibliotecas: `scikit-learn`, `pandas`, `shap`, `imbalanced-learn`, `simpledbf`.

---

## 🚀 Como Reproduzir os Resultados

Siga a ordem abaixo para garantir integridade do pipeline.

### Passo 1 — Engenharia de Features (SINAN)

```bash
# processa arquivos .dbf do SINAN, agrega por semana epidemiológica (ISO 8601)
python src/explore_sinan.py
```

**Entrada:** arquivos em `dados/brutos/SINAN_ZIKA/`  
**Saída:** `dados/processados/SINAN_ZIKA_AGREGADO_SEMANAL_UF.csv`

### Passo 2 — Treinamento e Validação (Modelo v7.4)

```bash
# rodar com busca de hiperparâmetros (RandomizedSearch)
python src/train.py --search --n_iter 50

# ou rodar com os parâmetros fixos finais
python src/train.py
```

**Saída:** modelos serializados (`.joblib`), conjuntos de teste processados e relatórios comparativos (SMOTE vs class_weight) no console e em `resultados_v7/`.

### Passo 3 — Análise de Drift e Explicabilidade

```bash
# gera curvas Precision-Recall, intervalos via Block Bootstrap e gráficos SHAP
python src/analysis.py
```

**Saída:** gráficos (`.png`) e tabelas de métricas (`.csv`) em `resultados_v7/`.

---

## 📊 Metodologia e Resultados Principais

O modelo final (`RandomForestClassifier`) foi treinado com `class_weight='balanced'` e validação temporal (TimeSeriesSplit). Semente global: `RANDOM_STATE = 42`.

| Cenário                    | Recall (Classe Confirmada) |
|---------------------------:|:--------------------------:|
| Pico Epidêmico (2015–2017) | 69.0% (IC 95%: 66.7%–71.7%) |
| Fora de Pico (2018–2024)   | 44.0% (IC 95%: 39.4%–48.9%) |

**Conclusões rápidas**
- Concept Drift confirmado: queda do sinal epidemiológico fora do pico.  
- Mitigação de desbalanceamento: `class_weight='balanced'` teve melhor recall (0.69) do que `SMOTE` (0.51) e `Threshold Tuning` (0.52).  
- Alinhamento temporal: uso de `isocalendar()` (ISO 8601) para semanas epidemiológicas, evitando vazamento temporal.  
- Reprodutibilidade garantida com `RANDOM_STATE = 42`.

---

## 📝 Citação

Se usar este código/metodologia em pesquisa, cite:

```bibtex
@misc{melo2025microcephaly,
  author       = {Melo, Deivy R. T.},
  title        = {Source code for predictive analysis of microcefalia cases},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/deivyrossi/microcephaly_prediction_analysis}},
  commit       = {4d123d9c5348b3a249497bd620cf36c07c59fda9}
}
```

---

## Contato

```text
Email: deivyrossi@gmail.com
```
