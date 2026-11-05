# 07 — Previsão de demanda com ML

> **Machine learning · UCI / Kaggle** · Status: **Planejado · standby**  
> Parte do portfólio [Beatriz Helena — Engenharia de IA](https://beatrizorlandi.github.io/projetos/pipeline-ml-demanda.html)

Planejamento de um ciclo completo de ML para demanda horária de bicicletas, com features causais e avaliação temporal.

**Stack principal:** `Python` · `PySpark` · `scikit-learn` · `MLflow` · `Unity Catalog` · `KaggleHub (planejado)`

---

## O Problema

Estimar a próxima janela de demanda exige reproduzir o que estaria disponível no momento da decisão, sem usar o futuro.

### Escopo Entregue

Design de aquisição UCI ou Kaggle, contratos, features, splits, baselines, treino, MLflow, inferência e monitoramento. Nenhum dataset foi baixado ou modelo treinado.

---

## Arquitetura e Fluxo

**1. UCI ou Kaggle** → **2. Landing + manifesto** → **3. Bronze / Silver** → **4. Features causais** → **5. Split temporal + treino** → **6. MLflow + gate** → **7. Inferência + monitoramento**

### Decisões Técnicas

- Escolher uma fonte por versão e não concatenar distribuições potencialmente sobrepostas.
- Excluir componentes do target e clima observado da hora futura das features operacionais iniciais.
- Separar treino e inferência e preservar o baseline se o candidato não demonstrar ganho.

---

## Validação e Critérios de Aceite

Aceite futuro: manifesto da fonte e licença, testes de availability_time, splits temporais congelados e comparação sob o mesmo orçamento. As metas do plano não são resultados de treinamento.


---

## Limites Conhecidos e Próximos Passos

### Limites Conhecidos
- Depende da aquisição externa de dados e das condições de acesso e licença.
- Não há métrica de predição, modelo registrado ou serving executado nesta trilha.

### Próximos Incrementos
- Adquirir e manifestar a fonte escolhida.
- Implementar baseline sazonal e features com disponibilidade temporal verificável.

---

## Estrutura do Repositório

- `informacoes_tecnicas/PLANEJAMENTO.md`: Planejamento técnico detalhado (requisitos, contratos, arquitetura, MLflow e validação).
- `00_PLANEJAMENTO.py`: Versão em notebook Databricks do planejamento técnico.
- Notebooks e módulos em Python/PySpark (quando aplicável ao estágio de implementação do estudo).
- `evidencias/` (quando aplicável): Resumo estático das execuções validadas no Databricks.

## Links

- **Portfólio Completo:** [https://beatrizorlandi.github.io](https://beatrizorlandi.github.io)
- **Estudo de Caso no Site:** [https://beatrizorlandi.github.io/projetos/pipeline-ml-demanda.html](https://beatrizorlandi.github.io/projetos/pipeline-ml-demanda.html)
- **Planejamento Técnico no Site:** [https://beatrizorlandi.github.io/documentacao/pipeline-ml-demanda.html](https://beatrizorlandi.github.io/documentacao/pipeline-ml-demanda.html)

## Execução Validada no Databricks (05/11/2026)

- **Notebook executado:** `01_pipeline_ml_demanda_horaria.py`
- **Tabelas geradas:**
  - Bronze: `workspace.estudos_bronze.bikes_raw` (4.320 horas)
  - Silver: `workspace.estudos_silver.bikes_horario`
  - Gold Features: `workspace.estudos_gold.bikes_features` (4.152 janelas com histórico completo de 168h e 14 features causais respeitando atraso de disponibilidade de 1h)
  - Gold Predições: `workspace.estudos_gold.bikes_predicoes` (831 horas de teste)
  - Monitoramento: `workspace.estudos_ops.bikes_monitoramento`
- **Modelo registrado no Unity Catalog:** `workspace.estudos_ai.demanda_bikes` (versão 1)
- **Métricas comparativas (MLflow):**
  - Baseline Média Global: `val_mae = 49,203` | `test_mae = 48,807`
  - Baseline Sazonal (Hora × Fim de Semana): `val_mae = 14,318` | `test_mae = 11,607`
  - `HistGradientBoostingRegressor` Causal: `val_mae = 13,452` (-6,05% vs. sazonal) | **`test_mae = 10,579`** (`test_rmse = 13,426`).
