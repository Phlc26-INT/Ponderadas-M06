# Pipeline de Sentimentos

Pipeline configuravel para analise de sentimentos em reviews PT-BR. O projeto e modular e reutilizavel: preprocessamento, vetorizacao e modelos classicos de ML sao configurados via dicionarios ou YAML.

## Funcionalidades

- Preprocessamento configuravel (lowercase, stopwords, stemming, lematização, tratamento de negação, remoção de URLs, normalização de emojis)
- Múltiplos vetorizadores (BoW, TF-IDF, TF-IDF + SVD, Word2Vec com media de embeddings)
- Múltiplos classificadores (Naive Bayes, Regressão Logística, LinearSVC, Random Forest, LightGBM)
- Selecao automática ou manual de hiperpârametros
- Split de treino/validação/teste com estratificação e avaliação final apenas no teste

## Estrutura do projeto

```
configs/
  baseline.yaml
scripts/
  run_experiments.py
streamlit_app/
  app.py
src/
  sentiment_pipeline/
    __init__.py
    preprocessing.py
    vectorization.py
    classification.py
    pipeline.py
    utils.py
```

## Configuração

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"
python -m spacy download pt_core_news_sm
```

## Inicio rápido

A configuração baseline já está ligada ao dataset fornecido:

- Coluna de texto: `review_text`
- Coluna de nota: `overall_rating`
- Regra de rótulo: 1-2 = negativo (0), 4-5 = positivo (1), 3 e descartado por padrao

Execute:

```bash
python scripts/run_experiments.py --config configs/baseline.yaml
```

As saídas são salvas em `experiments/results/`:

- `results.csv` com métricas de validacao por experimento
- `best_config.json` com a melhor configuração (com base no F1 macro de validação)
- `test_metrics.json` com as métricas finais no teste (opcional com `--skip-test`)

## Demo no Streamlit

Execute os experimentos primeiro para garantir que `best_config.json` exista e depois inicie a demo:

```bash
streamlit run streamlit_app/app.py
```

Envie um CSV, escolha a coluna de texto e o app adicionará as predições.

## Usando outro dataset

Se o seu dataset ja tiver rotulos, defina `label_col` e `label_mapping` no config:

```yaml
dataset:
  path: "data/my_dataset.csv"
  text_col: "text"
  label_col: "label"
  label:
    label_mapping:
      negative: 0
      positive: 1
```

## Observações sobre Word2Vec

Para embeddings pré-treinados, defina `pretrained_path` e `pretrained_binary` no config do vetorizador. Se você nao fornecer um caminho, a pipeline treina Word2Vec no corpus usando os parâmetros fornecidos.
