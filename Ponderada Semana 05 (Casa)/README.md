# Classificação Binária de Recomendação - B2W Reviews

Projeto adaptado do tutorial oficial do TensorFlow de [Classificação de texto básica](https://www.tensorflow.org/tutorials/keras/text_classification?hl=pt-br) para o conjunto de dados **B2W-Reviews01**.

O objetivo é prever, a partir do texto de uma avaliação de cliente, se ele recomendaria o produto a um amigo (coluna `recommend_to_a_friend`, com valores `Yes` ou `No`).

## Estrutura do projeto

```
b2w-reviews-classification/
├── data/
│   └── B2W-Reviews01.csv          # dataset de avaliações
├── models/                         # modelos treinados (gerados pelo notebook)
│   ├── export_model.keras
│   └── export_model.pkl
├── notebooks/
│   └── b2w_text_classification.ipynb
├── requirements.txt
└── README.md
```

## Como executar

1. Abra o notebook `notebooks/b2w_text_classification.ipynb` no Google Colab ou Jupyter local.
2. No Colab, faça o upload do CSV de `data/` ou ajuste a variável `DATA_PATH`.
3. Execute as células em ordem. O modelo treinado é salvo em `models/`.

## Função de inferência

Após o treinamento, a função `classificar_recomendacao(texto, pickle_path)` carrega o modelo salvo com `pickle` e devolve a classe predita (`Yes`/`No`) e a probabilidade.
