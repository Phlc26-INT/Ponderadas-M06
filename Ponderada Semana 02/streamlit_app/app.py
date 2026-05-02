# Codigo feito com auxilio de IA.
# Para um melhor entendimento, foi feito um notebook (NotebookLM) com o conteudo dos autoestudos.
# Explicacoes foram feitas de acordo com o que foi aprendido no notebook referenciado.

# Permite usar anotacoes de tipo mais simples.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import yaml

# Define caminhos e garante acesso ao pacote local.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importa componentes do pipeline.
from sentiment_pipeline.classification import Classifier
from sentiment_pipeline.preprocessing import Preprocessor
from sentiment_pipeline.utils import load_dataset_csv, stratified_split
from sentiment_pipeline.vectorization import Vectorizer

# Define caminhos usados pela demo.
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"
RESULTS_DIR = ROOT / "experiments" / "results"
BEST_CONFIG_PATH = RESULTS_DIR / "best_config.json"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"


# Le um arquivo YAML simples.
def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# Le um arquivo JSON simples.
def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Treina o pipeline completo e guarda em cache para a interface.
@st.cache_resource(show_spinner="Training model with best config...")
def train_pipeline() -> Tuple[Preprocessor, Vectorizer, Classifier, str, str]:
    # Valida se os arquivos principais existem.
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Config file not found: configs/baseline.yaml")
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "best_config.json not found. Run experiments first to generate it."
        )

    # Carrega configuracao e dados.
    cfg = read_yaml(CONFIG_PATH)
    dataset_cfg = cfg["dataset"]
    split_cfg = cfg.get("split", {})

    # Le o dataset e prepara a coluna de label.
    df = load_dataset_csv(
        dataset_cfg["path"],
        text_col=dataset_cfg["text_col"],
        label_col=dataset_cfg.get("label_col"),
        rating_col=dataset_cfg.get("rating_col"),
        label_cfg=dataset_cfg.get("label", {}),
    )

    # Separa treino e validacao para montar o treino final.
    X_train, X_val, _, y_train, y_val, _ = stratified_split(
        df,
        text_col=dataset_cfg["text_col"],
        label_col="label",
        train_size=split_cfg.get("train_size", 0.7),
        val_size=split_cfg.get("val_size", 0.15),
        test_size=split_cfg.get("test_size", 0.15),
        random_state=split_cfg.get("random_state", 42),
    )

    # Usa a melhor configuracao para treinar o modelo final.
    best_config = read_json(BEST_CONFIG_PATH)
    prep = Preprocessor(best_config.get("prep", {}))

    # Junta treino e validacao para o treino final.
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    # Preprocessa e vetoriza o texto.
    tokens_train = prep.transform(X_train_val)
    vec = Vectorizer(best_config.get("vec", {}))
    X_train_vec = vec.fit_transform(tokens_train)

    # Treina o classificador final.
    clf = Classifier(best_config.get("clf", {}))
    clf.fit(X_train_vec, y_train_val)

    return prep, vec, clf, dataset_cfg["text_col"], dataset_cfg["path"]


# Faz predicoes para um DataFrame enviado pelo usuario.
def predict_texts(
    df: pd.DataFrame, text_col: str, prep: Preprocessor, vec: Vectorizer, clf: Classifier
) -> pd.DataFrame:
    texts = df[text_col].fillna("").astype(str).tolist()
    tokens = prep.transform(texts)
    X_vec = vec.transform(tokens)
    preds = clf.predict(X_vec)
    output = df.copy()
    output["prediction"] = preds
    output["prediction_label"] = ["negative" if p == 0 else "positive" for p in preds]
    return output


# Formata metricas para exibicao.
def format_metric(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "--"


# Configura a pagina do Streamlit.
st.set_page_config(page_title="Sentiment Pipeline Demo", layout="wide")

# Injeta CSS para customizar a aparencia.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Newsreader:wght@300;400;600&display=swap');

      html, body, [class*="css"] {
        font-family: "Newsreader", serif;
                color: #141414;
      }

            [data-testid="stMarkdownContainer"] p {
                color: #1f1f1f;
            }

            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4,
            [data-testid="stMarkdownContainer"] h5,
            [data-testid="stMarkdownContainer"] h6 {
                color: #141414;
            }

            [data-testid="stMarkdownContainer"] a {
                color: #0b4f6c;
            }

            [data-testid="stFileUploader"] label {
                color: #141414;
                font-weight: 600;
            }

      [data-testid="stAppViewContainer"] {
                background: #f5f3ec;
      }

      .hero {
                background: #ffffff;
                border: 1px solid #c9c2b4;
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 18px 40px rgba(20, 20, 21, 0.12);
                color: #141414;
      }

      .hero h1 {
        font-family: "Space Grotesk", sans-serif;
        font-size: 34px;
        margin: 8px 0 8px;
                color: #141414;
      }

            .hero p {
                color: #1f1f1f;
            }

      .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
                border: 1px solid #c9c2b4;
                background: #f4efe6;
        font-family: "Space Grotesk", sans-serif;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
                color: #141414;
      }

      .card {
        background: #fff;
        border-radius: 18px;
                border: 1px solid #c9c2b4;
        padding: 16px;
                color: #141414;
      }

            .stButton > button {
                background: #141414;
                color: #fff;
                border: none;
                padding: 0.6rem 1.2rem;
                border-radius: 999px;
                font-weight: 600;
            }

            .stButton > button:hover {
                background: #2f2f2f;
                color: #fff;
            }

            .stButton > button:disabled {
                background: #5c5c5c;
                color: #f2f2f2;
            }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero section com titulo e descricao.
st.markdown(
    """
    <div class="hero">
      <div class="badge">Pipeline Demo</div>
      <h1>Sentiment Classification</h1>
      <p>Upload a CSV, pick the text column, and run predictions with the best configuration.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tenta treinar o pipeline e interrompe se houver erro.
try:
    prep, vec, clf, default_text_col, dataset_path = train_pipeline()
except Exception as exc:
    st.error(f"Model is not ready: {exc}")
    st.stop()

# Cartoes de informacao com dados basicos.
cols = st.columns(3)
with cols[0]:
    st.markdown("<div class='card'><strong>Training data</strong><br/>", unsafe_allow_html=True)
    st.write(dataset_path)
    st.markdown("</div>", unsafe_allow_html=True)
with cols[1]:
    st.markdown("<div class='card'><strong>Best config</strong><br/>", unsafe_allow_html=True)
    st.write(BEST_CONFIG_PATH.name)
    st.markdown("</div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown("<div class='card'><strong>Test metrics</strong><br/>", unsafe_allow_html=True)
    if TEST_METRICS_PATH.exists():
        metrics = read_json(TEST_METRICS_PATH)
        st.write(f"F1 macro: {format_metric(metrics.get('f1_macro'))}")
        st.write(f"Accuracy: {format_metric(metrics.get('accuracy'))}")
    else:
        st.write("Not found")
    st.markdown("</div>", unsafe_allow_html=True)

# Separador visual.
st.markdown("---")

# Estado interno para evitar recomputar sem necessidade.
if "predicting" not in st.session_state:
    st.session_state["predicting"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None
if "results_col" not in st.session_state:
    st.session_state["results_col"] = None
if "last_file" not in st.session_state:
    st.session_state["last_file"] = None

# Upload do CSV do usuario.
uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"])
if uploaded is not None:
    file_id = getattr(uploaded, "name", "uploaded")
    if st.session_state["last_file"] != file_id:
        st.session_state["results"] = None
        st.session_state["results_col"] = None
        st.session_state["last_file"] = file_id

    # Le o CSV e valida se ha colunas.
    data = pd.read_csv(uploaded, low_memory=False)
    if not len(data.columns):
        st.warning("CSV has no columns.")
        st.stop()

    # Define a coluna de texto padrao.
    default_idx = 0
    if default_text_col in data.columns:
        default_idx = list(data.columns).index(default_text_col)

    # Seletor de coluna de texto.
    text_col = st.selectbox(
        "Select the text column",
        options=list(data.columns),
        index=default_idx,
    )

    # Botao para executar predicao.
    run_clicked = st.button(
        "Run prediction",
        type="primary",
        disabled=st.session_state["predicting"],
    )

    if run_clicked:
        st.session_state["predicting"] = True
        try:
            with st.spinner("Running predictions..."):
                results = predict_texts(data, text_col, prep, vec, clf)
                display = results[[text_col, "prediction_label"]].rename(
                    columns={"prediction_label": "sentiment"}
                )
                st.session_state["results"] = display
                st.session_state["results_col"] = text_col
            st.success(f"Predicted {len(display)} rows.")
        finally:
            st.session_state["predicting"] = False

    # Mostra resultados e libera download.
    if st.session_state["results"] is not None:
        if st.session_state["results_col"] != text_col:
            st.info("Selection changed. Click Run prediction to refresh results.")
        else:
            st.dataframe(st.session_state["results"], use_container_width=True)

            csv_bytes = st.session_state["results"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
