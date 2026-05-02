# Permite usar anotacoes de tipo mais simples.
from __future__ import annotations

import copy
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Importa os componentes do pipeline.
from .classification import Classifier
from .preprocessing import Preprocessor
from .vectorization import Vectorizer


# Executa varios experimentos e devolve a melhor configuracao.
def run_experiments(
    experiment_configs: List[Dict],
    X_train,
    y_train,
    X_val,
    y_val,
    metric: str = "f1_macro",
) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    best_score = -1.0
    best_config = None

    # Percorre as configuracoes e mede desempenho na validacao.
    for cfg in experiment_configs:
        label = cfg.get("label", "experiment")
        start = time.perf_counter()

        # Preprocessa texto de treino e validacao.
        prep = Preprocessor(cfg.get("prep", {}))
        tokens_train = prep.transform(X_train)
        tokens_val = prep.transform(X_val)

        # Ajusta o vetorizador no treino e transforma a validacao.
        vec = Vectorizer(cfg.get("vec", {}))
        X_train_vec = vec.fit_transform(tokens_train)
        X_val_vec = vec.transform(tokens_val)

        # Treina o classificador e evita vazamento na validacao.
        clf_cfg = cfg.get("clf", {})
        clf = Classifier(clf_cfg)
        refit_full = None
        if clf_cfg.get("mode") in {"grid", "random"} and clf_cfg.get("search_on", "validation") == "validation":
            # Mantem a validacao fora do treino antes de pontuar.
            refit_full = False
        clf.fit(X_train_vec, y_train, X_val_vec, y_val, refit_on_full_train=refit_full)
        preds = clf.predict(X_val_vec)

        # Calcula metricas da validacao.
        f1_macro = f1_score(y_val, preds, average="macro")
        acc = accuracy_score(y_val, preds)
        elapsed = time.perf_counter() - start

        row = {
            "label": label,
            "f1_macro": f1_macro,
            "accuracy": acc,
            "elapsed_sec": round(elapsed, 3),
        }

        # Guarda os melhores parametros, se existirem.
        if clf.best_params_ is not None:
            row["best_params"] = json.dumps(clf.best_params_, sort_keys=True)
        rows.append(row)

        # Atualiza a melhor configuracao.
        score = row.get(metric)
        if score is not None and score > best_score:
            best_score = score
            best_config = _prepare_best_config(cfg, clf)

    # Deixa os resultados ordenados pela metrica escolhida.
    results = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    return results, best_config


# Repassa a melhor config e mede desempenho no teste final.
def evaluate_best_on_test(
    best_config: Dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> Dict:
    # Preprocessador escolhido pela melhor config.
    prep = Preprocessor(best_config.get("prep", {}))

    # Junta treino e validacao para treinar o modelo final.
    X_train_val = _concat_series(X_train, X_val)
    y_train_val = _concat_series(y_train, y_val)

    # Transforma treino+validacao e teste com o mesmo pre-processamento.
    tokens_train_val = prep.transform(X_train_val)
    tokens_test = prep.transform(X_test)

    # Ajusta o vetorizador no treino e aplica no teste.
    vec = Vectorizer(best_config.get("vec", {}))
    X_train_val_vec = vec.fit_transform(tokens_train_val)
    X_test_vec = vec.transform(tokens_test)

    # Treina o classificador final e faz predicoes no teste.
    clf = Classifier(best_config.get("clf", {}))
    clf.fit(X_train_val_vec, y_train_val)
    preds = clf.predict(X_test_vec)

    # Retorna metricas finais.
    return {
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "accuracy": accuracy_score(y_test, preds),
    }


# Atualiza a config com os melhores parametros encontrados.
def _prepare_best_config(cfg: Dict, clf: Classifier) -> Dict:
    best = copy.deepcopy(cfg)
    if clf.best_params_ is not None:
        best_clf = dict(best.get("clf", {}))
        best_clf["mode"] = "manual"
        best_clf["params"] = clf.best_params_
        best["clf"] = best_clf
    return best


# Une duas series e reorganiza o indice.
def _concat_series(a, b):
    return pd.concat([a, b], axis=0).reset_index(drop=True)
