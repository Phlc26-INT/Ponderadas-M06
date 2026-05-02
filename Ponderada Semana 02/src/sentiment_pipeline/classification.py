# Permite usar anotacoes de tipo mais simples.
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, ParameterSampler, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC

# Importa LightGBM se estiver disponivel.
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


# Wrapper simples para varios classificadores e modos de busca.
class Classifier:
    def __init__(self, config: dict):
        # Configuracoes gerais do classificador.
        self.config = config or {}
        self.model_name = self.config.get("model", "logreg")
        self.mode = self.config.get("mode", "manual")
        self.random_state = self.config.get("random_state", 42)
        self.scoring = self.config.get("scoring", "f1_macro")
        self.n_jobs = self.config.get("n_jobs", -1)
        self.cv = int(self.config.get("cv", 5))

        # Campos preenchidos depois do treino.
        self._model = None
        self.best_params_ = None
        self.best_score_ = None

    # Treina o modelo conforme o modo escolhido.
    def fit(self, X_train, y_train, X_val=None, y_val=None, refit_on_full_train: Optional[bool] = None):
        if self.mode == "manual":
            # Usa parametros fixos.
            params = self.config.get("params", {})
            self._model = self._build_model(params)
            self._model.fit(self._maybe_dense(X_train), y_train)
            return self

        if self.mode not in {"grid", "random"}:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Decide se a busca usa validacao separada ou CV.
        search_on = self.config.get("search_on", "validation")
        if search_on == "validation":
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val are required for validation search")
            # Testa combinacoes de parametros na validacao.
            best_params, best_score = self._search_on_validation(
                X_train, y_train, X_val, y_val
            )
            self.best_params_ = best_params
            self.best_score_ = best_score
            if refit_on_full_train is None:
                refit_full = self.config.get("refit_on_full_train", True)
            else:
                refit_full = refit_on_full_train
            # Reajusta o modelo com os melhores parametros.
            self._refit_after_search(X_train, y_train, X_val, y_val, refit_full)
            return self

        # Busca em CV com GridSearch ou RandomizedSearch.
        if self.mode == "grid":
            param_grid = self._get_param_grid()
            search = GridSearchCV(
                self._build_model({}),
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
            )
        else:
            param_dist = self._get_param_dist()
            search = RandomizedSearchCV(
                self._build_model({}),
                param_distributions=param_dist,
                scoring=self.scoring,
                n_iter=int(self.config.get("n_iter", 20)),
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # Ajusta e guarda o melhor estimador.
        search.fit(self._maybe_dense(X_train), y_train)
        self._model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        return self

    # Faz predicoes com o modelo treinado.
    def predict(self, X):
        if self._model is None:
            raise ValueError("Model is not fitted")
        return self._model.predict(self._maybe_dense(X))

    # Cria o modelo de acordo com o nome escolhido.
    def _build_model(self, params: Dict):
        if self.model_name == "multinomial_nb":
            return MultinomialNB(**params)
        if self.model_name == "bernoulli_nb":
            return BernoulliNB(**params)
        if self.model_name == "logreg":
            base = {"max_iter": 2000, "solver": "liblinear"}
            base.update(params)
            return LogisticRegression(**base)
        if self.model_name == "svc":
            base = {"max_iter": 2000}
            base.update(params)
            return LinearSVC(**base)
        if self.model_name == "rf":
            base = {"n_estimators": 200, "n_jobs": self.n_jobs, "random_state": self.random_state}
            base.update(params)
            return RandomForestClassifier(**base)
        if self.model_name == "lgbm":
            if LGBMClassifier is None:
                raise ImportError("lightgbm is required for model='lgbm'")
            base = {"n_estimators": 200, "n_jobs": self.n_jobs, "random_state": self.random_state}
            base.update(params)
            return LGBMClassifier(**base)
        raise ValueError(f"Unknown model: {self.model_name}")

    # Faz busca de hiperparametros usando um conjunto de validacao.
    def _search_on_validation(self, X_train, y_train, X_val, y_val) -> Tuple[Dict, float]:
        if self.mode == "grid":
            param_iter = ParameterGrid(self._get_param_grid())
        else:
            param_iter = ParameterSampler(
                self._get_param_dist(),
                n_iter=int(self.config.get("n_iter", 20)),
                random_state=self.random_state,
            )

        best_score = -1.0
        best_params = None
        for params in param_iter:
            model = self._build_model(params)
            model.fit(self._maybe_dense(X_train), y_train)
            preds = model.predict(self._maybe_dense(X_val))
            score = self._score(y_val, preds)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    # Reajusta o modelo com os melhores parametros.
    def _refit_after_search(self, X_train, y_train, X_val, y_val, refit_full: bool):
        params = self.best_params_ or {}
        if refit_full and X_val is not None:
            X_full, y_full = self._stack(X_train, y_train, X_val, y_val)
            self._model = self._build_model(params)
            self._model.fit(self._maybe_dense(X_full), y_full)
        else:
            self._model = self._build_model(params)
            self._model.fit(self._maybe_dense(X_train), y_train)

    # Junta treino e validacao, considerando matriz esparsa.
    def _stack(self, X_train, y_train, X_val, y_val):
        try:
            from scipy import sparse
        except ImportError:
            sparse = None

        if sparse is not None and sparse.issparse(X_train):
            X_full = sparse.vstack([X_train, X_val])
        else:
            X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        return X_full, y_full

    # Converte para denso quando o modelo precisa.
    def _maybe_dense(self, X):
        if self.model_name in {"rf"} and hasattr(X, "toarray"):
            return X.toarray()
        return X

    # Calcula a metrica definida na config.
    def _score(self, y_true, y_pred) -> float:
        if self.scoring == "f1_macro":
            return f1_score(y_true, y_pred, average="macro")
        raise ValueError(f"Unsupported scoring: {self.scoring}")

    # Retorna grid de parametros.
    def _get_param_grid(self) -> Dict:
        return self.config.get("param_grid") or self._default_param_grid()

    # Retorna distribuicao de parametros.
    def _get_param_dist(self) -> Dict:
        return self.config.get("param_dist") or self._default_param_grid()

    # Defaults simples caso nao haja grid definido.
    def _default_param_grid(self) -> Dict:
        if self.model_name in {"multinomial_nb", "bernoulli_nb"}:
            return {"alpha": [0.5, 1.0, 2.0]}
        if self.model_name == "logreg":
            return {"C": [0.5, 1.0, 2.0], "class_weight": [None, "balanced"]}
        if self.model_name == "svc":
            return {"C": [0.5, 1.0, 2.0], "class_weight": [None, "balanced"]}
        if self.model_name == "rf":
            return {
                "n_estimators": [200, 500],
                "max_depth": [None, 20],
                "min_samples_split": [2, 5],
            }
        if self.model_name == "lgbm":
            return {
                "n_estimators": [200, 500],
                "num_leaves": [31, 63],
                "learning_rate": [0.05, 0.1],
            }
        raise ValueError(f"Unknown model: {self.model_name}")
