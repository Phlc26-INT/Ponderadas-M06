# Permite usar anotacoes de tipo mais simples.
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Converte notas em rotulos binarios (positivo/negativo).
def binarize_ratings(
    ratings: pd.Series,
    pos_threshold: int = 4,
    neg_threshold: int = 2,
    drop_neutral: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    ratings = pd.to_numeric(ratings, errors="coerce")
    pos_mask = ratings >= pos_threshold
    neg_mask = ratings <= neg_threshold
    if drop_neutral:
        # Remove notas neutras e devolve labels e mascara.
        mask = pos_mask | neg_mask
        labels = pd.Series(np.where(pos_mask, 1, 0), index=ratings.index)
        labels = labels[mask]
        return labels, mask

    # Mantem todos os exemplos validos.
    mask = ratings.notna()
    labels = pd.Series(np.where(pos_mask, 1, 0), index=ratings.index)
    return labels, mask


# Mapeia labels de texto para numeros usando um dicionario.
def map_labels(series: pd.Series, mapping: Dict) -> pd.Series:
    labels = series.map(mapping)
    if labels.isna().any():
        raise ValueError("Label mapping produced NaN values. Check label_mapping.")
    return labels


# Carrega CSV e cria a coluna "label" conforme a config.
def load_dataset_csv(
    path: str,
    text_col: str,
    label_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    label_cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    label_cfg = label_cfg or {}
    df = pd.read_csv(path)

    # Usa label pronta ou cria a partir de notas.
    if label_col:
        labels = df[label_col]
        mapping = label_cfg.get("label_mapping")
        if mapping:
            labels = map_labels(labels, mapping)
        df = df.copy()
        df["label"] = labels
    else:
        if rating_col is None:
            raise ValueError("rating_col is required when label_col is not provided")
        labels, mask = binarize_ratings(
            df[rating_col],
            pos_threshold=label_cfg.get("pos_threshold", 4),
            neg_threshold=label_cfg.get("neg_threshold", 2),
            drop_neutral=label_cfg.get("drop_neutral", True),
        )
        df = df.loc[mask].copy()
        df["label"] = labels.values

    # Remove linhas invalidas e reseta indice.
    df = df.dropna(subset=[text_col, "label"]).reset_index(drop=True)
    return df


# Divide o dataset em treino, validacao e teste com estratificacao.
def stratified_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    # Separa features e labels.
    X = df[text_col]
    y = df[label_col]

    # Primeiro split: treino vs temporario.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=1 - train_size,
        stratify=y,
        random_state=random_state,
    )

    # Segundo split: validacao vs teste.
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
