# Permite usar anotacoes de tipo mais simples.
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Vetorizador com varias estrategias (bow, tfidf, svd, w2v).
class Vectorizer:
    def __init__(self, config: dict):
        # Guarda configuracoes e objetos internos.
        self.config = config or {}
        self.strategy = self.config.get("strategy", "tfidf")
        self._vectorizer = None
        self._svd = None
        self._w2v = None
        self._vector_size = None

    # Ajusta o vetorizador e transforma o treino.
    def fit_transform(self, token_lists: List[List[str]]):
        if self.strategy == "bow":
            return self._fit_transform_bow(token_lists)
        if self.strategy == "tfidf":
            return self._fit_transform_tfidf(token_lists)
        if self.strategy == "tfidf_svd":
            return self._fit_transform_tfidf_svd(token_lists)
        if self.strategy == "w2v":
            return self._fit_transform_w2v(token_lists)
        raise ValueError(f"Unknown vectorization strategy: {self.strategy}")

    # Transforma novos textos com o vetorizador ja treinado.
    def transform(self, token_lists: List[List[str]]):
        if self.strategy in {"bow", "tfidf"}:
            texts = self._join_tokens(token_lists)
            return self._vectorizer.transform(texts)
        if self.strategy == "tfidf_svd":
            texts = self._join_tokens(token_lists)
            tfidf = self._vectorizer.transform(texts)
            return self._svd.transform(tfidf)
        if self.strategy == "w2v":
            return self._average_vectors(token_lists)
        raise ValueError(f"Unknown vectorization strategy: {self.strategy}")

    # Junta tokens em strings para vetorizadores do sklearn.
    def _join_tokens(self, token_lists: List[List[str]]) -> List[str]:
        return [" ".join(tokens) for tokens in token_lists]

    # Bag of Words simples.
    def _fit_transform_bow(self, token_lists: List[List[str]]):
        texts = self._join_tokens(token_lists)
        self._vectorizer = CountVectorizer(
            binary=self.config.get("binary", False),
            max_features=self.config.get("max_features"),
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
        )
        return self._vectorizer.fit_transform(texts)

    # TF-IDF com configuracoes opcionais.
    def _fit_transform_tfidf(self, token_lists: List[List[str]]):
        texts = self._join_tokens(token_lists)
        ngram_range = tuple(self.config.get("ngram_range", (1, 1)))
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=self.config.get("sublinear_tf", False),
            ngram_range=ngram_range,
            max_features=self.config.get("max_features"),
            norm=self.config.get("norm", "l2"),
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
        )
        return self._vectorizer.fit_transform(texts)

    # TF-IDF seguido de SVD para reduzir dimensao.
    def _fit_transform_tfidf_svd(self, token_lists: List[List[str]]):
        tfidf = self._fit_transform_tfidf(token_lists)
        self._svd = TruncatedSVD(
            n_components=int(self.config.get("svd_components", 100)),
            random_state=self.config.get("random_state", 42),
        )
        return self._svd.fit_transform(tfidf)

    # Word2Vec com media dos vetores.
    def _fit_transform_w2v(self, token_lists: List[List[str]]):
        self._w2v = self._load_or_train_w2v(token_lists)
        self._vector_size = int(self._w2v.vector_size)
        return self._average_vectors(token_lists)

    # Carrega embeddings pre-treinados ou treina no corpus.
    def _load_or_train_w2v(self, token_lists: List[List[str]]):
        pretrained_path = self.config.get("pretrained_path")
        if pretrained_path:
            return self._load_pretrained(pretrained_path)
        if not self.config.get("train_on_corpus", True):
            raise ValueError("train_on_corpus must be True when pretrained_path is not set")

        # Treina Word2Vec com os parametros da config.
        model = Word2Vec(
            sentences=token_lists,
            vector_size=int(self.config.get("w2v_dim", 100)),
            window=int(self.config.get("w2v_window", 5)),
            min_count=int(self.config.get("w2v_min_count", 2)),
            sg=int(self.config.get("w2v_sg", 1)),
            workers=int(self.config.get("w2v_workers", os.cpu_count() or 2)),
            epochs=int(self.config.get("w2v_epochs", 10)),
        )
        return model.wv

    # Tenta carregar embeddings pre-treinados em dois formatos.
    def _load_pretrained(self, pretrained_path: str):
        path = Path(pretrained_path)
        binary = bool(self.config.get("pretrained_binary", False))
        try:
            return KeyedVectors.load(str(path))
        except Exception:
            return KeyedVectors.load_word2vec_format(str(path), binary=binary)

    # Faz a media dos vetores de cada texto.
    def _average_vectors(self, token_lists: List[List[str]]):
        vectors = np.zeros((len(token_lists), self._vector_size), dtype=np.float32)
        for idx, tokens in enumerate(token_lists):
            valid = [self._w2v[token] for token in tokens if token in self._w2v]
            if valid:
                vectors[idx] = np.mean(valid, axis=0)
        return vectors
