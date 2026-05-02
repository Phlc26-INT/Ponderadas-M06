# Permite usar anotacoes de tipo mais simples.
from __future__ import annotations

import re
from typing import Iterable, List

import emoji
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import wordpunct_tokenize


# Preprocessador configuravel para texto em PT-BR.
class Preprocessor:
    def __init__(self, config: dict):
        # Guarda as opcoes de preprocessamento.
        self.config = config or {}
        self.lowercase = self.config.get("lowercase", True)
        self.remove_urls = self.config.get("remove_urls", True)
        self.normalize_emojis = self.config.get("normalize_emojis", False)
        self.remove_punctuation = self.config.get("remove_punctuation", True)
        self.handle_negations = self.config.get("handle_negations", False)
        self.negation_window = int(self.config.get("negation_window", 3))
        self.remove_stopwords = self.config.get("remove_stopwords", False)
        self.normalization = self.config.get("normalization", None)

        # Lista de palavras de negacao usadas no marcador.
        self._negation_words = {"nao", "n\u00e3o", "nunca", "jamais", "nem"}
        self._demojize = self.normalize_emojis in (True, "demojize")

        # Carrega stopwords se solicitado.
        self._stopwords = None
        if self.remove_stopwords:
            self._stopwords = self._load_stopwords()

        # Define o tipo de normalizacao (stemming ou lematizacao).
        self._stemmer = None
        self._use_spacy = False
        self._nlp = None
        if self.normalization == "stemming":
            self._stemmer = self._load_stemmer()
        elif self.normalization == "lemmatization":
            self._use_spacy = True
            self._nlp = self._load_spacy()

    # Aplica o preprocessamento em uma lista de textos.
    def transform(self, texts: Iterable[str]) -> List[List[str]]:
        tokens_list = []
        for text in texts:
            tokens_list.append(self._process_text(text))
        return tokens_list

    # Processa um texto individual e retorna tokens.
    def _process_text(self, text: str) -> List[str]:
        if text is None:
            text = ""
        text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self._demojize:
            text = emoji.demojize(text)
            text = text.replace(":", " ")

        # Tokeniza com spaCy ou NLTK.
        if self._use_spacy:
            tokens = self._spacy_tokenize(text)
        else:
            tokens = wordpunct_tokenize(text)

        # Remove pontuacao se solicitado.
        if self.remove_punctuation:
            tokens = [t for t in tokens if any(ch.isalnum() for ch in t)]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]

        # Marca tokens negados se configurado.
        if self.handle_negations:
            tokens = self._mark_negations(tokens)

        return tokens

    # Tokenizacao usando spaCy com lematizacao.
    def _spacy_tokenize(self, text: str) -> List[str]:
        doc = self._nlp(text)
        tokens = []
        for tok in doc:
            if self.remove_punctuation and (tok.is_punct or tok.is_space):
                continue
            lemma = tok.lemma_
            if lemma == "-PRON-":
                lemma = tok.text
            tokens.append(lemma)
        return tokens

    # Carrega stopwords do NLTK.
    def _load_stopwords(self) -> set:
        try:
            sw = set(stopwords.words("portuguese"))
        except LookupError as exc:
            raise LookupError(
                "NLTK stopwords not found. Run: "
                "python -c \"import nltk; nltk.download('stopwords')\""
            ) from exc
        if self.remove_stopwords == "keep_negations":
            sw = sw.difference(self._negation_words)
        return sw

    # Carrega stemmer RSLP do NLTK.
    def _load_stemmer(self) -> RSLPStemmer:
        try:
            return RSLPStemmer()
        except LookupError as exc:
            raise LookupError(
                "NLTK RSLP stemmer not found. Run: "
                "python -c \"import nltk; nltk.download('rslp')\""
            ) from exc

    # Carrega modelo do spaCy para PT-BR.
    def _load_spacy(self):
        try:
            import spacy
        except ImportError as exc:
            raise ImportError("spaCy is required for lemmatization.") from exc
        try:
            return spacy.load("pt_core_news_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise OSError(
                "spaCy model not found. Run: python -m spacy download pt_core_news_sm"
            ) from exc

    # Marca tokens dentro da janela de negacao.
    def _mark_negations(self, tokens: List[str]) -> List[str]:
        result = []
        window = 0
        for token in tokens:
            if window > 0 and token not in self._negation_words:
                result.append(f"{token}_NEG")
                window -= 1
                continue
            result.append(token)
            if token in self._negation_words:
                window = self.negation_window
        return result
