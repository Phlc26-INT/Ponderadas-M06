# Reexporta classes e funcoes para facilitar os imports externos.
from .preprocessing import Preprocessor
from .vectorization import Vectorizer
from .classification import Classifier
from .pipeline import run_experiments, evaluate_best_on_test

# Lista publica do pacote.
__all__ = [
    "Preprocessor",
    "Vectorizer",
    "Classifier",
    "run_experiments",
    "evaluate_best_on_test",
]
