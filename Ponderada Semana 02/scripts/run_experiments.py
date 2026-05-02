# Permite usar anotacoes de tipo sem precisar importar sempre do __future__ em versoes antigas.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Define o caminho raiz do projeto e garante que o pacote local possa ser importado.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importa funcoes principais do pipeline.
from sentiment_pipeline.pipeline import evaluate_best_on_test, run_experiments
from sentiment_pipeline.utils import load_dataset_csv, stratified_split


# Carrega o YAML de configuracao.
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Fluxo principal do script: leitura de config, split, treino e avaliacao.
def main():
    # Le argumentos de linha de comando.
    parser = argparse.ArgumentParser(description="Run sentiment pipeline experiments")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    # Carrega a configuracao e o bloco do dataset.
    cfg = load_config(Path(args.config))
    dataset_cfg = cfg["dataset"]

    # Le o CSV e cria a coluna de label conforme a config.
    df = load_dataset_csv(
        dataset_cfg["path"],
        text_col=dataset_cfg["text_col"],
        label_col=dataset_cfg.get("label_col"),
        rating_col=dataset_cfg.get("rating_col"),
        label_cfg=dataset_cfg.get("label", {}),
    )

    # Faz a divisao estratificada em treino, validacao e teste.
    split_cfg = cfg.get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        df,
        text_col=dataset_cfg["text_col"],
        label_col="label",
        train_size=split_cfg.get("train_size", 0.7),
        val_size=split_cfg.get("val_size", 0.15),
        test_size=split_cfg.get("test_size", 0.15),
        random_state=split_cfg.get("random_state", 42),
    )

    # Executa os experimentos e escolhe a melhor config pelo F1 macro.
    results_df, best_config = run_experiments(
        cfg["experiments"], X_train, y_train, X_val, y_val
    )

    # Salva os resultados em disco.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "results.csv", index=False)

    # Salva a melhor configuracao encontrada.
    with (results_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    # Avalia no teste final, a menos que o usuario desative.
    if not args.skip_test:
        test_metrics = evaluate_best_on_test(
            best_config, X_train, y_train, X_val, y_val, X_test, y_test
        )
        with (results_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    # Mensagem final de confirmacao.
    print("Done. Results saved to", results_dir)


# Executa o fluxo principal quando o arquivo roda direto.
if __name__ == "__main__":
    main()
