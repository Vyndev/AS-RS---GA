"""
Experimento end-to-end: optimización GA + comparación de políticas.

Uso:
    python main.py                      # datos sintéticos
    python main.py --csv data/real.csv  # datos reales
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data import generate_synthetic, load_transactions, save_synthetic
from src.ga import DemandOptimizerGA
from src.optimizer import RandomPolicy, ABCPolicy, GAPolicy
from src.simulator import compare_policies

_LABELS = {"random": "Random", "abc": "ABC", "ga": "GA"}


def _print_results_table(results: pd.DataFrame) -> None:
    random_time = float(results[results["policy"] == "random"]["mean_travel_time"].iloc[0])
    sorted_r = results.sort_values("mean_travel_time").reset_index(drop=True)
    medals = ["1°", "2°", "3°"]
    line = "─" * 58
    print(f"\n{line}")
    print(f"  {'#':<4} {'Política':<10} {'T.medio (s)':<13} {'vs Random':<13} {'Bloqueo'}")
    print(line)
    for rank, (_, row) in enumerate(sorted_r.iterrows()):
        pol  = row["policy"]
        t    = row["mean_travel_time"]
        bl   = row["blocking_rate"]
        vs_r = f"{(random_time - t) / random_time * 100:+.2f}%" if pol != "random" else "  —"
        print(f"  {medals[rank]:<5} {_LABELS[pol]:<10} {t:<13.4f} {vs_r:<13} {bl:.2%}")
    print(line)


def main(csv_path: str = None, config: ExperimentConfig = None) -> dict:
    cfg = config or ExperimentConfig()
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    # 1. Datos
    # ────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("EXPERIMENTO AS/RS DOBLE PROFUNDIDAD — GA para asignación")
    print("=" * 70)

    if csv_path:
        print(f"\n[1] Cargando transacciones reales desde {csv_path}")
        df = load_transactions(csv_path)
    else:
        print(f"\n[1] Generando {cfg.synthetic.n_orders} transacciones sintéticas "
              f"con {cfg.synthetic.n_items} ítems")
        df = generate_synthetic(cfg.synthetic)
        save_synthetic(df, Path(cfg.results_dir) / "synthetic_transactions.csv")

    n = len(df)
    print(f"    Total: {n} transacciones  "
          f"(S={( df['operation']=='S').sum()}  R={(df['operation']=='R').sum()})")
    print(f"    Ítems distintos: {df['item_id'].nunique()}")

    # ────────────────────────────────────────────────────────────────────────
    # 2. Partición temporal
    # ────────────────────────────────────────────────────────────────────────
    split_idx = int(n * cfg.train_split)
    train_df  = df.iloc[:split_idx].copy().reset_index(drop=True)
    eval_df   = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"\n[2] Partición temporal (sin shuffle):")
    print(f"    Entrenamiento: {len(train_df)} filas  |  Evaluación: {len(eval_df)} filas")

    # ────────────────────────────────────────────────────────────────────────
    # 3. Optimización GA
    # ────────────────────────────────────────────────────────────────────────
    print("\n[3] Ejecutando Algoritmo Genético...")
    optimizer  = DemandOptimizerGA(cfg.ga)
    ga_metrics = optimizer.fit(train_df, cfg.rack, cfg.crane, cfg.costs, verbose=True)

    print(f"\n    Ítems optimizados : {ga_metrics['n_items']}")
    print(f"    Fitness inicial   : {ga_metrics['initial_fitness']:.4f}s")
    print(f"    Fitness final     : {ga_metrics['best_fitness']:.4f}s")
    print(f"    Mejora total      : {ga_metrics['total_improvement_pct']:.2f}%")

    top_items = sorted(optimizer.scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n    Top 10 ítems por score GA:")
    for item_id, score in top_items:
        print(f"      Ítem {item_id:3d}: {score:.3f}  {'█' * int(score * 30)}")

    # ────────────────────────────────────────────────────────────────────────
    # 4. Simulación comparativa
    # ────────────────────────────────────────────────────────────────────────
    print("\n[4] Comparando políticas...")
    policies = {
        "random": RandomPolicy(seed=cfg.synthetic.seed),
        "abc":    ABCPolicy(cfg.rack),
        "ga":     GAPolicy(optimizer.scores, cfg.rack, cfg.costs),
    }

    results = compare_policies(
        eval_df, policies, cfg.rack, cfg.crane,
        train_history=train_df, verbose=True
    )

    # ────────────────────────────────────────────────────────────────────────
    # 5. Resultados
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    _print_results_table(results)

    ga_row     = results[results["policy"] == "ga"].iloc[0]
    random_row = results[results["policy"] == "random"].iloc[0]
    abc_row    = results[results["policy"] == "abc"].iloc[0]

    impr_random = (random_row["mean_travel_time"] - ga_row["mean_travel_time"]) / random_row["mean_travel_time"] * 100
    impr_abc    = (abc_row["mean_travel_time"]    - ga_row["mean_travel_time"]) / abc_row["mean_travel_time"]    * 100

    print(f"\n  GA mejora {impr_random:.1f}% vs Random  |  {impr_abc:.1f}% vs ABC")

    out_path = Path(cfg.results_dir) / "comparison_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")

    return {
        "results":    results,
        "optimizer":  optimizer,
        "ga_metrics": ga_metrics,
        "eval_df":    eval_df,
        "train_df":   train_df,
        "improvement_vs_random_pct": impr_random,
        "improvement_vs_abc_pct":    impr_abc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experimento GA para AS/RS doble profundidad")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    main(csv_path=args.csv)
