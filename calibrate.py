"""
Calibración de los pesos de la función de costo mediante grid search.

Primero ejecuta el GA para obtener scores de demanda óptimos, luego evalúa
distintas combinaciones (w_distance, w_lane, w_neighbor) sobre el conjunto
de validación y reporta cuál minimiza el tiempo de viaje promedio.

Uso:
    python calibrate.py
"""
import itertools
from pathlib import Path
import pandas as pd

from src.config import ExperimentConfig, CostWeights
from src.data import generate_synthetic, load_transactions
from src.ga import DemandOptimizerGA
from src.optimizer import GAPolicy
from src.simulator import simulate


def grid_search_weights(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: ExperimentConfig,
    optimizer: DemandOptimizerGA,
    w_distance_grid=(0.5, 1.0, 2.0),
    w_lane_grid=(1.0, 3.0, 5.0),
    w_neighbor_grid=(0.0, 1.0, 2.0, 4.0)
) -> pd.DataFrame:
    """Grid search sobre los pesos de la función de costo.
    Reutiliza los scores ya evolucionados por el GA para cada combinación."""

    combos = list(itertools.product(w_distance_grid, w_lane_grid, w_neighbor_grid))
    print(f"\nEvaluando {len(combos)} combinaciones de pesos...")

    rows = []
    for i, (w_d, w_l, w_n) in enumerate(combos, 1):
        weights = CostWeights(w_distance=w_d, w_lane=w_l, w_neighbor=w_n)
        policy = GAPolicy(optimizer.scores, cfg.rack, weights)
        result = simulate(
            eval_df, policy, cfg.rack, cfg.crane,
            history_for_features=train_df, verbose=False
        )
        metrics = result.to_dict()
        rows.append({
            "w_distance": w_d,
            "w_lane": w_l,
            "w_neighbor": w_n,
            "mean_travel_time": metrics["mean_travel_time"],
            "blocking_rate": metrics["blocking_rate"],
            "total_travel_time": metrics["total_travel_time"]
        })
        print(f"  [{i:2d}/{len(combos)}] w=({w_d}, {w_l}, {w_n}) "
              f"→ tiempo promedio: {metrics['mean_travel_time']:.3f}, "
              f"bloqueos: {metrics['blocking_rate']:.2%}")

    return pd.DataFrame(rows).sort_values("mean_travel_time").reset_index(drop=True)


def main(csv_path: str = None):
    cfg = ExperimentConfig()

    print("=" * 70)
    print("CALIBRACIÓN DE PESOS DE LA FUNCIÓN DE COSTO (con GA)")
    print("=" * 70)

    if csv_path:
        df = load_transactions(csv_path)
    else:
        df = generate_synthetic(cfg.synthetic)

    split_idx = int(len(df) * cfg.train_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"\n[1/2] Ejecutando GA sobre {len(train_df)} transacciones de entrenamiento...")
    optimizer = DemandOptimizerGA(cfg.ga)
    ga_metrics = optimizer.fit(train_df, cfg.rack, cfg.crane, cfg.costs, verbose=True)
    print(f"  Mejor fitness GA: {ga_metrics['best_fitness']:.4f}s")

    print("\n[2/2] Grid search de pesos (scores GA fijos)...")
    grid = grid_search_weights(train_df, eval_df, cfg, optimizer)

    print("\n" + "=" * 70)
    print("MEJORES CONFIGURACIONES (ordenadas por tiempo de viaje promedio)")
    print("=" * 70)
    print(grid.head(10).to_string(index=False))

    best = grid.iloc[0]
    print(f"\nConfiguración óptima:")
    print(f"  w_distance = {best['w_distance']}")
    print(f"  w_lane     = {best['w_lane']}")
    print(f"  w_neighbor = {best['w_neighbor']}")
    print(f"  tiempo promedio: {best['mean_travel_time']:.3f}s")
    print(f"  tasa de bloqueo: {best['blocking_rate']:.2%}")

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.results_dir) / "grid_search_weights.csv"
    grid.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")

    return grid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    main(csv_path=args.csv)
