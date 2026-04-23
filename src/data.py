"""
Carga de datos reales (CSV) o generación sintética de transacciones.

Formato esperado del CSV de datos reales:
    columnas: order_id, item_id, operation
    - order_id: entero, número de orden en la secuencia (creciente)
    - item_id: entero, identificador del ítem
    - operation: 'S' (almacenamiento) o 'R' (recuperación)

Los datos sintéticos simulan un almacén con distribución Pareto de demanda
y cambios de régimen ocasionales (para justificar el uso de ML).
"""
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import SyntheticDataConfig


def load_transactions(csv_path: str) -> pd.DataFrame:
    """Carga transacciones reales desde CSV.
    Valida el esquema mínimo requerido."""
    df = pd.read_csv(csv_path)
    required = {"order_id", "item_id", "operation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    df = df.sort_values("order_id").reset_index(drop=True)
    valid_ops = {"S", "R"}
    invalid = set(df["operation"].unique()) - valid_ops
    if invalid:
        raise ValueError(f"Valores inválidos en 'operation': {invalid}. "
                         f"Solo se acepta 'S' (almacenar) o 'R' (recuperar).")
    return df


def generate_synthetic(cfg: SyntheticDataConfig) -> pd.DataFrame:
    """Genera una secuencia sintética de transacciones.

    Características:
    - Demanda Pareto: pocos ítems concentran muchos pedidos
    - Cambios de régimen: en cada pedido hay prob `regime_shift_prob` de que
      los rankings de popularidad cambien, generando variabilidad en el tiempo
      (esto es exactamente lo que justifica el uso de ML)
    - Operaciones: mezcla de almacenamiento y recuperación

    Esta generación está diseñada para que un predictor ML pueda encontrar
    patrones útiles que una clasificación ABC estática no capturaría.
    """
    rng = np.random.default_rng(cfg.seed)

    # Pesos iniciales Pareto sobre los ítems
    weights = rng.pareto(cfg.pareto_alpha, cfg.n_items) + 1
    weights /= weights.sum()
    perm = rng.permutation(cfg.n_items)
    weights = weights[perm]

    fill_high = int(cfg.rack_capacity * cfg.fill_high_pct)
    fill_low  = int(cfg.rack_capacity * cfg.fill_low_pct)

    records = []
    stored_items = []  # ítems actualmente en el rack
    phase = "fill"     # alternar entre "fill" y "drain"

    for order_id in range(cfg.n_orders):
        # Posible cambio de régimen: reasignar pesos
        if rng.random() < cfg.regime_shift_prob:
            weights = rng.pareto(cfg.pareto_alpha, cfg.n_items) + 1
            weights /= weights.sum()
            perm = rng.permutation(cfg.n_items)
            weights = weights[perm]

        fill_now = len(stored_items)

        # Cambio de fase cuando se alcanza el objetivo
        if phase == "fill"  and fill_now >= fill_high:
            phase = "drain"
        elif phase == "drain" and (fill_now <= fill_low or fill_now == 0):
            phase = "fill"

        p_store = (cfg.prob_storage_fill  if phase == "fill"
                   else cfg.prob_storage_drain)

        if rng.random() < p_store or not stored_items:
            if fill_now >= cfg.rack_capacity:
                continue
            op      = "S"
            item_id = int(rng.choice(cfg.n_items, p=weights))
            stored_items.append(item_id)
        else:
            op          = "R"
            stored_arr  = np.array(stored_items)
            item_weights = weights[stored_arr]
            item_weights /= item_weights.sum()
            idx     = int(rng.choice(len(stored_arr), p=item_weights))
            item_id = int(stored_arr[idx])
            stored_items.pop(idx)

        records.append({
            "order_id": order_id,
            "item_id": item_id,
            "operation": op
        })

    return pd.DataFrame(records)


def save_synthetic(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
