"""
Simulador de operación: toma una secuencia de transacciones y la ejecuta
aplicando una política de asignación, midiendo el tiempo total de viaje
y las métricas asociadas (bloqueos, reubicaciones).

Permite comparación contrafactual entre políticas sobre el mismo conjunto
de transacciones.
"""
from typing import List, Dict
import numpy as np
import pandas as pd

from src.config import RackConfig, CraneKinematics
from src.rack import RackState, Position
from src.optimizer import AllocationPolicy
from src.travel_time import storage_cycle_time, retrieval_cycle_time


class SimulationResult:
    """Resultado de una simulación."""
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.travel_times: List[float] = []
        self.storage_times: List[float] = []
        self.retrieval_times: List[float] = []
        self.blocked_count: int = 0
        self.retrieval_count: int = 0
        self.items_missing: int = 0
        # Ocupación del rack tras cada operación (sube con S, baja con R)
        self.fill_grade_series: List[float] = []
        # Costo medio de recuperación del estado del rack tras cada operación.
        self.avg_retrieval_cost_series: List[float] = []

    def to_dict(self) -> dict:
        return {
            "policy": self.policy_name,
            "n_operations": len(self.travel_times),
            "mean_travel_time": float(np.mean(self.travel_times)) if self.travel_times else np.nan,
            "median_travel_time": float(np.median(self.travel_times)) if self.travel_times else np.nan,
            "total_travel_time": float(np.sum(self.travel_times)),
            "mean_storage_time": float(np.mean(self.storage_times)) if self.storage_times else np.nan,
            "mean_retrieval_time": float(np.mean(self.retrieval_times)) if self.retrieval_times else np.nan,
            "blocking_rate": self.blocked_count / max(self.retrieval_count, 1),
            "blocked_count": self.blocked_count,
            "retrieval_count": self.retrieval_count,
            "items_missing": self.items_missing,
        }


def build_initial_rack(train_df: pd.DataFrame,
                       rack_cfg: RackConfig) -> "RackState":
    """Reconstruye el estado del rack al final del período de entrenamiento.

    Reproduce la secuencia de S/R del train_df con una política aleatoria fija
    para que el rack de evaluación empiece con los ítems reales que quedaron
    almacenados al terminar el entrenamiento.
    """
    rng = np.random.default_rng(0)
    rack = RackState(rack_cfg)

    for row in train_df.itertuples(index=False):
        op      = row.operation
        item_id = int(row.item_id)

        if op == "S":
            free = rack.free_positions()
            if not free:
                continue
            rack.store(free[int(rng.integers(0, len(free)))], item_id)

        elif op == "R":
            pos = rack.find_item(item_id)
            if pos is None:
                continue
            if rack.is_blocked(pos):
                ext_pos    = (pos[0], pos[1], 0)
                blocker_id = rack.get_item(ext_pos)
                rack.retrieve(ext_pos)
                free = rack.free_positions()
                if free:
                    rack.store(free[0], blocker_id)
            rack.retrieve(pos)

    return rack


def simulate(
    df: pd.DataFrame,
    policy: AllocationPolicy,
    rack_cfg: RackConfig,
    crane_cfg: CraneKinematics,
    history_for_features: pd.DataFrame = None,
    initial_rack: "RackState" = None,
    verbose: bool = False
) -> SimulationResult:
    """Ejecuta la secuencia de transacciones aplicando la política.

    Parámetros
    ----------
    df : transacciones a simular (columnas: order_id, item_id, operation)
    policy : política de asignación
    rack_cfg, crane_cfg : configuración del sistema
    history_for_features : historial de transacciones DISPONIBLE para el
        predictor. Típicamente es el dataset de entrenamiento + las
        transacciones ya procesadas en la simulación actual. Si es None,
        se usa df mismo acumulativo.
    verbose : imprime progreso cada N órdenes
    """
    if initial_rack is not None:
        rack = initial_rack.copy()
    else:
        rack = RackState(rack_cfg)
    result = SimulationResult(policy.name)

    # Historial que el predictor puede consultar (crece durante la simulación)
    if history_for_features is None:
        accumulated_history = pd.DataFrame(columns=["order_id", "item_id", "operation"])
    else:
        accumulated_history = history_for_features.copy()

    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        order_id = int(row.order_id)
        item_id = int(row.item_id)
        op = row.operation

        if op == "S":
            # Si el rack está lleno, saltamos la operación (en la realidad
            # habría que esperar o rechazar). Para fines experimentales
            # simplemente no se cuenta.
            if not rack.free_positions():
                continue
            pos = policy.decide(item_id, order_id, rack, accumulated_history)
            t = storage_cycle_time(pos, rack_cfg, crane_cfg, blocked=False)
            rack.store(pos, item_id)
            result.travel_times.append(t)
            result.storage_times.append(t)

        elif op == "R":
            pos = rack.find_item(item_id)
            if pos is None:
                # Ítem no está en el rack: operación inválida, se omite
                result.items_missing += 1
                continue

            blocked = rack.is_blocked(pos)
            t = retrieval_cycle_time(pos, rack_cfg, crane_cfg, blocked=blocked)

            if blocked:
                # Simulamos la reubicación del ítem del carril exterior
                exterior_pos = (pos[0], pos[1], 0)
                blocker = rack.get_item(exterior_pos)
                rack.retrieve(exterior_pos)
                # Buscar posición libre más cercana para reubicarlo
                free = rack.free_positions()
                # Excluir la posición de donde vamos a recuperar
                free = [p for p in free if p != pos]
                if free:
                    # Reubicar al más cercano al canal actual
                    def dist_to_pos(p):
                        return abs(p[0] - pos[0]) + abs(p[1] - pos[1])
                    free.sort(key=dist_to_pos)
                    rack.store(free[0], blocker)
                result.blocked_count += 1

            rack.retrieve(pos)
            result.travel_times.append(t)
            result.retrieval_times.append(t)
            result.retrieval_count += 1

        # Fill grade dinámico: sube con S, baja con R
        result.fill_grade_series.append(rack.fill_grade)

        # Costo medio de recuperación del estado actual del rack.
        # Usa numpy para obtener las posiciones ocupadas en una sola pasada.
        occupied = np.argwhere(rack.grid != -1)
        if len(occupied):
            avg_cost = float(np.mean([
                retrieval_cycle_time(
                    (int(c), int(r), int(l)), rack_cfg, crane_cfg,
                    blocked=rack.is_blocked((int(c), int(r), int(l)))
                )
                for c, r, l in occupied
            ]))
        else:
            avg_cost = 0.0
        result.avg_retrieval_cost_series.append(avg_cost)

        # Añadir la orden al historial acumulado
        accumulated_history = pd.concat([
            accumulated_history,
            pd.DataFrame([{"order_id": order_id, "item_id": item_id,
                           "operation": op}])
        ], ignore_index=True)

        if verbose and (i + 1) % 200 == 0:
            print(f"    {i+1}/{n} operaciones procesadas "
                  f"(fill={rack.fill_grade:.1%})")

    return result


def compare_policies(
    eval_df: pd.DataFrame,
    policies: Dict[str, AllocationPolicy],
    rack_cfg: RackConfig,
    crane_cfg: CraneKinematics,
    train_history: pd.DataFrame = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Compara varias políticas sobre el mismo conjunto de transacciones.
    Retorna un DataFrame con las métricas de cada política."""
    # Estado inicial del rack al final del entrenamiento (mismo para todas)
    initial_rack = None
    if train_history is not None:
        initial_rack = build_initial_rack(train_history, rack_cfg)

    rows = []
    for name, policy in policies.items():
        if verbose:
            print(f"\n  Simulando política: {name}")
        result = simulate(
            eval_df, policy, rack_cfg, crane_cfg,
            history_for_features=train_history,
            initial_rack=initial_rack,
            verbose=verbose
        )
        row = result.to_dict()
        rows.append(row)
    return pd.DataFrame(rows)
