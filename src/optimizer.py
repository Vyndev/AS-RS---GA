"""
Módulo 2: Optimizador de asignación.

Implementa la función de costo y las políticas de asignación:
    - RandomPolicy: asignación aleatoria (baseline Lerher 2010)
    - ABCPolicy: asignación por clases ABC estáticas
    - GAPolicy: política propuesta que usa scores evolucionados por el GA
"""
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import pandas as pd

from src.config import RackConfig, CostWeights, CraneKinematics
from src.rack import RackState, Position
from src.travel_time import (position_to_coords, io_coords,
                              storage_cycle_time, retrieval_cycle_time)


# ─────────────────────────────────────────────────────────────────────────────
# Componentes de la función de costo
# ─────────────────────────────────────────────────────────────────────────────

def distance_to_io_norm(pos: Position, rack: RackConfig) -> float:
    """Distancia Chebyshev al I/O, normalizada a [0, 1]."""
    x, y = position_to_coords(pos, rack)
    io = io_coords(rack)
    d = max(abs(x - io[0]) / max(rack.col_width, 1e-9),
            abs(y - io[1]) / max(rack.row_height, 1e-9))
    max_d = max(rack.n_columns, rack.n_rows)
    return d / max(max_d, 1)


def lane_penalty(pos: Position) -> float:
    """1.0 si es carril interior, 0.0 si exterior."""
    return 1.0 if pos[2] == 1 else 0.0


def compute_cost(score: float, pos: Position, neighbor_score: float,
                 rack: RackConfig, w: CostWeights) -> float:
    """Función de costo:
       costo = score × (w1·dist + w2·pen_carril + w3·score_vecino)
    """
    d = distance_to_io_norm(pos, rack)
    p = lane_penalty(pos)
    return score * (w.w_distance * d + w.w_lane * p + w.w_neighbor * neighbor_score)


# ─────────────────────────────────────────────────────────────────────────────
# Políticas de asignación
# ─────────────────────────────────────────────────────────────────────────────

class AllocationPolicy(ABC):
    """Interfaz común para las políticas de asignación."""
    name: str = "base"

    @abstractmethod
    def decide(self, item_id: int, order_id: int,
               rack_state: RackState, history: pd.DataFrame) -> Position:
        """Retorna la posición asignada para `item_id`."""
        raise NotImplementedError


class RandomPolicy(AllocationPolicy):
    """Benchmark 1: Asignación aleatoria (supuesto del modelo analítico)."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def decide(self, item_id, order_id, rack_state, history):
        free = rack_state.free_positions()
        if not free:
            raise RuntimeError("Rack lleno: no hay posiciones libres.")
        idx = self.rng.integers(0, len(free))
        return free[idx]


class ABCPolicy(AllocationPolicy):
    """Benchmark 2: Asignación por clases ABC estáticas.
    Los ítems clase A van a las posiciones más cercanas al I/O y al carril
    exterior. Los ítems clase C van a posiciones lejanas e interiores."""
    name = "abc"

    def __init__(self, rack: RackConfig):
        self.rack = rack

    def _get_abc_class(self, item_id: int, history: pd.DataFrame) -> str:
        if len(history) == 0:
            return "C"
        counts = history["item_id"].value_counts()
        if item_id not in counts.index:
            return "C"
        freq = counts[item_id]
        rank_pct = (counts < freq).mean()
        if rank_pct >= 0.80:
            return "A"
        if rank_pct >= 0.50:
            return "B"
        return "C"

    def decide(self, item_id, order_id, rack_state, history):
        free = rack_state.free_positions()
        if not free:
            raise RuntimeError("Rack lleno: no hay posiciones libres.")

        abc = self._get_abc_class(item_id, history)

        def sort_key(p):
            return (lane_penalty(p), distance_to_io_norm(p, self.rack))

        ranked = sorted(free, key=sort_key)
        if abc == "A":
            return ranked[0]
        if abc == "B":
            mid = len(ranked) // 2
            return ranked[mid]
        return ranked[-1]


class ExactOfflinePolicy(AllocationPolicy):
    """Oráculo greedy con conocimiento total del futuro.

    Para cada operación de almacenamiento, conoce la secuencia completa
    de recuperaciones futuras de ese ítem y elige la posición que minimiza:

        costo(pos) = t_almacenamiento(pos)
                   + n_recuperaciones_futuras(item) × t_recuperación(pos)

    A diferencia del Algoritmo Húngaro (asignación estática), este oráculo
    toma cada decisión en tiempo real con visibilidad perfecta del futuro,
    lo que garantiza que siempre supere a las políticas online (Random, ABC, GA)
    que no conocen la demanda futura.
    """
    name = "exact"

    def __init__(self, eval_df: pd.DataFrame, rack: RackConfig,
                 crane: CraneKinematics):
        self.rack = rack
        self.crane = crane
        # Índice: item_id → lista ordenada de order_ids de recuperación futura
        self._future_retrievals: Dict[int, list] = {}
        self._build_index(eval_df)

    def _build_index(self, eval_df: pd.DataFrame) -> None:
        """Pre-construye el índice de recuperaciones futuras por ítem."""
        retr = eval_df[eval_df["operation"] == "R"][["order_id", "item_id"]]
        for _, row in retr.iterrows():
            item_id = int(row["item_id"])
            self._future_retrievals.setdefault(item_id, []).append(int(row["order_id"]))
        for key in self._future_retrievals:
            self._future_retrievals[key].sort()

    def decide(self, item_id, order_id, rack_state, history):
        free = rack_state.free_positions()
        if not free:
            raise RuntimeError("Rack lleno: no hay posiciones libres.")

        # Recuperaciones futuras de este ítem a partir de esta orden
        all_retr = self._future_retrievals.get(item_id, [])
        n_future = sum(1 for o in all_retr if o > order_id)

        if n_future == 0:
            # Ítem no se volverá a recuperar: colocarlo lejos e interior
            return max(free, key=lambda p: storage_cycle_time(p, self.rack, self.crane))

        def position_cost(pos: Position) -> float:
            t_s = storage_cycle_time(pos, self.rack, self.crane)
            t_r = retrieval_cycle_time(pos, self.rack, self.crane, blocked=False)
            # Penalización por carril interior (mayor probabilidad de bloqueo)
            blocking_penalty = n_future * self.crane.t_extend if pos[2] == 1 else 0.0
            return t_s + n_future * t_r + blocking_penalty

        return min(free, key=position_cost)


class GAPolicy(AllocationPolicy):
    """Política propuesta: usa scores evolucionados por el GA en la función
    de costo con anticipación al ítem vecino del canal."""
    name = "ga"

    def __init__(self, scores: Dict[int, float], rack: RackConfig,
                 weights: CostWeights):
        self.scores = scores  # {item_id: score en [0,1]}
        self.rack = rack
        self.w = weights

    def _get_score(self, item_id: int) -> float:
        return self.scores.get(item_id, 0.5)

    def decide(self, item_id, order_id, rack_state, history):
        free = rack_state.free_positions()
        if not free:
            raise RuntimeError("Rack lleno: no hay posiciones libres.")

        score_i = self._get_score(item_id)

        best_pos, best_cost = None, float("inf")
        for pos in free:
            neighbor = rack_state.neighbor_item(pos)
            neighbor_score = 0.0 if neighbor == -1 else self._get_score(neighbor)
            c = compute_cost(score_i, pos, neighbor_score, self.rack, self.w)
            if c < best_cost:
                best_cost = c
                best_pos = pos

        return best_pos
