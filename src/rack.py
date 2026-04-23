"""
Estado del rack: gestiona las posiciones ocupadas y libres.
Cada posición se identifica por la tupla (col, row, lane).
lane=0 es el carril exterior (acceso directo).
lane=1 es el carril interior (potencialmente bloqueado).
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from src.config import RackConfig


Position = Tuple[int, int, int]  # (col, row, lane)


@dataclass
class RackState:
    """Mantiene el estado de ocupación del rack.

    El arreglo interno `grid` tiene forma (n_columns, n_rows, n_lanes)
    y almacena el item_id ocupante o -1 si está libre.
    """
    config: RackConfig

    def __post_init__(self):
        self.grid: np.ndarray = np.full(
            (self.config.n_columns, self.config.n_rows, self.config.n_lanes),
            fill_value=-1, dtype=np.int32
        )

    @property
    def fill_grade(self) -> float:
        """Proporción de posiciones ocupadas (z en la literatura)."""
        return np.sum(self.grid != -1) / self.config.total_positions

    def is_free(self, pos: Position) -> bool:
        col, row, lane = pos
        return self.grid[col, row, lane] == -1

    def get_item(self, pos: Position) -> int:
        """Retorna item_id en la posición, o -1 si está libre."""
        col, row, lane = pos
        return int(self.grid[col, row, lane])

    def store(self, pos: Position, item_id: int) -> None:
        col, row, lane = pos
        if not self.is_free(pos):
            raise ValueError(f"La posición {pos} ya está ocupada por "
                             f"item {self.grid[col, row, lane]}")
        self.grid[col, row, lane] = item_id

    def retrieve(self, pos: Position) -> int:
        col, row, lane = pos
        item = int(self.grid[col, row, lane])
        if item == -1:
            raise ValueError(f"La posición {pos} está vacía.")
        self.grid[col, row, lane] = -1
        return item

    def free_positions(self) -> List[Position]:
        """Lista de todas las posiciones libres del rack."""
        free = []
        for c in range(self.config.n_columns):
            for r in range(self.config.n_rows):
                for l in range(self.config.n_lanes):
                    if self.grid[c, r, l] == -1:
                        free.append((c, r, l))
        return free

    def neighbor_position(self, pos: Position) -> Position:
        """Retorna la posición del otro carril del mismo canal.
        En un sistema de doble profundidad (n_lanes=2), el vecino de lane=0
        es lane=1 y viceversa."""
        col, row, lane = pos
        return (col, row, 1 - lane)

    def neighbor_item(self, pos: Position) -> int:
        """Item en el carril vecino. -1 si está libre."""
        return self.get_item(self.neighbor_position(pos))

    def find_item(self, item_id: int) -> Optional[Position]:
        """Encuentra la posición de un item_id dado. None si no está."""
        locations = np.argwhere(self.grid == item_id)
        if len(locations) == 0:
            return None
        c, r, l = locations[0]
        return (int(c), int(r), int(l))

    def is_blocked(self, pos: Position) -> bool:
        """True si la posición está en el carril interior y el carril
        exterior del mismo canal está ocupado por otro ítem."""
        col, row, lane = pos
        if lane == 0:  # exterior nunca está bloqueado
            return False
        exterior_pos = (col, row, 0)
        return not self.is_free(exterior_pos)

    def copy(self) -> "RackState":
        """Copia profunda del estado, útil para simulación contrafactual."""
        new_state = RackState(self.config)
        new_state.grid = self.grid.copy()
        return new_state

    def __repr__(self) -> str:
        return (f"RackState(fill={self.fill_grade:.2%}, "
                f"occupied={int(np.sum(self.grid != -1))}/"
                f"{self.config.total_positions})")
