"""
Cálculo de tiempo de viaje del transelevador basado en Lerher et al. (2010).
Usa la métrica de Chebyshev (max entre horizontal y vertical) porque el
transelevador se mueve simultáneamente en ambos ejes.
"""
from typing import Tuple
from src.config import RackConfig, CraneKinematics


Position = Tuple[int, int, int]


def position_to_coords(pos: Position, rack: RackConfig) -> Tuple[float, float]:
    """Convierte una posición discreta (col, row, lane) a coordenadas
    físicas (x, y) en metros."""
    col, row, _ = pos
    x = col * rack.col_width
    y = row * rack.row_height
    return x, y


def io_coords(rack: RackConfig) -> Tuple[float, float]:
    return rack.io_col * rack.col_width, rack.io_row * rack.row_height


def chebyshev_time(p1: Tuple[float, float], p2: Tuple[float, float],
                   crane: CraneKinematics) -> float:
    """Tiempo de viaje entre dos puntos bajo la métrica de Chebyshev.
    El transelevador se mueve simultáneamente en X e Y; el tiempo total
    es el máximo entre ambos movimientos.
    Se ignora el perfil de aceleración para mantener el modelo simple y
    rápido; la corrección está dentro del ±5% reportado por Lerher."""
    x1, y1 = p1
    x2, y2 = p2
    t_h = abs(x2 - x1) / crane.v_horizontal
    t_v = abs(y2 - y1) / crane.v_vertical
    return max(t_h, t_v)


def storage_cycle_time(pos: Position, rack: RackConfig,
                       crane: CraneKinematics,
                       blocked: bool = False) -> float:
    """Tiempo de ciclo simple de almacenamiento SC.
    Ida al destino + depósito + regreso al I/O.
    Si la posición es interior y el exterior está ocupado, se suma la
    reubicación (este caso ocurre raramente en almacenamiento bajo la
    política de Lerher, pero se modela)."""
    io = io_coords(rack)
    dest = position_to_coords(pos, rack)

    t = 2 * chebyshev_time(io, dest, crane)  # ida y vuelta
    t += crane.t_pickup * 2                   # pickup en I/O, deposit en destino
    if pos[2] == 1:  # carril interior
        t += crane.t_extend * 2

    if blocked:
        # Reubicación: promedio empírico del 15% del tiempo base
        t += 0.15 * t

    return t


def retrieval_cycle_time(pos: Position, rack: RackConfig,
                         crane: CraneKinematics,
                         blocked: bool = False) -> float:
    """Tiempo de ciclo simple de recuperación SC.
    Si `blocked=True`, el ítem está en el carril interior y el exterior
    tiene otro ítem que debe reubicarse primero."""
    io = io_coords(rack)
    dest = position_to_coords(pos, rack)

    t = 2 * chebyshev_time(io, dest, crane)  # ida y vuelta
    t += crane.t_pickup * 2
    if pos[2] == 1:
        t += crane.t_extend * 2

    if blocked:
        # Término de reubicación: el transelevador saca el ítem bloqueante,
        # lo lleva a la celda libre más cercana y regresa. Aproximación:
        # ~60% de un ciclo completo adicional.
        relocation = 0.6 * 2 * chebyshev_time(io, dest, crane) + crane.t_pickup * 2
        t += relocation

    return t


def dual_command_time(store_pos: Position, retrieve_pos: Position,
                      rack: RackConfig, crane: CraneKinematics,
                      blocked_retrieval: bool = False) -> float:
    """Tiempo de ciclo doble DC: almacena en una posición y sin volver al I/O
    recupera de otra. Es más eficiente que dos SC separados."""
    io = io_coords(rack)
    p_store = position_to_coords(store_pos, rack)
    p_retr = position_to_coords(retrieve_pos, rack)

    t = (chebyshev_time(io, p_store, crane)
         + chebyshev_time(p_store, p_retr, crane)
         + chebyshev_time(p_retr, io, crane))
    t += crane.t_pickup * 4

    if store_pos[2] == 1:
        t += crane.t_extend * 2
    if retrieve_pos[2] == 1:
        t += crane.t_extend * 2

    if blocked_retrieval:
        relocation = 0.6 * 2 * chebyshev_time(io, p_retr, crane) + crane.t_pickup * 2
        t += relocation

    return t
