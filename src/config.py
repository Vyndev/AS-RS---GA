"""
Configuración del sistema AS/RS de doble profundidad.
Todos los parámetros del rack, cinemática del transelevador y modelo ML
están centralizados aquí.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RackConfig:
    """Parámetros del rack físico."""
    n_columns: int = 10           # X: canales horizontales
    n_rows: int = 4               # Y: niveles verticales
    n_lanes: int = 2              # Profundidad (2 = doble profundidad)
    col_width: float = 1.4        # metros
    row_height: float = 1.2       # metros
    io_col: int = 0               # columna del punto I/O (esquina)
    io_row: int = 0               # fila del punto I/O (esquina)

    @property
    def total_positions(self) -> int:
        return self.n_columns * self.n_rows * self.n_lanes


@dataclass
class CraneKinematics:
    """Parámetros cinemáticos del transelevador (S/R machine).
    Valores típicos: Lerher et al. (2010)."""
    v_horizontal: float = 2.0     # m/s
    v_vertical: float = 1.5       # m/s
    acceleration: float = 1.0     # m/s^2
    t_pickup: float = 6.0         # segundos para tomar/dejar una unidad
    t_extend: float = 2.0         # extensión del telescópico al carril interior


@dataclass
class CostWeights:
    """Pesos de la función de costo del optimizador.
    Se pueden calibrar mediante validación experimental."""
    w_distance: float = 1.0
    w_lane: float = 3.0           # penalización por asignar al carril interior
    w_neighbor: float = 2.0       # penalización por score alto del vecino


@dataclass
class GAConfig:
    """Hiperparámetros del Algoritmo Genético para optimización de scores."""
    population_size: int = 30
    n_generations: int = 50
    mutation_rate: float = 0.15    # probabilidad de mutar cada gen
    mutation_sigma: float = 0.05   # magnitud de la mutación gaussiana
    crossover_rate: float = 0.5    # cruce uniforme gen a gen
    tournament_k: int = 3          # participantes en cada torneo de selección
    eval_orders: int = 300         # transacciones usadas para evaluar fitness
    random_state: int = 42


@dataclass
class SyntheticDataConfig:
    """Parámetros para generación de datos sintéticos."""
    n_items: int = 60
    n_orders: int = 2000
    # Distribución Pareto 80/20: 20% de ítems concentran 80% de demanda
    pareto_alpha: float = 1.16
    # Probabilidad de que ocurra un cambio de régimen en la secuencia
    regime_shift_prob: float = 0.02
    # Capacidad total del rack (debe coincidir con RackConfig.total_positions)
    rack_capacity: int = 80
    # Fases de llenado/vaciado: el rack oscila entre fill_low y fill_high
    fill_high_pct: float = 0.85   # objetivo de la fase de llenado
    fill_low_pct: float = 0.10    # objetivo de la fase de vaciado
    # Probabilidad de S/R durante cada fase
    prob_storage_fill:  float = 0.90  # fase llenado: 90% almacenamientos
    prob_storage_drain: float = 0.10  # fase vaciado: 90% recuperaciones
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuración global del experimento."""
    rack: RackConfig = field(default_factory=RackConfig)
    crane: CraneKinematics = field(default_factory=CraneKinematics)
    costs: CostWeights = field(default_factory=CostWeights)
    ga: GAConfig = field(default_factory=GAConfig)
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    # Proporción de datos para entrenamiento (secuencial, no aleatoria)
    train_split: float = 0.75
    # Directorio de resultados
    results_dir: str = "results"
