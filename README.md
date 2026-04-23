# AS/RS Doble Profundidad — Optimización GA

Sistema de simulación y optimización de un almacén automatizado AS/RS (_Automated Storage and Retrieval System_) de **doble profundidad**, donde la política de asignación de posiciones se calibra mediante un **Algoritmo Genético (GA)**.

---

## Índice

1. [Descripción del problema](#1-descripción-del-problema)
2. [Arquitectura del proyecto](#2-arquitectura-del-proyecto)
3. [Instalación](#3-instalación)
4. [Ejecución rápida](#4-ejecución-rápida)
5. [Módulos](#5-módulos)
6. [Algoritmo Genético](#6-algoritmo-genético)
7. [Políticas de asignación](#7-políticas-de-asignación)
8. [Configuración](#8-configuración)
9. [Resultados y visualizaciones](#9-resultados-y-visualizaciones)

---

## 1. Descripción del problema

Un almacén AS/RS de **doble profundidad** tiene canales con dos carriles por nivel: el **carril exterior** (lane 0) con acceso directo, y el **carril interior** (lane 1) que queda bloqueado si el carril exterior del mismo canal está ocupado.

El objetivo es minimizar el **tiempo medio de viaje** del transelevador asignando los ítems a posiciones estratégicas según su demanda futura. Ítems de alta demanda deben ir cerca del punto I/O y preferentemente en el carril exterior.

El modelo de tiempo de viaje sigue la **métrica de Chebyshev** (movimiento simultáneo horizontal y vertical), basado en Lerher et al. (2010).

---

## 2. Arquitectura del proyecto

```
asrs_ml/
│
├── main.py              # Experimento completo: datos → GA → simulación → resultados
├── analyze.py           # Genera 8 figuras de presentación aisladas
├── calibrate.py         # Grid search para calibrar pesos de la función de costo
│
├── src/
│   ├── config.py        # Dataclasses de configuración (rack, grúa, GA, datos)
│   ├── rack.py          # Estado físico del rack (grid 3D, bloqueos, ocupación)
│   ├── travel_time.py   # Cálculo de tiempos de viaje (Chebyshev, ciclo simple/doble)
│   ├── data.py          # Generación sintética y carga de transacciones reales
│   ├── ga.py            # Optimizador GA (DemandOptimizerGA)
│   ├── optimizer.py     # Políticas de asignación: Random, ABC, GA, Exact
│   └── simulator.py     # Motor de simulación y comparación de políticas
│
└── results/             # Directorio de salida (CSV + figuras PNG)
```

---

## 3. Instalación

```bash
pip install numpy pandas matplotlib scipy
```

Python 3.10+ recomendado.

---

## 4. Ejecución rápida

### Experimento con datos sintéticos (configuración por defecto)
```bash
python main.py
```

### Experimento con datos reales
```bash
python main.py --csv data/mis_transacciones.csv
```

El CSV debe tener las columnas `order_id`, `item_id`, `operation` (`S`/`R`).

### Generar figuras de presentación
```bash
python analyze.py
python analyze.py --csv data/mis_transacciones.csv
```

### Calibrar pesos de la función de costo
```bash
python calibrate.py
```

---

## 5. Módulos

### `src/config.py` — Configuración centralizada

Todos los parámetros del sistema están definidos como dataclasses:

| Clase | Descripción |
|-------|-------------|
| `RackConfig` | Dimensiones del rack (columnas, filas, carriles, metros por celda) |
| `CraneKinematics` | Velocidades, aceleración y tiempos de pickup del transelevador |
| `CostWeights` | Pesos `w_distance`, `w_lane`, `w_neighbor` de la función de costo |
| `GAConfig` | Hiperparámetros del GA (población, generaciones, mutación, cruce) |
| `SyntheticDataConfig` | Parámetros del generador de datos sintéticos |
| `ExperimentConfig` | Configuración global que agrega todas las anteriores |

---

### `src/rack.py` — Estado del rack

`RackState` mantiene un array NumPy 3D con forma `(n_columns, n_rows, n_lanes)` donde cada celda contiene el `item_id` del ítem almacenado, o `-1` si está libre.

| Método | Descripción |
|--------|-------------|
| `store(pos, item_id)` | Almacena un ítem en la posición `(col, row, lane)` |
| `retrieve(pos)` | Extrae el ítem de una posición y la marca libre |
| `find_item(item_id)` | Busca la posición de un ítem por su ID |
| `is_blocked(pos)` | `True` si la posición es carril interior y el exterior está ocupado |
| `free_positions()` | Lista de todas las posiciones libres |
| `fill_grade` | Proporción de posiciones ocupadas (propiedad calculada) |
| `copy()` | Copia profunda del estado (para simulación contrafactual) |

**Convención de posiciones:** `(col, row, lane)` donde `lane=0` es el carril exterior y `lane=1` es el carril interior.

---

### `src/travel_time.py` — Tiempos de viaje

Implementa el modelo cinemático del transelevador basado en **Lerher et al. (2010)**.

| Función | Descripción |
|---------|-------------|
| `chebyshev_time(p1, p2, crane)` | Tiempo de desplazamiento entre dos puntos (movimiento simultáneo en X e Y) |
| `storage_cycle_time(pos, ...)` | Ciclo simple de almacenamiento: I/O → posición → I/O |
| `retrieval_cycle_time(pos, ...)` | Ciclo simple de recuperación, con penalización si hay bloqueo |
| `dual_command_time(s_pos, r_pos, ...)` | Ciclo doble: almacenamiento y recuperación en un solo viaje |

El parámetro `blocked=True` activa un término adicional que modela el tiempo de reubicación del ítem bloqueante:
- En almacenamiento: +15% del tiempo base.
- En recuperación: +60% de un ciclo Chebyshev adicional.

---

### `src/data.py` — Datos de transacciones

#### `generate_synthetic(cfg)`

Genera una secuencia de transacciones con las siguientes características:

1. **Distribución Pareto de demanda** (`pareto_alpha=1.16`): el 20% de los ítems concentra el 80% de los pedidos.
2. **Cambios de régimen** (`regime_shift_prob=0.02`): cada orden tiene un 2% de probabilidad de redistribuir completamente los pesos de demanda. Simula variaciones estacionales que hacen que una clasificación ABC estática sea subóptima.
3. **Llenado dinámico por fases**: el rack alterna entre fases de llenado (90% operaciones S) y vaciado (90% operaciones R), produciendo una ocupación oscilante entre el 10% y el 85%.

#### `load_transactions(csv_path)`

Carga datos reales desde CSV. Valida que existan las columnas `order_id`, `item_id`, `operation` y que `operation` solo contenga los valores `S` o `R`.

---

### `src/ga.py` — Algoritmo Genético

`DemandOptimizerGA` evoluciona un vector de scores de demanda por ítem. Ver [sección 6](#6-algoritmo-genético) para detalle completo.

---

### `src/optimizer.py` — Políticas de asignación

Ver [sección 7](#7-políticas-de-asignación).

---

### `src/simulator.py` — Motor de simulación

#### `simulate(df, policy, rack_cfg, crane_cfg, ...)`

Ejecuta la secuencia de transacciones aplicando una política de asignación:

- En cada operación `S`, llama a `policy.decide()` para obtener la posición y registra el tiempo de ciclo.
- En cada operación `R`, localiza el ítem, gestiona el bloqueo si procede (reubica el bloqueante) y registra el tiempo.
- Acumula un historial creciente de transacciones disponible para las políticas que lo necesiten.

Retorna un `SimulationResult` con métricas: tiempo medio/mediano/total, tasa de bloqueo, serie de fill grade.

#### `build_initial_rack(train_df, rack_cfg)`

Reconstruye el estado del rack al **final del período de entrenamiento** reproduciendo la secuencia de S/R con una política aleatoria fija. Garantiza que la simulación de evaluación comience con un nivel de ocupación realista en lugar de un rack vacío.

#### `compare_policies(eval_df, policies, ...)`

Evalúa múltiples políticas sobre el **mismo conjunto de transacciones** partiendo del **mismo estado inicial del rack**, permitiendo una comparación contrafactual justa.

---

## 6. Algoritmo Genético

### Representación

Cada **cromosoma** es un vector `s` de `n_items` valores en `[0, 1]`, donde `s[i]` es el score de demanda del ítem `i`. Un score alto indica que el ítem se espera recuperar con frecuencia; el optimizador usará ese score para asignarle posiciones preferentes (cerca del I/O, carril exterior).

### Función de fitness

```
f(s) = tiempo_medio_de_viaje(s)
     = promedio de t_k(s) para las últimas eval_orders transacciones de entrenamiento
```

Se ejecuta una mini-simulación con `GAPolicy` usando los scores del cromosoma. El objetivo es **minimizar** `f`.

### Operadores

| Operador | Detalle |
|----------|---------|
| **Selección** | Torneo de tamaño `k=3`: se eligen 3 individuos al azar y gana el de menor fitness |
| **Cruce** | Uniforme gen a gen: cada gen se toma del padre 1 con prob. `crossover_rate=0.5` |
| **Mutación** | Gaussiana por gen: `s[i] += N(0, 0.05)` con prob. `mutation_rate=0.15` por gen, clipeado a `[0,1]` |
| **Elitismo** | El 10% mejor de cada generación pasa directamente a la siguiente sin modificación |

### Salida de `fit()`

```python
{
    "n_items":               int,
    "n_generations":         int,
    "best_fitness":          float,   # tiempo medio [s] del mejor cromosoma
    "initial_fitness":       float,   # fitness antes de evolucionar (generación 0)
    "total_improvement_pct": float,   # mejora porcentual: (inicial - final) / inicial * 100
    "fitness_history":       list,    # mejor fitness por generación [gen 0 .. gen N]
    "mean_fitness_history":  list,    # fitness medio de la población por generación
    "worst_fitness_history": list,    # peor fitness de la población por generación
}
```

---

## 7. Políticas de asignación

Todas implementan la interfaz:
```python
AllocationPolicy.decide(item_id, order_id, rack_state, history) -> Position
```

### `RandomPolicy` (baseline)

Elige una posición libre al azar. Corresponde al supuesto del modelo analítico de Lerher et al. (2010) para el caso sin optimización.

### `ABCPolicy` (benchmark estático)

Clasifica cada ítem en A/B/C según su frecuencia histórica:
- **A** (top 20% por frecuencia): posición más cercana al I/O y carril exterior.
- **B** (20–50%): posición media.
- **C** (bottom 50%): posición más lejana e interior.

Limitación: la clasificación es estática y no se adapta a cambios de régimen en la demanda.

### `GAPolicy` (propuesta)

Usa los scores evolucionados por el GA en la **función de costo**:

```
costo(pos) = score_item * (w_distance * dist_norm(pos)
                         + w_lane    * pen_carril(pos)
                         + w_neighbor * score_vecino(pos))
```

Donde:
- `score_item` = score GA del ítem a almacenar en `[0, 1]`
- `dist_norm` = distancia Chebyshev al I/O, normalizada a `[0, 1]`
- `pen_carril` = 1.0 si carril interior, 0.0 si exterior
- `score_vecino` = score GA del ítem que ya ocupa el carril vecino del canal

Se elige la posición libre que **minimiza** esta expresión. Un ítem de alta demanda (score alto) acumulará mayor costo en posiciones lejanas o interiores, siendo empujado hacia posiciones favorables.

### `ExactOfflinePolicy` (oráculo de referencia)

Oráculo greedy con conocimiento total del futuro. Para cada operación S minimiza:
```
costo(pos) = t_almacenar(pos) + n_recuperaciones_futuras * t_recuperar(pos)
```
Incluido como referencia teórica; no es usable en producción al requerir conocimiento futuro.

---

## 8. Configuración

Todos los parámetros se modifican en `src/config.py` o instanciando un `ExperimentConfig` propio:

```python
from src.config import ExperimentConfig, GAConfig, RackConfig

cfg = ExperimentConfig(
    rack=RackConfig(n_columns=12, n_rows=5),
    ga=GAConfig(population_size=50, n_generations=100),
)
```

### Parámetros del rack (`RackConfig`)

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `n_columns` | 10 | Canales horizontales |
| `n_rows` | 4 | Niveles verticales |
| `n_lanes` | 2 | Carriles por canal (2 = doble profundidad) |
| `col_width` | 1.4 m | Ancho de cada canal |
| `row_height` | 1.2 m | Altura de cada nivel |
| `io_col` | 0 | Columna del punto I/O |
| `io_row` | 0 | Fila del punto I/O |

### Parámetros del transelevador (`CraneKinematics`)

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `v_horizontal` | 2.0 m/s | Velocidad horizontal |
| `v_vertical` | 1.5 m/s | Velocidad vertical |
| `t_pickup` | 6.0 s | Tiempo de toma/depósito de una unidad |
| `t_extend` | 2.0 s | Extensión del telescópico al carril interior |

### Pesos de la función de costo (`CostWeights`)

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `w_distance` | 1.0 | Peso del componente distancia al I/O |
| `w_lane` | 3.0 | Penalización por asignar al carril interior |
| `w_neighbor` | 2.0 | Penalización por score alto del ítem vecino |

Estos pesos pueden calibrarse con `calibrate.py`.

### Hiperparámetros GA (`GAConfig`)

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `population_size` | 30 | Individuos por generación |
| `n_generations` | 50 | Número de generaciones |
| `mutation_rate` | 0.15 | Probabilidad de mutar cada gen |
| `mutation_sigma` | 0.05 | Desviación estándar de la mutación gaussiana |
| `crossover_rate` | 0.5 | Probabilidad de cruce gen a gen |
| `tournament_k` | 3 | Participantes en cada torneo de selección |
| `eval_orders` | 300 | Transacciones usadas para evaluar fitness |

### Datos sintéticos (`SyntheticDataConfig`)

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `n_items` | 60 | Número de ítems distintos en el catálogo |
| `n_orders` | 2000 | Total de transacciones a generar |
| `pareto_alpha` | 1.16 | Parámetro de la distribución Pareto (80/20) |
| `regime_shift_prob` | 0.02 | Probabilidad de cambio de régimen por orden |
| `rack_capacity` | 80 | Capacidad máxima del rack |
| `fill_high_pct` | 0.85 | Umbral superior de llenado (fase → drain) |
| `fill_low_pct` | 0.10 | Umbral inferior de llenado (fase → fill) |

---

## 9. Resultados y visualizaciones

`analyze.py` genera 8 figuras PNG independientes en `results/`:

| Figura | Contenido |
|--------|-----------|
| `fig_01_policy_comparison.png` | Barras horizontales: tiempo medio y tasa de bloqueo por política |
| `fig_02_cumulative_time.png` | Tiempo de viaje acumulado con área de ahorro GA vs Random anotada |
| `fig_03_fill_grade.png` | Evolución dinámica del nivel de llenado del rack (crudo + suavizado) |
| `fig_04_travel_time_dist.png` | Histogramas superpuestos + KDE + líneas de media por política |
| `fig_05_ga_convergence.png` | Convergencia GA: mejor, media y peor fitness por generación |
| `fig_06_ga_improvement.png` | Mejora acumulada (%) + velocidad de aprendizaje por generación |
| `fig_07_ga_score_demand.png` | Scatter: score GA evolucionado vs frecuencia real de demanda + regresión lineal + Pearson r y Spearman rho |
| `fig_08_ga_rank_alignment.png` | Alineación de rankings: rank de demanda vs rank de score GA, coloreado por error absoluto de rango |

Las figuras 7 y 8 responden directamente a **"¿aprendió el GA los patrones de demanda?"**: una correlación alta y una nube densa en la diagonal de la figura 8 indican que el GA asignó scores altos a los ítems más demandados.

`main.py` genera adicionalmente `results/comparison_results.csv` con las métricas tabuladas de cada política.

`calibrate.py` genera `results/grid_search_weights.csv` con los resultados del grid search de pesos.
