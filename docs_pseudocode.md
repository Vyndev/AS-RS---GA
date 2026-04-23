# Diagramas de Flujo y Pseudocódigos — AS/RS + GA

Documento de referencia para entender el funcionamiento interno del sistema.  
Los diagramas usan sintaxis **Mermaid** (renderizan en GitHub, VS Code con extensión, o [mermaid.live](https://mermaid.live)).

---

## Índice

1. [Flujo general del experimento](#1-flujo-general-del-experimento)
2. [Generación de datos sintéticos](#2-generación-de-datos-sintéticos)
3. [Algoritmo Genético (GA)](#3-algoritmo-genético-ga)
4. [Motor de simulación](#4-motor-de-simulación)
5. [Política de asignación GAPolicy](#5-política-de-asignación-gapolicy)
6. [Comparación de políticas](#6-comparación-de-políticas)

---

## 1. Flujo general del experimento

### Diagrama de flujo

```mermaid
flowchart TD
    A([INICIO]) --> B[Cargar o generar\ntransacciones]
    B --> C[Dividir en\nTRAIN 75% / EVAL 25%]
    C --> D[Ejecutar GA\nsobre TRAIN]
    D --> E[Obtener scores\noptimizados por ítem]
    E --> F[Reconstruir estado\ndel rack al final de TRAIN]
    F --> G[Simular 3 políticas\nsobre EVAL con mismo rack inicial]
    G --> H{¿Política?}
    H -->|Random| I[Posición aleatoria]
    H -->|ABC| J[Posición por clase A/B/C]
    H -->|GA| K[Posición por función\nde costo + scores GA]
    I --> L[Medir tiempo de viaje]
    J --> L
    K --> L
    L --> M[Comparar métricas\ntiempo medio, bloqueo]
    M --> N[Guardar resultados CSV\n+ 8 figuras PNG]
    N --> Z([FIN])
```

### Pseudocódigo

```
EXPERIMENTO_PRINCIPAL(csv_path = null):

  1. DATOS
     si csv_path != null:
         df = cargar_CSV(csv_path)
     sino:
         df = generar_sintetico(n_ordenes=2000, n_items=60)

  2. PARTICION TEMPORAL (sin shuffle)
     train_df = df[ : 75% ]
     eval_df  = df[ 75% :  ]

  3. OPTIMIZACION GA
     optimizer = DemandOptimizerGA(config_ga)
     ga_metrics = optimizer.fit(train_df)
     scores = optimizer.scores          // {item_id -> score en [0,1]}

  4. ESTADO INICIAL DEL RACK
     rack_inicial = reconstruir_rack(train_df)

  5. SIMULACION COMPARATIVA
     para cada politica en {Random, ABC, GA}:
         resultado = simular(eval_df, politica, rack_inicial)

  6. RESULTADOS
     imprimir tabla comparativa
     guardar CSV y figuras
```

---

## 2. Generación de datos sintéticos

### Diagrama de flujo

```mermaid
flowchart TD
    A([INICIO]) --> B[Generar pesos Pareto\npara n_items ítems]
    B --> C[fase = FILL\nstored_items = lista vacía]
    C --> D{order_id < n_orders?}
    D -->|No| Z([FIN → DataFrame])
    D -->|Si| E{random < regime_shift_prob?}
    E -->|Si| F[Reasignar pesos Pareto\ncambio de régimen]
    E -->|No| G
    F --> G{Cambio de fase?}
    G -->|fill >= fill_high 85%| H[fase = DRAIN]
    G -->|drain <= fill_low 10%| I[fase = FILL]
    G -->|sin cambio| J
    H --> J
    I --> J
    J{fase = FILL?}
    J -->|Si| K[p_store = 0.90]
    J -->|No| L[p_store = 0.10]
    K --> M{random < p_store\no rack vacío?}
    L --> M
    M -->|Si → Almacenar| N[op = S\nitem = elegir por peso Pareto\nagregar a stored_items]
    M -->|No → Recuperar| O[op = R\nitem = elegir de stored_items\neliminar de stored_items]
    N --> P[Registrar orden: id, item, op]
    O --> P
    P --> D
```

### Pseudocódigo

```
GENERAR_SINTETICO(n_items, n_orders, pareto_alpha, regime_shift_prob):

  pesos = Pareto(pareto_alpha, n_items) + 1
  pesos = pesos / suma(pesos)              // normalizar a distribución

  stored_items = []
  fase = "FILL"

  para order_id = 0 hasta n_orders - 1:

    // Posible cambio de régimen
    si aleatorio() < regime_shift_prob:
        pesos = nuevo_Pareto(pareto_alpha, n_items)

    fill_actual = len(stored_items)

    // Cambio de fase
    si fase == "FILL"  y fill_actual >= 0.85 * capacidad:
        fase = "DRAIN"
    si fase == "DRAIN" y fill_actual <= 0.10 * capacidad:
        fase = "FILL"

    p_store = 0.90 si fase == "FILL" sino 0.10

    si aleatorio() < p_store o stored_items vacio:
        op      = "S"
        item_id = muestrear(n_items, probabilidades=pesos)
        stored_items.agregar(item_id)
    sino:
        op      = "R"
        item_id = muestrear(stored_items, probabilidades=pesos[stored_items])
        stored_items.eliminar(item_id)

    registros.agregar({order_id, item_id, op})

  retornar DataFrame(registros)
```

---

## 3. Algoritmo Genético (GA)

### Diagrama de flujo

```mermaid
flowchart TD
    A([INICIO fit]) --> B[Extraer item_ids únicos\ndel train_df]
    B --> C[eval_df = últimas eval_orders filas\nhistory_df = resto]
    C --> D[Inicializar población P\nP x n_items valores aleatorios en 0..1]
    D --> E[Evaluar fitness de cada individuo\nfitness = mini_simulacion con GAPolicy]
    E --> F[Guardar mejor cromosoma\nbest_chrom, best_fitness]
    F --> G{gen <= n_generations?}
    G -->|No| X[scores = best_chrom\nguardar historiales]
    X --> Z([FIN → retornar métricas])
    G -->|Si| H[Copiar elite top 10%\na new_pop]
    H --> I{len new_pop < P?}
    I -->|Si| J[Seleccionar padre1\npor torneo k=3]
    J --> K[Seleccionar padre2\npor torneo k=3]
    K --> L[Cruce uniforme:\nhijo_i = p1_i si rand < 0.5\nelse p2_i]
    L --> M[Mutacion gaussiana por gen:\nsi rand < 0.15: hijo_i += N 0 0.05\nclip a 0..1]
    M --> N[Agregar hijo a new_pop]
    N --> I
    I -->|No| O[pop = new_pop]
    O --> P[Evaluar fitness\nde toda la nueva población]
    P --> Q{gen_best < best_fitness?}
    Q -->|Si| R[Actualizar best_chrom\ny best_fitness]
    Q -->|No| S
    R --> S[Registrar historial\nbest, mean, worst]
    S --> G
```

### Pseudocódigo

```
GA_FIT(train_df, rack_cfg, crane_cfg, cost_weights):

  item_ids = items_unicos(train_df)
  n_items  = len(item_ids)

  eval_df    = train_df[ ultimas eval_orders filas ]
  history_df = train_df[ resto ]

  // ── Inicialización ─────────────────────────────────────────────
  poblacion = matriz_aleatoria(filas=pop_size, cols=n_items)  // valores en [0,1]

  FITNESS(cromosoma):
      scores  = {item_i: cromosoma[i] para cada item_i}
      policy  = GAPolicy(scores)
      result  = simular(eval_df, policy, history_df)
      retornar promedio(result.travel_times)

  fitness = [FITNESS(ind) para ind en poblacion]

  mejor_cromosoma = poblacion[ argmin(fitness) ]
  mejor_fitness   = min(fitness)
  historial_mejor = [mejor_fitness]

  // ── Ciclo evolutivo ────────────────────────────────────────────
  para gen = 1 hasta n_generations:

    nueva_pop = []

    // Elitismo: preservar el 10% mejor
    elite = poblacion[ argsort(fitness)[:pop_size//10] ]
    nueva_pop.agregar(elite)

    // Completar el resto con descendencia
    mientras len(nueva_pop) < pop_size:

        padre1 = TORNEO(poblacion, fitness, k=3)
        padre2 = TORNEO(poblacion, fitness, k=3)

        // Cruce uniforme
        hijo = []
        para i = 0 hasta n_items - 1:
            si aleatorio() < 0.5:
                hijo[i] = padre1[i]
            sino:
                hijo[i] = padre2[i]

        // Mutación gaussiana
        para i = 0 hasta n_items - 1:
            si aleatorio() < mutation_rate:
                hijo[i] = clip(hijo[i] + N(0, mutation_sigma), 0, 1)

        nueva_pop.agregar(hijo)

    poblacion = nueva_pop
    fitness   = [FITNESS(ind) para ind en poblacion]

    si min(fitness) < mejor_fitness:
        mejor_fitness   = min(fitness)
        mejor_cromosoma = poblacion[ argmin(fitness) ]

    historial_mejor.agregar(mejor_fitness)

  scores = {item_i: mejor_cromosoma[i]}
  retornar {scores, mejor_fitness, historial_mejor, ...}


TORNEO(poblacion, fitness, k):
  candidatos = muestra_aleatoria(poblacion, k)
  retornar candidato con menor fitness
```

---

## 4. Motor de simulación

### Diagrama de flujo

```mermaid
flowchart TD
    A([INICIO simular]) --> B{rack_inicial\ndado?}
    B -->|Si| C[rack = copia de rack_inicial]
    B -->|No| D[rack = rack vacío]
    C --> E
    D --> E[historial = train_history o vacío]
    E --> F{Siguiente\ntransacción?}
    F -->|No quedan| Z([FIN → SimulationResult])
    F -->|op = S| G{rack tiene\nposiciones libres?}
    G -->|No| H[Saltar operación]
    H --> U
    G -->|Si| I[pos = policy.decide\nitem_id, rack, historial]
    I --> J[t = tiempo_ciclo_almacenamiento pos]
    J --> K[rack.store pos, item_id]
    K --> L[registrar t en travel_times]
    L --> U
    F -->|op = R| M{item en rack?}
    M -->|No| N[items_missing ++\nSaltar]
    N --> U
    M -->|Si| O[pos = rack.find_item item_id]
    O --> P{rack.is_blocked pos?}
    P -->|Si| Q[Extraer bloqueante\nReubicar al free más cercano\nblocked_count ++]
    P -->|No| R
    Q --> R[t = tiempo_ciclo_recuperacion pos, blocked]
    R --> S[rack.retrieve pos]
    S --> T[registrar t en travel_times]
    T --> U[Actualizar fill_grade_series\nagregar orden al historial]
    U --> F
```

### Pseudocódigo

```
SIMULAR(df, policy, rack_cfg, crane_cfg, train_history, rack_inicial):

  rack      = copia(rack_inicial)  // mismo punto de partida para todas las políticas
  historial = train_history.copia()
  resultado = SimulationResult()

  para cada orden en df:
      item_id  = orden.item_id
      op       = orden.operation

      si op == "S":
          si rack.libre() == vacio: continuar

          pos = policy.decide(item_id, rack, historial)
          t   = storage_cycle_time(pos)
          rack.store(pos, item_id)
          resultado.travel_times.agregar(t)

      si op == "R":
          pos = rack.find_item(item_id)
          si pos == null:
              resultado.items_missing ++
              continuar

          bloqueado = rack.is_blocked(pos)

          si bloqueado:
              pos_ext    = (pos.col, pos.row, lane=0)
              bloqueante = rack.get_item(pos_ext)
              rack.retrieve(pos_ext)
              libre_mas_cercano = min(rack.free_positions(), distancia_a_pos)
              rack.store(libre_mas_cercano, bloqueante)
              resultado.blocked_count ++

          t = retrieval_cycle_time(pos, bloqueado)
          rack.retrieve(pos)
          resultado.travel_times.agregar(t)

      resultado.fill_grade_series.agregar(rack.fill_grade)
      historial.agregar(orden)

  retornar resultado
```

---

## 5. Política de asignación GAPolicy

### Diagrama de flujo

```mermaid
flowchart TD
    A([INICIO decide]) --> B[score_item = scores GA del item_id\ndefault 0.5 si desconocido]
    B --> C[free = rack.free_positions]
    C --> D[mejor_pos = null\nmejor_costo = infinito]
    D --> E{Para cada pos\nen free}
    E -->|pos disponible| F[vecino = item en el otro carril\ndel mismo canal]
    F --> G{vecino existe?}
    G -->|Si| H[score_vecino = scores GA del vecino]
    G -->|No| I[score_vecino = 0.0]
    H --> J
    I --> J[dist = distancia Chebyshev al I/O\nnormalizada a 0..1]
    J --> K[pen_carril = 1.0 si lane=1 sino 0.0]
    K --> L[costo = score_item x\nw_dist x dist +\nw_lane x pen_carril +\nw_neighbor x score_vecino]
    L --> M{costo < mejor_costo?}
    M -->|Si| N[mejor_costo = costo\nmejor_pos = pos]
    M -->|No| O
    N --> O{Quedan más\nposiciones?}
    O -->|Si| E
    O -->|No| Z([retornar mejor_pos])
```

### Pseudocódigo

```
GAPOLICY_DECIDE(item_id, rack, scores, pesos):

  score_item = scores.get(item_id, 0.5)
  libres     = rack.free_positions()

  mejor_pos  = null
  mejor_costo = +infinito

  para cada pos en libres:

      // Obtener score del ítem vecino en el mismo canal
      vecino       = rack.neighbor_item(pos)
      score_vecino = scores.get(vecino, 0.0) si vecino != -1 sino 0.0

      // Componentes del costo
      dist      = distancia_IO_normalizada(pos)   // en [0, 1]
      pen_carril = 1.0 si pos.lane == 1 sino 0.0

      costo = score_item * (
                  pesos.w_distance * dist
                + pesos.w_lane     * pen_carril
                + pesos.w_neighbor * score_vecino
              )

      si costo < mejor_costo:
          mejor_costo = costo
          mejor_pos   = pos

  retornar mejor_pos


// Intuición:
//   - score_item alto  → ítem muy demandado → acumula costo en pos lejanas
//     → el minimizador lo empuja a posiciones cercanas al I/O y carril exterior
//   - score_vecino alto → el vecino es muy demandado → asignar aquí bloquearía
//     su recuperación → el costo sube → se evita esa posición
```

---

## 6. Comparación de políticas

### Diagrama de flujo

```mermaid
flowchart TD
    A([INICIO compare_policies]) --> B[rack_inicial =\nreconstruir_rack train_df]
    B --> C{Para cada política\nen Random, ABC, GA}
    C --> D[resultado = simular\neval_df, política, rack_inicial]
    D --> E[metrics = resultado.to_dict\nn_ops, t_medio, t_mediano,\nbloqueo, items_missing]
    E --> F{Más\npolíticas?}
    F -->|Si| C
    F -->|No| G[Ordenar por t_medio]
    G --> H[Calcular mejora GA\nvs Random y vs ABC]
    H --> Z([retornar DataFrame de métricas])
```

### Pseudocódigo

```
COMPARE_POLICIES(eval_df, politicas, train_history):

  // Estado inicial idéntico para todas las políticas (comparación justa)
  rack_inicial = BUILD_INITIAL_RACK(train_history)

  resultados = []

  para cada (nombre, politica) en politicas:
      res = SIMULAR(eval_df, politica, rack_inicial=rack_inicial)
      resultados.agregar({
          policy:              nombre,
          mean_travel_time:    promedio(res.travel_times),
          median_travel_time:  mediana(res.travel_times),
          total_travel_time:   suma(res.travel_times),
          blocking_rate:       res.blocked_count / res.retrieval_count,
          items_missing:       res.items_missing
      })

  retornar DataFrame(resultados)


BUILD_INITIAL_RACK(train_df):
  // Reproduce la secuencia de entrenamiento con política aleatoria fija
  // para obtener el estado real del rack al terminar el entrenamiento

  rack = RackState(vacio)
  rng  = aleatorio(seed=0)

  para cada orden en train_df:
      si orden.op == "S":
          pos = rack.free_positions()[ aleatorio() ]
          rack.store(pos, orden.item_id)

      si orden.op == "R":
          pos = rack.find_item(orden.item_id)
          si pos == null: continuar
          si rack.is_blocked(pos):
              ext    = (pos.col, pos.row, lane=0)
              bloq   = rack.get_item(ext)
              rack.retrieve(ext)
              rack.store(rack.free_positions()[0], bloq)
          rack.retrieve(pos)

  retornar rack
```

---

## Resumen visual del sistema completo

```mermaid
flowchart LR
    subgraph DATOS
        A1[generate_synthetic\nPareto + fases] --> A2[(train_df\neval_df)]
    end

    subgraph GA
        B1[Población inicial\nscores aleatorios] --> B2[Evaluar fitness\nmini-simulación]
        B2 --> B3{Convergió?}
        B3 -->|No| B4[Selección torneo\nCruce uniforme\nMutación gaussiana\nElitismo]
        B4 --> B2
        B3 -->|Si| B5[(scores óptimos\npor ítem)]
    end

    subgraph SIMULACION
        C1[build_initial_rack\nrack con estado real] --> C2[Random]
        C1 --> C3[ABC]
        C1 --> C4[GAPolicy\nusa scores GA]
        C2 --> C5[(métricas)]
        C3 --> C5
        C4 --> C5
    end

    subgraph SALIDA
        D1[Tabla comparativa\nCSV]
        D2[8 figuras PNG\npresentación]
    end

    A2 --> GA
    A2 --> SIMULACION
    B5 --> C4
    C5 --> D1
    C5 --> D2
```
