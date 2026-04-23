"""
Módulo GA: Optimizador de scores de demanda mediante Algoritmo Genético.

Evoluciona un vector de scores en [0, 1] (uno por ítem) que minimiza
el tiempo medio de viaje en una mini-simulación sobre datos de entrenamiento.
El score de cada ítem indica su prioridad de demanda futura y reemplaza
directamente al predictor de Random Forest.
"""
import numpy as np
import pandas as pd
from typing import Dict

from src.config import GAConfig, RackConfig, CraneKinematics, CostWeights


class DemandOptimizerGA:
    """GA que evoluciona scores de demanda por ítem para minimizar tiempo de viaje.

    Cada cromosoma es un array de n_items valores en [0, 1], donde cada valor
    representa la prioridad de demanda predicha para un ítem. El GA minimiza
    el tiempo medio de viaje simulando las operaciones con esos scores.
    """

    def __init__(self, ga_cfg: GAConfig):
        self.cfg = ga_cfg
        self.scores: Dict[int, float] = {}
        self.item_ids: list = []
        # Historial del mejor individuo por generación
        self.fitness_history: list = []
        # Historial de la media y peor individuo (para graficar diversidad)
        self.mean_fitness_history: list = []
        self.worst_fitness_history: list = []
        self._is_fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        rack_cfg: RackConfig,
        crane_cfg: CraneKinematics,
        cost_weights: CostWeights,
        verbose: bool = True,
    ) -> dict:
        """Ejecuta el GA sobre datos de entrenamiento para encontrar scores óptimos."""
        from src.optimizer import GAPolicy
        from src.simulator import simulate

        self.item_ids = sorted(train_df["item_id"].unique().tolist())
        n_items = len(self.item_ids)
        idx_map = {item: i for i, item in enumerate(self.item_ids)}

        # Usar las últimas eval_orders filas para fitness (más representativas del futuro)
        n_eval = min(self.cfg.eval_orders, len(train_df))
        eval_df = train_df.iloc[-n_eval:].copy().reset_index(drop=True)
        history_df = train_df.iloc[:-n_eval].copy().reset_index(drop=True)

        rng = np.random.default_rng(self.cfg.random_state)

        def evaluate(chromosome: np.ndarray) -> float:
            scores_dict = {item: float(chromosome[i]) for item, i in idx_map.items()}
            policy = GAPolicy(scores_dict, rack_cfg, cost_weights)
            result = simulate(
                eval_df, policy, rack_cfg, crane_cfg,
                history_for_features=history_df, verbose=False,
            )
            return float(np.mean(result.travel_times))

        # Población inicial aleatoria en [0, 1]
        pop = rng.random((self.cfg.population_size, n_items))
        fitness = np.array([evaluate(ind) for ind in pop])

        best_idx = int(np.argmin(fitness))
        best_fitness = float(fitness[best_idx])
        best_chrom = pop[best_idx].copy()

        self.fitness_history = [best_fitness]
        self.mean_fitness_history = [float(fitness.mean())]
        self.worst_fitness_history = [float(fitness.max())]

        if verbose:
            print(f"  GA iniciado: {n_items} ítems, "
                  f"pop={self.cfg.population_size}, gen={self.cfg.n_generations}")
            print(f"  Gen   0: mejor={best_fitness:.4f}s  "
                  f"media={self.mean_fitness_history[0]:.4f}s  "
                  f"peor={self.worst_fitness_history[0]:.4f}s")

        elite_count = max(1, self.cfg.population_size // 10)

        for gen in range(1, self.cfg.n_generations + 1):
            new_pop = []

            # Elitismo: los mejores pasan directamente
            elite_idx = np.argsort(fitness)[:elite_count]
            for ei in elite_idx:
                new_pop.append(pop[ei].copy())

            # Generación de descendencia
            while len(new_pop) < self.cfg.population_size:
                p1 = self._tournament(pop, fitness, self.cfg.tournament_k, rng)
                p2 = self._tournament(pop, fitness, self.cfg.tournament_k, rng)

                # Cruce uniforme
                mask = rng.random(n_items) < self.cfg.crossover_rate
                child = np.where(mask, p1, p2)

                # Mutación gaussiana por gen
                mut_mask = rng.random(n_items) < self.cfg.mutation_rate
                child[mut_mask] += rng.normal(0, self.cfg.mutation_sigma, mut_mask.sum())
                child = np.clip(child, 0.0, 1.0)

                new_pop.append(child)

            pop = np.array(new_pop)
            fitness = np.array([evaluate(ind) for ind in pop])

            gen_best_idx = int(np.argmin(fitness))
            gen_best = float(fitness[gen_best_idx])

            if gen_best < best_fitness:
                best_fitness = gen_best
                best_chrom = pop[gen_best_idx].copy()

            self.fitness_history.append(best_fitness)
            self.mean_fitness_history.append(float(fitness.mean()))
            self.worst_fitness_history.append(float(fitness.max()))

            if verbose and (gen % 10 == 0 or gen == self.cfg.n_generations):
                print(f"  Gen {gen:3d}: mejor={best_fitness:.4f}s  "
                      f"media={self.mean_fitness_history[-1]:.4f}s  "
                      f"peor={self.worst_fitness_history[-1]:.4f}s")

        self.scores = {item: float(best_chrom[i]) for item, i in idx_map.items()}
        self._is_fitted = True

        total_improvement = (
            (self.fitness_history[0] - best_fitness) / self.fitness_history[0] * 100
        )

        return {
            "n_items": n_items,
            "n_generations": self.cfg.n_generations,
            "best_fitness": best_fitness,
            "initial_fitness": self.fitness_history[0],
            "total_improvement_pct": total_improvement,
            "fitness_history": self.fitness_history,
            "mean_fitness_history": self.mean_fitness_history,
            "worst_fitness_history": self.worst_fitness_history,
        }

    def predict_score(self, item_id: int, order_id: int = None,
                      history: pd.DataFrame = None) -> float:
        """Score en [0, 1] para un ítem. Interfaz compatible con GAPolicy."""
        if not self._is_fitted:
            raise RuntimeError("El optimizador GA no ha sido entrenado todavía.")
        return self.scores.get(item_id, 0.5)

    @staticmethod
    def _tournament(pop: np.ndarray, fitness: np.ndarray,
                    k: int, rng: np.random.Generator) -> np.ndarray:
        """Selección por torneo: retorna el mejor de k candidatos aleatorios."""
        candidates = rng.choice(len(pop), k, replace=False)
        best = candidates[int(np.argmin(fitness[candidates]))]
        return pop[best].copy()
