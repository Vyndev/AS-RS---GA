"""
Visualizaciones para presentación — AS/RS con Algoritmo Genético.

Genera 8 figuras aisladas, listas para incluir en diapositivas.

Figuras de comparación de políticas:
    fig_01_policy_comparison.png  — barras: tiempo + tasa de bloqueo
    fig_02_cumulative_time.png    — tiempo acumulado (divergencia entre políticas)
    fig_03_fill_grade.png         — evolución dinámica del llenado del rack
    fig_04_travel_time_dist.png   — distribución de tiempos por operación

Figuras de análisis del Algoritmo Genético:
    fig_05_ga_convergence.png     — curva de convergencia con diversidad poblacional
    fig_06_ga_improvement.png     — mejora acumulada + velocidad de aprendizaje
    fig_07_ga_score_demand.png    — correlación score GA vs demanda real
    fig_08_ga_rank_alignment.png  — alineación de rankings: GA vs demanda real

Uso:
    python analyze.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from src.config import ExperimentConfig
from src.data import generate_synthetic, load_transactions
from src.ga import DemandOptimizerGA
from src.optimizer import RandomPolicy, ABCPolicy, GAPolicy
from src.simulator import simulate, build_initial_rack

# ─────────────────────────────────────────────────────────────────────────────
# Estilo global
# ─────────────────────────────────────────────────────────────────────────────

PLT_STYLE = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "legend.frameon":    False,
    "legend.fontsize":   11,
}

C = {                          # paleta
    "random": "#9CA3AF",       # gris
    "abc":    "#1E3A8A",       # azul oscuro
    "ga":     "#D97706",       # ámbar
    "ga_mid": "#F59E0B",       # ámbar claro (media poblacional)
    "ga_lo":  "#FCD34D",       # ámbar muy claro (peor individuo)
    "pos":    "#059669",       # verde (mejora positiva)
    "neg":    "#DC2626",       # rojo (desalineación)
    "bg":     "#FFFBF5",       # fondo cálido
}
LABELS = {"random": "Random", "abc": "ABC", "ga": "GA"}
DPI    = 150


def _ax_clean(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22, linestyle="--", axis=grid_axis)


def _save(fig, path: Path):
    fig.patch.set_facecolor(C["bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(C["bg"])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURAS DE COMPARACIÓN DE POLÍTICAS
# ═════════════════════════════════════════════════════════════════════════════

def fig_policy_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Barras horizontales: tiempo medio + tasa de bloqueo, ordenadas por desempeño."""
    with plt.rc_context(PLT_STYLE):
        fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5))

        order  = ["ga", "abc", "random"]
        labels = [LABELS[p] for p in order]
        colors = [C[p] for p in order]
        times  = [float(summary[summary["policy"] == p]["mean_travel_time"].iloc[0])
                  for p in order]
        blocks = [float(summary[summary["policy"] == p]["blocking_rate"].iloc[0])
                  for p in order]
        random_time = float(summary[summary["policy"] == "random"]["mean_travel_time"].iloc[0])

        # ── Panel izquierdo: tiempo ──────────────────────────────────────────
        bars = ax_t.barh(labels, times, color=colors, height=0.50, edgecolor="white")
        for bar, val, pol in zip(bars, times, order):
            impr = (random_time - val) / random_time * 100
            suffix = f"  {val:.2f}s"
            if pol != "random":
                suffix += f"  ({impr:+.1f}%)"
            ax_t.text(val + max(times) * 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      suffix, va="center", fontsize=11,
                      fontweight="bold", color=C[pol])
        ax_t.set_xlim(0, max(times) * 1.22)
        ax_t.set_xlabel("Tiempo de viaje promedio (s)", fontsize=12)
        ax_t.set_title("Tiempo de viaje promedio", pad=10)
        _ax_clean(ax_t, "x")

        # ── Panel derecho: bloqueo ───────────────────────────────────────────
        bars2 = ax_b.barh(labels, [b * 100 for b in blocks],
                          color=colors, height=0.50, edgecolor="white")
        for bar, val in zip(bars2, blocks):
            ax_b.text(val * 100 + max(blocks) * 100 * 0.015,
                      bar.get_y() + bar.get_height() / 2,
                      f"{val*100:.1f}%", va="center",
                      fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Tasa de bloqueo (%)", fontsize=12)
        ax_b.set_title("Tasa de bloqueo", pad=10)
        _ax_clean(ax_b, "x")

        fig.suptitle("Comparación de Políticas de Asignación — AS/RS Doble Profundidad",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, output_path)


def fig_cumulative_time(results_by_policy: dict, output_path: Path) -> None:
    """Tiempo acumulado: divergencia total entre políticas."""
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))

        for pol in ["random", "abc", "ga"]:
            result = results_by_policy[pol]
            cum = np.cumsum(result.travel_times)
            ax.plot(cum, label=LABELS[pol], color=C[pol], linewidth=2.5)

        # Brecha final anotada
        cum_r = np.cumsum(results_by_policy["random"].travel_times)
        cum_g = np.cumsum(results_by_policy["ga"].travel_times)
        saving = cum_r[-1] - cum_g[-1]
        n_ops  = len(cum_r)
        ax.annotate(
            f"Ahorro GA vs Random\n{saving:.0f}s acumulados",
            xy=(n_ops - 1, cum_g[-1]),
            xytext=(n_ops * 0.72, (cum_r[-1] + cum_g[-1]) / 2),
            fontsize=11, color=C["ga"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["ga"], lw=1.5),
        )
        ax.set_xlabel("Número de operación", fontsize=12)
        ax.set_ylabel("Tiempo de viaje acumulado (s)", fontsize=12)
        ax.set_title("Tiempo de Viaje Acumulado por Política", pad=12)
        ax.legend(loc="upper left")
        _ax_clean(ax)
        fig.tight_layout()
        _save(fig, output_path)


def fig_fill_grade(results_by_policy: dict, output_path: Path) -> None:
    """Evolución dinámica del llenado del rack (oscilaciones llenado/vaciado)."""
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        for pol in ["random", "abc", "ga"]:
            result = results_by_policy[pol]
            fg  = np.array(result.fill_grade_series)
            ops = np.arange(len(fg))
            # Curva cruda muy transparente + suavizado
            ax.plot(ops, fg, color=C[pol], alpha=0.18, linewidth=0.8)
            window   = max(1, len(fg) // 25)
            fg_smooth = np.convolve(fg, np.ones(window) / window, mode="same")
            ax.plot(ops, fg_smooth, color=C[pol], linewidth=2.4, label=LABELS[pol])

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel("Número de operación", fontsize=12)
        ax.set_ylabel("Ocupación del rack", fontsize=12)
        ax.set_title("Evolución Dinámica del Llenado del Rack", pad=12)
        ax.annotate("↑ almacenamiento   ↓ recuperación",
                    xy=(0.01, 0.04), xycoords="axes fraction",
                    fontsize=9.5, color="#6B7280", style="italic")
        ax.legend(loc="upper right")
        _ax_clean(ax)
        fig.tight_layout()
        _save(fig, output_path)


def fig_travel_time_dist(results_by_policy: dict, output_path: Path) -> None:
    """Histogramas + KDE superpuestos del tiempo de viaje por operación."""
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))

        for pol in ["random", "abc", "ga"]:
            times = np.array(results_by_policy[pol].travel_times)
            ax.hist(times, bins=35, alpha=0.30, color=C[pol], edgecolor="white")
            # KDE
            kde_x = np.linspace(times.min(), times.max(), 300)
            kde   = stats.gaussian_kde(times)
            # Escalar KDE para que coincida visualmente con histograma
            ax.plot(kde_x, kde(kde_x) * len(times) * (times.max() - times.min()) / 35,
                    color=C[pol], linewidth=2.5, label=LABELS[pol])
            ax.axvline(times.mean(), color=C[pol], linewidth=1.5,
                       linestyle="--", alpha=0.8)

        ax.set_xlabel("Tiempo de viaje por operación (s)", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        ax.set_title("Distribución del Tiempo de Viaje por Operación", pad=12)
        ax.legend()
        _ax_clean(ax, "y")
        fig.tight_layout()
        _save(fig, output_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURAS DEL ALGORITMO GENÉTICO
# ═════════════════════════════════════════════════════════════════════════════

def fig_ga_convergence(ga_metrics: dict, output_path: Path) -> None:
    """Convergencia del GA: mejor / media / peor individuo por generación.

    Muestra cómo la población converge progresivamente hacia el óptimo y cómo
    la diversidad (brecha peor-mejor) se reduce con las generaciones.
    """
    with plt.rc_context(PLT_STYLE):
        best  = np.array(ga_metrics["fitness_history"])
        mean  = np.array(ga_metrics["mean_fitness_history"])
        worst = np.array(ga_metrics["worst_fitness_history"])
        gens  = np.arange(len(best))

        fig, ax = plt.subplots(figsize=(11, 5.5))

        ax.fill_between(gens, worst, best, alpha=0.10, color=C["ga"],
                        label="Rango población (peor – mejor)")
        ax.fill_between(gens, mean, best,  alpha=0.20, color=C["ga"])
        ax.plot(gens, worst, color=C["ga_lo"],  linewidth=1.2, linestyle=":",
                label="Peor individuo")
        ax.plot(gens, mean,  color=C["ga_mid"], linewidth=1.8, linestyle="--",
                label="Media población")
        ax.plot(gens, best,  color=C["ga"],     linewidth=3.0,
                label="Mejor individuo")

        # Anotaciones inicio / fin
        for gen_idx, label in [(0, f"Gen 0\n{best[0]:.3f}s"),
                                (len(gens)-1, f"Gen {len(gens)-1}\n{best[-1]:.3f}s")]:
            offset_x = len(gens) * 0.04 * (1 if gen_idx == 0 else -1)
            ax.annotate(label,
                        xy=(gen_idx, best[gen_idx]),
                        xytext=(gen_idx + offset_x * 3,
                                best[gen_idx] + (best.max() - best.min()) * 0.06),
                        fontsize=10, color=C["ga"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C["ga"], lw=1.3))

        ax.axhline(best[-1], color=C["ga"], linewidth=1, linestyle="-.", alpha=0.45)
        ax.set_xlabel("Generación", fontsize=12)
        ax.set_ylabel("Fitness — tiempo medio de viaje (s)", fontsize=12)
        ax.set_title("Convergencia del Algoritmo Genético", pad=12)
        ax.legend(loc="upper right")
        _ax_clean(ax)
        fig.tight_layout()
        _save(fig, output_path)


def fig_ga_improvement(ga_metrics: dict, output_path: Path) -> None:
    """Dos paneles: mejora acumulada (%) + velocidad de aprendizaje por generación.

    La velocidad revela en qué fase el GA aprende más rápido: generalmente las
    primeras generaciones dan los mayores saltos, y luego se producen mejoras
    incrementales de refinamiento.
    """
    with plt.rc_context(PLT_STYLE):
        best  = np.array(ga_metrics["fitness_history"])
        gens  = np.arange(len(best))
        impr  = (best[0] - best) / best[0] * 100   # mejora acumulada (%)
        delta = np.clip(-np.diff(best), 0, None)    # ganancia por generación (s)

        fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Mejora acumulada ─────────────────────────────────────────────────
        ax_a.plot(gens, impr, color=C["ga"], linewidth=2.8)
        ax_a.fill_between(gens, 0, impr, alpha=0.18, color=C["ga"])
        ax_a.axhline(impr[-1], color=C["ga"], linewidth=1,
                     linestyle="--", alpha=0.55)
        ax_a.text(gens[-1] * 0.55, impr[-1] * 1.06,
                  f"Mejora total: {impr[-1]:.2f}%",
                  fontsize=11, color=C["ga"], fontweight="bold")
        ax_a.set_xlabel("Generación", fontsize=12)
        ax_a.set_ylabel("Mejora acumulada vs Gen 0 (%)", fontsize=12)
        ax_a.set_title("Mejora Acumulada del Mejor Individuo", pad=10)
        ax_a.set_ylim(bottom=0)
        _ax_clean(ax_a)

        # ── Velocidad de aprendizaje ─────────────────────────────────────────
        bar_colors = [C["ga"] if d > 0 else "#E5E7EB" for d in delta]
        ax_d.bar(gens[1:], delta, color=bar_colors,
                 edgecolor="white", linewidth=0.5, width=0.85)
        # Suavizado tendencia
        w = max(3, len(delta) // 8)
        smooth = np.convolve(delta, np.ones(w) / w, mode="same")
        ax_d.plot(gens[1:], smooth, color="#92400E", linewidth=2,
                  linestyle="--", alpha=0.85, label=f"Tendencia (ventana {w})")
        ax_d.set_xlabel("Generación", fontsize=12)
        ax_d.set_ylabel("Reducción de fitness (s/gen)", fontsize=12)
        ax_d.set_title("Velocidad de Aprendizaje por Generación", pad=10)
        ax_d.legend()
        _ax_clean(ax_d, "y")

        fig.suptitle("Análisis de Aprendizaje del Algoritmo Genético",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, output_path)


def fig_ga_score_demand(optimizer_scores: dict, eval_df: pd.DataFrame,
                         output_path: Path) -> None:
    """Correlación entre el score evolucionado por el GA y la demanda real.

    Responde la pregunta clave: ¿aprendió el GA los patrones de demanda?
    Si la correlación es alta, los scores predicen correctamente qué ítems
    serán más demandados.
    """
    with plt.rc_context(PLT_STYLE):
        # Frecuencia de recuperación real (demanda en evaluación)
        retr_counts = (eval_df[eval_df["operation"] == "R"]
                       ["item_id"].value_counts().to_dict())
        total_retr  = max(sum(retr_counts.values()), 1)

        items    = sorted(optimizer_scores.keys())
        scores   = np.array([optimizer_scores[i] for i in items])
        freqs    = np.array([retr_counts.get(i, 0) / total_retr for i in items])

        # Cuartil de demanda para color
        quartiles = pd.qcut(freqs, 4, labels=False, duplicates="drop")
        cmap = plt.cm.YlOrBr
        point_colors = [cmap(0.25 + 0.22 * int(q)) for q in quartiles]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(freqs, scores, c=point_colors, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)

        # Línea de regresión
        slope, intercept, r, p, _ = stats.linregress(freqs, scores)
        x_line = np.linspace(freqs.min(), freqs.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color=C["ga"], linewidth=2, linestyle="--", alpha=0.8,
                label=f"Regresión  (r = {r:.3f})")

        # Spearman
        rho, p_sp = stats.spearmanr(freqs, scores)

        ax.set_xlabel("Frecuencia de recuperación real (fracción de ops)", fontsize=12)
        ax.set_ylabel("Score evolucionado por el GA [0, 1]", fontsize=12)
        ax.set_title("¿Aprendió el GA la Demanda Real?", pad=12)

        # Leyenda cuartiles
        legend_els = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(0.25 + 0.22 * q),
                   markersize=9, label=f"Q{q+1} demanda")
            for q in range(4)
        ]
        legend_els.append(
            Line2D([0], [0], color=C["ga"], linewidth=2,
                   linestyle="--", label=f"Regresión (r={r:.3f})"))
        ax.legend(handles=legend_els, loc="upper left", fontsize=10)

        ax.text(0.97, 0.05,
                f"Pearson  r = {r:.3f}  (p={p:.3f})\n"
                f"Spearman ρ = {rho:.3f}  (p={p_sp:.3f})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color="#374151",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.7, edgecolor="#D1D5DB"))
        _ax_clean(ax)
        fig.tight_layout()
        _save(fig, output_path)


def fig_ga_rank_alignment(optimizer_scores: dict, eval_df: pd.DataFrame,
                           output_path: Path) -> None:
    """Alineación de rankings: prioridad GA vs demanda real.

    Cada punto es un ítem. El eje X es el rango de demanda real (1 = más
    demandado), el eje Y es el rango de score GA (1 = score más alto).
    La diagonal perfecta indica que el GA aprendió el orden de demanda
    exactamente. Puntos alejados de la diagonal son ítems mal priorizados.
    """
    with plt.rc_context(PLT_STYLE):
        retr_counts = (eval_df[eval_df["operation"] == "R"]
                       ["item_id"].value_counts().to_dict())

        items  = sorted(optimizer_scores.keys())
        scores = np.array([optimizer_scores[i] for i in items])
        freqs  = np.array([retr_counts.get(i, 0) for i in items])

        n = len(items)
        # Rangos: 1 = mejor
        demand_rank = n + 1 - pd.Series(freqs).rank(method="average").values
        score_rank  = n + 1 - pd.Series(scores).rank(method="average").values

        rank_error  = np.abs(demand_rank - score_rank)
        rho, p_sp   = stats.spearmanr(demand_rank, score_rank)

        fig, ax = plt.subplots(figsize=(8, 7))

        # Diagonal de referencia
        ax.plot([1, n], [1, n], color="#9CA3AF", linewidth=1.5,
                linestyle="--", zorder=1, label="Alineación perfecta")

        # Puntos coloreados por error de rango
        sc = ax.scatter(demand_rank, score_rank,
                        c=rank_error, cmap="RdYlGn_r",
                        s=65, alpha=0.85, edgecolors="white",
                        linewidths=0.6, zorder=3,
                        vmin=0, vmax=n // 2)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.82)
        cbar.set_label("Error de ranking (posiciones)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        ax.set_xlabel("Rango de demanda real (1 = más demandado)", fontsize=12)
        ax.set_ylabel("Rango de score GA (1 = mayor prioridad)", fontsize=12)
        ax.set_title("Alineación del Ranking GA vs Demanda Real", pad=12)
        ax.legend(loc="upper left", fontsize=10)

        ax.text(0.97, 0.05,
                f"Spearman ρ = {rho:.3f}  (p={p_sp:.3f})\n"
                f"Error medio: {rank_error.mean():.1f} posiciones",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color="#374151",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.7, edgecolor="#D1D5DB"))
        _ax_clean(ax)
        fig.tight_layout()
        _save(fig, output_path)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(csv_path: str = None):
    cfg         = ExperimentConfig()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  VISUALIZACIONES PARA PRESENTACIÓN — AS/RS + GA")
    print("=" * 72)

    # ── Datos ─────────────────────────────────────────────────────────────────
    df = load_transactions(csv_path) if csv_path else generate_synthetic(cfg.synthetic)
    split_idx = int(len(df) * cfg.train_split)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    eval_df   = df.iloc[split_idx:].reset_index(drop=True)

    # ── GA ────────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Ejecutando GA  "
          f"(pop={cfg.ga.population_size}, gen={cfg.ga.n_generations})...")
    optimizer  = DemandOptimizerGA(cfg.ga)
    ga_metrics = optimizer.fit(train_df, cfg.rack, cfg.crane, cfg.costs, verbose=True)
    print(f"  Mejora GA: {ga_metrics['total_improvement_pct']:.2f}%  "
          f"({ga_metrics['initial_fitness']:.3f}s -> {ga_metrics['best_fitness']:.3f}s)")

    # ── Simulación ────────────────────────────────────────────────────────────
    print("\n[2/3] Simulando políticas...")
    initial_rack = build_initial_rack(train_df, cfg.rack)
    print(f"  Rack inicial: {initial_rack.fill_grade:.1%} de ocupación")

    policies = {
        "random": RandomPolicy(seed=cfg.synthetic.seed),
        "abc":    ABCPolicy(cfg.rack),
        "ga":     GAPolicy(optimizer.scores, cfg.rack, cfg.costs),
    }
    results_by_policy = {}
    summary_rows      = []
    for name, policy in policies.items():
        print(f"  [{name.upper():>6}]...", end="  ", flush=True)
        result = simulate(eval_df, policy, cfg.rack, cfg.crane,
                          history_for_features=train_df,
                          initial_rack=initial_rack, verbose=False)
        results_by_policy[name] = result
        d = result.to_dict()
        summary_rows.append(d)
        print(f"T.med={d['mean_travel_time']:.3f}s  bloqueo={d['blocking_rate']:.1%}")
    summary = pd.DataFrame(summary_rows)

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n[3/3] Generando figuras...")

    figs = [
        ("fig_01_policy_comparison.png",
         lambda p: fig_policy_comparison(summary, p),
         "Comparación de políticas (barras)"),
        ("fig_02_cumulative_time.png",
         lambda p: fig_cumulative_time(results_by_policy, p),
         "Tiempo acumulado"),
        ("fig_03_fill_grade.png",
         lambda p: fig_fill_grade(results_by_policy, p),
         "Llenado dinámico del rack"),
        ("fig_04_travel_time_dist.png",
         lambda p: fig_travel_time_dist(results_by_policy, p),
         "Distribución de tiempos"),
        ("fig_05_ga_convergence.png",
         lambda p: fig_ga_convergence(ga_metrics, p),
         "Convergencia GA (mejor/media/peor)"),
        ("fig_06_ga_improvement.png",
         lambda p: fig_ga_improvement(ga_metrics, p),
         "Mejora acumulada + velocidad"),
        ("fig_07_ga_score_demand.png",
         lambda p: fig_ga_score_demand(optimizer.scores, eval_df, p),
         "Score GA vs demanda real (correlación)"),
        ("fig_08_ga_rank_alignment.png",
         lambda p: fig_ga_rank_alignment(optimizer.scores, eval_df, p),
         "Alineación de rankings"),
    ]

    for fname, fn, desc in figs:
        fn(results_dir / fname)
        print(f"  OK  {fname:<40}  {desc}")

    print(f"\n  8 figuras guardadas en: {results_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    main(csv_path=args.csv)
