"""
Evaluación del modelo de búsqueda semántica (multilingual-e5-base).

Métricas y visualizaciones aplicadas (basadas en ejercicios del curso IPN CIC):
  - Tabla de scores de similitud por consulta   — Ejercicio regresión logística
  - Barras de similitud con color condicional   — Tarea 1
  - np.diff sobre scores (caída de relevancia)  — Tarea 1
  - Subplots con height_ratios                  — Tarea 1
  - Distribución de scores (histograma)         — Tarea 1
  - Precisión@K para cada consulta test         — Ejercicio regresión logística

Uso:
    python eval_buscador.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from buscador import BuscadorLudoteca


# Consultas de prueba: (texto, complejidad esperada, jugadores esperados)
# Cada tupla define qué esperamos encontrar para poder medir calidad
CONSULTAS_TEST = [
    {
        "consulta":       "quiero algo rápido y divertido para toda la familia",
        "complejidad_ok": ["Bajo"],
        "top_k":          5,
        "etiqueta":       "Familiar/rápido",
    },
    {
        "consulta":       "juego complejo con mucha estrategia y planificación",
        "complejidad_ok": ["Alto", "Medio"],
        "top_k":          5,
        "etiqueta":       "Estrategia compleja",
    },
    {
        "consulta":       "algo de deducción o misterio donde tengas que adivinar",
        "complejidad_ok": ["Bajo", "Medio", "Alto"],
        "top_k":          5,
        "etiqueta":       "Deducción/misterio",
    },
    {
        "consulta":       "juego corto de 15 minutos para calentar antes de jugar",
        "complejidad_ok": ["Bajo"],
        "top_k":          5,
        "etiqueta":       "Calentamiento rápido",
    },
    {
        "consulta":       "quiero algo competitivo uno contra uno, largo y que requiera pensar mucho",
        "complejidad_ok": ["Alto", "Medio"],
        "top_k":          5,
        "etiqueta":       "Competitivo 1v1",
    },
]


def evaluar_consulta(buscador, consulta_cfg):
    """Ejecuta una consulta y devuelve resultados con scores."""
    resultados = buscador.buscar_juegos(
        consulta_cfg["consulta"],
        top_k=consulta_cfg["top_k"],
    )
    return resultados


def calcular_precision_at_k(resultados, complejidades_ok):
    """
    Precision@K: qué fracción de los top-K resultados tienen
    la complejidad que esperamos para esta consulta.
    """
    if not resultados:
        return 0.0
    aciertos = sum(1 for r in resultados if r["Complejidad"] in complejidades_ok)
    return aciertos / len(resultados)


def tabla_resumen(resultados_por_consulta):
    """Tabla de métricas por consulta — como la tabla de betas del ejercicio."""
    filas = []
    for cfg, resultados in resultados_por_consulta:
        scores = [float(r["Similitud"]) for r in resultados]
        p_at_k = calcular_precision_at_k(resultados, cfg["complejidad_ok"])
        filas.append({
            "Consulta":         cfg["etiqueta"],
            "Score máx":        f"{max(scores):.3f}" if scores else "—",
            "Score medio":      f"{np.mean(scores):.3f}" if scores else "—",
            "Score mín":        f"{min(scores):.3f}" if scores else "—",
            f"Precision@{cfg['top_k']}": f"{p_at_k*100:.0f}%",
        })
    df = pd.DataFrame(filas)
    print("--- MÉTRICAS DEL BUSCADOR SEMÁNTICO ---")
    print(df.to_string(index=False))
    print()
    return df


def graficar_scores_por_consulta(resultados_por_consulta):
    """
    Subplot: barras de similitud + np.diff de los scores.
    Inspirado directamente en Tarea 1 (edades + diferencias).
    """
    n = len(resultados_por_consulta)
    figure, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n),
                                gridspec_kw={"width_ratios": [3, 1]})
    if n == 1:
        axes = [axes]

    for i, (cfg, resultados) in enumerate(resultados_por_consulta):
        nombres = [r["Nombre"][:18] for r in resultados]
        scores  = np.array([float(r["Similitud"]) for r in resultados])
        complejidades = [r["Complejidad"] for r in resultados]

        # Colores condicionales según complejidad esperada
        colores = [
            "mediumseagreen" if c in cfg["complejidad_ok"] else "tomato"
            for c in complejidades
        ]

        # Barras de similitud
        bars = axes[i][0].bar(nombres, scores, color=colores)
        axes[i][0].set_title(f'"{cfg["etiqueta"]}"', fontsize=10, fontweight="bold")
        axes[i][0].set_ylabel("Similitud coseno")
        axes[i][0].set_ylim(0, 1.0)
        axes[i][0].tick_params(axis="x", rotation=30)
        for bar, s in zip(bars, scores):
            axes[i][0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{s:.3f}", ha="center", va="bottom", fontsize=8
            )

        # np.diff de los scores (Tarea 1: caída de relevancia entre posiciones)
        diff_scores = np.diff(scores)
        colores_diff = [("steelblue" if d >= 0 else "mediumpurple") for d in diff_scores]
        axes[i][1].bar(range(len(diff_scores)), diff_scores, color=colores_diff)
        axes[i][1].axhline(0, color="black", linewidth=0.7)
        axes[i][1].set_title("Δ Score (np.diff)", fontsize=9)
        axes[i][1].set_ylabel("Caída de relevancia")
        axes[i][1].set_xlabel("Posición")

    plt.suptitle("Análisis de scores por consulta semántica", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("eval_buscador_scores.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("💾 Guardado: eval_buscador_scores.png\n")


def graficar_precision_at_k(resultados_por_consulta):
    """Barras de Precision@K por consulta — como las métricas del ejercicio de regresión."""
    etiquetas = [cfg["etiqueta"] for cfg, _ in resultados_por_consulta]
    precisiones = [
        calcular_precision_at_k(resultados, cfg["complejidad_ok"]) * 100
        for cfg, resultados in resultados_por_consulta
    ]

    colores = [
        "mediumseagreen" if p >= 80 else ("orange" if p >= 50 else "tomato")
        for p in precisiones
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(etiquetas, precisiones, color=colores)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="Umbral 80%")
    ax.set_title("Precision@K por tipo de consulta\n(% resultados con complejidad esperada)")
    ax.set_ylabel("Precision@K (%)")
    ax.set_ylim(0, 115)
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    for bar, p in zip(bars, precisiones):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{p:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig("eval_buscador_precision.png", dpi=150)
    plt.show()
    print("💾 Guardado: eval_buscador_precision.png\n")


def graficar_distribucion_scores(resultados_por_consulta):
    """
    Distribución global de todos los scores de similitud.
    Histograma + np.diff — Tarea 1.
    """
    todos_los_scores = []
    for _, resultados in resultados_por_consulta:
        todos_los_scores.extend([float(r["Similitud"]) for r in resultados])

    scores = np.array(todos_los_scores)
    scores_sorted = np.sort(scores)
    diff_scores = np.diff(scores_sorted)

    figure, axis = plt.subplots(2, 1, height_ratios=(2, 1), figsize=(10, 7))

    # Histograma de distribución
    axis[0].hist(scores, bins=15, color="steelblue", alpha=0.8, edgecolor="white")
    axis[0].axvline(np.mean(scores), color="tomato", linestyle="--",
                    label=f"Media: {np.mean(scores):.3f}")
    axis[0].axvline(np.median(scores), color="orange", linestyle="--",
                    label=f"Mediana: {np.median(scores):.3f}")
    axis[0].set_title("Distribución global de scores de similitud")
    axis[0].set_xlabel("Similitud coseno")
    axis[0].set_ylabel("Frecuencia")
    axis[0].legend()

    # np.diff sobre scores ordenados (Tarea 1)
    axis[1].bar(
        np.arange(len(diff_scores)),
        diff_scores,
        color=[("steelblue" if d >= 0 else "mediumpurple") for d in diff_scores],
        width=1.0
    )
    axis[1].axhline(0, color="black", linewidth=0.7)
    axis[1].set_title("Variación entre scores ordenados (np.diff)")
    axis[1].set_ylabel("Δ Score")
    axis[1].set_xlabel("Índice")

    plt.tight_layout()
    plt.savefig("eval_buscador_distribucion.png", dpi=150)
    plt.show()
    print("💾 Guardado: eval_buscador_distribucion.png\n")


def imprimir_resultados_detallados(resultados_por_consulta):
    """Muestra cada consulta con sus resultados como tabla — como display() del ejercicio."""
    for cfg, resultados in resultados_por_consulta:
        print(f"\n🔍 Consulta: \"{cfg['consulta']}\"")
        print(f"   Complejidades esperadas: {cfg['complejidad_ok']}")
        print(f"   Precision@{cfg['top_k']}: "
              f"{calcular_precision_at_k(resultados, cfg['complejidad_ok'])*100:.0f}%")
        df = pd.DataFrame([{
            "Nombre":       r["Nombre"],
            "Complejidad":  r["Complejidad"],
            "Tiempo (min)": r["Tiempo"],
            "Jugadores":    f"{r['Min']}-{r['Max']}",
            "Similitud":    r["Similitud"],
            "¿Esperado?":   "✅" if r["Complejidad"] in cfg["complejidad_ok"] else "❌",
        } for r in resultados])
        print(df.to_string(index=False))


def evaluar():
    print("📦 Cargando buscador semántico...")
    try:
        buscador = BuscadorLudoteca()
        print("✅ Modelo E5 cargado.\n")
    except Exception as e:
        print(f"❌ No se pudo cargar el buscador: {e}")
        return

    # Ejecutar todas las consultas de prueba
    resultados_por_consulta = []
    for cfg in CONSULTAS_TEST:
        resultados = evaluar_consulta(buscador, cfg)
        resultados_por_consulta.append((cfg, resultados))

    imprimir_resultados_detallados(resultados_por_consulta)
    print()
    tabla_resumen(resultados_por_consulta)
    graficar_scores_por_consulta(resultados_por_consulta)
    graficar_precision_at_k(resultados_por_consulta)
    graficar_distribucion_scores(resultados_por_consulta)


if __name__ == "__main__":
    evaluar()
