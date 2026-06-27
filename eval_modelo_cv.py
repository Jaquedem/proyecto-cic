"""
Evaluación completa del modelo de visión EfficientNet-B4.

Métricas aplicadas (basadas en ejercicios del curso IPN CIC):
  - Accuracy, Precision, Recall, F1 (sklearn) — Ejercicio regresión logística
  - Matriz de Confusión con heatmap seaborn    — Ejercicio regresión logística
  - Barras por clase con color condicional     — Tarea 1 (np, matplotlib)
  - np.diff sobre accuracies por clase         — Tarea 1
  - Distribución de confianza (histograma)     — Tarea 1
  - Subplots con height_ratios                 — Tarea 1

Uso:
    python eval_modelo_cv.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

DATASET_DIR = "./dataset"
MODEL_DIR   = "./modelo_cv"


def cargar_modelo():
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Dispositivo: {device.upper()}")

    extractor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()

    with open(Path(MODEL_DIR) / "label_map.json", encoding="utf-8") as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    label2id = maps["label2id"]

    return model, extractor, id2label, label2id, device


def predecir_dataset(model, extractor, label2id, device):
    """Recorre todo el dataset y devuelve y_true, y_pred y confianzas."""
    import torch
    from PIL import Image

    y_true, y_pred, confianzas = [], [], []

    clases = sorted(Path(DATASET_DIR).iterdir())
    total_imgs = 0
    print("\n📂 Procesando imágenes...")

    for clase_dir in clases:
        if not clase_dir.is_dir():
            continue
        nombre = clase_dir.name
        if nombre not in label2id:
            print(f"   ⚠️  '{nombre}' no está en el modelo, saltando.")
            continue

        true_id = label2id[nombre]
        imgs = [p for p in sorted(clase_dir.glob("*"))
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = extractor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_id = int(np.argmax(probs))
                confianza = float(probs[pred_id])

                y_true.append(true_id)
                y_pred.append(pred_id)
                confianzas.append(confianza)
                total_imgs += 1
            except Exception as e:
                print(f"   ⚠️  Error en {img_path.name}: {e}")

    print(f"   ✅ {total_imgs} imágenes procesadas.\n")
    return np.array(y_true), np.array(y_pred), np.array(confianzas)


def tabla_metricas(y_true, y_pred, id2label):
    """Tabla de métricas globales (como en el ejercicio de regresión logística)."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    exactitud   = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    sensibilidad = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1          = f1_score(y_true, y_pred, average="macro", zero_division=0)

    resultados = pd.DataFrame({
        "Métrica":    ["Exactitud (Accuracy)", "Precisión (Precision)", "Sensibilidad (Recall)", "Puntuación F1"],
        "Valor":      [exactitud, precision, sensibilidad, f1],
        "Porcentaje": [f"{v*100:.2f}%" for v in [exactitud, precision, sensibilidad, f1]],
    })

    print("--- MÉTRICAS GLOBALES DEL MODELO CV ---")
    print(resultados.to_string(index=False))
    print()
    return exactitud, precision, sensibilidad, f1


def graficar_confusion(y_true, y_pred, id2label):
    """Heatmap de matriz de confusión — igual que en el ejercicio de regresión logística."""
    from sklearn.metrics import confusion_matrix

    clases_ordenadas = sorted(id2label.keys())
    nombres = [id2label[i] for i in clases_ordenadas]

    cm = confusion_matrix(y_true, y_pred, labels=clases_ordenadas)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=nombres,
        yticklabels=nombres,
    )
    plt.title("Matriz de Confusión — Identificador de Juegos")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Clase Real")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("eval_confusion_matrix.png", dpi=150)
    plt.show()
    print("💾 Guardado: eval_confusion_matrix.png\n")


def graficar_por_clase(y_true, y_pred, id2label, confianzas):
    """
    Subplot doble — inspirado en Tarea 1 (subplots con height_ratios):
      - Arriba: barras de accuracy por clase con color condicional
      - Abajo:  np.diff de las accuracies (variación entre clases)
    """
    from sklearn.metrics import accuracy_score

    clases_ids = sorted(id2label.keys())
    nombres    = [id2label[i] for i in clases_ids]
    accuracies = []

    for cid in clases_ids:
        mask = y_true == cid
        if mask.sum() == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(accuracy_score(y_true[mask], y_pred[mask]))

    accs = np.array(accuracies)
    diffs = np.diff(accs)   # igual que np.diff de Tarea 1

    # Colores condicionales (verde ≥80%, naranja ≥50%, rojo <50%)
    colores = [
        "mediumseagreen" if a >= 0.8 else ("orange" if a >= 0.5 else "tomato")
        for a in accs
    ]
    colores_diff = [
        "steelblue" if d >= 0 else "mediumpurple"
        for d in diffs
    ]

    figure, axis = plt.subplots(2, 1, height_ratios=(3, 1), figsize=(12, 7))

    # Subplot 1 — Accuracy por clase
    bars = axis[0].bar(nombres, accs * 100, color=colores)
    axis[0].axhline(80, color="gray", linestyle="--", linewidth=1, label="Umbral 80%")
    axis[0].set_title("Accuracy por clase — Modelo EfficientNet-B4")
    axis[0].set_ylabel("Accuracy (%)")
    axis[0].set_ylim(0, 110)
    axis[0].legend()
    for bar, a in zip(bars, accs):
        axis[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{a*100:.0f}%",
            ha="center", va="bottom", fontsize=8
        )
    axis[0].tick_params(axis="x", rotation=35)

    # Subplot 2 — np.diff de las accuracies (Tarea 1)
    axis[1].bar(range(len(diffs)), diffs * 100, color=colores_diff)
    axis[1].axhline(0, color="black", linewidth=0.8)
    axis[1].set_title("Variación de accuracy entre clases (np.diff)")
    axis[1].set_ylabel("Δ Accuracy (%)")
    axis[1].set_xlabel("Clases (consecutivas)")

    plt.tight_layout()
    plt.savefig("eval_accuracy_por_clase.png", dpi=150)
    plt.show()
    print("💾 Guardado: eval_accuracy_por_clase.png\n")

    return nombres, accs


def graficar_metricas_por_clase(y_true, y_pred, id2label):
    """Barras agrupadas de Precision, Recall y F1 por clase."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    clases_ids = sorted(id2label.keys())
    nombres    = [id2label[i] for i in clases_ids]

    prec = precision_score(y_true, y_pred, average=None, labels=clases_ids, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, labels=clases_ids, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, labels=clases_ids, zero_division=0)

    x = np.arange(len(nombres))
    ancho = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - ancho, prec * 100, ancho, label="Precisión",    color="steelblue")
    ax.bar(x,         rec  * 100, ancho, label="Sensibilidad", color="mediumseagreen")
    ax.bar(x + ancho, f1   * 100, ancho, label="F1",           color="mediumpurple")

    ax.set_title("Precisión, Sensibilidad y F1 por clase")
    ax.set_ylabel("Valor (%)")
    ax.set_xlabel("Clase")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("eval_metricas_por_clase.png", dpi=150)
    plt.show()
    print("💾 Guardado: eval_metricas_por_clase.png\n")


def graficar_confianza(confianzas, y_true, y_pred):
    """
    Histograma de distribución de confianza separando aciertos y errores.
    Inspirado en los histogramas de frecuencia de Tarea 1.
    """
    aciertos = confianzas[y_true == y_pred]
    errores  = confianzas[y_true != y_pred]

    fig, axis = plt.subplots(2, 1, height_ratios=(2, 1), figsize=(10, 7))

    # Histogramas superpuestos
    axis[0].hist(aciertos, bins=20, color="mediumseagreen", alpha=0.7, label=f"Aciertos (n={len(aciertos)})")
    axis[0].hist(errores,  bins=20, color="tomato",         alpha=0.7, label=f"Errores  (n={len(errores)})")
    axis[0].set_title("Distribución de confianza del modelo")
    axis[0].set_ylabel("Frecuencia")
    axis[0].set_xlabel("Confianza (probabilidad)")
    axis[0].axvline(np.mean(confianzas), color="navy", linestyle="--",
                    label=f"Media total: {np.mean(confianzas):.2f}")
    axis[0].legend()

    # np.diff sobre confianzas ordenadas (Tarea 1)
    conf_sorted = np.sort(confianzas)
    diff_conf   = np.diff(conf_sorted)
    axis[1].bar(
        np.arange(len(diff_conf)),
        diff_conf,
        color=[("steelblue" if d >= 0 else "mediumpurple") for d in diff_conf],
        width=1.0
    )
    axis[1].axhline(0, color="black", linewidth=0.7)
    axis[1].set_title("Variación entre confianzas ordenadas (np.diff)")
    axis[1].set_ylabel("Δ Confianza")
    axis[1].set_xlabel("Índice (imagen)")

    plt.tight_layout()
    plt.savefig("eval_distribucion_confianza.png", dpi=150)
    plt.show()
    print("💾 Guardado: eval_distribucion_confianza.png\n")


def resumen_final(exactitud, precision, sensibilidad, f1, nombres, accs):
    """Tabla resumen final igual que la tabla de betas del ejercicio de regresión logística."""
    df = pd.DataFrame({
        "Clase":           nombres,
        "Accuracy (%)":    [f"{a*100:.1f}" for a in accs],
        "Estado":          ["✅" if a >= 0.8 else ("⚠️" if a >= 0.5 else "❌") for a in accs],
    })
    print("--- RESUMEN POR CLASE ---")
    print(df.to_string(index=False))
    print()
    print(f"  Accuracy global  : {exactitud*100:.2f}%")
    print(f"  Precisión macro  : {precision*100:.2f}%")
    print(f"  Sensibilidad macro: {sensibilidad*100:.2f}%")
    print(f"  F1 macro         : {f1*100:.2f}%")
    print()
    if exactitud >= 0.80:
        print("✅ Modelo listo para producción (accuracy ≥ 80%)")
    elif exactitud >= 0.60:
        print("⚠️  Accuracy aceptable. Más fotos o más épocas pueden mejorar el modelo.")
    else:
        print("❌ Accuracy baja. Recomendado: reentrenar con más datos y bf16=True.")


def evaluar():
    try:
        model, extractor, id2label, label2id, device = cargar_modelo()
    except Exception as e:
        print(f"❌ No se pudo cargar el modelo: {e}")
        return

    y_true, y_pred, confianzas = predecir_dataset(model, extractor, label2id, device)
    if len(y_true) == 0:
        print("❌ No se encontraron imágenes en el dataset.")
        return

    exactitud, precision, sensibilidad, f1 = tabla_metricas(y_true, y_pred, id2label)
    graficar_confusion(y_true, y_pred, id2label)
    nombres, accs = graficar_por_clase(y_true, y_pred, id2label, confianzas)
    graficar_metricas_por_clase(y_true, y_pred, id2label)
    graficar_confianza(confianzas, y_true, y_pred)
    resumen_final(exactitud, precision, sensibilidad, f1, nombres, accs)


if __name__ == "__main__":
    evaluar()
