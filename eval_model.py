"""
Evaluación rápida del modelo entrenado.
Carga el modelo de ./modelo_cv/ y mide accuracy por clase sobre todo el dataset.

Uso:
    python eval_model.py
"""

import json
from pathlib import Path
from collections import defaultdict

DATASET_DIR = "./dataset"
MODEL_DIR   = "./modelo_cv"


def evaluar():
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Dispositivo: {device.upper()}")

    print(f"\n📦 Cargando modelo desde '{MODEL_DIR}'...")
    extractor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    with open(Path(MODEL_DIR) / "label_map.json", encoding="utf-8") as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()}
    label2id = maps["label2id"]

    print(f"✅ Modelo cargado — {len(id2label)} clases\n")

    # Recorrer dataset completo
    correctas = defaultdict(int)
    totales   = defaultdict(int)

    clases = sorted(Path(DATASET_DIR).iterdir())
    for clase_dir in clases:
        if not clase_dir.is_dir():
            continue
        nombre = clase_dir.name
        if nombre not in label2id:
            print(f"⚠️  Clase '{nombre}' no está en el modelo, saltando.")
            continue

        true_id = label2id[nombre]
        imgs = sorted(clase_dir.glob("*"))
        imgs = [p for p in imgs if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = extractor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                pred_id = logits.argmax(-1).item()
                totales[nombre] += 1
                if pred_id == true_id:
                    correctas[nombre] += 1
            except Exception as e:
                print(f"  ⚠️  Error en {img_path.name}: {e}")

    # Mostrar resultados
    print("=" * 55)
    print(f"{'CLASE':<25} {'ACIERTOS':>8}  {'BARRA':<22} {'ACC':>5}")
    print("=" * 55)

    total_ok = 0
    total_n  = 0
    for nombre in sorted(totales.keys()):
        ok = correctas[nombre]
        n  = totales[nombre]
        acc = ok / n if n > 0 else 0
        barra = "█" * int(acc * 20)
        print(f"  {nombre:<23} {ok:>3}/{n:<3}  {barra:<22} {acc*100:>4.0f}%")
        total_ok += ok
        total_n  += n

    print("=" * 55)
    global_acc = total_ok / total_n if total_n > 0 else 0
    print(f"  {'TOTAL':<23} {total_ok:>3}/{total_n:<3}  {'█'*int(global_acc*20):<22} {global_acc*100:>4.0f}%")
    print("=" * 55)

    if global_acc >= 0.80:
        print("\n✅ Modelo listo para producción (≥80%)")
    elif global_acc >= 0.60:
        print("\n⚠️  Accuracy aceptable pero mejorable. Considera más fotos o más épocas.")
    else:
        print("\n❌ Accuracy baja. Recomendado: reentrenar con bf16=True y más datos.")


if __name__ == "__main__":
    evaluar()
