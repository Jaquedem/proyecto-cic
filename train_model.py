"""
Script de entrenamiento del modelo de reconocimiento de juegos de mesa.

Estructura esperada del dataset:
    dataset/
        Love letter/
            foto1.jpg
            foto2.jpg
            ...
        Carcassonne/
            foto1.jpg
            ...
        (una carpeta por juego, nombre = nombre del juego)

Uso:
    python train_model.py

El modelo entrenado se guarda en ./modelo_cv/
"""

import os
import json
from pathlib import Path

DATASET_DIR = "./dataset"
OUTPUT_DIR = "./modelo_cv"
MODEL_BASE = "google/efficientnet-b0"   # liviano, rápido de entrenar
NUM_EPOCHS = 5
BATCH_SIZE = 16
IMG_SIZE = 224
LEARNING_RATE = 5e-5


def verificar_dataset():
    path = Path(DATASET_DIR)
    if not path.exists():
        print(f"❌ Carpeta '{DATASET_DIR}' no encontrada.")
        print("   Crea la carpeta y dentro una subcarpeta por cada juego con sus fotos.")
        return None

    clases = [d for d in path.iterdir() if d.is_dir()]
    if len(clases) == 0:
        print("❌ No hay subcarpetas de juegos en el dataset.")
        return None

    total_imgs = 0
    for clase in clases:
        imgs = list(clase.glob("*.jpg")) + list(clase.glob("*.jpeg")) + list(clase.glob("*.png"))
        total_imgs += len(imgs)
        print(f"  📂 {clase.name}: {len(imgs)} imágenes")

    print(f"\n✅ Dataset: {len(clases)} clases, {total_imgs} imágenes en total.")
    return [c.name for c in sorted(clases)]


def entrenar():
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
        from datasets import Dataset
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("   Instala con: pip install transformers datasets torch torchvision Pillow")
        return

    print("🔍 Verificando dataset...")
    etiquetas = verificar_dataset()
    if etiquetas is None:
        return

    label2id = {label: i for i, label in enumerate(etiquetas)}
    id2label = {i: label for i, label in enumerate(etiquetas)}

    print(f"\n📥 Cargando procesador de imágenes ({MODEL_BASE})...")
    extractor = AutoImageProcessor.from_pretrained(MODEL_BASE)

    # Indexar rutas (no cargar pixels en RAM — HF Dataset no serializa numpy 3D bien)
    print("🖼️  Indexando imágenes...")
    rutas, labels = [], []
    for label in etiquetas:
        carpeta = Path(DATASET_DIR) / label
        for img_path in sorted(carpeta.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            rutas.append(str(img_path))
            labels.append(label2id[label])

    if len(rutas) == 0:
        print("❌ No se encontraron imágenes.")
        return

    print(f"✅ {len(rutas)} imágenes indexadas.")

    # División train/val 80-20
    import random
    indices = list(range(len(rutas)))
    random.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    def hacer_dataset(idx):
        return Dataset.from_dict({
            "ruta": [rutas[i] for i in idx],
            "label": [labels[i] for i in idx],
        })

    ds_train = hacer_dataset(train_idx)
    ds_val = hacer_dataset(val_idx)

    def preprocess(batch):
        imgs = [Image.open(r).convert("RGB") for r in batch["ruta"]]
        procesado = extractor(images=imgs, return_tensors="pt")
        batch["pixel_values"] = [procesado["pixel_values"][i] for i in range(len(imgs))]
        return batch

    ds_train = ds_train.map(preprocess, batched=True, batch_size=BATCH_SIZE, remove_columns=["ruta"])
    ds_val   = ds_val.map(preprocess, batched=True, batch_size=BATCH_SIZE, remove_columns=["ruta"])
    ds_train.set_format("torch", columns=["pixel_values", "label"])
    ds_val.set_format("torch", columns=["pixel_values", "label"])

    print(f"\n🧠 Cargando modelo base ({MODEL_BASE})...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_BASE,
        num_labels=len(etiquetas),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": float(accuracy)}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    print(f"\n🚀 Iniciando entrenamiento ({NUM_EPOCHS} épocas)...\n")
    trainer.train()

    print("\n💾 Guardando modelo...")
    trainer.save_model(OUTPUT_DIR)
    extractor.save_pretrained(OUTPUT_DIR)

    # Guardar etiquetas
    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(etiquetas))

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ¡Modelo guardado en '{OUTPUT_DIR}'!")
    print("   Reinicia el servidor FastAPI para que el identificador lo use automáticamente.")


if __name__ == "__main__":
    entrenar()
