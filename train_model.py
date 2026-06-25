"""
Script de entrenamiento del modelo de reconocimiento de juegos de mesa.

Estructura esperada del dataset:
    dataset/
        Love letter/
            foto1.jpg
            foto2.jpg
            ...
        (una carpeta por juego, nombre = nombre del juego)

Uso:
    python train_model.py

El modelo entrenado se guarda en ./modelo_cv/
"""

import os
import json
import random
from pathlib import Path

DATASET_DIR = "./dataset"
OUTPUT_DIR  = "./modelo_cv"
MODEL_BASE  = "google/efficientnet-b0"
NUM_EPOCHS  = 20
BATCH_SIZE  = 16
LEARNING_RATE = 2e-4


def verificar_dataset():
    path = Path(DATASET_DIR)
    if not path.exists():
        print(f"❌ Carpeta '{DATASET_DIR}' no encontrada.")
        return None

    clases = [d for d in path.iterdir() if d.is_dir()]
    if not clases:
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
        from torchvision import transforms
        from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("   Instala con: pip install transformers accelerate torch torchvision Pillow")
        return

    print("🔍 Verificando dataset...")
    etiquetas = verificar_dataset()
    if etiquetas is None:
        return

    label2id = {label: i for i, label in enumerate(etiquetas)}
    id2label = {i: label for i, label in enumerate(etiquetas)}

    print(f"\n📥 Cargando procesador de imágenes ({MODEL_BASE})...")
    extractor = AutoImageProcessor.from_pretrained(MODEL_BASE)

    # Indexar rutas
    print("🖼️  Indexando imágenes...")
    rutas, labels = [], []
    for label in etiquetas:
        carpeta = Path(DATASET_DIR) / label
        for img_path in sorted(carpeta.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            rutas.append(str(img_path))
            labels.append(label2id[label])

    if not rutas:
        print("❌ No se encontraron imágenes.")
        return

    print(f"✅ {len(rutas)} imágenes indexadas.")

    # División estratificada 80-20 (misma proporción por clase)
    from collections import defaultdict
    por_clase = defaultdict(list)
    for idx, lbl in enumerate(labels):
        por_clase[lbl].append(idx)

    train_idx, val_idx = [], []
    for lbl_indices in por_clase.values():
        random.shuffle(lbl_indices)
        split = max(1, int(len(lbl_indices) * 0.8))
        train_idx.extend(lbl_indices[:split])
        val_idx.extend(lbl_indices[split:])

    print(f"   Train: {len(train_idx)} | Val: {len(val_idx)}")

    # Augmentation solo para entrenamiento
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.RandomGrayscale(p=0.05),
    ])

    # Dataset dinámico — augmentation se aplica de nuevo cada época
    class GameDataset(torch.utils.data.Dataset):
        def __init__(self, indices, augment=False):
            self.indices  = indices
            self.augment  = augment

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            idx = self.indices[i]
            img = Image.open(rutas[idx]).convert("RGB")
            if self.augment:
                img = aug(img)
            pv = extractor(images=img, return_tensors="pt")["pixel_values"][0]
            return {"pixel_values": pv, "labels": labels[idx]}

    ds_train = GameDataset(train_idx, augment=True)
    ds_val   = GameDataset(val_idx,   augment=False)

    print(f"\n🧠 Cargando modelo base ({MODEL_BASE})...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_BASE,
        num_labels=len(etiquetas),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, lbls = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == lbls).mean()

        # Accuracy por clase
        por_clase_acc = {}
        for cls_id, cls_name in id2label.items():
            mask = lbls == cls_id
            if mask.sum() > 0:
                por_clase_acc[cls_name] = float((preds[mask] == lbls[mask]).mean())

        print("\n  Accuracy por clase:")
        for nombre, a in sorted(por_clase_acc.items()):
            barra = "█" * int(a * 20)
            print(f"    {nombre:<25} {barra:<20} {a*100:.0f}%")

        return {"accuracy": float(acc)}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        dataloader_num_workers=0,
        warmup_ratio=0.1,
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

    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(etiquetas))

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ¡Modelo guardado en '{OUTPUT_DIR}'!")
    print("   Reinicia el servidor FastAPI para que el identificador lo use automáticamente.")


if __name__ == "__main__":
    entrenar()
