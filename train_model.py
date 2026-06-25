"""
Script de entrenamiento del modelo de reconocimiento de juegos de mesa.

Estructura esperada del dataset:
    dataset/
        Love letter/
            foto1.jpg
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
from collections import defaultdict

DATASET_DIR = "./dataset"
OUTPUT_DIR  = "./modelo_cv"
MODEL_BASE  = "google/efficientnet-b4"
NUM_EPOCHS  = 25
BATCH_SIZE  = 32
LEARNING_RATE = 1e-4

# Estas variables se rellenan en entrenar() y son necesarias a nivel de módulo
# para que GameDataset sea picklable en Windows (multiprocessing spawn)
_rutas    = []
_labels   = []
_extractor = None
_aug       = None


# Nivel de módulo — requerido por Windows multiprocessing para poder pickle la clase
class GameDataset:
    def __init__(self, indices, augment=False):
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        from PIL import Image
        idx = self.indices[i]
        img = Image.open(_rutas[idx]).convert("RGB")
        if self.augment and _aug is not None:
            img = _aug(img)
        pv = _extractor(images=img, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pv, "labels": _labels[idx]}


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
    global _rutas, _labels, _extractor, _aug

    try:
        import torch
        from torchvision import transforms
        from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
        import numpy as np
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("   Instala con: pip install transformers accelerate torch torchvision Pillow")
        return

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)")
    else:
        print("⚠️  No se detectó GPU, entrenando en CPU (más lento).")

    print("🔍 Verificando dataset...")
    etiquetas = verificar_dataset()
    if etiquetas is None:
        return

    label2id = {label: i for i, label in enumerate(etiquetas)}
    id2label  = {i: label for i, label in enumerate(etiquetas)}

    print(f"\n📥 Cargando procesador de imágenes ({MODEL_BASE})...")
    _extractor = AutoImageProcessor.from_pretrained(MODEL_BASE)

    # Augmentation para entrenamiento
    _aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.RandomGrayscale(p=0.05),
    ])

    # Indexar rutas
    print("🖼️  Indexando imágenes...")
    for label in etiquetas:
        carpeta = Path(DATASET_DIR) / label
        for img_path in sorted(carpeta.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            _rutas.append(str(img_path))
            _labels.append(label2id[label])

    if not _rutas:
        print("❌ No se encontraron imágenes.")
        return

    print(f"✅ {len(_rutas)} imágenes indexadas.")

    # División estratificada 80-20
    por_clase = defaultdict(list)
    for idx, lbl in enumerate(_labels):
        por_clase[lbl].append(idx)

    train_idx, val_idx = [], []
    for lbl_indices in por_clase.values():
        random.shuffle(lbl_indices)
        split = max(1, int(len(lbl_indices) * 0.8))
        train_idx.extend(lbl_indices[:split])
        val_idx.extend(lbl_indices[split:])

    print(f"   Train: {len(train_idx)} | Val: {len(val_idx)}")

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

        print("\n  Accuracy por clase:")
        for cls_id, cls_name in sorted(id2label.items()):
            mask = lbls == cls_id
            if mask.sum() > 0:
                a = float((preds[mask] == lbls[mask]).mean())
                barra = "█" * int(a * 20)
                print(f"    {cls_name:<25} {barra:<20} {a*100:.0f}%")

        return {"accuracy": float(acc)}

    # num_workers=4 en GPU (multiprocessing acelerado), 0 en CPU para evitar overhead
    num_workers = 4 if use_gpu else 0

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
        report_to="none",
        dataloader_num_workers=num_workers,
        warmup_steps=50,
        fp16=use_gpu,
        dataloader_pin_memory=use_gpu,
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
    _extractor.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(etiquetas))

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ¡Modelo guardado en '{OUTPUT_DIR}'!")
    print("   Reinicia el servidor FastAPI para que el identificador lo use automáticamente.")


if __name__ == "__main__":
    entrenar()
