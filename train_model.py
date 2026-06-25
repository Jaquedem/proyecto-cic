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

# Variables globales — module-level para que GameDataset sea picklable en Windows
_tensors   = []   # tensores float32 en rango [0,1] — sin normalizar todavía
_labels    = []
_normalize = None  # torchvision.transforms.Normalize, se asigna en entrenar()


class GameDataset:
    def __init__(self, indices, augment=False):
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import torchvision.transforms.functional as TF
        idx = self.indices[i]
        pv = _tensors[idx].clone()  # [C, H, W] en [0, 1]

        if self.augment:
            if random.random() > 0.5:
                pv = TF.hflip(pv)
            angle = random.uniform(-20, 20)
            pv = TF.rotate(pv, angle)
            pv = TF.adjust_brightness(pv, 1 + random.uniform(-0.4, 0.4))
            pv = TF.adjust_contrast(pv,   1 + random.uniform(-0.4, 0.4))
            pv = TF.adjust_saturation(pv, 1 + random.uniform(-0.3, 0.3))
            pv = TF.adjust_hue(pv, random.uniform(-0.05, 0.05))
            pv = pv.clamp(0.0, 1.0)

        # Normalizar DESPUÉS del augmentation (no antes)
        if _normalize is not None:
            pv = _normalize(pv)

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
    global _tensors, _labels, _normalize

    try:
        import torch
        from PIL import Image
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
    extractor = AutoImageProcessor.from_pretrained(MODEL_BASE)

    # Preparar normalización y resize usando los parámetros del extractor
    img_size = extractor.size.get("height", 380)
    _normalize = transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
    to_tensor  = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # convierte a [C,H,W] float en [0,1]
    ])

    # Pre-procesar TODAS las imágenes a tensores [0,1] sin normalizar (≈450 MB RAM)
    # La normalización se aplica en __getitem__ DESPUÉS del augmentation
    print(f"🖼️  Pre-procesando imágenes a tensores {img_size}×{img_size}...")
    for label in etiquetas:
        carpeta = Path(DATASET_DIR) / label
        for img_path in sorted(carpeta.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                pv = to_tensor(img)  # [C,H,W] float en [0,1]
                _tensors.append(pv)
                _labels.append(label2id[label])
            except Exception as e:
                print(f"  ⚠️  Saltando {img_path.name}: {e}")

    if not _tensors:
        print("❌ No se encontraron imágenes.")
        return

    print(f"✅ {len(_tensors)} tensores en memoria.")

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
        use_safetensors=True,   # evita pytorch_model.bin → no activa CVE-2025-32434
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

    num_workers = 0  # Windows spawn no hereda globals del proceso principal

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
        bf16=use_gpu,   # bfloat16: mismo rendimiento que fp16 pero sin NaN
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
    extractor.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(etiquetas))

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ ¡Modelo guardado en '{OUTPUT_DIR}'!")
    print("   Reinicia el servidor FastAPI para que el identificador lo use automáticamente.")


if __name__ == "__main__":
    entrenar()
