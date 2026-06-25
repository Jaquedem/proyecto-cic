"""
Script de captura de fotos para el dataset.

Abre la cámara y guarda frames en dataset/<juego>/ con nombre automático.

Uso:
    python capturar_dataset.py

Controles:
    ESPACIO  → capturar foto
    N        → siguiente juego
    P        → juego anterior
    Q / ESC  → salir
"""

import cv2
import os
from pathlib import Path
from datetime import datetime

DATASET_DIR = "./dataset"
CLASES = sorted([
    d.name for d in Path(DATASET_DIR).iterdir() if d.is_dir()
]) if Path(DATASET_DIR).exists() else []


def contar_fotos(clase: str) -> int:
    carpeta = Path(DATASET_DIR) / clase
    return len(list(carpeta.glob("*.jpg")) + list(carpeta.glob("*.png")))


def capturar():
    if not CLASES:
        print(f"❌ No se encontraron carpetas en '{DATASET_DIR}'.")
        print("   Crea al menos una subcarpeta con el nombre del juego.")
        return

    idx_clase = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n📸 Captura de dataset iniciada")
    print("   ESPACIO = capturar | N = siguiente clase | P = anterior | Q/ESC = salir\n")

    flash_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo frame.")
            break

        clase_actual = CLASES[idx_clase]
        n_fotos = contar_fotos(clase_actual)

        display = frame.copy()

        # Flash blanco al capturar
        if flash_frames > 0:
            overlay = display.copy()
            overlay[:] = (255, 255, 255)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            flash_frames -= 1

        # Overlay de información
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(display, f"Clase: {clase_actual}  [{idx_clase+1}/{len(CLASES)}]",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Fotos guardadas: {n_fotos}",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        cv2.putText(display, "ESPACIO=capturar  N=siguiente  P=anterior  Q=salir",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Guía de encuadre
        cx, cy = w // 2, h // 2
        size = min(w, h) * 2 // 3
        cv2.rectangle(display,
                      (cx - size // 2, cy - size // 2),
                      (cx + size // 2, cy + size // 2),
                      (0, 230, 255), 2)

        cv2.imshow("Captura de Dataset — Bombunny", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            # Guardar foto
            carpeta = Path(DATASET_DIR) / clase_actual
            carpeta.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
            ruta = carpeta / f"{clase_actual}_{ts}.jpg"
            cv2.imwrite(str(ruta), frame)
            flash_frames = 6
            print(f"  ✅ [{clase_actual}] {ruta.name}  (total: {n_fotos + 1})")

        elif key == ord("n") or key == ord("N"):
            idx_clase = (idx_clase + 1) % len(CLASES)
            print(f"  ➡️  Clase: {CLASES[idx_clase]}")

        elif key == ord("p") or key == ord("P"):
            idx_clase = (idx_clase - 1) % len(CLASES)
            print(f"  ⬅️  Clase: {CLASES[idx_clase]}")

        elif key in (ord("q"), ord("Q"), 27):  # Q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n📊 Resumen final:")
    for clase in CLASES:
        n = contar_fotos(clase)
        barra = "█" * (n // 5)
        estado = "✅" if n >= 80 else "⚠️ " if n >= 40 else "❌"
        print(f"  {estado} {clase:<28} {n:>4} fotos  {barra}")
    print()


if __name__ == "__main__":
    capturar()
