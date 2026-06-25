"""
Normaliza las imágenes de portadas para visualización consistente:
- Elimina bordes blancos excesivos (trim automático)
- Centra en un canvas cuadrado de 300x300 con fondo blanco suave
- Guarda las versiones procesadas sobreescribiendo las originales

Uso: python preprocess_images.py
"""
import os
from PIL import Image, ImageOps
import numpy as np

INPUT_DIR = "app/static/img/portadas"
SIZE = (300, 300)
BG_COLOR = (255, 255, 255)
TRIM_THRESHOLD = 240   # píxeles más claros que esto se consideran "borde blanco"
PADDING = 12           # píxeles de margen interior tras el trim


def trim_whitespace(img: Image.Image) -> Image.Image:
    """Recorta bordes blancos/muy claros."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    data = np.array(img)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Máscara: píxeles que NO son fondo (no blancos y no transparentes)
    not_bg = ~((r > TRIM_THRESHOLD) & (g > TRIM_THRESHOLD) & (b > TRIM_THRESHOLD)) & (a > 10)

    rows = np.any(not_bg, axis=1)
    cols = np.any(not_bg, axis=0)

    if not rows.any() or not cols.any():
        return img   # imagen completamente blanca, devolver tal cual

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Pequeño margen para no cortar demasiado pegado
    margin = 4
    rmin = max(0, rmin - margin)
    rmax = min(data.shape[0] - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(data.shape[1] - 1, cmax + margin)

    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def fit_in_square(img: Image.Image, size: tuple, padding: int, bg: tuple) -> Image.Image:
    """Escala con aspect ratio y centra en canvas cuadrado."""
    img = img.convert("RGBA")
    usable = (size[0] - padding * 2, size[1] - padding * 2)
    img.thumbnail(usable, Image.LANCZOS)

    canvas = Image.new("RGBA", size, (*bg, 255))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    canvas.paste(img, offset, mask=img)
    return canvas.convert("RGB")


def procesar():
    archivos = [f for f in os.listdir(INPUT_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith(".")]

    print(f"Procesando {len(archivos)} imágenes...\n")

    for nombre in sorted(archivos):
        ruta = os.path.join(INPUT_DIR, nombre)
        try:
            img = Image.open(ruta)
            original_size = img.size

            # 1. Trim
            img = trim_whitespace(img)
            # 2. Ajustar a cuadrado normalizado
            img = fit_in_square(img, SIZE, PADDING, BG_COLOR)

            # Guardar como PNG
            base = os.path.splitext(nombre)[0]
            out_path = os.path.join(INPUT_DIR, base + ".png")
            img.save(out_path, "PNG", optimize=True)

            # Si el nombre original era .jpg/.jpeg, eliminar el archivo antiguo
            if not nombre.endswith(".png"):
                os.remove(ruta)

            print(f"  ✅ {nombre}: {original_size[0]}x{original_size[1]} → 300x300")
        except Exception as e:
            print(f"  ❌ {nombre}: {e}")

    print("\n¡Listo! Todas las imágenes normalizadas a 300x300.")


if __name__ == "__main__":
    procesar()
