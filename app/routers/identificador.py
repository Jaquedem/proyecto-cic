from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import base64
import numpy as np

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

CSV_PATH   = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "coleccion_juegos.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "modelo_cv")

LABEL_TO_NOMBRE = {
    "deep_dreams":       "Deep Dreams",
    "deep_sea":          "Deep Sea",
    "exploding_kittens": "Exploding kittens",
    "fantasma_blitz":    "Fantasma blitz",
    "reyes_dela_rosca":  "Reyes de la Rosca",
    "spotit_clasico":    "Spot it! Classic",
    "taco_gato_cabra":   "Taco Gato Cabra Queso Pizza",
    "unstable_unicorns": "Unstable unicorns",
    "virus":             "Virus",
}

_extractor = None
_modelo    = None
_id2label  = None


def cargar_modelo():
    global _extractor, _modelo, _id2label
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        import torch
        import json
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _extractor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        _modelo = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
        _modelo.eval()
        if torch.cuda.is_available():
            _modelo.cuda()

        map_path = os.path.join(MODEL_PATH, "label_map.json")
        with open(map_path, encoding="utf-8") as f:
            maps = json.load(f)
        _id2label = {int(k): v for k, v in maps["id2label"].items()}
        return True
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return False


def predecir_imagen(img):
    """Inferencia con TTA: original + flip horizontal + 2 crops → promedia probabilidades."""
    import torch
    import torchvision.transforms.functional as TF
    from torchvision import transforms

    img_size = _extractor.size.get("height", 380)
    to_tensor  = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    normalize  = transforms.Normalize(mean=_extractor.image_mean, std=_extractor.image_std)

    variantes = [
        img,                          # original
        TF.hflip(img),                # flip horizontal
        TF.center_crop(img, min(img.size)),  # crop cuadrado central
    ]

    device = next(_modelo.parameters()).device
    logits_acum = None

    for v in variantes:
        pv = normalize(to_tensor(v)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = _modelo(pixel_values=pv).logits
        probs = torch.softmax(logits, dim=-1)
        logits_acum = probs if logits_acum is None else logits_acum + probs

    probs_mean = (logits_acum / len(variantes)).squeeze(0).cpu().numpy()
    pred_id    = int(np.argmax(probs_mean))
    confianza  = float(probs_mean[pred_id])
    nombre     = _id2label[pred_id]
    return nombre, round(confianza * 100, 1)


def get_info_juego(nombre: str) -> dict | None:
    df = pd.read_csv(CSV_PATH)
    nombre_csv = LABEL_TO_NOMBRE.get(nombre, nombre.replace("_", " "))
    fila = df[df["Nombre del juego"].str.lower() == nombre_csv.lower()]
    if fila.empty:
        return None
    row = fila.iloc[0]
    return {
        "Nombre": row["Nombre del juego"],
        "Complejidad": row["Nivel de complejidad"],
        "Premisa": row["Premisa del juego"],
        "Min": int(row["min jugadores"]),
        "Max": int(row["max jugadores"]),
        "Tiempo": int(row["Tiempo de juego (min)"]),
        "Foto": row.get("foto", ""),
    }


@router.get("/", response_class=HTMLResponse)
async def identificador_page(request: Request):
    modelo_listo = os.path.exists(MODEL_PATH) and any(os.scandir(MODEL_PATH))
    return templates.TemplateResponse(request, "identificador.html", {
        "modelo_listo": modelo_listo,
    })


@router.post("/predecir", response_class=JSONResponse)
async def predecir(request: Request):
    if _modelo is None:
        modelo_ok = cargar_modelo()
        if not modelo_ok:
            return {"estado": "sin_modelo", "mensaje": "Modelo no entrenado todavía."}

    try:
        from PIL import Image
        import io

        body = await request.json()
        img_data = body.get("imagen", "")
        if img_data.startswith("data:image"):
            img_data = img_data.split(",", 1)[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        nombre, confianza = predecir_imagen(img)
        info = get_info_juego(nombre)

        return {
            "estado": "ok",
            "nombre": nombre,
            "confianza": confianza,
            "info": info,
        }
    except Exception as e:
        return {"estado": "error", "mensaje": str(e)}
