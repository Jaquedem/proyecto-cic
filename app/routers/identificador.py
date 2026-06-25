from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import base64
import numpy as np

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "coleccion_juegos.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "modelo_cv")

_modelo = None
_etiquetas = None


def cargar_modelo():
    """Carga el modelo CV entrenado si existe."""
    global _modelo, _etiquetas
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        from transformers import pipeline
        _modelo = pipeline("image-classification", model=MODEL_PATH)
        labels_path = os.path.join(MODEL_PATH, "labels.txt")
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                _etiquetas = [l.strip() for l in f.readlines()]
        return True
    except Exception:
        return False


def get_info_juego(nombre: str) -> dict | None:
    df = pd.read_csv(CSV_PATH)
    fila = df[df["Nombre del juego"].str.lower() == nombre.lower()]
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
        "Youtube": row.get("youtube", ""),
    }


@router.get("/", response_class=HTMLResponse)
async def identificador_page(request: Request):
    modelo_listo = os.path.exists(MODEL_PATH) and any(os.scandir(MODEL_PATH))
    return templates.TemplateResponse("identificador.html", {
        "request": request,
        "modelo_listo": modelo_listo,
    })


@router.post("/predecir", response_class=JSONResponse)
async def predecir(request: Request):
    """Recibe un frame en base64, ejecuta inferencia y devuelve la predicción."""
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

        resultados = _modelo(img)
        top = resultados[0]
        nombre = top["label"]
        confianza = round(top["score"] * 100, 1)

        info = get_info_juego(nombre)

        return {
            "estado": "ok",
            "nombre": nombre,
            "confianza": confianza,
            "info": info,
        }
    except Exception as e:
        return {"estado": "error", "mensaje": str(e)}
