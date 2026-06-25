from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "coleccion_juegos.csv")


def cargar_juegos():
    df = pd.read_csv(CSV_PATH)
    return df


@router.get("/", response_class=HTMLResponse)
async def ruleta_page(request: Request):
    df = cargar_juegos()
    complejidades = sorted(df["Nivel de complejidad"].unique().tolist())
    max_tiempo = int(df["Tiempo de juego (min)"].max())
    max_jugadores = int(df["max jugadores"].max())
    return templates.TemplateResponse(request, "ruleta.html", {
        "complejidades": complejidades,
        "max_tiempo": max_tiempo,
        "max_jugadores": max_jugadores,
        "total": len(df),
    })


@router.get("/filtrar", response_class=JSONResponse)
async def filtrar_juegos(
    jugadores: int = None,
    tiempo_max: int = None,
    complejidad: str = None,
):
    df = cargar_juegos()

    if jugadores and jugadores > 0:
        df = df[(df["min jugadores"] <= jugadores) & (df["max jugadores"] >= jugadores)]

    if tiempo_max and tiempo_max > 0:
        df = df[df["Tiempo de juego (min)"] <= tiempo_max]

    if complejidad and complejidad != "Todas":
        df = df[df["Nivel de complejidad"] == complejidad]

    juegos = df[["Nombre del juego", "Nivel de complejidad", "Tiempo de juego (min)",
                 "min jugadores", "max jugadores", "Premisa del juego", "foto"]].to_dict(orient="records")
    return {"juegos": juegos, "total": len(juegos)}
