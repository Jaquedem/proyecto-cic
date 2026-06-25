from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os

from app.routers import recomendador, ruleta, identificador

app = FastAPI(title="Bombunny - Ludoteca")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(recomendador.router, prefix="/recomendador")
app.include_router(ruleta.router, prefix="/ruleta")
app.include_router(identificador.router, prefix="/identificador")

CSV_PATH = "coleccion_juegos.csv"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    total = 0
    if os.path.exists(CSV_PATH):
        total = len(pd.read_csv(CSV_PATH))
    return templates.TemplateResponse("index.html", {"request": request, "total_juegos": total})
