from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from buscador import BuscadorLudoteca

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

_buscador: BuscadorLudoteca | None = None

def get_buscador() -> BuscadorLudoteca:
    global _buscador
    if _buscador is None:
        _buscador = BuscadorLudoteca()
    return _buscador


@router.get("/", response_class=HTMLResponse)
async def recomendador_page(request: Request):
    return templates.TemplateResponse(request, "recomendador.html", {
        "resultados": [],
        "consulta": "",
        "buscando": False,
    })


@router.post("/buscar", response_class=HTMLResponse)
async def buscar(
    request: Request,
    consulta: str = Form(...),
    num_jugadores: int = Form(None),
):
    buscador = get_buscador()
    resultados = buscador.buscar_juegos(
        consulta,
        num_jugadores=num_jugadores if num_jugadores and num_jugadores > 0 else None,
        top_k=4,
    )
    return templates.TemplateResponse(request, "recomendador.html", {
        "resultados": resultados,
        "consulta": consulta,
        "num_jugadores": num_jugadores,
        "buscando": True,
    })
