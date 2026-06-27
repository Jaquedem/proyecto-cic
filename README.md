# Bombunny 🐰💣 — Ludoteca Inteligente

Aplicación web para gestionar y explorar una colección de juegos de mesa, con tres módulos principales: recomendador semántico, ruleta aleatoria e identificador por cámara.

Construida con **FastAPI + Jinja2** en el backend y HTML/CSS/JS vanilla en el frontend.

---

## Estructura del proyecto

```
proyecto-cic/
│
├── app/                          # Aplicación FastAPI
│   ├── main.py                   # Punto de entrada: crea la app, monta rutas y archivos estáticos
│   ├── routers/
│   │   ├── recomendador.py       # Módulo de búsqueda semántica (POST /recomendador/buscar)
│   │   ├── ruleta.py             # Módulo de ruleta aleatoria (GET /ruleta/filtrar)
│   │   └── identificador.py     # Módulo de identificación por cámara (POST /identificador/predecir)
│   ├── templates/
│   │   ├── base.html             # Layout base: navbar + estructura HTML común
│   │   ├── index.html            # Pantalla de inicio con las tres opciones
│   │   ├── recomendador.html     # UI del buscador semántico
│   │   ├── ruleta.html           # UI de la ruleta con filtros y animación
│   │   └── identificador.html   # UI de la cámara con captura y predicción
│   └── static/
│       ├── css/styles.css        # Estilos globales (variables de color, navbar, botones)
│       └── img/                  # Imágenes estáticas (logo.png, mascota.png, portadas de juegos)
│
├── buscador.py                   # Clase BuscadorLudoteca: carga el modelo E5, genera embeddings y busca por similitud semántica
├── coleccion_juegos.csv          # Dataset principal: nombre, complejidad, jugadores, tiempo, premisa, foto
│
├── modelo_local_e5/              # Modelo multilingual-e5-base descargado localmente (modo offline)
├── modelo_cv/                    # Modelo EfficientNet-B4 entrenado para identificar juegos por foto
│
├── dataset/                      # Fotos de entrenamiento organizadas por carpetas (una por juego)
│   ├── exploding_kittens/
│   ├── virus/
│   └── ...
│
├── train_model.py                # Entrena EfficientNet-B4 con transfer learning sobre el dataset local
├── eval_model.py                 # Evaluación básica: accuracy por clase en terminal
├── eval_modelo_cv.py             # Evaluación completa del modelo CV: métricas sklearn + 4 gráficas exportadas
├── eval_buscador.py              # Evaluación del buscador semántico: Precision@K + 3 gráficas exportadas
├── capturar_dataset.py           # Script OpenCV para capturar fotos con la webcam y construir el dataset
│
├── model.py                      # Descarga multilingual-e5-base de HuggingFace y lo guarda en modelo_local_e5/
├── juegos.py                     # Diccionario fuente con los datos de los juegos (usado para generar el CSV)
├── preprocess_images.py          # Normaliza las portadas: recorta márgenes blancos y centra en canvas 300×300
│
├── run.py                        # Lanzador simple: uvicorn.run(app, host="0.0.0.0", port=8000)
└── requirements.txt              # Dependencias Python del proyecto
```

---

## Módulos

### 1. Recomendador semántico (`/recomendador`)

El usuario describe en lenguaje natural qué tipo de juego busca. El motor convierte la consulta y las descripciones de todos los juegos en vectores semánticos usando **multilingual-e5-base** y devuelve los más similares por coseno.

- **Filtros opcionales:** número exacto de jugadores y nivel de complejidad (Bajo / Medio / Alto). Ambos se aplican como filtros duros antes de la búsqueda semántica.
- **Formato de texto indexado:** `"passage: {nombre}. Complejidad: {nivel}. Jugadores: X a Y. Duración: Z minutos. {premisa}"`
- **Formato de consulta:** `"query: {texto del usuario}"`

Archivos clave: `buscador.py`, `app/routers/recomendador.py`, `modelo_local_e5/`

---

### 2. Ruleta aleatoria (`/ruleta`)

Filtra la colección por número de jugadores, rango de tiempo (min/max en minutos) y complejidad, y elige un juego al azar entre los que cumplen los criterios.

Archivos clave: `app/routers/ruleta.py`, `coleccion_juegos.csv`

---

### 3. Identificador por cámara (`/identificador`)

Accede a la webcam del usuario, captura un fotograma y lo envía al servidor como imagen JPEG. El modelo **EfficientNet-B4** fine-tuneado predice qué juego es y devuelve el nombre, la confianza y la premisa del juego desde el CSV.

- **TTA (Test-Time Augmentation):** promedia predicciones sobre 3 variantes (original, volteo horizontal, recorte central) para mayor robustez.
- La cámara se muestra espejada (`transform: scaleX(-1)`) para efecto de espejo natural.

Archivos clave: `app/routers/identificador.py`, `modelo_cv/`, `train_model.py`

---

## Instalación y ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar el modelo semántico (solo la primera vez, requiere internet)
python model.py

# 3. Arrancar el servidor
python -m uvicorn app.main:app --reload
```

La app queda disponible en `http://localhost:8000`.

---

## Entrenamiento y evaluación del modelo CV

```bash
# Capturar fotos con la webcam (50+ por clase recomendado)
python capturar_dataset.py

# Entrenar el modelo
python train_model.py

# Evaluación completa con métricas y gráficas
python eval_modelo_cv.py

# Evaluación del buscador semántico
python eval_buscador.py
```

El dataset debe tener la estructura `dataset/{nombre_clase}/foto_xxx.jpg`. El modelo se guarda en `modelo_cv/`.

Los scripts de evaluación exportan las siguientes imágenes:

| Archivo generado | Contenido |
|---|---|
| `eval_confusion_matrix.png` | Heatmap de la matriz de confusión del modelo CV |
| `eval_accuracy_por_clase.png` | Accuracy por clase + variación entre clases (np.diff) |
| `eval_metricas_por_clase.png` | Precisión, Sensibilidad y F1 agrupados por clase |
| `eval_distribucion_confianza.png` | Distribución de confianza: aciertos vs errores |
| `eval_buscador_scores.png` | Similitud coseno por consulta + caída de relevancia |
| `eval_buscador_precision.png` | Precision@K por tipo de consulta semántica |
| `eval_buscador_distribucion.png` | Distribución global de scores del modelo E5 |

---

## Personalización visual

| Elemento | Dónde cambiarlo |
|----------|----------------|
| Tamaño del logo en la navbar | `app/static/css/styles.css` → `.navbar-logo { width: 48px; height: 48px; }` |
| Tamaño de la mascota en el inicio | `app/templates/index.html` → `.hero-mascot-img { width: 170px; height: 170px; }` |
| Colores globales | `app/static/css/styles.css` → variables CSS en `:root` |
| Logo e imagen de mascota | `app/static/img/logo.png` y `app/static/img/mascota.png` |

---

## Dependencias principales

| Librería | Uso |
|----------|-----|
| `fastapi` | Framework web y servidor de rutas |
| `uvicorn` | Servidor ASGI |
| `jinja2` | Plantillas HTML |
| `pandas` | Lectura del CSV de juegos |
| `sentence-transformers` | Motor de búsqueda semántica (E5) |
| `transformers` | Modelo EfficientNet-B4 para CV |
| `torch` + `torchvision` | Inferencia y preprocesado de imágenes |
| `Pillow` | Manipulación de imágenes en el servidor |
| `opencv-python` | Captura de webcam en `capturar_dataset.py` |
