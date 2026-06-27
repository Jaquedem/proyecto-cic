# Bombunny — Guía de Presentación

> Proyecto final del curso **Inteligencia Artificial con Python**
> Instituto Politécnico Nacional — Centro de Investigación en Cómputo
> Jaqueline Tinoco Vázquez

---

## ¿Qué es Bombunny?

Bombunny es una **ludoteca inteligente**: una aplicación web que ayuda a los usuarios de una colección de juegos de mesa a encontrar el juego perfecto para cada momento.

El problema que resuelve es real: cuando tienes muchos juegos y no sabes cuál sacar, tomar la decisión puede ser difícil. Bombunny lo resuelve desde tres ángulos distintos, cada uno usando una técnica de inteligencia artificial diferente.

---

## Los tres módulos del sistema

### 1. Recomendador semántico
El usuario escribe en lenguaje natural lo que busca ("quiero algo rápido y divertido para cuatro personas"). El sistema entiende el significado de esa frase y la compara contra las descripciones de todos los juegos de la colección, devolviendo los más relevantes aunque no compartan palabras exactas.

### 2. Ruleta aleatoria
Cuando el usuario no tiene preferencia y prefiere dejar que el azar decida, puede filtrar por número de jugadores, tiempo disponible y dificultad, y el sistema elige un juego al azar entre los que cumplen los criterios.

### 3. Identificador por cámara
El usuario apunta la cámara a la caja de un juego de mesa y el sistema reconoce automáticamente qué juego es, mostrando su nombre y descripción. Funciona en tiempo real desde el navegador.

---

## Herramientas utilizadas

### Lenguaje y entorno

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **Python** | Lenguaje de programación de propósito general | Base de todo el proyecto: backend, modelos, scripts de entrenamiento y evaluación |
| **pip / venv** | Gestor de paquetes y entornos virtuales | Instalar y aislar las dependencias del proyecto |

---

### Backend y servidor web

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **FastAPI** | Framework web moderno para Python, asíncrono | Define las rutas de la aplicación (`/recomendador`, `/ruleta`, `/identificador`) y sirve las respuestas |
| **Uvicorn** | Servidor ASGI (interfaz entre el sistema operativo y FastAPI) | Ejecuta la aplicación y la pone disponible en `localhost:8000` |
| **Jinja2** | Motor de plantillas HTML | Genera las páginas web dinámicas combinando HTML con datos del servidor |

---

### Datos y análisis

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **Pandas** | Librería de análisis de datos con estructuras tipo tabla (DataFrames) | Leer el CSV de la colección de juegos y aplicar filtros (por jugadores, tiempo, complejidad) |
| **NumPy** | Librería de computación numérica con arreglos eficientes | Operaciones vectoriales en los scripts de evaluación; `np.diff` para analizar variaciones entre clases |

---

### Visualización

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **Matplotlib** | Librería de visualización de datos 2D | Generar todas las gráficas de evaluación: barras por clase, histogramas, subplots |
| **Seaborn** | Librería de visualización estadística construida sobre Matplotlib | Generar el heatmap de la matriz de confusión con colores y anotaciones automáticas |

---

### Métricas de evaluación

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **scikit-learn** | Librería de machine learning clásico para Python | Calcular las métricas de evaluación del modelo: accuracy, precision, recall, F1 y la matriz de confusión |

---

### Inteligencia Artificial — Módulo de recomendación

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **sentence-transformers** | Librería para generar representaciones vectoriales de texto | Cargar el modelo E5 y generar embeddings de las descripciones de los juegos y de las consultas del usuario |
| **multilingual-e5-base** | Modelo de lenguaje preentrenado por Microsoft (familia E5) | Convierte texto en vectores de 768 dimensiones que capturan el significado semántico; permite comparar frases sin necesidad de palabras exactas en común |
| **Similitud coseno** | Métrica matemática entre vectores | Mide qué tan "cerca" están dos vectores semánticos; cuanto más cercano a 1.0, más similares son los significados |

---

### Inteligencia Artificial — Módulo de visión por computadora

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **HuggingFace Transformers** | Librería que centraliza modelos de IA preentrenados | Cargar EfficientNet-B4 y su procesador de imágenes; gestionar el guardado y carga del modelo entrenado |
| **EfficientNet-B4** | Red neuronal convolucional de Google, diseñada para ser eficiente en recursos | Base del modelo de identificación; fue re-entrenada (fine-tuning) con fotos de los juegos de la colección |
| **Transfer Learning** | Técnica de IA donde se reutiliza un modelo ya entrenado como punto de partida | EfficientNet-B4 ya sabe reconocer objetos en general; solo se re-entrenó la última capa para reconocer los 9 juegos específicos de la colección |
| **PyTorch** | Framework de deep learning de Meta | Realiza la inferencia (predicción) en el servidor cuando llega una foto de la cámara |
| **torchvision** | Extensión de PyTorch para procesado de imágenes | Redimensiona, convierte a tensor y normaliza las imágenes antes de pasarlas al modelo |
| **TTA (Test-Time Augmentation)** | Técnica para mejorar predicciones promediando variantes de la misma imagen | El modelo predice sobre la foto original, la misma voltada horizontalmente y un recorte central; se promedian los tres resultados para mayor robustez |

---

### Captura de datos

| Herramienta | Qué es | Para qué se usa en el proyecto |
|---|---|---|
| **OpenCV** | Librería de visión por computadora en tiempo real | En `capturar_dataset.py`: accede a la webcam, muestra una guía visual y guarda las fotos del dataset de entrenamiento |
| **Pillow (PIL)** | Librería de manipulación de imágenes para Python | Abre y convierte imágenes en el servidor antes de pasarlas al modelo de CV |

---

## Las gráficas de evaluación

### Modelo de visión (EfficientNet-B4)
Generadas ejecutando `python eval_modelo_cv.py`

---

#### `eval_confusion_matrix.png` — Matriz de Confusión
**Qué muestra:** Una tabla de calor donde las filas son las clases reales y las columnas son las predicciones del modelo. Los números en la diagonal principal son los aciertos; cualquier número fuera de la diagonal es un error.

**Cómo leerla:** Si la celda (Virus, Exploding Kittens) tiene un valor alto, significa que el modelo confunde Virus con Exploding Kittens frecuentemente. Una matriz ideal tiene todos los valores concentrados en la diagonal.

**De dónde viene la técnica:** Ejercicio de regresión logística del curso (predicción de aprobación de estudiantes).

---

#### `eval_accuracy_por_clase.png` — Accuracy por clase + variación
**Qué muestra:** Dos subgráficas:
- **Arriba:** barras de porcentaje de aciertos por cada juego. Verde = ≥80% (listo para producción), naranja = ≥50% (aceptable), rojo = <50% (requiere más datos).
- **Abajo:** `np.diff` de esas barras — la diferencia de accuracy entre clases consecutivas. Indica qué tan dispareja es la calidad del modelo entre clases.

**De dónde viene la técnica:** Tarea 1 del curso (diferencias de edades con `np.diff` + subplots con `height_ratios`).

---

#### `eval_metricas_por_clase.png` — Precisión, Sensibilidad y F1 por clase
**Qué muestra:** Tres barras agrupadas por cada clase del modelo:
- **Precisión:** de todas las veces que el modelo dijo "este juego es X", ¿qué porcentaje realmente era X?
- **Sensibilidad (Recall):** de todas las fotos reales de X, ¿qué porcentaje identificó correctamente?
- **F1:** media armónica entre precisión y sensibilidad. Penaliza cuando una es muy alta y la otra muy baja.

**De dónde viene la técnica:** Ejercicio de regresión logística del curso (métricas de clasificación con sklearn).

---

#### `eval_distribucion_confianza.png` — Distribución de confianza
**Qué muestra:** Dos subgráficas:
- **Arriba:** histograma de la probabilidad con la que el modelo hizo cada predicción, separando aciertos (verde) y errores (rojo). Un modelo bueno tiene los aciertos con alta confianza y los errores con baja confianza.
- **Abajo:** `np.diff` sobre las confianzas ordenadas — muestra en qué rango de probabilidad ocurren los saltos más grandes.

**De dónde viene la técnica:** Tarea 1 del curso (histogramas de frecuencia + `np.diff`).

---

### Modelo semántico (multilingual-e5-base)
Generadas ejecutando `python eval_buscador.py`

---

#### `eval_buscador_scores.png` — Scores de similitud por consulta
**Qué muestra:** Para cada una de las 5 consultas de prueba, un par de subgráficas:
- **Izquierda:** barras de similitud coseno de los top resultados. Verde = el resultado tiene la complejidad esperada para esa consulta; rojo = no la tiene.
- **Derecha:** `np.diff` de los scores — muestra qué tan rápido cae la relevancia entre el primer resultado y los siguientes (cuanto más plano, más "seguros" son todos los resultados).

**De dónde viene la técnica:** Tarea 1 del curso.

---

#### `eval_buscador_precision.png` — Precision@K por tipo de consulta
**Qué muestra:** Para cada tipo de consulta, qué porcentaje de los resultados devueltos tienen la complejidad que esperaríamos. Por ejemplo, si busco "algo complejo y estratégico" y los 5 resultados son todos de complejidad Baja, la Precision@5 es 0%.

**Por qué importa:** Mide si el modelo semántico entiende correctamente la intención del usuario más allá de las palabras exactas.

**De dónde viene la técnica:** Ejercicio de regresión logística del curso (métricas de clasificación).

---

#### `eval_buscador_distribucion.png` — Distribución global de scores
**Qué muestra:** Dos subgráficas:
- **Arriba:** histograma de todos los scores de similitud generados en las consultas de prueba, con líneas verticales marcando la media y la mediana. Indica si el modelo tiende a ser "seguro" (scores altos) o "inseguro" (scores bajos o dispersos).
- **Abajo:** `np.diff` sobre los scores ordenados — muestra en qué zona de similitud se concentran los saltos más abruptos de relevancia.

**De dónde viene la técnica:** Tarea 1 del curso.

---

## Flujo técnico completo del sistema

```
Usuario escribe consulta
        │
        ▼
  [multilingual-e5-base]
  Convierte texto → vector de 768 dimensiones
        │
        ▼
  Similitud coseno vs todos los juegos del CSV
        │
        ▼
  Top-K resultados ordenados por relevancia
        │
        ▼
  (Filtros duros opcionales: jugadores, complejidad)
        │
        ▼
  Resultado mostrado en pantalla


Usuario apunta cámara a un juego
        │
        ▼
  Captura de fotograma (JavaScript → base64)
        │
        ▼
  [EfficientNet-B4 fine-tuned]
  Extrae características visuales → vector → clasificación
        │
        ▼
  TTA: promedio de 3 predicciones
        │
        ▼
  Nombre del juego + confianza + descripción del CSV
```

---

## Métricas clave del proyecto

> Esta sección es para que completes con los valores reales que obtuviste al correr los scripts de evaluación.

### Modelo de visión (EfficientNet-B4)

| Métrica | Valor |
|---|---|
| Accuracy global | _(por completar)_ |
| Precisión macro | _(por completar)_ |
| Sensibilidad macro | _(por completar)_ |
| F1 macro | _(por completar)_ |
| Clase con mejor accuracy | _(por completar)_ |
| Clase con peor accuracy | _(por completar)_ |

### Modelo semántico (multilingual-e5-base)

| Consulta de prueba | Precision@5 |
|---|---|
| Familiar / rápido | _(por completar)_ |
| Estrategia compleja | _(por completar)_ |
| Deducción / misterio | _(por completar)_ |
| Calentamiento rápido | _(por completar)_ |
| Competitivo 1v1 | _(por completar)_ |

---

## Discusión de resultados

> _(Por completar por Jaqueline)_

---

## Conclusiones

> _(Por completar por Jaqueline)_
