import os
from sentence_transformers import SentenceTransformer

def descargar():
    MODEL_NAME = 'intfloat/multilingual-e5-base'
    LOCAL_MODEL_PATH = './modelo_local_e5'
    
    print(f"📥 Iniciando la descarga del modelo '{MODEL_NAME}' desde Hugging Face...")
    
    try:
        # Descarga el modelo a la memoria ram
        modelo = SentenceTransformer(MODEL_NAME)
        
        # Lo guarda en la carpeta local de tu proyecto
        print(f"Guardando en: {LOCAL_MODEL_PATH}...")
        modelo.save(LOCAL_MODEL_PATH)
        
        print("Descarga completa:)")

        
    except Exception as e:
        print(f"Error durante la descarga: {e}")

if __name__ == "__main__":
    descargar()