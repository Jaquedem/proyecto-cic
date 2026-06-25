import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

class BuscadorLudoteca:
    def __init__(self, csv_path="coleccion_juegos.csv"):
        # 1. Cargamos los datos de la colección
        self.df = pd.read_csv(csv_path)
        
        # 2. Cargamos el modelo desde la carpeta local (Modo Offline)
        self.model = SentenceTransformer('./modelo_local_e5')
        
        # 3. Preparamos los textos para el modelo basados en la premisa
        self.textos_indexar = [f"passage: {premisa}" for premisa in self.df["Premisa del juego"]]
        
        # 4. Generamos los embeddings de todos los juegos
        self.embeddings_juegos = self.model.encode(self.textos_indexar, convert_to_tensor=True)

    def buscar_juegos(self, consulta_usuario, num_jugadores=None, top_k=3):
        # 1. Aplicamos filtro duro por número de jugadores si se especifica
        df_filtrado = self.df
        indices_validos = list(range(len(self.df)))
        
        if num_jugadores is not None:
            # Filtramos donde el número de jugadores esté en el rango permitido por el juego
            filtro = (self.df['min jugadores'] <= num_jugadores) & (self.df['max jugadores'] >= num_jugadores)
            df_filtrado = self.df[filtro]
            indices_validos = df_filtrado.index.tolist()
            
        if df_filtrado.empty:
            return []

        # 2. Convertimos la duda del usuario en un vector semántico
        consulta_formateada = f"query: {consulta_usuario}"
        embedding_consulta = self.model.encode(consulta_formateada, convert_to_tensor=True)
        
        # 3. Filtramos los vectores matemáticos para evaluar SOLO los juegos válidos
        embeddings_filtrados = self.embeddings_juegos[indices_validos]
        similitudes = util.cos_sim(embedding_consulta, embeddings_filtrados)[0]
        
        # 4. Obtenemos los mejores resultados del subconjunto que sí cumple la regla
        k_real = min(top_k, len(df_filtrado))
        top_resultados = torch.topk(similitudes, k=k_real)
        
        resultados = []
        for score, idx_filtrado in zip(top_resultados.values, top_resultados.indices):
            # Mapeamos el índice del filtro de vuelta al DataFrame original
            idx_original = indices_validos[idx_filtrado.item()]
            fila = self.df.iloc[idx_original]
            resultados.append({
                "Nombre": fila["Nombre del juego"],
                "Complejidad": fila["Nivel de complejidad"],
                "Premisa": fila["Premisa del juego"],
                "Min": fila["min jugadores"],
                "Max": fila["max jugadores"],
                "Similitud": f"{score.item():.2f}"
            })
            
        return resultados

# --- PRUEBA LOCAL EN TERMINAL ---
if __name__ == "__main__":
    buscador = BuscadorLudoteca()
    
    # Especificamos explícitamente que somos 4 personas
    personas = 2
    consulta = "Quiero un juego de larga duración para 2 jugadores que sea competitivo y que te haga analizar tus movimientos"
    
    print(f"\n🔍 Pregunta del usuario: '{consulta}' (Buscando para {personas} jugadores...)")
    
    # Le pasamos el parámetro 'num_jugadores=personas' a la función
    sugerencias = buscador.buscar_juegos(consulta, num_jugadores=personas, top_k=2)
    
    print("\n🐰 Bombunny sugiere:")
    if not sugerencias:
        print("❌ No encontré juegos en la ludoteca que admitan esa cantidad de jugadores.")
    else:
        for i, juego in enumerate(sugerencias, 1):
            print(f"\n{i}. {juego['Nombre']} (Complejidad: {juego['Complejidad']} | Rango: {juego['Min']}-{juego['Max']} jugadores)")
            print(f"   Por qué coincide: {juego['Premisa']}")
            print(f"   Score de coincidencia: {juego['Similitud']}")