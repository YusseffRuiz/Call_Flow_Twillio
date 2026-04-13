import pandas as pd
import os
from dotenv import load_dotenv
from RAG_CORE.retrieval_module import RetrievalModule


# ==========================================
# CONFIGURACIÓN MULTI-FAISS
# ==========================================
DRIVE_URL = "https://docs.google.com/spreadsheets/d/1pAciTjKST3Yn0KMnmZM02-xXWwwdJShn/export?format=xlsx"
BASE_DB_DIR = "./vector_dbs"

load_dotenv("Documents/tokens.env")  # busca .env en el cwd
HF_TOKEN = os.getenv("HF_TOKEN")

# Diccionario de enrutamiento: { "Nombre_Hoja_Excel" : "nombre_carpeta_faiss" }
RAG_CONFIG = {
    "UNIDADES": "unidades",
    # "SERV FUNERARIOS": "funerarias", # <-- Listo para el futuro
    #  "SERVICIOS" "servicios",
}

# Configuración de tu modelo (Ajusta al que uses actualmente en RetrievalModule)
MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'  # Ejemplo, pon el tuyo
DEVICE = "cpu"  # Cambiar a 'cuda' si el servidor de actualización tiene GPU



def descargar_excel():
    try:
        xls = pd.ExcelFile(DRIVE_URL)
        print(" -> ✅ Excel descargado con éxito.")
        return xls
    except Exception as e:
        print(f"[CRITICO] Error al descargar Excel: {e}")
        return None

def actualizar_bases_de_conocimiento():
    print("[1/3] Descargando base de datos desde Google Drive...")
    xls = descargar_excel()
    if not xls: return

    if not os.path.exists(BASE_DB_DIR):
        os.makedirs(BASE_DB_DIR)

    for hoja, carpeta_destino in RAG_CONFIG.items():
        if hoja not in xls.sheet_names:
            print(f" ⚠️ Advertencia: La hoja '{hoja}' no se encontró. Saltando...")
            continue

        print(f"\n[2/3] Generando Colección: {carpeta_destino.upper()}")

        retrieval_module = RetrievalModule(database_path=xls, hf_token=HF_TOKEN, model_name=MODEL_NAME, origin_sheet=hoja)

        db_path = os.path.join(BASE_DB_DIR, carpeta_destino)

        retrieval_module.initialize(path_to_database=db_path ,save_db=True, score_threshold=0.34, percentile=0.9)
        print(f" -> ✅ DataFrame guardado en: {db_path}/kb_df.pkl")

        csv_path = os.path.join(db_path, f"auditoria_{carpeta_destino}.csv")
        retrieval_module.kb_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f" -> 👁️ Tabla de auditoría exportada en: {csv_path}")

        # Opción B (Alternativa): Guardarlo como Excel directo (.xlsx)
        # excel_path = os.path.join(db_path, f"auditoria_{carpeta_destino}.xlsx")
        # kb_df.to_excel(excel_path, index=False)
        # print(f" -> 👁️ Tabla de auditoría exportada en: {excel_path}")


    print("\n[3/3] 🚀 Actualización finalizada exitosamente.")


if __name__ == "__main__":
    actualizar_bases_de_conocimiento()