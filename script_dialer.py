import os

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from dotenv import load_dotenv

# --- CONFIGURACIÓN ---
# La URL de tu App Principal (donde está el endpoint /twilio/start_call)
load_dotenv("Documents/keys.env")
API_BASE_URL = "https://tu-dominio-ngrok.appspot.com"
EXCEL_PATH = "Documents/lista_pacientes.xlsx"
MAX_CONCURRENT_CALLS = 1  # Ajustado a tu capacidad de 3-5 llamadas
API_KEY = os.getenv("API_KEY_INTERNA", "")


def disparar_llamada_individual(paciente: Dict):
    endpoint = f"{API_BASE_URL}/twilio/start_call"

    # Preparamos el encabezado de seguridad
    headers = {
        "API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "to": paciente["Teléfono"],
        "nombre": paciente["Nombre"]
    }

    #Imprimir las pruebas

    # try:
    #     # Enviamos la petición incluyendo los headers
    #     response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
    #
    #     if response.status_code == 200:
    #         print(f"✅ [OK] Llamada autorizada para: {paciente['Nombre']}")
    #     else:
    #         print(f"⚠️ [RECHAZADO] {paciente['Nombre']}: {response.status_code}")
    #
    # except Exception as e:
    #     print(f"❌ Error: {e}")


def ejecutar_campaña():
    # 1. Cargar y limpiar datos
    try:
        df = pd.read_excel(EXCEL_PATH)
        # Limpieza rápida de números
        df['Teléfono'] = df['Teléfono'].astype(str).str.replace(r'\s+', '', regex=True)
        pacientes = df.to_dict('records')
    except Exception as e:
        print(f"Error al leer el Excel: {e}")
        return
    print("Pacientes: ", pacientes)
    print("Telefonos: ", df['Teléfono'])
    print(f"🚀 Iniciando campaña para {len(pacientes)} pacientes...")
    print(f"Limitando a {MAX_CONCURRENT_CALLS} hilos simultáneos.")

    # 2. Orquestación con hilos
    # Usamos ThreadPoolExecutor para no saturar la red ni tu servidor
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
        executor.map(disparar_llamada_individual, pacientes)

    print("\n--- Campaña finalizada ---")


if __name__ == "__main__":
    ejecutar_campaña()
