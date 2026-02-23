import os
import sys

import streamlit as st
import pandas as pd
import json
import glob
import plotly.express as px
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from RAG_CORE.rag_utils.mappings import SYN, MX_STATES


# Configuración profesional de la página
st.set_page_config(page_title="CORA Business Intelligence", layout="wide", page_icon="📈")

st.title("📊 CORA: Reporte Ejecutivo de Operaciones")
st.markdown("### Análisis de métricas para Medical Life (Finanzas & Ventas)")


# Función robusta de carga de datos
def load_cora_logs():
    files = glob.glob("logs/*.json")
    all_data = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    all_data.append(json.loads(line))
                except:
                    continue
    return pd.json_normalize(all_data)


df = load_cora_logs()

# --- SECCIÓN 1: MÉTRICAS FINANCIERAS (KPIs) ---
st.divider()
st.header("💰 Indicadores Financieros")
m1, m2, m3, m4 = st.columns(4)

with m1:
    total_cost = df['stats.cost'].sum()
    st.metric("Inversión en Tokens (Total)", f"${total_cost:.4f} USD", delta_color="inverse")
with m2:
    avg_call_cost = df['stats.cost'].mean()
    st.metric("Costo Promedio / Turno", f"${avg_call_cost:.5f} USD")
with m3:
    total_turns = len(df)
    st.metric("Volumen de Interacciones", f"{total_turns} turnos")
with m4:
    # ROI Proyectado: Comparando vs costo de agente humano (aprox $0.15 USD por turno) --- Ejemplo
    human_cost_est = total_turns * 0.15
    savings = human_cost_est - total_cost
    st.metric("Ahorro Estimado vs Humano", f"${savings:.2f} USD", delta="Eficiencia")

# --- SECCIÓN 2: DESEMPEÑO TÉCNICO Y EXPERIENCIA DE USUARIO ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⏱️ Análisis de Latencias (Promedio)")
    # Muestra cuánto tiempo real espera el usuario
    lat_data = {
        'Etapa': ['ASR (Voz)', 'LLM (Pensar)', 'E2E (Total)'],
        'Segundos': [df['latencies.asr'].mean(), df['latencies.llm_first_token'].mean(), df['latencies.e2e'].mean()]
    }
    fig_lat = px.bar(lat_data, x='Etapa', y='Segundos', color='Etapa', text_auto='.2s')
    st.plotly_chart(fig_lat, use_container_width=True)

with col_right:
    st.subheader("🎯 Calidad de Entendimiento (NER/RAG)")
    # Identificamos respuestas fallidas o fuera de dominio
    oos_count = df[df['texts.bot'].str.contains("out_of_scope")].shape[0]
    success_count = total_turns - oos_count

    fig_pie = px.pie(
        names=['Éxito (RAG)', 'Fuera de Alcance (OOS)'],
        values=[success_count, oos_count],
        color_discrete_sequence=['#2ecc71', '#e74c3c'],
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- SECCIÓN 3: INTELIGENCIA DE VENTAS ---
st.divider()
st.header("🗺️ Inteligencia de Mercado: Análisis de Demanda")


CIUDADES_PROYECTO = list(MX_STATES.keys())
SERVICIOS_PROYECTO = list(set(SYN.values()))

# Función para extraer menciones dinámicamente
def extraer_menciones(text, lista_keywords):
    if not isinstance(text, str): return "Otros"
    for word in lista_keywords:
        if word.lower() in text.lower():
            return word
    return "No especificado"


# Creamos columnas de análisis en el DataFrame
df['Ciudad_Detectada'] = df['texts.user'].apply(lambda x: extraer_menciones(x, CIUDADES_PROYECTO))
df['Servicio_Detectado'] = df['texts.user'].apply(lambda x: extraer_menciones(x, SERVICIOS_PROYECTO))
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.subheader("📍 Sedes con mayor interés")
    ciudades_count = df[df['Ciudad_Detectada'] != "No especificado"]['Ciudad_Detectada'].value_counts().reset_index()
    ciudades_count.columns = ['Ciudad', 'Interacciones']

    fig_ciudades = px.pie(ciudades_count, values='Interacciones', names='Ciudad',
                          hole=0.4, title="Distribución Geográfica de Consultas",
                          color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_ciudades, use_container_width=True)

with col_v2:
    st.subheader("🩺 Servicios más solicitados")
    servicios_count = df[df['Servicio_Detectado'] != "No especificado"][
        'Servicio_Detectado'].value_counts().reset_index()
    servicios_count.columns = ['Servicio', 'Frecuencia']

    fig_servicios = px.bar(servicios_count, x='Frecuencia', y='Servicio', orientation='h',
                           title="Top de Especialidades Médicas",
                           text_auto=True, color='Frecuencia',
                           color_continuous_scale='GnBu')
    st.plotly_chart(fig_servicios, use_container_width=True)

st.dataframe(df[['turn_idx', 'texts.user', 'texts.bot', 'stats.cost']].tail(10), use_container_width=True)