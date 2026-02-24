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


def load_cora_logs():
    files = glob.glob("logs/*.json")
    all_data = []
    for f in files:
        # Extraemos el Call ID del nombre del archivo
        filename = os.path.basename(f).split('.')[0]

        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    record = json.loads(line)
                    record['call_id'] = filename  # Inyectamos el ID
                    all_data.append(record)
                except:
                    continue

    df_raw = pd.json_normalize(all_data)
    # Limpieza estándar de métricas
    cols_metrics = ['latencies.e2e', 'latencies.asr', 'latencies.llm_first_token', 'latencies.ttfb_internal',
                    'stats.cost']
    for col in cols_metrics:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
    return df_raw


df = load_cora_logs()


# --- BARRA LATERAL (FILTROS) ---
st.sidebar.header("🔍 Filtros de Auditoría")

# Filtro por Ciudad (Dinámico desde mappings)
ciudades_sidebar = ["Todos"] + sorted([e.capitalize() for e in MX_STATES.keys()])
sede_sel = st.sidebar.selectbox("Filtrar por Sede detectada:", ciudades_sidebar)

# Buscador de Call ID
search_call = st.sidebar.text_input("Buscar por Call ID (Twilio SID):")

# Aplicación de filtros al DataFrame principal
df_filtered = df.copy()
if sede_sel != "Todos":
    # Usamos la lógica de extracción que definimos para la sección de ventas
    df_filtered = df_filtered[df_filtered['texts.user'].str.contains(sede_sel, case=False, na=False)]

if search_call:
    df_filtered = df_filtered[df_filtered['call_id'].str.contains(search_call, case=False, na=False)]


# --- SECCIÓN 1: MÉTRICAS FINANCIERAS (KPIs) ---
st.divider()
st.header("💰 Indicadores Financieros")
m1, m2, m3, m4 = st.columns(4)

with m1:
    total_cost = df_filtered['stats.cost'].sum()
    st.metric("Inversión en Tokens (Total)", f"${total_cost:.4f} USD", delta_color="inverse")
with m2:
    avg_call_cost = df_filtered['stats.cost'].mean()
    st.metric("Costo Promedio / Turno", f"${avg_call_cost:.5f} USD")
with m3:
    total_turns = len(df_filtered)
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
        'Segundos': [df_filtered['latencies.asr'].mean(), df_filtered['latencies.llm_first_token'].mean(), df_filtered['latencies.e2e'].mean()]
    }
    fig_lat = px.bar(lat_data, x='Etapa', y='Segundos', color='Etapa', text_auto='.2s')
    st.plotly_chart(fig_lat, use_container_width=True)

with col_right:
    st.subheader("🎯 Calidad de Entendimiento (NER/RAG)")
    # Identificamos respuestas fallidas o fuera de dominio
    oos_count = df_filtered[df_filtered['texts.bot'].str.contains("out_of_scope")].shape[0]
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
df_filtered['Ciudad_Detectada'] = df_filtered['texts.user'].apply(lambda x: extraer_menciones(x, CIUDADES_PROYECTO))
df_filtered['Servicio_Detectado'] = df_filtered['texts.user'].apply(lambda x: extraer_menciones(x, SERVICIOS_PROYECTO))
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.subheader("📍 Sedes con mayor interés")
    ciudades_count = df_filtered[df_filtered['Ciudad_Detectada'] != "No especificado"]['Ciudad_Detectada'].value_counts().reset_index()
    ciudades_count.columns = ['Ciudad', 'Interacciones']

    fig_ciudades = px.pie(ciudades_count, values='Interacciones', names='Ciudad',
                          hole=0.4, title="Distribución Geográfica de Consultas",
                          color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_ciudades, use_container_width=True)

with col_v2:
    st.subheader("🩺 Servicios más solicitados")
    servicios_count = df_filtered[df_filtered['Servicio_Detectado'] != "No especificado"][
        'Servicio_Detectado'].value_counts().reset_index()
    servicios_count.columns = ['Servicio', 'Frecuencia']

    fig_servicios = px.bar(servicios_count, x='Frecuencia', y='Servicio', orientation='h',
                           title="Top de Especialidades Médicas",
                           text_auto=True, color='Frecuencia',
                           color_continuous_scale='GnBu')
    st.plotly_chart(fig_servicios, use_container_width=True)

st.dataframe(df_filtered[['turn_idx', 'texts.user', 'texts.bot', 'stats.cost']].tail(10), use_container_width=True)

# --- SECCIÓN 4: SALUD DEL SISTEMA (TURNOS CRÍTICOS) ---
st.divider()
st.header("⚠️ Salud del Sistema: Turnos Críticos")
st.write("Identificación automática de interacciones con latencia alta o costos elevados.")

# Definición de umbrales críticos
LATENCY_THRESHOLD = 5.0  # Segundos
COST_THRESHOLD = 0.005  # USD

# Filtramos los casos que requieren atención inmediata
criticos = df_filtered[
    (df_filtered['latencies.e2e'] > LATENCY_THRESHOLD) |
    (df_filtered['stats.cost'] > COST_THRESHOLD) |
    (df_filtered['texts.bot'].str.contains("izcalli", case=False))  # Detección del bug de repetición
    ].copy()

if not criticos.empty:
    # Formatear tabla para visualización incluyendo el CALL ID
    display_criticos = criticos[[
        'call_id', 'turn_idx', 'latencies.e2e', 'latencies.asr', 'stats.cost', 'texts.user', 'texts.bot'
    ]].sort_values(by='latencies.e2e', ascending=False)

    st.subheader("📋 Registro de Incidentes Detectados")
    st.dataframe(
        display_criticos.style.highlight_max(subset=['latencies.e2e'], color='#ff4b4b')
        .format({'stats.cost': '{:.4f}', 'latencies.e2e': '{:.2f}s', 'latencies.asr': '{:.2f}s'}),
        use_container_width=True
    )

    st.info(f"Se detectaron {len(criticos)} turnos críticos. Utiliza el **call_id** para rastrear el audio en Twilio.")
else:
    st.success("✅ No se detectaron turnos críticos. El sistema opera bajo los parámetros de calidad (SLA).")

# --- FOOTER ---
st.divider()
st.caption("Analytics Engine v1.0 | Medical Life IA")