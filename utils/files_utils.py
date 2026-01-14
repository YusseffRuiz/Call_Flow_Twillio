import math
from rapidfuzz import fuzz
import numpy as np
import pandas as pd
import re
from RAG_CORE.rag_utils.mappings import SYN, SERVICIOS_CANONICOS, SERVICE_LEXICON
from RAG_CORE.rag_utils.search_utils import norm_txt


# Búsqueda de la hoja que nos interesa
def load_sheet_by_name(xlsx_path: str, target: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    # match insensible a mayúsculas / tildes / espacios, aunque las hojas de excel no te permitan tildes.
    def norm(s):
        return re.sub(r'\s+', ' ', str(s).strip().lower())
    names = xl.sheet_names
    target_norm = norm(target)
    # exacto primero
    for name in names:
        if norm(name) == target_norm:
            return xl.parse(name)
    # fuzzy search: contiene la palabra
    for name in names:
        if norm(target_norm) in norm(name):
            return xl.parse(name)
    raise ValueError(f"No se encontró la hoja '{target}' en {xlsx_path}. Encontradas: {names}")



def _clean_time(s: str) -> str:
    s = (s or "").strip()
    # normaliza formatos "7:00", "07:00", "7", etc.
    m = re.match(r'^\s*(\d{1,2})(?::?(\d{2}))?\s*$', s)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2) or 0)
        return f"{h:02d}:{mi:02d}"
    # si ya viene con :, lo dejamos tal cual
    return s

def _pair_to_range(a: str, b: str) -> str:
    # Rango de horas: cambiar de par de columnas a rango
    a2 = _clean_time(a)
    b2 = _clean_time(b)
    if a2 and b2:
        return f"{a2}–{b2}"
    return (a2 or b2 or "").strip()

def _find_span_columns(df: pd.DataFrame, base_label: str) -> tuple[str|None, str|None]:
    """
    Busca dos columnas contiguas cuyo nombre empiece por base_label (case-insensitive)
    o que una sea 'Unnamed: k' contigua a la anterior.
    Devuelve (col_entrada, col_salida) si las encuentra.
    """
    cols = list(df.columns)
    base_norm = base_label.lower()
    for i, c in enumerate(cols):
        c_norm = str(c).strip().lower()
        if c_norm.startswith(base_norm):
            # candidato a ENTRADA en i; la SALIDA suele ser i+1 (Unnamed o similar)
            if i + 1 < len(cols):
                return cols[i], cols[i+1]
    # fallback: buscar por regex más la columna contigua
    for i in range(len(cols)-1):
        if base_norm in str(cols[i]).strip().lower():
            return cols[i], cols[i+1]
    return None, None

def consolidate_horarios_row(row: pd.Series, df: pd.DataFrame) -> str:
    # L-V: 2 columnas (entrada/salida)
    lv_in_col, lv_out_col = _find_span_columns(df, "LUNES A VIERNES")
    lv = ""
    if lv_in_col:
        lv = _pair_to_range(str(row.get(lv_in_col, "") or ""), str(row.get(lv_out_col, "") or ""))

    # Sábado: 2 columnas
    sa_in_col, sa_out_col = _find_span_columns(df, "HORARIO SÁBADO")
    sa = ""
    if sa_in_col:
        sa = _pair_to_range(str(row.get(sa_in_col, "") or ""), str(row.get(sa_out_col, "") or ""))

    # Domingo: puede venir en una sola col (o a veces también en par; soportamos ambos)
    dom_col_exact = None
    for c in df.columns:
        if str(c).strip().lower().startswith("horario domingo"):
            dom_col_exact = c
            break
    do = ""
    if dom_col_exact:
        val = str(row.get(dom_col_exact, "") or "")
        # si viniera en dos columnas contiguas, intentamos también
        do_in, do_out = _find_span_columns(df, "HORARIO DOMINGO")
        if do_in and do_out:
            val = _pair_to_range(str(row.get(do_in, "") or ""), str(row.get(do_out, "") or "")) or val
        do = val

    # Laboratorio: una sola columna
    lab_col = None
    for c in df.columns:
        if str(c).strip().lower().startswith("horario de laboratorio"):
            lab_col = c; break
    lab = str(row.get(lab_col, "") or "") if lab_col else ""

    parts = []
    if lv:  parts.append(f"L-V {lv}")
    if sa:  parts.append(f"S {sa}")
    if do:  parts.append(f"D {do}")
    if lab: parts.append(f"Lab: {lab}")
    return " | ".join(p for p in parts if p)



## Parsing de servicios
# ordena claves de SYN por longitud descendente para “frase más larga primero”
SYN_SORTED = sorted(SYN.keys(), key=lambda x: len(x), reverse=True)
SERVICE_CANON = set(SERVICIOS_CANONICOS)

def parse_servicios(texto: str) -> list[str]:
    """
    Convierte un texto libre de servicios en claves canónicas.
    Conserva solo valores en SERVICIOS_CANONICOS y deduplica.
    """
    if texto is None or (isinstance(texto, float) and math.isnan(texto)):
        base = ""
    else:
        base = norm_txt(texto)

    # Split por separadores comunes, pero también probamos matching directo por frases
    # para capturar cosas como “examen de la vista”, “graduacion de lentes”, etc.
    hits = []

    # 1) matching por frases (prioriza n-gramas largos)
    temp = base
    for key in SYN_SORTED:
        if key in temp:
            hits.append(SYN[key])
            # opcional: evitar dobles conteos sobre el mismo tramo
            temp = temp.replace(key, " ")

    # 2) split por lista (por si venían separados)
    parts = re.split(r'[,*;|/()\n-]+', base)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p in SYN:
            hits.append(SYN[p])

    # dedup + solo canónicos válidos
    out = []
    seen = set()
    for h in hits:
        if h in SERVICE_CANON and h not in seen:
            seen.add(h)
            out.append(h)
    return out

def add_info_services(texto, servicios):
    if texto is None or (isinstance(texto, float) and math.isnan(texto)):
        base = ""
    else:
        base = norm_txt(texto)

        # Split por separadores comunes, pero también probamos matching directo por frases
        # para capturar cosas como “examen de la vista”, “graduacion de lentes”, etc.
    hits = []

    # 1) matching por frases (prioriza n-gramas largos)
    temp = base
    for key in SYN_SORTED:
        if key in temp:
            hits.append(SYN[key])
            # opcional: evitar dobles conteos sobre el mismo tramo
            temp = temp.replace(key, " ")

    # 2) split por lista (por si venían separados)
    parts = re.split(r'[,*;|/()\n-]+', base)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p in SYN:
            hits.append(SYN[p])

    # dedup + solo canónicos válidos
    seen = set()
    for h in hits:
        if h in SERVICE_CANON and h not in seen and h not in servicios:
            seen.add(h)
            servicios.append(h)
    return servicios


def resolve_service(query: str, embeddings_model=None, topn: int = 3):
    q = norm_txt(query)
    # print(q)

    # 1) regex fuerte
    regex_scores = {svc: 1.0 if any(re.search(pat, q) for pat in cfg.get("re", [])) else 0.0
                    for svc, cfg in SERVICE_LEXICON.items()}
    # print(regex_scores)

    # 2) fuzzy por sinónimos
    fuzzy_scores = {}
    for svc, cfg in SERVICE_LEXICON.items():
        best = 0.0
        for token in cfg.get("syn", []):
            best = max(best, fuzz.token_set_ratio(q, norm_txt(token)) / 100.0)
        fuzzy_scores[svc] = best
    # print(fuzzy_scores)

    # 3) embeddings (query vs. descripción del servicio)
    embed_scores = {svc: 0.0 for svc in SERVICE_LEXICON.keys()}
    if embeddings_model is not None:
        qv = np.array(embeddings_model.embed_query(q), dtype=float)
        qn = np.linalg.norm(qv) + 1e-9
        for svc, cfg in SERVICE_LEXICON.items():
            dv = np.array(embeddings_model.embed_query(norm_txt(cfg.get("desc",""))), dtype=float)
            cs = float(np.dot(qv, dv) / (qn * (np.linalg.norm(dv) + 1e-9)))
            embed_scores[svc] = cs
    # print(embed_scores)

    # 4) ensemble
    W = {"regex": 0.5, "fuzzy": 0.3, "embed": 0.2}
    scores = {svc: W["regex"]*regex_scores[svc] + W["fuzzy"]*fuzzy_scores[svc] + W["embed"]*embed_scores[svc]
              for svc in SERVICE_LEXICON.keys()}
    # print(scores)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    # print(ranked)
    # desempate / umbral
    if not ranked:
        return {"service": None, "score": 0.0, "candidates": []}
    if len(ranked) >= 3 and (ranked[0][1] - ranked[2][1]) < 0.08:
        return {"service": None, "score": ranked[0][1], "candidates": ranked}
    if ranked[0][1] < 0.35:
        return {"service": None, "score": ranked[0][1], "candidates": ranked}
    return {"service": ranked[0][0], "score": ranked[0][1], "candidates": ranked}
