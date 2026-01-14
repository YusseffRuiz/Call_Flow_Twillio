import re
import unicodedata
import hashlib

# --- utilidades básicas ---
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def norm_txt(s: str) -> str:
    s = strip_accents(s or '').lower()
    # s = re.sub(r'[^\w\s]', ' ', s)
    # s = re.sub(r"[,.#\-]+", ' ', s)
    # s = re.sub(r'\s+', ' ', s).strip()
    return s.strip()

# --- Utilidades de extraccion de localidad ---
VIAL_TOKENS = r"(av\.?|av|avenida|calle|carr\.?|carretera|blvd\.?|bulevar|prol\.?|prolongacion|prolongación|andador|and\.?|col\.?|col|colonia|mz|mza|manzana|lt|lote|cp|c\.p\.|cp\.?|fracc\.?|fraccionamiento|barrio|col |col\.|\#)"
VIAL_RE = re.compile(fr"\b{VIAL_TOKENS}\b", re.IGNORECASE)
VIAL_START_RE = re.compile(fr"^\s*{VIAL_TOKENS}\b", re.IGNORECASE)

# Números de vivienda típicos (y afines)
HOUSE_NUM_TOKENS = r"(no\.?|n[úu]m\.?|#|int\.?|ext\.?|mz\.?|manzana|lt\.?|lote|depto\.?|dep\.?|edif\.?|edificio)"
HOUSE_NUM_RE = re.compile(fr"\b{HOUSE_NUM_TOKENS}\b", re.IGNORECASE)

# Patrón de número “tipo puerta”: 12, 12A, 12-3, 12-14, 114B, s/n
HOUSE_NUM_VALUE_RE = re.compile(r"\b(\d{1,4}([a-zA-Z])?(-\d{1,4})?|s/?n)\b", re.IGNORECASE)

# C.P. / CP / Código Postal (5 dígitos). Detecta variantes “C.P.”, “CP”, “C P”
CP_FLAG_RE = re.compile(r"\b(c\.?\s?p\.?|c[oó]digo\s+postal)\b", re.IGNORECASE)
CP_5DIGIT_RE = re.compile(fr"\b\d{5}\b")



def extract_cp(dom: str):
    if not isinstance(dom,str): return None
    m = re.search(r"\b(?:C\.?P\.?\s*)?(\d{5})\b", dom, flags=re.I)
    return m.group(1) if m else None

def clean_address(txt: str) -> str:
    """
    Elimina segmentos 'palabra vial + nombre' hasta coma/punto y coma/salto de línea.
    'calle X ...' 'av. miguel hidalgo ...' hasta próximo separador
    """
    t = norm_txt(txt)
    return re.sub(fr"\b{VIAL_TOKENS}\b[^,;\n]*", "", t)

def split_segments(txt_norm: str) -> list[str]:
    """
    Segmenta de derecha a izquierda, útil para detectar '..., municipio, estado'.
    """
    parts = [p.strip() for p in re.split(r"[,;\n]+", txt_norm) if p.strip()]
    return parts  # derecha = parts[-1]

def prefer_tail_state(parts: list[str], value_list: dict, segments: int=4, estado=True):
    """
    Busca el estado en los últimos 2-3 segmentos (mayor probabilidad).
    Devuelve (estado_canon, score 100/0).
    Estado = True - > Funciona en busqueda de estados con dict Estados. Revisar estructura:
        Estado_canon: [lista de variantes]
    Estado = False - > Usando el dict Municipios y sus alias. Estructura:
        alias: (municipio_canon, estado_canon)
    """
    if len(parts) < 4:  # Buscar entonces en toda la string
        segments = len(parts)
    tail = parts[-segments:] if len(parts) >= segments else parts
    tail_txt = " | ".join(tail)
    if estado:
        for full, variants in value_list.items():
            for variant in variants:
                variant = norm_txt(variant)
                if re.search(fr"\b{re.escape(variant)}\b", tail_txt):
                    return norm_txt(full), 100
    else:
        for alias, (muni, est) in value_list.items():
            if re.search(fr"\b{re.escape(alias)}\b", tail_txt):
                return norm_txt(muni), 100

    return None, 0

def segment_has_vial_prefix(seg: str) -> bool:
    """¿El segmento inicia con una palabra de vialidad?"""
    return bool(VIAL_START_RE.search(seg or ""))

def find_token_in_tail(parts: list[str], token: str, segments: int = 3) -> bool:
    """True si token aparece como palabra en alguno de los últimos N segmentos."""
    token = norm_txt(token)
                        # Si la direccion es muy corta, significa que no hay numeros o la direccion esta incompleta
    if len(parts) < 4:  # Buscar entonces en toda la string
        segments = len(parts)
    tail = parts[-segments:] if len(parts) >= segments else parts
    tail_txt = " | ".join(tail)
    return bool(re.search(fr"\b{re.escape(token)}\b", tail_txt))

def _has_house_number_like(text: str) -> bool:
    """¿Aparece un indicio de número de vivienda (token o valor típico)?"""
    if HOUSE_NUM_RE.search(text):               # 'No.', '#', 'int', 'mz', 'lt', etc.
        return True
    # Números típicos de puerta (12, 12A, 12-3, 12-14, s/n)
    if HOUSE_NUM_VALUE_RE.search(text):
        # Cuidado: si es 5 dígitos y está cerca de CP, podría ser código postal
        # (esto lo filtra looks_like_street_context más abajo con ventana/CP)
        return True
    return False

def _looks_like_cp(text: str) -> bool:
    """
    ¿hay un CP explícito en la ventana (CP + 5 dígitos) o un 5 dígitos sin prefijo?
    """
    # si hay bandera CP y 5 dígitos, seguro es C.P.
    if CP_FLAG_RE.search(text) and CP_5DIGIT_RE.search(text):
        return True
    # 5 dígitos sueltos podrían ser CP; no siempre, pero mejor tratarlos como CP para no
    # confundirlos con número de puerta.
    if CP_5DIGIT_RE.search(text):
        return True
    return False

def looks_like_street_context(full_txt_norm: str, match_span: tuple[int,int]) -> bool:
    """
    True si el match (p.ej. 'benito juarez', 'hidalgo', 'morelos', 'cuauhtemoc') luce
    como parte de una vía (calle/av/blvd), considerando:
      - palabra vial ± 2-3 tokens
      - presencia de indicios de número de vivienda (No., #, int, mz/lt, 12A, 10-12, s/n)
      - y evitando confundir con C.P. (5 dígitos)
    """
    start, end = match_span
    left = full_txt_norm[:start]
    right = full_txt_norm[end:]

    # Ventanas un poco más amplias en caracteres para capturar 'No.' y números
    ltxt = left[-30:]
    rtxt = right[:30]

    # 1) ¿Hay contexto vial cercano a la izquierda/derecha?
    left_has_vial  = bool(VIAL_RE.search(ltxt))
    right_has_vial = bool(VIAL_RE.search(rtxt))

    # 2) ¿Hay indicios de número de vivienda cerca (No., #, 12A, 10-12, s/n)?
    left_has_num  = _has_house_number_like(ltxt)
    right_has_num = _has_house_number_like(rtxt)

    # 3) ¿Se ve un C.P. en la misma ventana? (para no confundir 5 dígitos como puerta)
    left_looks_cp  = _looks_like_cp(ltxt)
    right_looks_cp = _looks_like_cp(rtxt)

    # Heurística principal:
    # - Si hay palabra vial y (número de vivienda) en el mismo lado del match
    #   y NO parece ser C.P., tratamos el match como 'calle', no entidad geo.
    if left_has_vial and left_has_num and not left_looks_cp:
        return True
    if right_has_vial and right_has_num and not right_looks_cp:
        return True

    # Heurística secundaria:
    # - Si hay palabra vial y el match está a 0-2 tokens de distancia del siguiente token,
    #   también considerar 'calle'.
    def _few_tokens_between(vial_side: str) -> bool:
        toks = re.split(r"\s+", vial_side.strip())
        return len([t for t in toks if t]) <= 3  # ~2-3 tokens

    if left_has_vial:
        # tokens entre última vial y match
        chunk = re.sub(rf".*\b{VIAL_TOKENS}\b", "", ltxt, flags=re.IGNORECASE)
        if _few_tokens_between(chunk):
            return True
    if right_has_vial:
        # tokens entre match y primera vial a la derecha
        chunk = re.sub(rf"^\s*\b{VIAL_TOKENS}\b.*", "", rtxt, flags=re.IGNORECASE)
        if _few_tokens_between(chunk):
            return True

    # 4) Caso: no hay palabra vial, pero hay patrón claro de número de vivienda junto al match,
    #    y NO parece C.P. (ej. 'Hidalgo 114', 'Morelos #450', 'Cuauhtemoc s/n')
    #    Mira solo 15-20 chars alrededor inmediato.
    around = full_txt_norm[max(0, start-20): min(len(full_txt_norm), end+20)]
    if _has_house_number_like(around) and not _looks_like_cp(around):
        return True

    return False

def normalize_tel(t: str):
    digits = re.sub(r"\D","", str(t or ""))
    if digits.startswith("52") and len(digits)==12: digits = digits[2:]
    if len(digits)==10: return f"+52 {digits[:3]} {digits[3:6]} {digits[6:]}"
    return str(t or "")

# --- Utilidades para ID --- #
def slugify(text: str) -> str:
    if not text:
        return ""
    # normaliza -> ascii básico
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    # solo [a-z0-9-], colapsa separadores
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:80]  # corta por si es larguísimo

def short_hash(*parts: str, length: int = 8) -> str:
    base = "||".join(p or "" for p in parts)
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return h[:length]

def make_canonical_id(nombre_unidad: str, direccion: str) -> str:
    """
    IMPORTANTE: Actualmente el ID canonico se realiza por medio de "nombre de unidad + direccion + hash de 8 digitos.
    En el futuro, si se desea integrar con otros sistemas, puede usarse un ID tipo UUID v4 o si el sistema crece,
    aumentar el numero de hash a 9.
    """
    s = slugify(nombre_unidad)
    h = short_hash(nombre_unidad, direccion, length=8)
    return f"{s}-{h}"


def alias_token_pattern(aliases) -> re.Pattern: # Cambia de lista a Regex
    """
    Construye un patrón regex que matchee cualquiera de los aliases como token completo,
    permitiendo punto final opcional (ej. 'tlax.').
    """
    punct = r"\s,.;:()/\-–—|"
    alts = []
    for a in aliases:
        a = strip_accents(a).lower().strip()
        if not a:
            continue
        # si el alias ya termina en punto, aceptamos exactamente ese; si no, punto opcional
        if a.endswith("."):
            alts.append(re.escape(a))
        else:
            alts.append(re.escape(a) + r"\.?")
    # límites “suaves” de token: inicio o separador, y fin o separador
    # pat = rf"(?i)(?P<sep_l>^[{punct}]|(?<=^[^{punct}])[{punct}]|^)" \
    #       rf"(?P<alias>(?:{'|'.join(alts)}))" \
    #       rf"(?P<sep_r>[$punct]|$)"
    # Simplificado: usamos lookarounds prácticos con grupos para limpiar limpio
    return re.compile(rf"(?i)(?<!\w)(?:{'|'.join(alts)})(?!\w)")