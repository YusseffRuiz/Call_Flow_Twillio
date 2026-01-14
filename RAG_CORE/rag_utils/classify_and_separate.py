import math
import re

# mapeo de headers → clave canónica de especialidad
ESPECIALIDADES_CANON = {
    "MÉDICO GENERAL": "medico_general",
    "MEDICO GENERAL": "medico_general",
    "DENTISTA": "odontologia",
    "ODONTOLOGO": "odontologia",
    "ODONTOLOGÍA": "odontologia",
    "OPTOMETRISTA": "optometrista",
    "NUTRIOLOGO": "nutricion",
    "NUTRIÓLOGO": "nutricion",
    "PSICOLOGO": "psicologia",
    "PSICÓLOGO": "psicologia",
    # agrega variantes si aparecen en tu hoja
}

# formatos aceptados en Horarios
HOUR = r"(?:[01]?\d|2[0-3]):[0-5]\d"
RANGE = re.compile(rf"{HOUR}\s*([-–]|[aA])\s*{HOUR}")
HOUR_ONLY = re.compile(rf"{HOUR}")

def normalize_comidas(s: str) -> list[str]:
    if not s or s.startswith("*") or s=="x" or s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    # lista numerada → partes
    parts = re.split(r"\d+\)\s*", s)
    out = []
    for p in parts:
        p = p.strip(" -\n")
        if not p:
            continue
        m = RANGE.search(p)
        if m:
            out.append(m.group().replace(" ", ""))
        else:
            # si hay una sola hora, no asumimos rango; puedes decidir guardar tal cual
            m2 = HOUR_ONLY.search(p)
            if m2:
                out.append(m2.group())
    # dedup
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

def normalize_header(h: str) -> str:
    h = (h or "").strip().upper()
    h = re.sub(r"\s+", " ", h)
    return h

def parse_comidas_from_row(row: dict, header_sequence: list[str]) -> dict:
    """
    Lee la fila tal como viene del DataFrame y recorre headers en orden para
    asociar 'COMIDA' a la especialidad inmediatamente anterior.
    Devuelve dict: { "medico_general":"15:00-16:00", "odontologia":"13:00-14:00", ... }
    """
    comidas = {}
    last_especialidad_key = None

    for hdr in header_sequence:
        h = normalize_header(hdr)
        val = str(row.get(hdr, "") or "").strip()
        if not val:
            # vacío: no cambia contexto
            continue

        if h == "COMIDA":
            if last_especialidad_key:
                val = normalize_comidas(val)
                if len(val) == 0:
                    val = "No disponibles, Consultar en Clinica"
                comidas[last_especialidad_key] = val
            continue

        # ¿es una especialidad conocida?
        if h in ESPECIALIDADES_CANON:
            last_especialidad_key = ESPECIALIDADES_CANON[h]
            continue

        # otros headers no relevantes: resetea contexto
        last_especialidad_key = last_especialidad_key

    return comidas



# --- Utils para telefonia ---

# patrones básicos
RE_EXT = re.compile(r"(ext|extension)\s*[:#\-]?\s*(\d{1,6})", re.IGNORECASE)
RE_NUM = re.compile(r"\+?52?\s*[\-.\s]?\s*(\(?\d{2,3}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{4})")

def _only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def normalize_mx_10d(digits: str) -> str:
    """
    Regla del proyecto: no usar +52.
    - Si viene con 52 (12 dígitos), quitarlo -> 10 dígitos.
    - Si ya son 10, se queda.
    - Si no es 10 o 12, se descarta.
    """
    if not digits:
        return ""
    if len(digits) == 12 and digits.startswith("52"):
        digits = digits[2:]
    if len(digits) == 10:
        return digits
    return ""

def extract_phone_numbers(text: str) -> list[dict]:
    """
    Devuelve lista de dicts: [{"number":"5512345678","ext":"123"}, ...]
    - Detecta varios en una cadena con ruido (saltos, barras, comas, etc).
    - Deduplica por (number,ext).
    """
    if not text:
        return []
    t = str(text)

    # 1) extraer extensiones (puede haber más de una)
    exts = [m.group(2) for m in RE_EXT.finditer(t)]

    # 2) extraer candidatos de números
    raw_candidates = []
    for m in RE_NUM.finditer(t):
        raw = m.group(0)
        digits = _only_digits(raw)
        norm = normalize_mx_10d(digits)
        if norm:
            raw_candidates.append(norm)

    # 3) split adicional por separadores obvios
    for chunk in re.split(r"[\/|,;\n]+", t):
        digits = _only_digits(chunk)
        norm = normalize_mx_10d(digits)
        if norm:
            raw_candidates.append(norm)

    # 4) dedup y asocia ext (si hay, la pegamos al primero; si son varias, repartimos)
    uniq = []
    seen = set()
    if raw_candidates:
        # reparte ext(s) si existen, de forma estable
        for i, num in enumerate(raw_candidates):
            ext = exts[i] if i < len(exts) else ""
            key = (num, ext)
            if key not in seen:
                seen.add(key)
                uniq.append({"number": num, "ext": ext})

    return uniq

def phones_to_display(phones: list[dict]) -> str:
    """
    Para mostrar en GUI/answer: "5512345678" o "5512345678 ext 123"
    Unir con " / ".
    """
    if not phones:
        return ""
    parts = []
    for p in phones:
        if p.get("ext"):
            parts.append(f'{p["number"]} ext {p["ext"]}')
        else:
            parts.append(p["number"])
    return " / ".join(parts)



def get_phones(phones_number: str):
    """
    :param phones_number: string de numeros telefonicos
    :return: phones_uniq: <- lista normalizada,telefono_display: <- string para mostrar
    """
    phones = []
    phones.extend(extract_phone_numbers(phones_number))

    # dedup final por (number, ext)
    seen = set()
    phones_uniq = []
    for p in phones:
        key = (p["number"], p.get("ext", ""))
        if key not in seen:
            seen.add(key)
            phones_uniq.append(p)

    telefono_display = phones_to_display(phones_uniq)
    return telefono_display, phones_uniq

