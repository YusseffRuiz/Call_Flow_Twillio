############################ IMPLEMENTACION RAILGUARDS###############
from RAG_CORE.rag_utils.mappings import MEDICAL_DIAG_KEYWORDS, URGENT

def apply_business_rules(hit):
    m = hit["metadata"] if "metadata" in hit else hit
    tipo = m.get("tipo_sede","")
    programa = m.get("programa","").lower()
    servicios = set(m.get("servicios_lista", []))

    # Policia => solo medicamentos/farmacia
    if programa == "policia":
        servicios = {s for s in servicios if s in {"farmacia", "medicamentos"}}
        m["servicios_lista"] = sorted(list(servicios))
        m["costo_consulta"] = ""   # no aplica
        # mensaje recordatorio
        m["nota_regla"] = "Programa Policía: solo dispensación de medicamentos."

    # Farmacia / Botiquín => solo medicamentos
    if tipo in {"farmacia_subrogada", "botiquin_policiaco"}:
        servicios = {s for s in servicios if s in {"farmacia", "medicamentos"}}
        m["servicios_lista"] = sorted(list(servicios))
        m["costo_consulta"] = ""  # no consultas

    # Subrogadas: consulta $0 (si no es solo medicamentos)
    if programa == "subrogadas" and tipo not in {"farmacia_subrogada", "botiquin_policiaco"}:
        m["costo_consulta"] = "$0"

    # X mi: resaltar mastografía si existe
    if tipo == "x_mi" and "mastografia_sin_dolor" in servicios:
        m["nota_regla"] = "Campaña X mi: mastografía sin dolor disponible."

    # Tlaxcala activos/pensionados: recordatorio de elegibilidad
    if programa in {"tlaxcala_activos", "tlaxcala_pensionados"}:
        m["nota_regla"] = "Elegible solo para personal activo o pensionado de Gobierno de Tlaxcala."

    return m


    # TODO Agregar todas las reglas de los servicios que se ofrecen.

def is_medical_diagnosis_request(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in MEDICAL_DIAG_KEYWORDS)

def is_urgent(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in URGENT)
