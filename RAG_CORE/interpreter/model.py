from typing import Any, Dict, List


def make_empty_interpretation() -> Dict[str, Any]:
    """
    Estructura estándar de salida del intérprete de la query.
    """
    return {
        "intent": "buscar_servicio",
        "service_id": None,
        "location": {
            "estado": None,
            "municipio": None,
        },
        "flags": {
            "degradaciones": [],         # ex. ["sin_municipio", "relajar_estado"]
            "fast_path_candidate": False # UME/terapias, etc.
        },
        "confidence": {
            "service": 0.0,
            "estado": 0.0,
            "municipio": 0.0,
        },
        "debug": {
            "service_candidates": [],   # [(svc, score), ...]
            "service_raw_scores": {},  # opcional si luego quieres exponer más
            "geo": {},                 # lo que devuelva el geo-intérprete
        },
    }
