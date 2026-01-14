# RAG_CORE/interpreter/location_interpreter.py

from typing import Any, Callable, Dict, Optional

from RAG_CORE import geo_location
from RAG_CORE.rag_utils.search_utils import norm_txt


class LocationInterpreter:
    """
    Interpreta estado/municipio a partir de la query usando:
      - GeoFuzzy (ambigüedad estado/muni)
      - helpers de RetrievalModule (extract_state_from_query, extract_municipality_from_query)

    La idea es centralizar aquí lo que hoy está disperso en RetrievalModule.ask.
    """

    def __init__(
        self,
        geo: geo_location.GeoFuzzy,
        extract_state_fn: Callable[[str], Optional[str]],
        extract_muni_fn: Callable[[str, Optional[str]], Optional[Dict[str, Any]]],
    ):
        self.geo = geo
        self.extract_state_fn = extract_state_fn
        self.extract_muni_fn = extract_muni_fn

    def interpret(self, query: str, q_norm: Optional[str] = None) -> Dict[str, Any]:
        """
        Devuelve un dict con:
        {
          "estado": str|None,
          "municipio": str|None,
          "conf_estado": float,
          "conf_municipio": float,
          "degradaciones": [..],
          "debug": {"amb":..., "muni_info":...}
        }
        """
        if q_norm is None:
            q_norm = norm_txt(query)

        degradaciones = []

        # --- 1) Estado con tu helper actual ---
        detected_state = self.extract_state_fn(query)  # p.ej. "campeche" o None

        # --- 2) Ambigüedad estado/muni con GeoFuzzy ---
        amb_active = False
        amb = self.geo.detect_state_muni_ambiguity(q_norm)
        force_state = None
        suppress_muni = False

        if amb:
            force_state = amb["state"]
            if not self.geo.has_explicit_muni_marker(q_norm):
                suppress_muni = True            # interpretamos como estado, no municipio
                amb_active = True
                degradaciones.append("ambig_state_muni")

        # --- 3) Municipio (si no lo suprimimos) ---
        muni_info = None
        if not suppress_muni:
            muni_info = self.extract_muni_fn(query, detected_state)

        # --- 4) Resolver estado final ---
        estado_final = None
        conf_estado = 0.0

        if force_state:
            estado_final = force_state
            conf_estado = 1.0
        elif detected_state:
            estado_final = detected_state.title()
            conf_estado = 1.0  # tu helper es bastante estricto; lo tratamos como alta confianza

        # --- 5) Resolver municipio final ---
        municipio_final = None
        conf_muni = 0.0

        if muni_info:
            # Caso ambigüedad multi-estado -> aquí sólo lo reportamos, decisión la toma arriba
            if "ambiguous" in muni_info:
                # No elegimos uno; dejamos que RetrievalModule lo trate como "preguntar al usuario"
                degradaciones.append("municipio_ambiguo")
            else:
                municipio_final = muni_info.get("municipio")
                # Normalmente viene un score (WRatio); lo normalizamos a [0,1]
                score = float(muni_info.get("score", 100))
                conf_muni = max(0.0, min(score / 100.0, 1.0))

                # Si hay estado en muni_info y no habíamos fijado estado, lo usamos
                if not estado_final and muni_info.get("estado"):
                    estado_final = muni_info["estado"]
                    conf_estado = max(conf_estado, conf_muni)

        # Si no hay muni pero sí query habla de "en X" etc., lo podemos marcar como degradación suave
        if not municipio_final:
            degradaciones.append("sin_municipio")

        return {
            "estado": estado_final,
            "municipio": municipio_final,
            "conf_estado": conf_estado,
            "conf_municipio": conf_muni,
            "degradaciones": degradaciones,
            "debug": {
                "amb": amb,
                "muni_info": muni_info,
                "q_norm": q_norm,
                "amb_active": amb_active,
            },
        }
