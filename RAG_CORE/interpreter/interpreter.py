from typing import Any, Dict, List

from RAG_CORE.interpreter.model import make_empty_interpretation
from RAG_CORE.interpreter.service_interpreter import ServiceInterpreter
from RAG_CORE.interpreter.location_interpreter import LocationInterpreter
from RAG_CORE.rag_utils.mappings import SERVICE_LEXICON  # para detectar UME/terapias


class Interpreter:
    """
    Une ServiceInterpreter + LocationInterpreter y devuelve
    un JSON de intención listo para usar en el retriever.
    """

    def __init__(
        self,
        service_interpreter: ServiceInterpreter,
        location_interpreter: LocationInterpreter,
    ):
        self.service_interpreter = service_interpreter
        self.location_interpreter = location_interpreter

    def parse(self, query: str) -> Dict[str, Any]:
        out = make_empty_interpretation()

        # 1) Intent (por ahora buscar_servicio)
        out["intent"] = "buscar_servicio"

        # 2) Servicio
        svc_res = self.service_interpreter.interpret(query)
        service = svc_res.get("service")
        out["service_id"] = service
        out["confidence"]["service"] = float(svc_res.get("score", 0.0) or 0.0)
        out["debug"]["service_candidates"] = svc_res.get("candidates", [])

        # 3) Ubicación
        loc_res = self.location_interpreter.interpret(query)
        out["location"]["estado"] = loc_res["estado"]
        out["location"]["municipio"] = loc_res["municipio"]
        out["confidence"]["estado"] = loc_res["conf_estado"]
        out["confidence"]["municipio"] = loc_res["conf_municipio"]
        out["flags"]["degradaciones"] = loc_res["degradaciones"]
        out["debug"]["geo"] = loc_res["debug"]

        # 4) Fast-path UME / terapias alternativas
        candidates = [c[0] for c in out["debug"]["service_candidates"]]
        if (
            service in ["ozonoterapia", "fototerapia", "cosmeatria", "hiperbarica", "laserterapia"]
            or "ume_alternativas" in candidates
        ):
            out["flags"]["fast_path_candidate"] = True

        return out
