from typing import Any, Dict
from utils import files_utils  # mismo import que ya usas en retrieval_module


class ServiceInterpreter:
    """
    Envoltorio ligero sobre files_utils.resolve_service.
    Estandarizacion de la salida, wrapper.
    """

    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model

    def interpret(self, query: str, topn: int = 3) -> Dict[str, Any]:
        """
        Devuelve exactamente lo que hoy devuelve resolve_service, pero
        encapsulado en una clase por claridad futura.

        Ej:
        {
          "service": "optometrista" | None,
          "score": 0.82,
          "candidates": [("optometrista", 0.82), ("ume_alternativas", 0.61), ...]
        }
        """
        return files_utils.resolve_service(
            query=query,
            embeddings_model=self.embeddings_model,
            topn=topn,
        )
