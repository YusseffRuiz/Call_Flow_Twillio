import numpy as np
import pandas as pd
import datetime
from collections import defaultdict

import torch
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz
from typing import Literal

## Librerias LLM
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from RAG_CORE.rag_utils.mappings import MX_STATES, MUNICIPALITY_ALIASES, MUNI_BY_STATE_MINI, build_muni_aliases_from_catalog
from RAG_CORE.rag_utils.mappings import merge_aliases, add_services_from_staff, AMBIG_STATE_MUNI, SERVICE_LEXICON
from RAG_CORE.railguards_helper import apply_business_rules
from RAG_CORE import geo_location
from utils import files_utils
from RAG_CORE.rag_utils.search_utils import *
from RAG_CORE.rag_utils.classify_and_separate import parse_comidas_from_row, get_phones
from RAG_CORE.interpreter.service_interpreter import ServiceInterpreter
from RAG_CORE.interpreter.location_interpreter import LocationInterpreter
from RAG_CORE.interpreter.interpreter import Interpreter

class RetrievalModule:
    def __init__(self, database_path, hf_token, model_name, origin_sheet="Unidades", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.database_path = database_path
        self.hf_token = hf_token
        self.origin_sheet = origin_sheet
        self.model_name = model_name
        self.device = device

        # self.kb_df = self.build_kb() # 1ra version
        self.df = files_utils.load_sheet_by_name(self.database_path, self.origin_sheet)
        self.kb_df = None
        self.geo_df = None
        self.docs = None
        self.vectorstore = None
        self.by_state = None
        self.all_munis = None
        self.state_lookup = {}
        self.geo = None
        self.score_threshold = None
        self.percentile = None
        self.fast_path = False # Opcional, mas veloz, menos recursos, menos "inteligente"

        self.IDX_BY_STATE = defaultdict(set)  # estado_norm -> set(id)
        self.IDX_BY_MUNI = defaultdict(set)  # (estado_norm, muni_norm) -> set(id)
        self.IDX_BY_SERVICE = defaultdict(set)  # servicio -> set(id)
        self.ID_TO_DOC = {}  # id -> Document
        # --- Intérpretes de servicio y ubicación ---
        self.service_interpreter = None
        self.location_interpreter = None
        self.interpreter = None

        # --- BM25 / híbrido ---
        self.bm25 = None
        self._bm25_corpus_tokens = []
        self._bm25_ids = []


    def initialize(self, path_to_database="kb_faiss_langchain", save_db = False, load_db = False, score_threshold=0.36, percentile = 0.85):
        """
        :param path_to_database: Path para guardar o cargar la base de datos
        :param save_db: True si queremos crear una nueva DB
        :param load_db: True si queremos cargar una database, si este parametro esta en True, se invalida el load
        :param score_treshold: Treshold minimo de confianza
        :param percentile: Quartile de fiabilidad de la respuesta, este para filtrar el score inicial
        """

        model_kwargs = {"device": self.device, "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": False}  # Set to True if normalization is desired

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        self.score_threshold = score_threshold
        self.percentile = percentile

        # Declaracion de la geolocalizacion

        # 1) alias automáticos desde el catálogo por estado
        state_aliases = build_muni_aliases_from_catalog(MUNI_BY_STATE_MINI)
        # 2) mézclalos con tus alias “manuales” (siglas, abreviaturas raras)
        muni_alias_all = merge_aliases(state_aliases, MUNICIPALITY_ALIASES)

        # -------- Geo fuzzy bootstrap ----------
        self.geo = geo_location.GeoFuzzy(
            state_list=MX_STATES,
            muni_by_state=MUNI_BY_STATE_MINI,
            municipality_aliases=muni_alias_all,
            ambiguity_list=AMBIG_STATE_MUNI,
            state_score_cut=70,
            muni_score_cut=70
        )

        docs, self.kb_df, self.geo_df = self.rows_to_documents_unidades(self.geo)

        if load_db:
            self.load_kb(kb_path=path_to_database, embeddings=self.embeddings)
        else:
            self.vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.COSINE,  # usar coseno requiere normalize_embeddings=True
            )
            if save_db:
                self.save_kb(db_name=path_to_database)
        #
        # crea un lookup de variante_normalizada -> estado_canonico
        for canon, variants in MX_STATES.items():
            for v in variants:
                self.state_lookup[norm_txt(v)] = canon

        self.by_state, self.all_munis = self.build_geo_lexicons()
        self.docs = docs
        # --- Intérpretes de servicio y ubicación ---
        self.service_interpreter = ServiceInterpreter(self.embeddings)
        self.location_interpreter = LocationInterpreter(
            geo=self.geo,
            extract_state_fn=self.extract_state_from_query,
            extract_muni_fn=self.extract_municipality_from_query,
        )
        self.interpreter = Interpreter(
            service_interpreter=self.service_interpreter,
            location_interpreter=self.location_interpreter,
        )

        # índices por metadatos (fast_path)
        self._build_metadata_indexes()

        # índice BM25 sobre searchable_text
        self._build_bm25_index()


    def save_kb(self, db_name="kb_faiss_langchain"):
        self.vectorstore.save_local(db_name)

    def load_kb(self, kb_path, embeddings):
        self.vectorstore = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)

    def _build_metadata_indexes(self):

        for d in self.docs:
            m = d.metadata
            mid = str(m.get("id") or d.metadata.get("nombre_oficial") or d.page_content[:50])
            self.ID_TO_DOC[mid] = d

            st = norm_txt(m.get("estado") or "")
            mu = norm_txt(m.get("municipio") or "")
            for svc in (m.get("servicios_lista") or []):
                self.IDX_BY_SERVICE[norm_txt(svc)].add(mid)

            if st:
                self.IDX_BY_STATE[st].add(mid)
                if mu:
                    self.IDX_BY_MUNI[(st, mu)].add(mid)

    def _build_bm25_index(self):
        """
        Construye un índice BM25 sobre la columna `searchable_text` de kb_df.
        Usamos el id canónico como ancla para mapear de vuelta a Document.
        """
        if self.kb_df is None or "searchable_text" not in self.kb_df.columns:
            return

        texts = self.kb_df["searchable_text"].fillna("").astype(str).tolist()
        ids = self.kb_df["id"].astype(str).tolist()

        corpus_tokens = [norm_txt(t).split() for t in texts]

        self._bm25_corpus_tokens = corpus_tokens
        self._bm25_ids = ids
        self.bm25 = BM25Okapi(corpus_tokens)

    def _bm25_search(self, query: str, k: int = 40):
        """
        Ejecuta búsqueda BM25 en searchable_text.
        Devuelve list[(Document, score_float)] ordenada desc.
        """
        if self.bm25 is None:
            return []

        tokens = norm_txt(query).split()
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scores = np.array(scores, dtype=float)

        # índices ordenados por score desc
        idx_sorted = np.argsort(scores)[::-1][:k]

        results = []
        for idx in idx_sorted:
            s = float(scores[idx])
            if s <= 0:
                continue
            mid = self._bm25_ids[idx]
            d = self.ID_TO_DOC.get(mid)
            if d is None:
                continue
            results.append((d, s))

        return results

    def _hybrid_fusion_rrf(
        self,
        vec_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        k_out: int = 40,
        alpha: float = 0.5,
        k_rrf: int = 60,
    ):
        """
        Fusión híbrida BM25 + vector mediante Reciprocal Rank Fusion.

        alpha ~ peso de BM25 vs vector:
          - alpha = 0.5 -> ambos igual
          - alpha > 0.5 -> favorece BM25
        """
        # map: id -> rank
        def to_rank_map(results):
            rank_map = {}
            for rank, (d, _) in enumerate(results, start=1):
                mid = str(d.metadata.get("id") or d.metadata.get("nombre_oficial") or id(d))
                if mid not in rank_map:
                    rank_map[mid] = rank
            return rank_map

        r_vec = to_rank_map(vec_results)
        r_bm  = to_rank_map(bm25_results)

        all_ids = set(r_vec.keys()) | set(r_bm.keys())
        fused = []

        for mid in all_ids:
            d = self.ID_TO_DOC.get(mid)
            if d is None:
                continue

            score = 0.0
            if mid in r_vec:
                score += (1.0 - alpha) / (k_rrf + r_vec[mid])
            if mid in r_bm:
                score += alpha / (k_rrf + r_bm[mid])

            fused.append((d, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k_out]


    def rows_to_documents_unidades(self, geolocation):
        df_geo = _finalize_geo(self.df, geolocation, min_state=60, min_muni=60)

        # -------- Construcción de Documents ----------

        # Genera el Document para usarse en el vectorstore
        docs = []
        # Generamos tambien la KB
        rows = []
        # normaliza encabezados (quita espacios rara vez, mantiene acentos)
        self.df = self.df.rename(columns=lambda c: str(c).strip())
        today = datetime.date.today()
        header_sequence = list(self.df.columns)  # mantiene el orden original

        for _, r in df_geo.iterrows():
            nombre_unidad = str(r.get("UNIDAD", "") or "").strip()
            if not nombre_unidad:
                continue

            programa = str(r.get("MODELO DE NEGOCIO", "") or "").strip()
            # Extraccion de telefonia, del campo telefono y extraer de otros campos si se encuentra (agregar manual)
            telefono_raw = (r.get("TELÉFONO", ""))
            #extra_phone_1 = str(r.get("INFORMACIÓN ADICIONAL", "") or "").strip()
            telefono_display, phones_uniq = get_phones(telefono_raw)
            #
            direccion = str(r.get("DIRECCIÓN DE UNIDAD", "") or "").strip()
            info_extra = str(r.get("INFORMACIÓN ADICIONAL", "") or "").strip()
            referencias = str(r.get("REFERENCIAS DE LA UNIDAD", "") or "").strip()
            servicios = files_utils.parse_servicios(r.get("SERVICIOS", ""))
            servicios, extra_med_info = add_services_from_staff(r, servicios) # Se verifica si faltaron servicios por agregar.
            servicios = files_utils.add_info_services(r.get('INFORMACIÓN ADICIONAL', ''), servicios) # Agrega todavia mas servicios si faltaron
            comidas = parse_comidas_from_row(r, header_sequence)

            doc_id = make_canonical_id(nombre_unidad, direccion)
            horarios_txt = files_utils.consolidate_horarios_row(r, self.df)

            estado = str(r.get("estado", "") or "").title()
            municipio = str(r.get("municipio", "") or "").title()

            row = {
                "id": doc_id,  #
                "nombre_oficial": nombre_unidad,
                # "programa": programa,
                "estado": estado,
                "municipio": municipio,
                "direccion_corta": direccion,
                "horarios_texto": horarios_txt,
                "comidas" : comidas if comidas else None,
                "telefono": telefono_display,
                "servicios_lista": servicios,
                "informacion_adicional": info_extra,
                "datos_extras_medicos": extra_med_info,
                "referencias_unidad": referencias,
                # "fuente": f"{self.database_path}:Sheet1:{self.origin_sheet}",
                # "ultima_actualizacion": today, "tenant": "ML-base",
            }

            # page_content ordenado (servicios → tipo → ubicación → ...), como definimos
            page_content = (
                f"passage: "
                f"id: {doc_id}"
                f"nombre: {nombre_unidad}"
                f"servicios: {', '.join(servicios)}"
                f"ubicacion: {municipio}, {estado}"
                # f"tipo: {programa}"
                f"direccion: {direccion}"
                f"horario: {horarios_txt}"
                f"tel: {telefono_display}"
                f"referencias: {referencias}"
            )

            meta = {
                "id": doc_id,
                "nombre_oficial": nombre_unidad,
                # "programa": programa,
                "estado": estado,
                "municipio": municipio,
                "direccion_corta": direccion,
                "horarios_texto": horarios_txt,
                "comidas" : comidas if comidas else None,
                "telefono": telefono_display,
                "servicios_lista": servicios,
                "informacion_adicional": info_extra,
                "datos_extras_medicos": extra_med_info,
                # "referencias_unidad": referencias,
            }

            docs.append(Document(page_content=page_content, metadata=meta))
            row["searchable_text"] = build_searchable_text(row)
            rows.append(row)
        kb_df = pd.DataFrame(rows)
        kb_df.head(10)
        # print(kb_df.head(10))
        return docs, kb_df, df_geo

    def build_geo_lexicons(self):
        by_state = defaultdict(lambda: defaultdict(set))
        all_munis = defaultdict(set)  # norm_muni -> set((display, estado))
        for _, r in self.kb_df.iterrows():
            est = str(r["estado"]).strip()
            mun = str(r["municipio"]).strip()
            if not est or not mun:
                continue
            ne, nm = norm_txt(est), norm_txt(mun)
            by_state[ne][nm].add(mun)  # display original
            all_munis[nm].add((mun, est))  # (display, estado)
        return by_state, all_munis

    # Metodos de extraccion de datos de la pregunta

    def extract_state_from_query(self, query: str) -> str | None:
        q = norm_txt(query)
        # búsqueda por variantes exactas contenidas en el query
        # (para evitar falsos matches, usamos límites de palabra "suaves")
        for variant_norm, canon in self.state_lookup.items():
            # construimos regex que busque la variante como token/segmento
            pat = r'(^|[\s,;:()\-])' + re.escape(variant_norm) + r'([\s,;:()\-]|$)'
            if re.search(pat, q):
                return canon  # devolvemos el nombre canónico p.ej. "tlaxcala"
        return None

    def fuzzy_find_muni_in_state(self, q: str, state_norm: str, score_cutoff=88):
        # devuelve (display_muni, estado, score) o None
        if state_norm not in self.by_state:
            return None
        candidates = list(self.by_state[state_norm].keys())  # norm munis del estado
        if not candidates:
            return None
        match = process.extractOne(
            norm_txt(q), candidates, scorer=fuzz.WRatio, score_cutoff=score_cutoff
        )
        if not match:
            return None
        nm, score, _ = match
        display = sorted(self.by_state[state_norm][nm])[0]
        # Recuperar estado en formato display (primera coincidencia)
        disp_state = None
        for d_m, d_e in self.all_munis[nm]:
            if norm_txt(d_e) == state_norm:
                disp_state = d_e
                return display, disp_state or state_norm.title(), score

    def fuzzy_find_muni_any_state(self, q: str, score_cutoff=90):
        # busca en todos los estados, útil cuando no detectaste estado
        candidates = list(self.all_munis.keys())
        match = process.extractOne(norm_txt(q), candidates, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if not match:
            return None
        nm, score, _ = match
        # si el nombre existe en varios estados, devuelve lista y pide desambiguar
        options = sorted(list(self.all_munis[nm]))
        return options, score  # [ (display_muni, display_estado), ... ], score

    # Extraer el municipio directamente de la query
    def extract_municipality_from_query(self, query: str, detected_state: str | None = None):
        qn = norm_txt(query)

        # 0) alias directos
        alias_hit = alias_lookup(qn)
        if alias_hit:
            return {"municipio": alias_hit[0], "estado": alias_hit[1], "method": "alias"}

        # 1) patrones triviales "en X", "cerca de X", "de X"
        m = re.search(r'(?:en|cerca de|de|por)\s+([a-záéíóúñ\s\.]+?)(?:\?|\.|,|$)', query, flags=re.I)
        span_txt = m.group(1) if m else query  # si no hay patrón, intentamos con todo el query

        # 2) si tenemos estado, intentamos dentro del estado
        if detected_state:
            state_norm = norm_txt(detected_state)
            hit = self.fuzzy_find_muni_in_state(span_txt, state_norm, score_cutoff=88)
            if hit:
                muni, est, score = hit
                return {"municipio": muni, "estado": est, "method": "fuzzy_in_state", "score": score}

        # 3) sin estado: fuzzy en todos los estados
        any_hit = self.fuzzy_find_muni_any_state(span_txt, score_cutoff=90)
        if any_hit:
            options, score = any_hit
            if len(options) == 1:
                muni, est = options[0]
                return {"municipio": muni, "estado": est, "method": "fuzzy_any", "score": score}
            else:
                # ambigüedad: múltiples estados
                return {"ambiguous": options, "score": score, "method": "ambiguous_any"}

        return None

    def _all_docs(self):
        """
        Devuelve la lista de `Document` indexados. Compatible con FAISS LangChain.
        """
        # Si usas LC FAISS:
        try:
            store = self.vectorstore.docstore._dict
            return list(store.values())
        except Exception:
            # Si guardaste una copia propia:
            return getattr(self.vectorstore, "docs", [])


    def _find_ume(self, max_n: int = 5):
        """
        Busca UME directamente en el docstore.
        """

        def _is_ume_meta(m) -> bool:
            name = (m.get("nombre_oficial") or "").lower()
            return ("ume" in name) or ("unidad médica y estética" in name) or (
                    "unidad medica y estetica" in name)
        docs = self._all_docs()

        hits = [d for d in docs if _is_ume_meta(d.metadata)]
        return hits[:max_n]

    def fast_filter_by_metadata(self, query: str, filtros: dict | None = None, N_min: int = 3, N_max: int = 10):
        """
        Devuelve lista de Document (sin scores) usando SOLO índices por metadatos.
        Sin uso del Retrieval. Mucho mas veloz, menos certero.
        Si no hay suficientes (>=N_min), devuelve [] y dejamos que corra el pipeline vectorial.
        USO PARA PRUEBAS en maquinas lentas
        Regresa los docs y una bandera de si se pregunto por un servicio UME o no.
        """
        filtros = filtros or {}
        # 1) Detectar intención básica
        interp = self.interpreter.parse(query)

        service = interp["service_id"]
        detected_state = interp["location"]["estado"]

        # Si es el nombre de UME, forzar filtro por nombre de la unidad=ume (y NO exigir un servicio concreto)
        if service == "ume_alternativas": # Regresa sin cambios, buscaremos solo UME
            docs = self._all_docs()
            for d in docs:
                if re.search(r"\b(SOY+\s+TU+\s+SALUD+\s+UNIDAD+\s+M[É|E]DICA+\s+Y+\s+EST[É|E]TICA+)\b", norm_txt(d.metadata.get("nombre_oficial")).upper()):
                    return d, True # UME Flag

        muni_info = interp["debug"]["geo"]["muni_info"]

        # 2) Armar “requested” tolerante (como en ask actual)
        req_state = filtros.get("estado") or (detected_state if detected_state else "")
        req_muni = filtros.get("municipio") or (
            muni_info.get("municipio") if (muni_info and "municipio" in muni_info) else "")

        st_norm = norm_txt(req_state)
        mu_norm = norm_txt(req_muni)
        svc_norm = norm_txt(service) if service else ""

        # 3) Armar sets candidatos
        sets = []
        if st_norm:
            sets.append(self.IDX_BY_STATE.get(st_norm, set()))
        if st_norm and mu_norm:
            sets.append(self.IDX_BY_MUNI.get((st_norm, mu_norm), set()))
        if svc_norm:
            sets.append(self.IDX_BY_SERVICE.get(svc_norm, set()))

        if not sets:
            return [], False  # sin filtros útiles => no hacemos fast path

        # 4) Intersección (si solo hay uno, úsalo)
        cand_ids = sets[0].copy()
        for s in sets[1:]:
            cand_ids &= s

        # Si no hay intersección, relajar municipio primero (degradación rápida)
        if not cand_ids and st_norm and svc_norm:
            cand_ids = self.IDX_BY_STATE.get(st_norm, set()) & self.IDX_BY_SERVICE.get(svc_norm, set())

        if not cand_ids:
            # como fallback mínimo: solo por estado o solo por servicio (el que exista)
            cand_ids = self.IDX_BY_STATE.get(st_norm, set()) if st_norm else self.IDX_BY_SERVICE.get(svc_norm, set())

        if not cand_ids:
            return [], False

        # 5) Heurística de orden simple (prioriza exactitud de municipio/servicio)
        def score_id(mid: str) -> int:
            d = self.ID_TO_DOC[mid]
            mm = d.metadata
            sc = 0
            if svc_norm and svc_norm in [norm_txt(x) for x in (mm.get("servicios_lista") or [])]:
                sc += 3
            if st_norm and norm_txt(mm.get("estado") or "") == st_norm:
                sc += 2
            if mu_norm and norm_txt(mm.get("municipio") or "") == mu_norm:
                sc += 2
            return sc

        mids_sorted = sorted(cand_ids, key=score_id, reverse=True)
        mids_sorted = mids_sorted[:N_max]

        # criterio de salida rápida
        if len(mids_sorted) >= N_min:
            return [self.ID_TO_DOC[mid] for mid in mids_sorted], False
        return [], False

    # Función principal, búsqueda de datos.
    def ask(self, query: str, top_k: int = 20, filtros: dict | None = None,
            max_to_show: int = 10,
            require_margin: float | None = 0.06,  # diferencia top1-top2; None para desactivar
            return_docs: bool = False,
            retrieval_mode: Literal["similarity","mmr", "bm25","hybrid"] = "hybrid", # (Opcional: Especificar el tipo de retriever)
            # Puede ser: mmr, similarity, bm25, hybrid
            hybrid_alpha: float = 0.6,
            memoria=None
            ):
        """
        - Recupera k*2 candidatos de FAISS (LangChain).
        - Aplica:
          a) Filtro por servicio si la consulta lo menciona.
          b) Filtro por metadatos (estado/municipio/programa/tipo_sede...).
          c) Umbral por percentil dinámico y mínimo absoluto.
          d) (Opcional) chequeo de margen top1-top2.
        - Devuelve lista de hasta max_to_show resultados formateados.
        """
        def get_line_output(meta):
            line = (
                f"• {meta.get('id', 'N/A')} — {meta.get('municipio', 'N/A')}, {meta.get('estado', 'N/A')} | "
                f"{meta.get('direccion_corta', 'N/A')} | Horario: {meta.get('horarios_texto', 'No disponible')} "
                f"| Tel: {meta.get('telefono', 'No disponible')} | Horario de Comidas: {meta['comidas']} | "
                f"Servicios: {', '.join(meta.get('servicios_lista', [])) or 'Consultar en sede'} | "
                f"Consulta: {meta.get('costo_consulta') or 'Consultar en sede'} | "
                f"Medicamentos: {meta.get('costo_medicamentos') or 'Consultar en sede'} "
                f"Datos sobre los médicos extra: {meta.get('datos_extras_medicos')}"
            )
            if meta.get("nota_regla"):
                line += f" — {meta['nota_regla']}"
            return line

        def ret_docs(docs_flag, reply, docs, max_show=None):
            if docs_flag:
                # devolver lista de Document (ya filtrados y ordenados)
                if docs is not None:
                    if max_show is not None: return reply, [d for (d, _) in docs[:max_show]]
                    return reply, [d for (d, _) in docs]
                else:
                    return reply, [None, None]
            else:
                return reply

        def ok(meta):
            return all(str(meta.get(k, "")).lower() == str(v).lower()
                       for k, v in filtros.items() if v)

        filtros = filtros or {}

        if memoria:
            last_docs = memoria.get_last_docs()
            # print(last_docs)
            if last_docs:
                rescored = _rescore_docs_with_query(self.embeddings, normalize_query_e5(query), last_docs)
                rescored_sorted = sorted(rescored, key=lambda x: x[1], reverse=True)

                # Aplica reglas y formato
                docs = rescored_sorted[:max_to_show]
                formatted = [get_line_output(d.metadata) for d, _ in docs]
                answer = "\n".join(formatted)

                resp = {"question": query, "answer": answer, "hits": [{"metadata": d.metadata} for d, _ in docs]}
                return (resp, [d for d, _ in docs]) if return_docs else resp

        # 1) Normalizar query para e5
        q = normalize_query_e5(query)
        q_norm = _norm_simple(query)

        interp = self.interpreter.parse(query)
        # print(interp)

        service = interp["service_id"]
        estado_interp = interp["location"]["estado"]
        muni_interp = interp["location"]["municipio"]

        # print(f"Servicio interpretado: {service} | Estado: {estado_interp} | Municipio: {muni_interp}")

        # Path alterno, Fast Path, busqueda rapida sin uso de muchos recursos.

        # 1) FAST PATH por metadatos (si habilitado)
        if self.fast_path:
            fast_docs, ume_flag = self.fast_filter_by_metadata(query, filtros, N_min=3, N_max=max_to_show * 2)
            if fast_docs:
                if ume_flag:
                    lines = [get_line_output(fast_docs.metadata)]
                    header = f"Cualquier tipo de terapias alternativas se pueden ver en UME."
                    answer = header + "\n" + "\n".join(lines)

                    resp = {"question": query,
                            "answer": answer,
                            "hits": [{"metadata": fast_docs.metadata}]}
                    return (resp, fast_docs) if return_docs else resp
                # Desde aquí puedes saltarte el vector percentile/threshold y formatear de una vez:
                enhanced = []
                for d in fast_docs[:max_to_show]:
                    m = {"metadata": d.metadata.copy()}
                    m["metadata"] = apply_business_rules(m)
                    # enhanced.append((m["metadata"], m["score"]))
                    enhanced.append(m["metadata"])

                lines = [get_line_output(m) for m in enhanced]

                header = "Opciones encontradas (vía búsqueda rápida por metadatos):"
                answer = header + "\n" + "\n".join(lines)
                resp = {"question": query,
                        "answer": answer,
                        "hits": [{"metadata": d} for d, _ in fast_docs[:max_to_show]]}
                        # "hits": [{"metadata": d} for d in fast_docs[:max_to_show]]}
                return ret_docs(docs_flag=return_docs, reply=resp, docs=fast_docs, max_show=max_to_show)

        # 2) Recuperar más (3 veces top_k) de lo necesario para filtrar
        vec_results = []
        bm25_results = []

        # --- Vectorial (FAISS + embeddings) ---
        if retrieval_mode in ("similarity", "mmr", "hybrid"):
            if retrieval_mode == "similarity" or retrieval_mode == "hybrid":
                vec_results = self.vectorstore.similarity_search_with_score(
                    q, k=max(40, top_k * 3))
            else:  # "mmr" o "hybrid"
                mmr_docs = self.vectorstore.max_marginal_relevance_search(
                    q, k=top_k, fetch_k=max(40, top_k * 3), lambda_mult=0.7,)
                if not mmr_docs and retrieval_mode != "hybrid":
                    resp = {"question": query, "answer": "No encontré resultados.", "hits": []}
                    return ret_docs(docs_flag=return_docs, reply=resp, docs=mmr_docs, max_show=max_to_show)

                if mmr_docs:
                    vec_results = _rescore_docs_with_query(self.embeddings, q, mmr_docs)

        # --- BM25 puro ---
        if retrieval_mode in ("bm25", "hybrid"):
            bm25_results = self._bm25_search(query, k=max(40, top_k * 3))

        # --- Seleccionar / fusionar resultados según modo ---
        if retrieval_mode == "similarity" or retrieval_mode == "mmr":
            results = vec_results
        elif retrieval_mode == "bm25":
            results = bm25_results
        elif retrieval_mode == "hybrid":
            # si uno de los dos viene vacío, usar el que sí tenga datos
            results = self._hybrid_fusion_rrf(
                vec_results=vec_results,
                bm25_results=bm25_results,
                k_out=max(40, top_k * 3),
                alpha=hybrid_alpha,
            ) if vec_results and bm25_results else vec_results or bm25_results
        else:
            results = []

        if not results:
            resp = {"question": query, "answer": "No encontré resultados.", "hits": []}
            return ret_docs(docs_flag=return_docs, reply=resp, docs=results, max_show=max_to_show)

        # 3) Filtrar por servicio si corresponde (Filtrado por servicio de la misma query)
        # print("Servicio a dar (Interpreter): " + str(service))

        # Si es el nombre de UME, forzar filtro por nombre de la unidad=ume (y NO exigir un servicio concreto)
        is_ume_intent = service in SERVICE_LEXICON["ume_alternativas"]["syn"]
        if is_ume_intent or service == "ume_alternativas":
            # Canonicaliza estado (si usas tu helper)
            ume_docs = self._find_ume(max_n=max_to_show)

            if not ume_docs:
                # Sin UME en ninguna parte (raro). Devuelve mensaje claro.
                msg = "No encontré una Unidad Médica y Estética (UME) en la base."
                return ({"question": query, "answer": msg, "hits": []}, []) if return_docs else {"question": query,
                                                                                                 "answer": msg,
                                                                                                 "hits": []}
            lines = [get_line_output(d.metadata) for d in ume_docs]
            header = "Cualquier tipo de terapias alternativas se pueden ver en UME."
            answer = header + "\n" + "\n".join(lines)

            resp = {"question": query,
                    "answer": answer,
                    "hits": [{"metadata": d.metadata, "score": 100} for d in ume_docs]}

            return (resp, ume_docs) if return_docs else resp

        if service:
            results = [(d, s) for (d, s) in results if service in (d.metadata.get("servicios_lista"))]
            if not results:
                resp = {"question": query, "answer": f"No encontré sedes con el servicio solicitado ({service}).", "hits": []}
                return ret_docs(docs_flag=return_docs, reply=resp, docs=results, max_show=max_to_show)

        # 4) Filtrar por metadatos exactos (post-filtro)
        amb_active = "ambig_state_muni" in interp["flags"]["degradaciones"]
        amb = interp["debug"]["geo"]["amb"]

        detected_state = estado_interp
        muni_info = interp["debug"]["geo"]["muni_info"]

        if detected_state and not filtros.get("estado"):
            filtros["estado"] = detected_state

        if muni_info and muni_interp:
            filtros["municipio"] = muni_interp
            if not filtros.get("estado") and muni_info.get("estado"):
                filtros["estado"] = muni_info["estado"]

        if filtros:
            results = [(d, s) for (d, s) in results if ok(d.metadata)]
            if not results:
                resp = {"question": query, "answer": "No encontré resultados con esos filtros.", "hits": []}
                return ret_docs(docs_flag=return_docs, reply=resp, docs=results, max_show=max_to_show)

        # 5) Percentil dinámico + mínimo absoluto
        # Manejar si son pocos resultados
        n = len(results)
        scores = np.array([s for _, s in results], dtype=float)

        if retrieval_mode in ("bm25", "hybrid"):
            kept_sorted = sorted(results, key=lambda x: x[1], reverse=True)
            threshold = 0.0
        else:
            if n <= 5:
                relaxed_min = max(0.36, self.score_threshold - 0.07)
                threshold = max(relaxed_min, self.score_threshold)
                kept = [(d, s) for (d, s) in results if s >= relaxed_min]
            else:
                perc_cut = float(np.quantile(scores, self.percentile))
                threshold = max(perc_cut, self.score_threshold)
                kept = [(d, s) for (d, s) in results if s >= threshold]
            if not kept:
                fallback = float(np.quantile(scores, 0.60)) if len(scores) else self.score_threshold
                threshold2 = max(fallback, self.score_threshold)
                kept = [(d, s) for (d, s) in results if s >= threshold2]
                if not kept:
                    resp = {"question": query, "answer": "No encontré resultados con suficiente confianza.", "hits": []}
                    return ret_docs(docs_flag=return_docs, reply=resp, docs=kept, max_show=max_to_show)

            if not kept:
                # si quedó vacío, relajar un poco al percentil 0.5 como fallback
                resp = {"question": query, "answer": "No encontré resultados con suficiente confianza.", "hits": []}
                return ret_docs(docs_flag=return_docs, reply=resp, docs=kept, max_show=max_to_show)

            kept_sorted = sorted(kept, key=lambda x: x[1], reverse=True)


        # --- boosts (opcional pero recomendado)
        requested_state = filtros.get("estado") or (detected_state or "")
        requested_muni = filtros.get("municipio") or (muni_interp or "")

        state_for_boost = requested_state or (detected_state or None)
        muni_for_boost = requested_muni or (muni_interp or None)

        kept_sorted = apply_small_boosts(
            kept_sorted,
            service=service,
            state=state_for_boost,
            municipality=muni_for_boost,
        )

        # aplicar reglas de negocio
        enhanced = []
        for d, s in kept_sorted:
            m = {"metadata": d.metadata.copy(), "score": float(s)}
            m["metadata"] = apply_business_rules(m)
            enhanced.append((m["metadata"], m["score"]))

        # margen top1-top2 para decidir si pedir aclaración
        suggest = ""
        if require_margin is not None and len(enhanced) >= 2:
            margin = enhanced[0][1] - enhanced[1][1]
            if margin < require_margin:
                suggest = " ¿Podrías especificar el municipio?"

        # nota por ambigüedad (estado vs municipio)
        if amb_active and not self.geo.has_explicit_muni_marker(q_norm):
            suggest = suggest + (f"\nNota: '{amb['token'].title()}' puede referirse al estado o al municipio de "
                    f"{amb['muni']}. Por ahora te muestro opciones en el estado. "
                    f"¿Te refieres al estado completo o al municipio de {amb['muni']}?").strip()

        # nota de degradación visible, para mostrar mas resultados cuando no sale ninguno, esta muy pobre. No implmentar.
        # degradation_note = ""
        # if degraded:
        #     if req_muni_canon:
        #         degradation_note = " (relajé el filtro de municipio para mostrar opciones más cercanas)."
        #     else:
        #         degradation_note = " (relajé filtros para no dejarte sin opciones)."

        lines = [get_line_output(m) for m, _ in enhanced[:max_to_show]]
        # Output del sistema.

        #header = f"Opciones encontradas (umbral={threshold:.2f}, percentil={percentile * 100:.0f}%){degradation_note}:"
        header = f"Opciones encontradas (umbral={threshold:.2f}, percentil={self.percentile * 100:.0f}%):"
        answer = header + "\n" + "\n".join(lines) + suggest

        resp = {"question": query,
                    "answer": answer,
                    "hits": [{"metadata": d.metadata, "score": float(s)} for d, s in kept_sorted[:max_to_show]]}
        return ret_docs(docs_flag=return_docs, reply=resp, docs=kept_sorted, max_show=max_to_show)



def build_searchable_text(row: dict) -> str:
    return " | ".join(filter(None, [
        row["id"],
        # row["programa"],
        f"{row['municipio']}, {row['estado']}",
        row["direccion_corta"],
        f"horario: {row['horarios_texto']}",
        f"tel: {row['telefono']}",
        "servicios: " + ", ".join(row["servicios_lista"])
    ]))

def normalize_query_e5(q: str) -> str:
    # puedes agregar reemplazos de sinónimos si deseas
    return "query: " + q.strip()

def alias_lookup(q_norm: str):
    for k, (mun, est) in MUNICIPALITY_ALIASES.items():
        if k in q_norm:
            return mun, est
    return None

def _finalize_geo(df_in: pd.DataFrame, geo: geo_location.GeoFuzzy, min_state=60, min_muni=85) -> pd.DataFrame:
    estados, municipios, dbg = [], [], []
    for _, r in df_in.iterrows():
        addr = str(r.get("DIRECCIÓN DE UNIDAD","") or "")
        st, mu, d = geo.extract_from_address(addr)
        # aplica umbrales
        st_ok = st if (d.get("state_score", 0) >= min_state) else ""
        mu_ok = mu if (d.get("muni_score", 0) >= min_muni) else ""
        #

        if not st_ok:
            nombre_unidad = str(r.get("UNIDAD", "") or "").strip()
            st_name, mu_name, _ = geo.extract_from_address(norm_txt(nombre_unidad))
            # Umbral Nuevamente
            st_ok = st if (d.get("state_score", 0) >= min_state) else ""
            mu_ok = mu if (d.get("muni_score", 0) >= min_muni) else ""

        estados.append(st_ok or "")
        municipios.append(mu_ok or "")
        dbg.append(d)

    df = df_in.copy()
    df["estado"] = estados
    df["municipio"] = municipios
    df["_geo_debug"] = dbg  # útil en pruebas; puedes quitarlo luego
    return df

# -------------------------------------------------------------

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity entre dos vectores 1D."""
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))

def _rescore_docs_with_query(embeddings_model, query_text: str, docs: list):
    """
    Re-scorea una lista de Documents devolviendo [(doc, score_cosine)].
    - Usa el MISMO modelo de embeddings que el vectorstore (consistencia).
    - Para pocos docs (<=50) es suficientemente rápido.
    """
    q_vec = np.array(embeddings_model.embed_query(query_text), dtype=float)
    # embed_documents puede ser costoso para muchos; aquí son pocos
    doc_vecs = embeddings_model.embed_documents([d.page_content for d in docs])
    pairs = []
    for d, v in zip(docs, doc_vecs):
        score = _cos_sim(q_vec, np.array(v, dtype=float))
        pairs.append((d, score))
    # ordenar mayor a menor (cosine más alto es mejor)
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def apply_small_boosts(pairs,  # list[(Document, score_float)] de los results
                       service: str | None = None,
                       state: str | None = None,
                       municipality: str | None = None) -> list[tuple]:
    """
    Aplica pequeños boosts al score del documento según match exacto en metadata.
    Retorna lista [(doc, new_score)].
    """
    BOOST_SERVICE = 0.03
    BOOST_MUNI    = 0.02
    BOOST_STATE   = 0.01

    out = []
    for d, s in pairs:
        meta = getattr(d, "metadata", {}) or {}
        score = float(s)

        if service:
            svcs = meta.get("servicios_lista") or []
            if isinstance(svcs, (list, tuple)) and service in svcs:
                score += BOOST_SERVICE

        if municipality:
            if str(meta.get("municipio","")).lower() == str(municipality).lower():
                score += BOOST_MUNI

        if state:
            if str(meta.get("estado","")).lower() == str(state).lower():
                score += BOOST_STATE

        out.append((d, score))

    # Reordenar por nuevo score
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _norm_simple(s: str) -> str:
    import unicodedata
    if s is None: return ""
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+"," ", s.lower()).strip()

