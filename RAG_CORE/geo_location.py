from typing import Dict, List, Tuple, Optional
from rapidfuzz import process, fuzz

from RAG_CORE.rag_utils.search_utils import *

"""
# Features a desarrollar
- Agregar sistema de medición de distancias entre estados.
"""

class GeoFuzzy:
    def __init__(
        self,
        state_list: Dict[str, List[str]],
        muni_by_state = None,
        municipality_aliases = None,
        ambiguity_list = None,
        state_score_cut: int = 90,
        muni_score_cut: int = 85,
    ):
        """
        state_aliases: {"nuevo leon":["nuevo leon","nl","nvo leon",...], "cdmx":[...], ...}
        muni_by_state: {"nuevo leon":["monterrey","san pedro garza garcia",...], "cdmx":[...]}
        municipality_aliases: { simil: ('full', estado)}
        """


        # lookup alias -> canon
        self.state_list = state_list
        self.ambiguity_list = ambiguity_list

        self.state_lookup = {}
        for canon, aliases in state_list.items():
            for a in aliases + [canon]:
                self.state_lookup[norm_txt(a)] = norm_txt(canon)

        self.state_candidates = sorted(set(self.state_lookup.values()))

        # 🔧 normaliza llaves y valores del catálogo por estado
        self.muni_by_state = {}
        if muni_by_state: #  -> Estado: {alias, (muni_canon, estado)}
            for est, aliases in muni_by_state.items():
                muni = {}
                for k, (m, e) in aliases.items():
                    muni[norm_txt(k)] = (norm_txt(m), norm_txt(e))
                self.muni_by_state[norm_txt(est)] = muni

        # alias municipio -> (muni_canon, estado_canon)
        self.muni_alias = {}
        if municipality_aliases:
            for k, (m, e) in municipality_aliases.items():
                self.muni_alias[norm_txt(k)] = (norm_txt(m), norm_txt(e))

        self.state_score_cut = state_score_cut
        self.muni_score_cut = muni_score_cut

    def detect_state(self, text: str) -> Tuple[Optional[str], int]:
        tokens = set(text.split())
        for alias_norm, canon in self.state_lookup.items():
            if ' ' in alias_norm:
                if alias_norm in text:
                    return canon, 100
            else:
                if alias_norm in tokens:
                    return canon, 100

        cand, score, _ = process.extractOne(text, self.state_candidates, scorer=fuzz.token_set_ratio)
        if score >= self.state_score_cut:
            return cand, score
        return None, score

    def detect_municipality_alias_first(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Si aparece un alias de municipio, devuelve (municipio_canon, estado_canon).
        Verifica que no sea una calle y si parece ser de CDMX solamente lo acepta si dice CDMX
        """
        q = clean_address(text) # ya esta normalizado, sin cosas extras
        parts_clean = split_segments(q)
        segment = (len(parts_clean)//2+1)
        ## Primero Buscar palabra "municipio"
        muni_visible, m = _muni_marker_location(text.lower()) # Uso de text, debido a que tiene toda la direccion
        if m:
            new_add = text[m.end():]
            for alias, (muni, est) in self.muni_alias.items():
                if alias in new_add:
                    return muni, est

        # busca alias como substring
        # Alias es el string a buscar, muni es el correcto
        for alias, (muni, est) in self.muni_alias.items():
            if alias in q:
                m = re.search(fr"\b{re.escape(alias)}\b", q)
                if m:
                    ## Detectar si el municipio/estado esta en la lista de ambiguedad, de ser así,
                    ## Remover el estado canónico y buscar y hay otro municipio, si no, dejar el que estaba.
                    ambiguous = self.ambiguity_list.get(norm_txt(est))
                    if ambiguous:
                        edo_alias = self.state_list.get(norm_txt(est))
                        edo_alias = alias_token_pattern(edo_alias)
                        n = edo_alias.search(q)
                        if n:
                            q = (q[:n.start()] + " " + q[n.end():]).strip()
                            parts_clean = split_segments(q)
                    if looks_like_street_context(text, m.span()):
                        if not find_token_in_tail(parts_clean, alias, segment):
                            return None, None
                    elif not looks_like_street_context(text, m.span()): # Busqueda del municipio al final de la addr
                        (muni_tmp,_) = prefer_tail_state(parts_clean, self.muni_alias, segment, estado=False) # Regresa lugar,score
                        if muni_tmp is not None:
                            if not verify_cdmx(muni, est, q):
                                return None, None
                            muni = muni_tmp
                        # else:
                        #     print("Failing tail state")
                else:
                    if est=="cdmx" and not verify_cdmx(muni, est, q):
                        return None, None
                return muni, est
        return None, None

    def detect_municipality(self, text: str, state_canon: str):
        pool = self.muni_by_state.get(norm_txt(state_canon)) # Obtiene una pool de los aliases del estado.
        ambiguous = self.ambiguity_list.get(norm_txt(state_canon))
        if ambiguous:
            edo_alias = self.state_list.get(norm_txt(state_canon))
            edo_alias = alias_token_pattern(edo_alias)
            m = edo_alias.search(text)
            if m:
                text = (text[:m.start()] + " " + text[m.end():]).strip()
        if not pool:
            print("Estado no encontrado")
            return None, 0, []
        # 1) substring exacto
        for alias, (m, e) in pool.items():
            if alias in text:
                top = process.extract(text, pool, scorer=fuzz.token_set_ratio, limit=5)
                return m, 100, [(mm, sc) for mm, sc, _ in top]
        # 2) fuzzy
        best, score, _ = process.extractOne(text, pool, scorer=fuzz.token_set_ratio)
        if score >= self.muni_score_cut:
            top = process.extract(text, pool, scorer=fuzz.token_set_ratio, limit=5)
            return best, score, [(mm, sc) for mm, sc, _ in top]
        return None, score, []

    def extract_from_address(self, address: str):
        """
        Devuelve (estado_canon, municipio_canon, debug)
        Estrategia:
          1) Segmentar por comas y leer de derecha a izquierda.
          2) Estado: prioriza cola con MX_STATES (variantes/abreviaturas).
          3) Municipio: alias primero; si no, fuzzy dentro del estado, por cola de segmentos.
          4) Anti-calle: sólo descartar si el match está en un segmento que COMIENZA con vialidad.
        """
        debug = {"path": "", "state_score": 0, "muni_score": 0, "candidates": []}

        addr_clean = norm_txt(address or "")
        parts_clean = split_segments(addr_clean)  # segmentos crudos (para anti-calle por segmento)
        segments = (len(parts_clean)//2+1)

        # 0) Municipio por alias (en TODOS los segmentos, de derecha a izquierda)
        m_alias, e_alias = self.detect_municipality_alias_first(addr_clean)
        if m_alias and e_alias:
            debug.update(path="municipio_alias", state_score=100, muni_score=100)
            return e_alias, m_alias, debug

        # 1) Estado por cola (preferente)
        st_tail, st_tail_score = prefer_tail_state(parts_clean, self.state_list, segments)

        # 2) Estado general (tu heurística regular, pero sobre el texto normalizado)
        st_gen, st_gen_score = self.detect_state(addr_clean)

        # Decidir estado
        state = st_tail or st_gen
        debug["state_score"] = 100 if st_tail else st_gen_score

        # Anti-calle para estado: sólo descartar si el estado aparece en un segmento que empieza con vialidad
        if state:
            m = re.search(fr"\b{re.escape(state)}\b", addr_clean)
            if m and looks_like_street_context(addr_clean, m.span()):
                # si NO aparece en la cola limpia, trátalo como calle
                if not find_token_in_tail(parts_clean, state, segments):
                    state = None
                    debug["state_score"] = 0

        if not state:
            debug["path"] = "no_state"
            return None, None, debug

        # 3) Municipio dentro del estado (prioriza cola de segmentos)
        #    Usa tu detect_municipality(texto, estado) pero pásale SOLO la cola
        muni, muni_score, top = self.detect_municipality(" | ".join(parts_clean), state)

        # Si no encontró, intenta con el string normalizado (pero cuidado con falsos)
        if not muni:
            muni, muni_score, top = self.detect_municipality(addr_clean, state)

        # Anti-calle para municipio: igual regla, por segmento
        if muni:
            m = re.search(fr"\b{re.escape(muni)}\b", addr_clean)
            if m and looks_like_street_context(addr_clean, m.span()):
                if not find_token_in_tail(parts_clean, muni, segments):
                    # muni, muni_score, top = None, 0, []
                    pass
            if _is_cdmx_borough(muni):
                if state and norm_txt(state) == "cdmx" and not mentions_cdmx(address):
                    muni, muni_score, top = None, 0, []  # ignorar ese alias
        debug.update(path="state_then_muni", muni_score=muni_score, candidates=top)
        return state, muni, debug

    def detect_state_muni_ambiguity(self, query_norm: str):
        """
        Devuelve dict con {'state','muni','token'} si la query contiene un nombre
        que puede ser ESTADO y MUNICIPIO, sin pistas de 'estado'/'municipio'.
        """
        # pistas que anulan la ambigüedad
        if re.search(r"\b(estado|edo\.?|e\.?do\.?)\b", query_norm):
            return None
        if re.search(r"\b(municipio|ciudad|capital|mpio\.?)\b", query_norm):
            return None

        for st, muni in self.ambiguity_list.items():
            if re.search(rf"\b{re.escape(st)}\b", query_norm):
                return {"state": st.title(), "muni": muni.title(), "token": st}
        return None

    @staticmethod
    def has_explicit_muni_marker(q_norm: str) -> bool:
        # pistas que indican que sí quiso municipio/ciudad
        return bool(re.search(_MUNICIPIO_TAG_RE, q_norm))


CDMX_TOKENS = r"(cdmx|ciudad\s+de\s+m[e]xico|d\.?f\.?|df|mexico\s+city)"
CDMX_PAT = re.compile(rf"\b{CDMX_TOKENS}\b", re.IGNORECASE)

def mentions_cdmx(text_norm: str) -> bool:
    m = re.search(CDMX_PAT, text_norm)
    if m:
        return True
    else:
        return False

def _is_cdmx_borough(alcaldia: str) -> bool:
    return norm_txt(alcaldia) in CDMX_BOROUGHS # Verifica si la alcaldia es parte de la CDMX

def verify_cdmx(alcaldia: str, estado: str, direccion: str) -> bool:
    # --- ANTI-CDMX ---
    # Si el alias apunta a una alcaldía CDMX pero:
    #   - ya detectaste un estado y NO es CDMX, y
    #   - el texto NO menciona CDMX,
    # entonces descártalo (probablemente es una calle o colonia).
    if _is_cdmx_borough(alcaldia): # Verifica si la alcaldia es parte de la CDMX
        if estado and norm_txt(estado) == "cdmx" and not mentions_cdmx(direccion):
            return False  # ignorar ese alias pues no pertenece a CDMX
        elif estado and norm_txt(estado) == "cdmx" and mentions_cdmx(direccion):
            return True # Si es municipio de CDMX y menciona la CDMX
        else:
            return False
    else:
        # CDMX es el unico lugar donde se mapean colonias a una delegacion (por ahora)
        return False

CDMX_BOROUGHS = {
# Álvaro Obregón
    "alvaro obregon": ("alvaro obregon","cdmx"),
    "a obregon": ("alvaro obregon","cdmx"),
    "ao": ("alvaro obregon","cdmx"),
    # Azcapotzalco
    "azcapotzalco": ("azcapotzalco","cdmx"),
    "azca": ("azcapotzalco","cdmx"),
    # Benito Juárez
    "benito juarez": ("benito juarez","cdmx"),
    "benito juares": ("benito juarez","cdmx"),
    "bj": ("benito juarez","cdmx"),
    # Coyoacán
    "coyoacan": ("coyoacan","cdmx"),
    # Cuajimalpa
    "cuajimalpa": ("cuajimalpa de morelos","cdmx"),
    "cuajimalpa de morelos": ("cuajimalpa de morelos","cdmx"),
    # Cuauhtémoc
    "cuauhtemoc": ("cuauhtemoc","cdmx"),
    "Roma" : ("cuauhtemoc","cdmx"),
    # Gustavo A. Madero
    "gustavo a madero": ("gustavo a. madero","cdmx"),
    "gustavo a. madero": ("gustavo a. madero","cdmx"),
    "g a madero": ("gustavo a. madero","cdmx"),
    "gam": ("gustavo a. madero","cdmx"),
    # Iztacalco
    "iztacalco": ("iztacalco","cdmx"),
    # Iztapalapa
    "iztapalapa": ("iztapalapa","cdmx"),
    "izta": ("iztapalapa","cdmx"),
    # Magdalena Contreras
    "magdalena contreras": ("magdalena contreras","cdmx"),
    # Miguel Hidalgo
    "miguel hidalgo": ("miguel hidalgo","cdmx"),
    "mh": ("miguel hidalgo","cdmx"),
    # Milpa Alta
    "milpa alta": ("milpa alta","cdmx"),
    # Tláhuac
    "tlahuac": ("tlahuac","cdmx"),
    # Tlalpan
    "tlalpan": ("tlalpan","cdmx"),
    # Venustiano Carranza
    "venustiano carranza": ("venustiano carranza","cdmx"),
    "v carranza": ("venustiano carranza","cdmx"),
    "vcarranza": ("venustiano carranza","cdmx"),
    "vc": ("venustiano carranza","cdmx"),
    # Xochimilco
    "xochimilco": ("xochimilco","cdmx"),
}

_MUNICIPIO_TAG_RE = re.compile(
    r'\b(municipio|mpio\.?|mcpio\.?|muni)\b',
    flags=re.IGNORECASE
)


def _muni_marker_location(q_norm: str):
    if GeoFuzzy.has_explicit_muni_marker(q_norm):
        m = re.search(_MUNICIPIO_TAG_RE, q_norm)
        return True, m
    else:
        return False, None