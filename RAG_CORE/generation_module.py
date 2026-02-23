import asyncio
import re

import tiktoken
# import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import time

from mistralai import Mistral
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#################################Augmentation and Generation ##########################
# Planteamiento del Modelo de LLM
def _try_tiktoken_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

_enc = _try_tiktoken_encoding()

def count_tokens(text: str) -> int:
    """Cuenta tokens usando tiktoken si está disponible; si no, aproxima por palabras."""
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    # fallback: aprox 1 token ≈ 0.75 palabras en español
    return int(len(text.split()) / 0.75) + 1


class GenerationModuleLlama:
    def __init__(self, llm_model):# , device="cuda" if torch.cuda.is_available() else "cpu", configFile = None):
        """
        :param model_name: model name or model path
        :param device: if gpu his available
        """
        self.debug = None
        self.memoria = None
        self.initial_prompt = None
        self.retrieval = None
        self.follow_up_model = None
        self.llm_model = llm_model
        self.memoria = None
        # print(f"Module Created!, gpu layers: {gpu_layers}.")

    def initialize(self, initial_prompt=None, retrieval=None, debug=False, max_tokens=1024):
        if initial_prompt is not None:
            self.initial_prompt = initial_prompt
        else:
            self.initial_prompt = INIT_PROMPT_LLAMA
        self.retrieval = retrieval
        self.debug = debug
        self.follow_up_model = IntentionDetector(retrieval=self.retrieval, threshold=0.5, debug=debug)
        self.memoria = ConversationalMemory(max_tokens=max_tokens)


    def build_llama2_prompt(self, context: str, question: str, historial: str = None) -> str:
        # Plantilla oficial LLaMA-2 chat
        if historial is not None:
            return (
                f"[INST] <<SYS>>{self.initial_prompt}<< / SYS >>"
                f"# HISTORIAL: {historial}"
                f"# CONTEXTO: {context}"
                f"# PREGUNTA: {question}. RECUERDA: UN SOLO PÁRRAFO, NADA DE LISTAS, NADA DE NÚMEROS. (ejemplo: decir 'uno' en lugar de '1')."
                f"[/ INST]"
            )
        else:
            return (
                f"[INST] <<SYS>>\n{self.initial_prompt.strip()}\n<</SYS>>\n\n"
                f"# CONTEXTO\n{context.strip()}\n\n"
                f"# PREGUNTA\n{question.strip()}. RECUERDA: UN SOLO PÁRRAFO, NADA DE LISTAS, NADA DE NÚMEROS. (ejemplo: decir 'uno' en lugar de '1')."
                f"[/INST]\n"
            )


    ## Función principal de respuestas interpretadas por el sistema RAG
    def rag_answer(self, query: str):
        """
        Regresa la respuesta del sistema LLM y un bool indicando si ya termino la interaccion (True) o si continua (False).
        """
        # Uso de nuestra función ask previamente desarrollada. - Retrieval
        query = query.lower()

        # Detección simple de fin de sesión
        if self.follow_up_model.detect_exit_intent(query):
            self.memoria.clear()
            return "Ha sido un gusto ayudarte. ¡Que tengas buen día! ¡Adiós!", True

        docs = self.memoria.get_last_docs() # revisamos si hay informacion previa
        used_cached_docs = docs is not None and len(docs) > 0

        follow_up = self.follow_up_model.is_follow_up(query, self.memoria.get_recent_turns())
        # Verificación adicional: ¿el usuario cambió de intención a pesar de ser follow-up?
        if follow_up and not self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns()):
            if self.debug:
                print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
            follow_up = False  # Forzamos modo nueva búsqueda
            docs = []  # Limpiamos docs previos para no arrastrar contexto viejo
        if self.debug:
            print("Follow up: ", follow_up)

        if not follow_up:
            # Validamos si el usuario sigue en la misma intención, aunque no sea un follow-up directo
            if self.should_continue_context(query, used_cached_docs):
                follow_up = True
                if self.debug:
                    print(
                        "[DEBUG] No se detectó follow-up directo, pero sí continuidad semántica. Se mantiene contexto.")
            else:
                resp, docs = self.retrieval.ask(query, return_docs=True, memoria=None)

                if not docs:
                    self.memoria.add_turn("user", query)
                    self.memoria.add_turn("assistant", "No encontré información suficiente en la base.")
                    return "No encontré información suficiente en la base.", False

                self.memoria.set_last_docs(docs)
                if self.debug:
                    print("[DEBUG] Nueva búsqueda realizada, se guardan nuevos documentos.")
                self.initial_prompt = INIT_PROMPT_LLAMA

        else:
            # 4. En caso de follow-up, usar los documentos previos
            if not used_cached_docs:
                if self.debug:
                    print("[DEBUG] Follow-up detectado, pero no hay documentos previos.")
                self.memoria.add_turn("user", query)
                self.memoria.add_turn("assistant", "No tengo contexto previo suficiente.")
                return "¿Podrías especificar a qué unidad o tema te refieres?", False

            matches = self.follow_up_model.match_sucursal_from_input(user_input=query, docs=docs)
            if matches:
                docs = [doc for doc, _ in matches]
                self.memoria.set_last_docs(docs)
                self.initial_prompt = DETAILS_PROMPT
                if self.debug:
                    print(f"[DEBUG] Se detectaron {len(matches)} coincidencias por follow-up:")
                    for doc in docs:
                        nombre = doc.metadata.get("nombre_oficial", "Sin nombre")
                        municipio = doc.metadata.get("municipio", "Sin municipio")
                        estado = doc.metadata.get("estado", "Sin estado")
                        print(f" - {nombre} ({municipio}, {estado})")
            else:
                # Solo considerar cambio de intención si NO hubo match
                is_same_topic = self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns())
                if self.debug:
                    print(f"[DEBUG] Similitud semántica entre queries: {is_same_topic}")
                if not is_same_topic:
                    if self.debug:
                        print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
                    follow_up = False
                    docs = []
                    self.memoria.set_last_docs([])
                else:
                    self.initial_prompt = CONTINUOUS_PROMPT_LLAMA
                    if self.debug:
                        print("[DEBUG] Follow-up válido sin coincidencia específica. Se mantiene el contexto actual.")

        context = build_context_from_docs(docs, full=follow_up)

        historial = "\n".join([f"{t['role'].title()}: {t['content']}" for t in self.memoria.get_recent_turns()])
        context_tokens = count_tokens(context)
        historial_tokens = count_tokens(historial)
        self.memoria.trim_if_exceeds_tokens(context_tokens, historial_tokens)

        # Armado de prompt
        prompt_value = self.build_llama2_prompt(context=context, question=query, historial=historial)

        if self.debug:
            print("Finished Context, tokens number: ", count_tokens(prompt_value))

        # Generation
        t0 = time.perf_counter()
        out = self.llm_model(
            prompt=prompt_value,
            stop=["</s>"],
            max_tokens=512,
            echo=False,
            stream=False
        )
        t1 = time.perf_counter()

        ## Debugging
        if self.debug:
            print("Finished invoke, time: ", t1 - t0, " s")
            print("Prompt completo:\n", prompt_value)
            # print("Respuesta tokens:", len(out["choices"][0]["text"].split()))
            # print("Respuesta completa:\n", out["choices"][0]["text"])

        try:
            out_text = out["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[ERROR] No se pudo extraer texto de la salida del modelo: {e}")
            out_text = resp.get("answer", "") or str(out)  # último recurso
        # Actualizar memoria
        self.memoria.add_turn("user", query)
        self.memoria.add_turn("assistant", out_text)

        return out_text, False

    ## Función de respuestas rag optimizada para llamadas en tiempo real
    async def rag_answer_stream(self, query: str):
        """
        Regresa la respuesta del sistema LLM y un bool indicando si ya termino la interaccion (True) o si continua (False).
        """
        query = query.lower()

        docs = self.memoria.get_last_docs()  # revisamos si hay informacion previa
        used_cached_docs = docs is not None and len(docs) > 0

        follow_up = self.follow_up_model.is_follow_up(query, self.memoria.get_recent_turns())
        # Verificación adicional: ¿el usuario cambió de intención a pesar de que parece ser follow-up?
        if follow_up and not self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns()):
            if self.debug:
                print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
            follow_up = False  # Forzamos modo nueva búsqueda
            docs = []  # Limpiamos docs previos para no arrastrar contexto viejo
        if self.debug:
            print("Follow up: ", follow_up)
        loop = asyncio.get_event_loop()
        if not follow_up:
            # Validamos si el usuario sigue en la misma intención, aunque no sea un follow-up directo
            if self.should_continue_context(query, used_cached_docs):
                follow_up = True
                resp, docs = await loop.run_in_executor(None, lambda: self.retrieval.ask(query, return_docs=True, memoria=self.memoria))
                # resp, docs = self.retrieval.ask(query, return_docs=True, memoria=self.memoria)
                if self.debug:
                    print(
                        "[DEBUG] No se detectó follow-up directo, pero sí continuidad semántica. Se mantiene contexto.")
            else:
                resp, docs = await loop.run_in_executor(None, lambda: self.retrieval.ask(query, return_docs=True,
                                                                                         memoria=self.memoria))
                is_relevant = any(doc.metadata.get('score', 1.0) <= 0.50 for doc in docs) if docs else False  # 0.0 es max relevant

                has_lexicon_match = self.follow_up_model.is_service_in_scope(query)

                if has_lexicon_match or not is_relevant:
                    if self.debug:
                        print(f"[DEBUG] No hizo match con los temas especificados: {has_lexicon_match} o"
                              f" no fue relevante: {is_relevant}")
                    yield {"text": "out_of_scope", "end_session": False}
                    return

                if not docs or not is_relevant:
                    self.memoria.add_turn("user", query)
                    self.memoria.add_turn("assistant", "No encontré información suficiente en la base.")
                    if self.debug:
                        tmp_docs = True if docs else False
                        score_tmp = 1.0 if not docs else max(doc.metadata.get('score', 1.0) for doc in docs)
                        print(f"[DEBUG] No se encontró información suficiente debido a la falta de docs ({tmp_docs})"
                              f"o a is_relevant ({is_relevant}) con score: {score_tmp}")
                    yield {"text": "No encontré información suficiente en la base.", "end_session": False}
                    return

                self.memoria.set_last_docs(docs)
                if self.debug:
                    print("[DEBUG] Nueva búsqueda realizada, se guardan nuevos documentos.")
                self.initial_prompt = INIT_PROMPT_LLAMA

        else:
            # 4. En caso de follow-up, usar los documentos previos
            if not used_cached_docs:
                if self.debug:
                    print("[DEBUG] Follow-up detectado, pero no hay documentos previos.")
                self.memoria.add_turn("user", query)
                self.memoria.add_turn("assistant", "No tengo contexto previo suficiente.")
                # yield {"text": "Podrías especificar a qué unidad o tema te refieres?", "end_session": False}
                follow_up = False
                # loop = asyncio.get_event_loop()
                resp, docs = await loop.run_in_executor(None, lambda: self.retrieval.ask(query, return_docs=True, memoria=None))
                # resp, docs = self.retrieval.ask(query, return_docs=True, memoria=None)

            matches = self.follow_up_model.match_sucursal_from_input(user_input=query, docs=docs)
            if matches:
                docs = [doc for doc, _ in matches]
                self.memoria.set_last_docs(docs)
                self.initial_prompt = DETAILS_PROMPT
                if self.debug:
                    print(f"[DEBUG] Se detectaron {len(matches)} coincidencias por follow-up:")
                    for doc in docs:
                        nombre = doc.metadata.get("nombre_oficial", "Sin nombre")
                        municipio = doc.metadata.get("municipio", "Sin municipio")
                        estado = doc.metadata.get("estado", "Sin estado")
                        print(f" - {nombre} ({municipio}, {estado})")
            else:
                # Solo considerar cambio de intención si NO hubo match
                is_same_topic = self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns())
                if self.debug:
                    print(f"[DEBUG] Similitud semántica entre queries: {is_same_topic}")
                if not is_same_topic:
                    if self.debug:
                        print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
                    follow_up = False
                    docs = []
                    self.memoria.set_last_docs([])
                else:
                    self.initial_prompt = CONTINUOUS_PROMPT_LLAMA
                    if self.debug:
                        print(
                            "[DEBUG] Follow-up válido sin coincidencia específica. Se mantiene el contexto actual.")

        context = build_context_from_docs(docs, full=follow_up)

        historial = "\n".join([f"{t['role'].title()}: {t['content']}" for t in self.memoria.get_recent_turns()])
        context_tokens = count_tokens(context)
        historial_tokens = count_tokens(historial)
        self.memoria.trim_if_exceeds_tokens(context_tokens, historial_tokens)

        # Armado de prompt
        prompt_value = self.build_llama2_prompt(context=context, question=query, historial=historial)

        if self.debug:
            print("Finished Context, tokens number: ", count_tokens(prompt_value))

        # Generation
        t0 = time.perf_counter()
        out = self.llm_model(
            prompt=prompt_value,
            stop=["</s>"],
            max_tokens=512,
            stream=True
        )
        t1 = time.perf_counter()

        ## Debugging
        if self.debug:
            print("Finished invoke, time: ", t1 - t0, " s")
            print("Prompt completo:\n", prompt_value)
            # print("Respuesta tokens:", len(out["choices"][0]["text"].split()))
            # print("Respuesta completa:\n", out["choices"][0]["text"])

        sentence_buffer = ""
        full_response_text = ""
        word_count = 0

        for chunk in out:
            token = chunk["choices"][0]["text"]
            sentence_buffer += token

            if " " in token:
                word_count += 1
            # Detectamos fin de oración para disparar el TTS
            if any(punct in token for punct in ["!", "?", "\n"]):
                clean_sentence = sentence_buffer.strip()
                if clean_sentence:
                    if self.debug:
                        print(clean_sentence)
                    clean_sentence = re.sub(r'^\d+[\.\-]\s*', '', clean_sentence)
                    # Quita asteriscos de negrita que el TTS lee como "asterisco"
                    clean_sentence = clean_sentence.replace("*", "")
                    full_response_text += " " + clean_sentence
                    yield {"text": clean_sentence, "end_session": False}
                    sentence_buffer = ""
                    word_count = 0

        # Entregar el resto si quedó algo en el buffer
        if sentence_buffer.strip():
            last_sentence = sentence_buffer.strip()
            full_response_text += " " + last_sentence
            yield {"text": last_sentence, "end_session": False}

        # Actualizar memoria
        self.memoria.add_turn("user", query)
        self.memoria.add_turn("assistant", full_response_text.strip())


    def should_continue_context(self, query: str, used_cached_docs: bool) -> bool:
        """
        Determina si se debe mantener el contexto anterior aunque no se haya detectado un follow-up directo.
        Usa is_follow_up_user como respaldo semántico.
        """
        if not used_cached_docs:
            return False

        try:
            return self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns())
        except Exception as e:
            print(f"[DEBUG] Error al validar continuidad de contexto: {e}")
            return False


class GenerationModuleMistral(GenerationModuleLlama):
    def __init__(self, llm_model=None, api_key=None):
        super().__init__(llm_model)
        self.model_name = "mistral-small-latest"
        self.api_key = api_key

    def initialize(self, initial_prompt=None, retrieval=None, debug=False, max_tokens=1024):
        super().initialize(initial_prompt, retrieval, debug, max_tokens)
        self.llm_model = Mistral(api_key=self.api_key)

    async def rag_answer_stream(self, query: str):
        """
        Regresa la respuesta del sistema LLM y un bool indicando si ya termino la interaccion (True) o si continua (False).
        """
        query = query.lower()

        docs = self.memoria.get_last_docs()  # revisamos si hay informacion previa
        used_cached_docs = docs is not None and len(docs) > 0

        follow_up = self.follow_up_model.is_follow_up(query, self.memoria.get_recent_turns())
        # Verificación adicional: ¿el usuario cambió de intención a pesar de que parece ser follow-up?
        if follow_up and not self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns()):
            if self.debug:
                print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
            follow_up = False  # Forzamos modo nueva búsqueda
            docs = []  # Limpiamos docs previos para no arrastrar contexto viejo
        if self.debug:
            print("Follow up: ", follow_up)
        loop = asyncio.get_event_loop()
        if not follow_up:
            # Validamos si el usuario sigue en la misma intención, aunque no sea un follow-up directo
            if self.should_continue_context(query, used_cached_docs):
                follow_up = True
                resp, docs = await loop.run_in_executor(None, lambda: self.retrieval.ask(query, return_docs=True, memoria=self.memoria))
                # resp, docs = self.retrieval.ask(query, return_docs=True, memoria=self.memoria)
                if self.debug:
                    print(
                        "[DEBUG] No se detectó follow-up directo, pero sí continuidad semántica. Se mantiene contexto.")
            else:
                # loop = asyncio.get_event_loop()
                resp, docs = await loop.run_in_executor(None, lambda: self.retrieval.ask(query, return_docs=True,
                                                                                         memoria=self.memoria))

                is_relevant = any(doc.metadata.get('score', 1.0) <= 0.50 for doc in docs) if docs else False # 0.0 es max relevant
                has_lexicon_match = self.follow_up_model.is_service_in_scope(query)

                if has_lexicon_match or not is_relevant:
                    if self.debug:
                        print(f"[DEBUG] No hizo match con los temas especificados: {has_lexicon_match} o"
                              f" no fue relevante: {is_relevant}")
                    yield {"text": "out_of_scope", "end_session": False}
                    return

                if not docs or not is_relevant:
                    self.memoria.add_turn("user", query)
                    self.memoria.add_turn("assistant", "No encontré información suficiente en la base.")
                    if self.debug:
                        tmp_docs = True if docs else False
                        score_tmp = 1.0 if not docs else max(doc.metadata.get('score', 1.0) for doc in docs)
                        print(f"[DEBUG] No se encontró información suficiente debido a la falta de docs ({tmp_docs})"
                              f"o a is_relevant ({is_relevant}) con score: {score_tmp}")
                    yield {"text": "No encontré información suficiente en la base.", "end_session": False}
                    return

                self.memoria.set_last_docs(docs)
                if self.debug:
                    print("[DEBUG] Nueva búsqueda realizada, se guardan nuevos documentos.")
                self.initial_prompt = INIT_PROMPT_LLAMA

        else:
            # 4. En caso de follow-up, usar los documentos previos
            if not used_cached_docs:
                if self.debug:
                    print("[DEBUG] Follow-up detectado, pero no hay documentos previos.")
                self.memoria.add_turn("user", query)
                self.memoria.add_turn("assistant", "No tengo contexto previo suficiente.")
                # yield {"text": "Podrías especificar a qué unidad o tema te refieres?", "end_session": False}
                follow_up = False
                # loop = asyncio.get_event_loop()
                resp, docs = await loop.run_in_executor(None, lambda: self.retrieval.ask(query, return_docs=True, memoria=None))
                # resp, docs = self.retrieval.ask(query, return_docs=True, memoria=None)

            matches = self.follow_up_model.match_sucursal_from_input(user_input=query, docs=docs)
            if matches:
                docs = [doc for doc, _ in matches]
                self.memoria.set_last_docs(docs)
                self.initial_prompt = DETAILS_PROMPT
                if self.debug:
                    print(f"[DEBUG] Se detectaron {len(matches)} coincidencias por follow-up:")
                    for doc in docs:
                        nombre = doc.metadata.get("nombre_oficial", "Sin nombre")
                        municipio = doc.metadata.get("municipio", "Sin municipio")
                        estado = doc.metadata.get("estado", "Sin estado")
                        print(f" - {nombre} ({municipio}, {estado})")
            else:
                # Solo considerar cambio de intención si NO hubo match
                is_same_topic = self.follow_up_model.is_follow_up_user(query, self.memoria.get_recent_turns())
                if self.debug:
                    print(f"[DEBUG] Similitud semántica entre queries: {is_same_topic}")
                if not is_same_topic:
                    if self.debug:
                        print("[DEBUG] Cambio de intención detectado. Se reinicia contexto y búsqueda.")
                    follow_up = False
                    docs = []
                    self.memoria.set_last_docs([])
                else:
                    self.initial_prompt = CONTINUOUS_PROMPT_LLAMA
                    if self.debug:
                        print(
                            "[DEBUG] Follow-up válido sin coincidencia específica. Se mantiene el contexto actual.")

        context = build_context_from_docs(docs, full=follow_up)

        historial = "\n".join([f"{t['role'].title()}: {t['content']}" for t in self.memoria.get_recent_turns()])
        context_tokens = count_tokens(context)
        historial_tokens = count_tokens(historial)
        self.memoria.trim_if_exceeds_tokens(context_tokens, historial_tokens)

        # Armado de prompt
        messages = self.build_mistral_prompt(context=context, question=query, historial=self.memoria.get_recent_turns())

        if self.debug:
            print("Finished Context, tokens number: ", context_tokens + historial_tokens)

        # Generation
        t0 = time.perf_counter()
        # Mistral soporta async stream nativamente
        stream_response = await self.llm_model.chat.stream_async(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=150,
            frequency_penalty=1.1,
            presence_penalty=0.6,
            top_p=0.9,
        )
        t1 = time.perf_counter()

        ## Debugging
        if self.debug:
            print("Finished invoke, time: ", t1 - t0, " s")
            # print("Prompt completo:\n", prompt_value)
            # print("Respuesta tokens:", len(out["choices"][0]["text"].split()))
            # print("Respuesta completa:\n", out["choices"][0]["text"])

        sentence_buffer = ""
        full_response_text = ""

        async for chunk in stream_response:
            token = chunk.data.choices[0].delta.content
            if token is None: continue

            sentence_buffer += token

            # Detectamos fin de oración para disparar el TTS
            if any(punct in token for punct in ["!", "?", "\n"]):
                clean_sentence = sentence_buffer.strip()
                if clean_sentence:
                    if self.debug:
                        print(clean_sentence)
                    # Quita asteriscos de negrita que el TTS lee como "asterisco"
                    clean_sentence = clean_sentence.replace("*", "")
                    full_response_text += " " + clean_sentence
                    yield {"text": clean_sentence, "end_session": False}
                    sentence_buffer = ""
                    word_count = 0

        # Entregar el resto si quedó algo en el buffer
        if sentence_buffer.strip():
            full_response_text += " " + sentence_buffer.strip()
            yield {"text": sentence_buffer.strip(), "end_session": False}

        # Actualizar memoria
        self.memoria.add_turn("user", query)
        self.memoria.add_turn("assistant", full_response_text.strip())

    def build_mistral_prompt(self, context: str, question: str, historial: list = None):
        # Plantilla oficial Mistral
        messages = []

        system_content = self.initial_prompt + "CONTEXTO: " + context
        messages.append({"role": "system", "content": system_content})
        if historial:
            for turn in historial:
                # turn['role'] debe ser 'user' o 'assistant'
                messages.append({"role": turn["role"], "content": turn["content"]})

        # Pregunta actual
        messages.append({"role": "user", "content": question})

        return messages


INIT_PROMPT_LLAMA = """
    Eres CORA, asistente de Medical Life, empresa proveedora de servicios médicos. Responde estrictamente en español mexicano.
    Si piden hablar con asesor responde: TRANSFER_CALL. Usa SOLO el CONTEXTO. Si no hay datos di: No cuento con esa informacion.

    TU RESPUESTA DEBE:
    - No incluyas presentación o tu nombre, ya que es un seguimiento de la misma conversación.
    - Iniciar con "Por supuesto" o "Claro".
    - Mencionar las sedes (MAXIMO TRES) indicando nombre, municipio y estado.
    - Si hay mas de tres sedes, mencionar cuantas mas existen y preguntar si desea conocerlas.
    - Escribir TODO en un solo párrafo, sin guiones, sin asteriscos y sin usar números (ejemplo: decir "uno" en lugar de "1").
    - No uses puntos ni comillas dentro del párrafo, solo al final.
    - Usa frases cortas y directas.
    - No realices abreviaciones en estados o ciudades (ejemplo: si dice "edo. de Mexico", cambiar a "estado de mexico").
    - Finaliza siempre preguntando si desea más información de alguna de las sedes(como horarios o ubicación exacta).
    - La respuesta debe ser breve (3 a 6 líneas).

    Ejemplo de estilo: Claro encontré la sede centro en querétaro y la sede norte en el municipio de monterrey para el servicio de farmacia, desea saber horarios o ubicación exacta de alguna?
    """

DETAILS_PROMPT = """
    Eres un asistente de Medical Life. El usuario quiere detalles de una unidad específica.
    Si piden asesor responde: TRANSFER_CALL.
    Responde SOLO con el CONTEXTO en un solo párrafo continuo.
    NO te presentes. NO saludes. Inicia con una frase de confirmación como "claro!" y ve directo a los datos.
    Finaliza preguntando si desea más información de alguna de las sedes o si desea terminar la comunicación.

    REGLAS FONÉTICAS:
    - NOMBRES Y DIRECCIONES: Escríbelos completo. Traduce "Col." a "colonia", "No." a "número", "Av." a "avenida".
    - NUNCA menciones el id de las sucursales.
    - No realices abreviaciones en estados o ciudades (ejemplo: si dice "edo. de Mexico", cambiar a "estado de mexico").
    - HORARIOS: Di los días completos. Ejemplo: "de lunes a viernes de nueve de la mañana a seis de la tarde".
    - TELÉFONOS: Escribe los números con letras uno por uno (ejemplo: cinco cinco dos dos).
    - ESTILO: Sin listas, sin puntos, sin guiones, sin abreviaciones.
    - ESTRUCTURA: Confirmación directa con los datos y termina preguntando si necesita algo más.

    Ejemplo: La sede se encuentra en la colonia centro número diez y su teléfono es cinco cinco uno dos tres cuatro, desea más información o quiere terminar la llamada?
    """

CONTINUOUS_PROMPT_LLAMA = """
    Eres CORA de Medical Life. Responde SOLO con el CONTEXTO. Si piden asesor responde: TRANSFER_CALL.
    Estas en una conversación fluida, ve directo al grano, dando un parafraseo corto de la pregunta del cliente. 

    INSTRUCCIONES:
    - Responde en un solo párrafo de máximo tres líneas.
    - Prohibido usar listas, viñetas, guiones o caracteres especiales.
    - Escribe números con letras.
    - No realices abreviaciones en estados o ciudades (ejemplo: si dice "edo. de Mexico", cambiar a "estado de mexico").
    - Si falta información di: No tengo esos datos por ahora.
    - Finaliza con una pregunta corta como: Alguna otra duda o desea terminar.

    Ejemplo: Encontré la información de farmacia en la sede de campeche municipio de escarcega desea saber algo más de esta unidad o prefiere terminar
    """

def build_context_from_docs(docs, full=False):
    parts = []
    for d in docs:
        m = d.metadata
        if full:
            snippet = d.page_content
            parts.append(f"{snippet}. ")
        else:
            # parts.append(f"{m['id']} — {m['municipio']}, {m['estado']} | Servicios: {', '.join(m.get('servicios_lista', []))}")
            parts.append(f"{m['municipio']}, {m['estado']} | Servicios: {', '.join(m.get('servicios_lista', []))}")
    return "\n---\n".join(parts)


class ConversationalMemory:
    """
    Clase dedicada a almacenar una conversacion en formato:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Metodos
    - Añadir nuevos turnos

    - Obtener el historial más reciente (get_recent_turns(n))

    - Limpiar el historial (al cerrar sesión)

    - Guardar el último set de documentos entregados por el retrieval (para follow-ups)

    """
    def __init__(self, max_tokens=4096):
        """
        max_turns: cantidad de turnos de ida y vuelta a mantener (user + assistant)
        """
        self.turns = []
        self.last_docs = []
        # self.max_turns = max_turns
        self.last_user_query = []
        self.last_assistant_query = []
        self.max_tokens = max_tokens

    def add_turn(self, role, content):
        """Agrega un nuevo mensaje al historial"""
        self.turns.append({"role": role, "content": content})
        if role == "user":
            self.last_user_query = content
        elif role == "assistant":
            self.last_assistant_query = content
        # if len(self.turns) > self.max_turns * 2:  # user + assistant = 2 turnos por ciclo
        #     self.turns = self.turns[-self.max_turns * 2:]

    def get_recent_turns(self):
        """Retorna los turnos recientes (user + assistant alternados)"""
        return self.turns

    def set_last_docs(self, docs):
        """Guarda los documentos del retrieval más reciente"""
        self.last_docs = docs

    def get_last_docs(self):
        return self.last_docs

    def clear(self):
        """ Limpia memoria e historial"""
        self.turns = []
        self.last_docs = []

    def get_last_assistant_response(self):
        return self.last_assistant_query


    def trim_if_exceeds_tokens(self, tokens_so_far: int, tokens_next_context: int):
        """
        Recorta últimos turnos si la suma actual + contexto futuro excede el 90% del límite de tokens.

        Parámetros:
            - tokens_so_far: tokens del historial actual
            - tokens_next_context: tokens esperados del siguiente contexto
            - max_tokens: límite total aceptable del modelo
        """
        total_expected = tokens_so_far + tokens_next_context
        threshold = int(self.max_tokens * 0.9)

        if total_expected >= threshold and len(self.turns) >= 2:
            # Borra los dos turnos más antiguos (usuario y asistente)
            self.turns = self.turns[2:]


class IntentionDetector:
    def __init__(self, retrieval=None, threshold=0.65, debug=True):
        if retrieval is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.model = retrieval
        self.threshold = threshold
        self.debug = debug
        self.intent_examples = {
            "repeat": ["repite eso", "que dijiste", "no entendi", "puedes repetir", "me repites", "me puedes repetir", "no escuche", "no te escuche"],
            "presence": ["sigues ahi", "hola", "me escuchas", "estas ahi"],
            "wait": ["espera", "dame un segundo", "un momento", "aguanta", "dame un minuto", "un minuto"],
            "audio_complaint": ["te escuchas mal", "está trabado", "audio robótico", "se escucha mal", "no entendi", "no te entiendo"],
            "asesor": ["humano", "asesor", "quiero hablar con un asesor"]
        }
        self.domain_queries = [
            "localizar sucursales sedes unidades", # Logística
            "servicios médicos salud clinica hospital laboratorio dentista optometrista lentes enfermo",         # Salud
            "horarios de atención dirección ubicación mapa dónde están",    # Información
            "quiero saber información de la empresa sucursales disponibles" # Intención
        ]

        self.precomputed_embeddings = {}
        for intent, phrases in self.intent_examples.items():
            # Generamos una lista de vectores para cada intención
            self.precomputed_embeddings[intent] = [
                self.model.embeddings.embed_query(phrase) for phrase in phrases
            ]

        self.exit_examples = [
            "ya terminé", "eso es todo", "puedes cortar la llamada", "terminamos", "no tengo mas preguntas", "adiós",
            "es todo", "seria todo", "no necesito nada", "nada", "adios",
        ]
        self.emb_refs = [self.model.embeddings.embed_query(e) for e in self.exit_examples]

        from RAG_CORE.rag_utils.mappings import SERVICE_LEXICON
        self.compiled_items = self._compile_lexicon(SERVICE_LEXICON)


    def is_follow_up(self, user_input: str, memory_turns: list) -> bool:
        # Follow up semantica
        if not memory_turns:
            return False

        # 1. Heurístico: frases típicas de seguimiento
        heuristics = [
            r"\bs[ií]\b", r"\besa\b", r"\bla de\b", r"\bquiero más info\b",
            r"\bde la\b", r"\besa sede\b", r"\bhorario\b", r"\bdirección\b",
            r"\bdónde está\b", r"\bteléfono\b", r"\bubicación\b", r"\bcuál es el horario\b",
            r"\bme repites\b", r"\bdónde queda\b"
        ]
        for pattern in heuristics:
            if re.search(pattern, user_input):
                return True

        # 2. Última respuesta del asistente (más relevante que la pregunta anterior del usuario)
        last_assistant = next((t["content"] for t in reversed(memory_turns) if t["role"] == "assistant"), None)
        if not last_assistant:
            return False

        # 3. Similaridad semántica entre nuevo input y última respuesta
        embedding_a = self.model.embeddings.embed_query(user_input)
        last_assistant_text = (
            last_assistant["content"]["choices"][0]["text"]
            if isinstance(last_assistant, dict) and isinstance(last_assistant.get("content"), dict)
            else last_assistant.get("content") if isinstance(last_assistant, dict)
            else str(last_assistant)
        )
        # print("Last text: ", last_assistant_text)
        embedding_b = self.model.embeddings.embed_query(str(last_assistant_text))
        sim = cosine_similarity([embedding_a], [embedding_b])[0][0]

        return sim >= self.threshold

    def is_follow_up_user(self, user_input: str, memory_turns: list) -> bool:
        # Follow up de intencion
        if not memory_turns:
            return False

            # Heurístico básico (puedes dejarlo, pero como fallback)
        heuristics = [r"\bs[ií]\b", r"\besa\b", r"\bla de\b", r"\bde la\b", r"\besa sede\b", r"\bmas detalles\b"]
        if any(re.search(h, user_input.lower()) for h in heuristics):
            return True

        # Obtener última pregunta del usuario
        last_user_query = None
        for turn in reversed(memory_turns):
            if turn["role"] == "user":
                last_user_query = turn["content"]
                break

        if not last_user_query:
            return False

        # Comparar embeddings
        if self.debug:
            print(f"[DEBUG] LAST QUERY: {last_user_query}")
            print(f"[DEBUG] USER QUERY: {user_input}")
        emb_current = self.model.embeddings.embed_query(user_input)
        emb_previous = self.model.embeddings.embed_query(last_user_query)
        sim = cosine_similarity([emb_current], [emb_previous])[0][0]
        if self.debug:
            print(f"[DEBUG] Similitud semántica entre queries: {sim:.3f}")

        return sim >= self.threshold  # threshold por defecto: 0.65–0.75

    def match_sucursal_from_input(self, user_input: str, docs: list, threshold: float = 0.65, top_k: int = 1):
        """
        Dada una entrada del usuario y una lista de documentos (sucursales),
        devuelve la(s) sucursal(es) que más se parezcan con base en embeddings.

        Parámetros:
            - user_input: texto del usuario (ej. "de la Gustavo Madero")
            - docs: lista de Document con metadatos (output de last_docs)
            - encoder: Usa el vectorstore y embeddings del sistema.
            - threshold: umbral mínimo de similitud
            - top_k: cuántas coincidencias devolver (por default solo 1)

        Retorna:
            Lista de Document (las mejores coincidencias)
        """
        if not docs:
            return []

            # Paso 1: preparar corpus reducido (texto representativo por doc)
        temp_docs = []
        for d in docs:
            m = d.metadata
            # Concatenamos nombre oficial + municipio + estado (ajustable)
            doc_text = f"{m.get('nombre_oficial', '')} {m.get('municipio', '')} {m.get('estado', '')}"
            # Creamos Document temporal con mismo ID y score dummy
            temp_docs.append(Document(page_content=doc_text, metadata=m))

        # Paso 2: construir FAISS temporal en memoria con solo estos docs
        faiss_local = FAISS.from_documents(temp_docs, embedding=self.model.embeddings)

        # Paso 3: búsqueda en el vectorstore temporal
        results = faiss_local.similarity_search_with_score(user_input, k=top_k)
        matches = [(doc.metadata, score) for doc, score in results if score >= threshold]

        # Paso 4: devolver el documento original completo
        docs_out = []
        for meta, score in matches:
            for d in docs:
                if d.metadata.get("id") == meta.get("id"):
                    docs_out.append((d, score))
                    break

        return docs_out

    def detect_exit_intent(self, user_input: str) -> bool:
        # heurístico rápido
        TRESHOLD = 0.45
        exit_patterns = [r"\b(terminar|eso es todo|cortar|no tengo más preguntas|terminar|es todo|seria todo)\b"]
        if any(re.search(p, user_input.lower()) for p in exit_patterns):
            return True

        # comparación semántica
        emb_user = self.model.embeddings.embed_query(user_input)
        scores = [cosine_similarity([emb_user], [e])[0][0] for e in self.emb_refs]
        if self.debug:
            print("[DEBUG] Nivel de deteccion de finalizacion: ", scores, " Fin: ", max(scores) >= TRESHOLD)
        return max(scores) >= TRESHOLD  # umbral ajustable

    def get_semantic_intent(self, query: str, threshold: float = 0.45) -> str:
        # 1. Solo generamos UN embedding (el de la pregunta actual)
        query_emb = self.model.embeddings.embed_query(query)

        best_intent = "rag_required"
        max_score = 0

        scores = []
        # 2. Comparamos contra lo que ya tenemos en memoria RAM
        for intent, example_embs in self.precomputed_embeddings.items():
            scores = [cosine_similarity([query_emb], [ex_emb])[0][0] for ex_emb in example_embs]
            current_max = max(scores)

            if current_max > max_score and current_max >= threshold:
                max_score = current_max
                best_intent = intent
        if self.debug:
            print("[DEBUG] Nivel de deteccion de intent: ", scores, " Intent: ", best_intent, ", ", max(scores) >=threshold)

        return best_intent
    @staticmethod
    def _compile_lexicon(lexicon):
        """Compila todos los patrones RE del lexicón en una sola lista."""
        patterns = []
        for service, data in lexicon.items():
            for regex_str in data.get("re", []):
                patterns.append(re.compile(regex_str, re.IGNORECASE))
        return patterns

    def is_service_in_scope(self, text: str) -> bool:
        """
        Capa 1: Verifica mediante Regex si hay un servicio médico explícito.
        """
        return any(pattern.search(text) for pattern in self.compiled_items)