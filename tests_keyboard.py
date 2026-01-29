from __future__ import annotations
import os
from dotenv import load_dotenv
from llama_cpp import Llama


# Your existing modules (from your repo)
from RAG_CORE.generation_module import GenerationModuleLlama
from RAG_CORE.retrieval_module import RetrievalModule
import campaign_data.utils_script as MESSAGES
load_dotenv("Documents/keys.env")
# Language
LANGUAGE = os.getenv("AGENT_LANGUAGE", "es")

LLM_MODEL_FILE = "../HF_Agents/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
HF_TOKEN = os.getenv("HF_TOKEN")

MEDICAL_EXTENDED = "Documents/medical_life_real.xlsx"

VECTOR_MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())


# -----------------------------
# Text Script (MVP)
# -----------------------------
WELCOME_CAMPAING = [MESSAGES.MSG_1, MESSAGES.MSG_2, MESSAGES.MSG_3, MESSAGES.MID_MESSAGE_1, MESSAGES.MID_MESSAGE_2,
                 MESSAGES.MID_MESSAGE_3, MESSAGES.MID_MESSAGE_4, MESSAGES.MID_MESSAGE_5, MESSAGES.LOCATION_MESSAGE,
                 MESSAGES.CLOSE_MESSAGE_1, MESSAGES.CLOSE_MESSAGE_2]

GREETING_MSG = ("¡Hola! Mi nombre es Cora, soy el asistente de Medical Laif para resolver tus dudas y es un placer"
                " atender tu llamada. Dime, ¿tienes alguna duda sobre un servicio o localización de algún centro?")


def flow(llm_model=None):
    print("buen día, comenzamos con las pruebas")
    while True:
        try:
           pregunta = input("👤 Usuario: ").strip()
           if pregunta.strip():
               respuesta, finish_flag = llm_model.rag_answer(query=pregunta)
               print("🤖 Cora:\n", respuesta)
               print("-" * 50 + "\n\n")

               # Esperar nueva pregunta
               if "adios" in respuesta.lower() or finish_flag:
                   break
           else:
               print("😶 No se entendió lo que dijiste.")

        except KeyboardInterrupt:
            print("\n👋 Conversación terminada.")
            break


def main():
    gpu_layers = -1  # 20 a 30 funcionan en nuestra GPU NVIDIA RTX4090 8 GB, -1 settea a GPU
    config = {'max_new_tokens': 256, 'context_length': 2500, 'temperature': 0.45, "gpu_layers": gpu_layers,
              "threads": os.cpu_count()}

    DEBUG = True

    """
    Augmentation and Generation Portion
    """
    retrieval_module = RetrievalModule(database_path=MEDICAL_EXTENDED, hf_token=HF_TOKEN, model_name=VECTOR_MODEL_NAME)
    retrieval_module.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold=0.34,
                                percentile=0.9)
    llm_model = Llama(model_path=LLM_MODEL_FILE,
                      n_ctx=config["context_length"],
                      # The max sequence length to use - note that longer sequence lengths require much more resources
                      n_threads=config["threads"],
                      # The number of CPU threads to use, tailor to your system and the resulting performance
                      n_gpu_layers=gpu_layers,
                      temperature=config["temperature"],
                      n_batch=512,
                      use_mlock=False,
                      use_mmap=True,
                      f16_kv=True,
                      verbose=False
                      )
    llm_module = GenerationModuleLlama(llm_model)
    llm_module.initialize(initial_prompt=MESSAGES.INIT_PROMPT_LLAMA, retrieval=retrieval_module, debug=DEBUG,
                          max_tokens=config["context_length"]/2)

    flow(llm_module)

if __name__ == "__main__":
    main()
