from llama_cpp import Llama
import os
from dotenv import load_dotenv

load_dotenv("Documents/keys.env")
LLM_MODEL_FILE = "../HF_Agents/mistral-7b-instruct-v0.2.Q4_K_M.gguf"


gpu_layers = -1  # 20 a 30 funcionan en nuestra GPU NVIDIA RTX4090 8 GB, -1 settea a GPU
config = {'max_new_tokens': 256, 'context_length': 2500, 'temperature': 0.45, "gpu_layers": gpu_layers,
              "threads": os.cpu_count()}

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


# Your existing modules (from your repo)
import asyncio
from RAG_CORE.generation_module import GenerationModuleLlama
from RAG_CORE.retrieval_module import RetrievalModule
from voices.TTS_engine import Speaker

# Language
LANGUAGE = os.getenv("AGENT_LANGUAGE", "es")

HF_TOKEN = os.getenv("HF_TOKEN")
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")

MEDICAL_EXTENDED = "Documents/medical_life_real.xlsx"

VECTOR_MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

def play_start_sound(speaker=None, dg_client=None):
    speaker.play_text("¡Hola! Medical Laif, Cora Pruebas de Velocidad. Hazme las preguntas que tengas", dg_client=dg_client)

# -----------------------------
# Text Script (MVP)
# -----------------------------

def flow(llm_model=None, speaker=None):
    if speaker is not None:
        play_start_sound(speaker=speaker)
    else:
        print("buen día, comenzamos con las pruebas")
    while True:
        try:
           pregunta = input("👤 Usuario: ").strip()
           if pregunta.strip():
               respuesta, finish_flag = llm_model.rag_answer(query=pregunta)
               print("🤖 Cora:\n", respuesta)
               if speaker is not None:
                    speaker.tts_to_wav(text=respuesta.text)
               print("-" * 50 + "\n\n")

               # Esperar nueva pregunta
               if "adios" in respuesta.lower() or finish_flag:
                   print('bye bye')
                   break
           else:
               print("😶 No se entendió lo que dijiste.")
               interaction = "No entendí, ¿podrías repetirme tu pregunta? "
               speaker.tts_to_wav(text=interaction)

        except KeyboardInterrupt:
            print("\n👋 Conversación terminada.")
            break


async def async_flow(llm_module, speaker = None, agent=None):
    play_start_sound(speaker=speaker, dg_client=agent)
    print("🏥 Medical Life - Cora (Pruebas de Velocidad)")

    while True:
        try:
            pregunta = input("👤 Usuario: ").strip()
            if not pregunta:
                continue

            # Usamos el generador asíncrono que definimos previamente
            print("🤖 Cora: ", end="", flush=True)

            # Consumimos el stream del LLM
            async for response in llm_module.rag_answer_stream(pregunta):
                sentence = response['text']
                print(sentence, end=" ", flush=True)
                if speaker is not None:
                    await speaker.play_text(sentence, agent)

                if response['end_session']:
                    print("\n👋 Sesión terminada por el sistema.")
                    return

            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            break

def main():


    DEBUG = True

    """
    Augmentation and Generation Portion
    """
    retrieval_module = RetrievalModule(database_path=MEDICAL_EXTENDED, hf_token=HF_TOKEN, model_name=VECTOR_MODEL_NAME)
    retrieval_module.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold=0.34,
                                percentile=0.9)

    llm_module = GenerationModuleLlama(llm_model)
    llm_module.initialize(retrieval=retrieval_module, debug=DEBUG,
                          max_tokens=config["context_length"]/2)

    load_dotenv("Documents/keys.env")
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DG_API_KEY:
        raise Exception("DEEPGRAM_API_KEY not found")
    tts = Speaker(engine="DG", dg_api_key=DG_API_KEY)


    asyncio.run(async_flow(llm_module=llm_module, speaker=tts))


if __name__ == "__main__":
    main()
