import asyncio

from dotenv import load_dotenv
from voices.TTS_engine import TextToSpeech
from llama_cpp import Llama
import os
from RAG_CORE.generation_module import GenerationModuleLlama
from RAG_CORE.retrieval_module import RetrievalModule

load_dotenv("Documents/keys.env")
LLM_MODEL_FILE = "../HF_Agents/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEBUG = True


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

DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DG_API_KEY:
    raise Exception("DEEPGRAM_API_KEY not found")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise Exception("HF_TOKEN not found")
MEDICAL_EXTENDED = "Documents/medical_life_real.xlsx"

VECTOR_MODEL_NAME = 'jinaai/jina-embeddings-v2-base-es'

class ConversationManager:
    async def main(self, llm_model=None):
        # Loop indefinitely until "goodbye" is detected
        tts = TextToSpeech(dg_api_key=DG_API_KEY)
        tts.speak("¡Hola! Medical Laif, Cora Pruebas de Velocidad. Hazme las preguntas que tengas")
        while True:
            ## Anotar Pregunta
            pregunta = input("👤 Usuario: ").strip()
            # Check for "goodbye" to exit the loop
            if pregunta.strip():
                respuesta, finish_flag = llm_model.rag_answer(query=pregunta)
                print("🤖 Cora:\n", respuesta)
                tts.speak(respuesta)
                if "adios" in respuesta.lower() or finish_flag:
                    print('bye bye')
                    break
                print("Ya se hablo")
            else:
                print("😶 No se entendió lo que dijiste.")
                interaction = "No entendí, ¿podrías repetirme tu pregunta? "
                tts.speak(text=interaction)




if __name__ == "__main__":
    retrieval_module = RetrievalModule(database_path=MEDICAL_EXTENDED, hf_token=HF_TOKEN, model_name=VECTOR_MODEL_NAME)
    retrieval_module.initialize(load_db=True, path_to_database="kb_faiss_langchain", score_threshold=0.34,
                                percentile=0.9)

    llm_module = GenerationModuleLlama(llm_model)
    llm_module.initialize(retrieval=retrieval_module, debug=DEBUG,
                          max_tokens=config["context_length"] / 2)
    manager = ConversationManager()
    asyncio.run(manager.main(llm_model=llm_module))
    # print("All ok")