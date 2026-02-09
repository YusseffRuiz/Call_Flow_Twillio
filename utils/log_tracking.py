import os
import time
import json

class CallMetrics:
    def __init__(self):
        self.turns = []
        self.total_call_cost = 0.0

    def initialize(self, call_id):
        self.call_id = call_id
        self.start_time = time.perf_counter()
        self.call_start_perf = time.perf_counter()  # Para duración total de la llamada
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def calculate_cost(self, tokens_llm, audio_duration):
        # Calculo de costo por llamada total en USD
        cost_llm = (tokens_llm / 1000) * 0.002  # $0.002 por 1k tokens
        cost_tts = (tokens_llm / 1000) * 0.03  # Deepgram $0.03 por 1k tokens https://deepgram.com/pricing
        cost_twilio = audio_duration * (0.004 / 60)  # Costo de llamada de twilio $0.004 por minuto
        # Podemos prescindir del ASR.
        return round(cost_llm + cost_tts + cost_twilio, 5)

    def log_turn(self, metrics):
        """
        metrics: dict con todos los timestamps y textos del turno
        """

        t0 = metrics.get('t0', time.perf_counter())
        t1 = metrics.get('t1', t0)
        t2 = metrics.get('t2', t1)
        t3 = metrics.get('t3', t2)
        bot_text = metrics.get('bot_text', "")
        user_text = metrics.get('user_text', "")
        t_end = metrics.get('t_end_audio', t3)

        # Latencia E2E: Desde que el usuario dejó de hablar (T0) hasta que el primer audio salió (T3)
        e2e = t3 - t0

        # TTFB: Desde el primer token LLM (T2) hasta salida audio (T3)
        ttfb = t3 - t2

        # LLM TPS
        num_tokens = len(bot_text) / 4
        # Duración del audio generado (aproximación o real)
        audio_len = t_end - t3
        turn_cost = self.calculate_cost(num_tokens, audio_len)
        self.total_call_cost += turn_cost

        turn_data = {
            "turn_idx": len(self.turns) + 1,
            "t0_user_end": t0,
            "t1_asr_end": t1,
            "t2_llm_first_token": t2,
            "t3_tts_first_chunk": t3,
            "latencies": {
                "e2e": round(e2e, 3),
                "asr": round(t1 - t0, 3),
                "llm_first_token": round(t2 - t1, 3),
                "ttfb_internal": round(ttfb, 3),
            },
            "texts": {
                "user": user_text,
                "bot": bot_text
            },
            "stats": {
                "cost": turn_cost,
                "was_misunderstood": metrics['was_misunderstood']
            }
        }
        self.turns.append(turn_data)
        self._save_to_jsonl(turn_data)

    def _save_to_jsonl(self, data):
        with open(f"logs/call_{self.call_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")