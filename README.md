# Twilio Real-Time Voice Agent – Medical Life MVP

## Overview
Este proyecto implementa un agente de voz en tiempo real usando telefonía via Twilio Streams y WebSockets.
El flujo de la arquitectura para contestar preguntas durante la llamada contiene: 
XTTS (clonación de voz) o Deepgram, ASR (automatic speech recognition) y LLM+RAG.

Es un proyecto privado en México.

La versión actual es un MVP con las siguientes características:

- Audio Bidireccional (Twilio + Servidor)
- TTS playback (usando TTS o Deepgram)
- Detección de turnos y silencios (utilizando VAD)
- Barge-in (Interrupciones del cliente)
- End-to-end: ASR → RAG → LLM → TTS loop 

En este punto, se preparó para que se utilice ngrok como punto de exposición pública para Twilio.

## Arquitectura del sistema
    ┌─────────────┐
    │   Twilio    │
    │  Phone Call │
    └──────┬──────┘
           │ Media Streams (μ-law 8kHz)
           ▼
    ┌───────────────────────────────┐
    │ FastAPI WebSocket Server      │
    │ /twilio/ws                    │
    │                               │
    │  - TurnDetector (VAD)         │
    │  - Barge-in logic             │
    │  - Audio buffering            │
    └──────┬───────────┬───────────┘
           │           │
           │           │
           ▼           ▼
    ┌─────────────┐   ┌─────────────────────┐
    │     ASR     │   │  XTTS/ DeepGram     │
    │ faster-     │   │  Voice Cloning TTS  │
    │ whisper     │   │  (Coqui XTTS)       │
    └──────┬──────┘   └─────────┬───────────┘
           │                    │
           ▼                    │
    ┌────────────────────────────┴───────┐
    │           LLM + RAG                 │
    │  - FAISS / embeddings (Spanish)     │
    │  - Medical knowledge base           │
    │  - LLaMA / Mistral GGUF             │
    └─────────────────────────────────────┘

## Componentes
1. Twilio Media Streams
2. Turn Detection & Voice Activity Detection (VAD)
   - Uso de webrtcvad 
   - threshold de deteccion de voz via RMS de energía.
   - Detección de silencio total (800 ms)
3. Barge-In Handling
   - Por medio de VAD y del RMS de energía.
   - Envía clear a Twilio para interrumpir bot.
4. Text-to-Speech (XTTS)
   - Uso de Coqui XTTS.
   - Clonación de voz y uso de voces base.
   - Sample rate de salida ~24 kHz
   - Downsampled a 8 kHz para Twilio
   - Se espera una latencia de 20 ms para "tiempo real".
5. Automatic Speech Recognition (ASR)
   - Implementado por Fast-whisper
6. LLM + RAG
   - Retrieval por medio de FAISS index.
     - Embeddings en español: jina-embeddings-v2-base-es
     - Uso de un dataset externo en Excel.
   - Generator:
     - Uso de modelo GGUF local.
     - Uso de CUDA local.
     - Uso de Llama.cpp como backend.
     - Si no se utiliza CUDA, el sistema es demasiado lento.


**El sistema soporta un switch de configuración entre ejecución Local (Edge) y Cloud (Performance):**
1. Nivel Base (Local/Gratis)
   - LLM: Llama-cpp (GGUF) corriendo en CUDA local.
   - Ventaja: Privacidad total de datos y costo cero de inferencia.
   - Limitación: Requiere hardware GPU de alto nivel para mantener latencia < 2s.
2. Nivel Pro (Cloud/Pago)
   - LLM: Mistral AI Cloud (vía API asíncrona).
   - XTTS: Inferencia optimizada con Caché de Audios de Sistema (Fillers).
   - Ventaja: Escalabilidad, mayor inteligencia semántica y respuestas casi instantáneas.

**Switch de Modelo**

Mediante variables de entorno se controla el uso Pro o el Base:

`VERSION_PAGA = True`


## Uso del sistema
1. Instalar dependencias:

2. Declarar variables de ambiente en Documents/keys.env

```
    PUBLIC_BASE_URL=https://<your-ngrok-id>.ngrok-free.dev
    TWILIO_ACCOUNT_SID=ACxxxxxxxx
    TWILIO_AUTH_TOKEN=xxxxxxxx
    TWILIO_FROM_NUMBER=+1xxxxxxxx
    TWILIO_TO_NUMBER=+52xxxxxxxx
    AGENT_LANGUAGE=es
```

3. Iniciar tunel (ngrok)

    ```ngrok http 8000```

4. Correr servidor

    ```uvicorn main:app --host 0.0.0.0 --port 8000```
   
5. Correr el inicio de la llamada
```angular2html
$body = @{ to = "+phonenumber" } | ConvertTo-Json 
Invoke-RestMethod -Method Post -Uri "https://ngrok-dev-web.com/twilio/start_call"  -ContentType "application/json" -Body $body 
```


## Status Actual
- ✅ Streaming en tiempo real validado.
- ✅ XTTS clonación de voz y voces base.
- ✅ ASR Funcional.
- ✅ Barge-in funcional.
- ✅ Integración RAG + LLM. 
- ✅ Ruteo a un asesor o teléfono real. 
- ✅ Safeguards básicos (cambio de tema o no se escuchó). 
- ✅ Analítica de llamadas. 
- ⚠️ Tunear sensibilidad al ruido.
- ⚠️ Sin intento de llamada/ Sin recuperación si hay fallas.

## Siguientes pasos
- Tresholds adaptativos para VAD (Detección de ruido).
- Reintento de llamadas y flujo de fallo.
- Deployment sin ngrok (HTTPS/WSS)
- Deployment en servidor