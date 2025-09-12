from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import asyncio
import time

OLLAMA_URL = "http://ollama:11434"
app = FastAPI()

# Map OpenAI model names to Ollama local models
MODEL_MAP = {
    "gpt-3.5-turbo": "llama2",
    "gpt-4": "mistral",
    # Add aliases as needed
}

def map_model(model: str) -> str:
    return MODEL_MAP.get(model, model)

def openai_error(message: str, code: str = "invalid_request_error"):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )

# --------------------------
# Chat Completions
# --------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = map_model(body.get("model", "llama2"))
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        return openai_error("`messages` is required.")

    ollama_payload = {"model": model, "messages": messages, "stream": stream}

    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            async def event_stream():
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
            if resp.status_code != 200:
                return openai_error(f"Ollama error: {resp.text}")

            data = resp.json()
            now = int(time.time())

            return JSONResponse({
                "id": f"chatcmpl-{now}",
                "object": "chat.completion",
                "created": now,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": data.get("message", {}),
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(str(data.get("message", {}))),
                    "total_tokens": len(str(messages)) + len(str(data.get("message", {})))
                }
            })

# --------------------------
# Completions
# --------------------------
@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model = map_model(body.get("model", "llama2"))
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)

    if not prompt:
        return openai_error("`prompt` is required.")

    ollama_payload = {"model": model, "prompt": prompt, "stream": stream}

    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            async def event_stream():
                async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=ollama_payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=ollama_payload)
            if resp.status_code != 200:
                return openai_error(f"Ollama error: {resp.text}")

            data = resp.json()
            now = int(time.time())

            return JSONResponse({
                "id": f"cmpl-{now}",
                "object": "text_completion",
                "created": now,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": data.get("response", ""),
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(data.get("response", "")),
                    "total_tokens": len(prompt) + len(data.get("response", ""))
                }
            })

# --------------------------
# Embeddings
# --------------------------
@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = map_model(body.get("model", "llama2"))
    input_text = body.get("input", "")

    if not input_text:
        return openai_error("`input` is required.")

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": model, "prompt": input_text})
        if resp.status_code != 200:
            return openai_error(f"Ollama error: {resp.text}")

        data = resp.json()
        now = int(time.time())

        return JSONResponse({
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": data.get("embedding", []),
                "index": 0
            }],
            "model": model,
            "usage": {
                "prompt_tokens": len(input_text),
                "total_tokens": len(input_text)
            }
        })

# --------------------------
# Models (list available Ollama models)
# --------------------------
@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/tags")
        if resp.status_code != 200:
            return openai_error(f"Ollama error: {resp.text}")

        data = resp.json()
        models = []
        for i, m in enumerate(data.get("models", [])):
            models.append({
                "id": m.get("name", f"model-{i}"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "permission": []
            })
        return {"object": "list", "data": models}

# --------------------------
# Moderations (stub)
# --------------------------
@app.post("/v1/moderations")
async def moderations(request: Request):
    # Just always return "safe" for now
    body = await request.json()
    input_text = body.get("input", "")
    return {
        "id": f"modr-{int(time.time())}",
        "object": "moderation",
        "model": "moderation-stub",
        "results": [{
            "categories": {
                "sexual": False,
                "violence": False,
                "hate": False,
                "self-harm": False
            },
            "category_scores": {
                "sexual": 0.0,
                "violence": 0.0,
                "hate": 0.0,
                "self-harm": 0.0
            },
            "flagged": False
        }]
    }

# --------------------------
# Audio Transcriptions (stub)
# --------------------------
@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile, model: str = Form(...)):
    # No Ollama speech model yet, so stub
    return {
        "text": f"[Transcription not available in Ollama shim, got file: {file.filename}]"
    }
