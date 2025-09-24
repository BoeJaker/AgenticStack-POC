from fastapi import FastAPI, Request, HTTPException
import httpx
import subprocess
import os

app = FastAPI()

# ---------------- CONFIG ----------------
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "").split(",")
GPU_NAMES = os.environ.get("GPU_NAMES", "").split(",")
GPU_IDS = list(map(int, os.environ.get("GPU_IDS", "0").split(",")))
GPU_INSTANCES_PER_GPU = int(os.environ.get("GPU_INSTANCES_PER_GPU", "1"))

# Ollama container URLs
OLLAMA_INSTANCES = {"cpu": "http://ollama-cpu:11434"}
GPU_CONTAINERS = []

for gpu_name, gpu_id in zip(GPU_NAMES, GPU_IDS):
    for idx in range(GPU_INSTANCES_PER_GPU):
        cname = f"ollama-{gpu_name}-{idx}"
        OLLAMA_INSTANCES[cname] = f"http://{cname}:11434"
        GPU_CONTAINERS.append(cname)

# ---------------- MODEL INFO ----------------
HEAVY_MODELS = ["llama2:70b", "llama2:70b-q4", "mistral:30b"]
MODEL_VRAM_REQ = {
    "llama2:13b": 10,
    "llama2:70b-q4": 24,
    "mistral:30b": 16,
    "mistral:7b": 6,
}

active_requests = {name: 0 for name in OLLAMA_INSTANCES.keys()}

# RAM-disk paths
RAMDISK_PATHS = {"cpu": "/mnt/ramdisk_cpu"}
for gpu_name in GPU_NAMES:
    for idx in range(GPU_INSTANCES_PER_GPU):
        RAMDISK_PATHS[f"ollama-{gpu_name}-{idx}"] = f"/mnt/ramdisk_{gpu_name}_{idx}"

# ---------------- HELPERS ----------------
def get_gpu_free_vram(gpu_index: int) -> float:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            f"--id={gpu_index}"
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return float(output) / 1024
    except Exception:
        return 0.0

def is_model_in_ram(instance_name: str, model_name: str) -> bool:
    path = os.path.join(RAMDISK_PATHS[instance_name], model_name.replace(":", "_"))
    return os.path.exists(path)

def choose_instance(model_name: str):
    candidates = []

    if model_name in HEAVY_MODELS:
        for gpu_name, gpu_id in zip(GPU_NAMES, GPU_IDS):
            for idx in range(GPU_INSTANCES_PER_GPU):
                cname = f"ollama-{gpu_name}-{idx}"
                free_vram = get_gpu_free_vram(gpu_id)
                if free_vram >= MODEL_VRAM_REQ.get(model_name, 8):
                    candidates.append(cname)
        if not candidates:
            candidates.append("cpu")
    else:
        candidates.append("cpu")

    # Prefer RAM-loaded models
    in_ram_candidates = [c for c in candidates if is_model_in_ram(c, model_name)]
    if in_ram_candidates:
        candidates = in_ram_candidates

    return min(candidates, key=lambda x: active_requests[x])

# ---------------- ROUTES ----------------
@app.api_route("/run/{model_name}", methods=["GET", "POST", "PUT", "DELETE"])
async def run_model(model_name: str, request: Request):
    instance_name = choose_instance(model_name)
    target_url = f"{OLLAMA_INSTANCES[instance_name]}/run/{model_name}"

    active_requests[instance_name] += 1
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            body = await request.body()
            headers = dict(request.headers)
            response = await client.request(
                request.method, target_url, content=body, headers=headers
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_requests[instance_name] -= 1

# Catch-all route for everything else → transparent pass-through
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(full_path: str, request: Request):
    # Default route prefers CPU if no model-specific logic
    instance_name = "cpu"
    target_url = f"{OLLAMA_INSTANCES[instance_name]}/{full_path}"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            body = await request.body()
            headers = dict(request.headers)
            response = await client.request(
                request.method, target_url, content=body, headers=headers
            )
            return httpx.Response(
                status_code=response.status_code,
                content=response.content,
                headers=response.headers
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Status endpoint
@app.get("/status")
def status():
    status_info = {}
    for name in OLLAMA_INSTANCES.keys():
        path = RAMDISK_PATHS.get(name, "")
        ram_loaded = [f for f in os.listdir(path)] if os.path.exists(path) else []
        status_info[name] = {
            "active_requests": active_requests.get(name, 0),
            "models_in_ram": ram_loaded
        }
    return status_info
