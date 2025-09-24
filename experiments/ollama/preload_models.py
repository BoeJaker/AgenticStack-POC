import os
import shutil
import subprocess
import time

# RAM-disk path inside container
RAMDISK_PATH = os.environ.get("OLLAMA_MODEL_PATH", "/mnt/ramdisk")
# List of models to preload
MODELS_TO_PRELOAD = [
    "llama2:13b",
    "llama2:70b-q4",
    "mistral:7b",
    "mistral:30b",
]

def model_ram_path(model_name):
    # replace ":" with "_" to match folder naming in RAM-disk
    return os.path.join(RAMDISK_PATH, model_name.replace(":", "_"))

def load_model(model_name):
    """Pull the model into RAM using Ollama CLI"""
    path = model_ram_path(model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"[INFO] Preloading model: {model_name} → {path}")
    # Ollama load command
    cmd = ["ollama", "pull", model_name, "--path", path]
    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO] Successfully loaded {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")

def main():
    # Wait a few seconds for Ollama service to be ready
    time.sleep(5)
    for model in MODELS_TO_PRELOAD:
        load_model(model)
    print("[INFO] All models preloaded into RAM-disk.")

if __name__ == "__main__":
    main()
