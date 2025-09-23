from jinja2 import Environment, FileSystemLoader
import os

# Load config from .env
from dotenv import load_dotenv
load_dotenv()

GPU_NAMES = os.getenv("GPU_NAMES", "").split(",")
GPU_IDS = list(map(str, os.getenv("GPU_IDS", "").split(",")))
GPU_INSTANCES_PER_GPU = int(os.getenv("GPU_INSTANCES_PER_GPU", "1"))
CPU_CORES = os.getenv("CPU_CORES", "8")
CPU_MEMORY = os.getenv("CPU_MEMORY", "64G")
CPU_RAMDISK_SIZE = os.getenv("CPU_RAMDISK_SIZE", "64G")
GPU_RAMDISK_SIZE = os.getenv("GPU_RAMDISK_SIZE", "64G")
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "")

gpu_list = list(zip(GPU_NAMES, GPU_IDS))

env = Environment(loader=FileSystemLoader("."))
template = env.get_template("docker-compose.yml.j2")

rendered = template.render(
    gpu_list=gpu_list,
    gpu_instances_per_gpu=GPU_INSTANCES_PER_GPU,
    CPU_CORES=CPU_CORES,
    CPU_MEMORY=CPU_MEMORY,
    CPU_RAMDISK_SIZE=CPU_RAMDISK_SIZE,
    GPU_RAMDISK_SIZE=GPU_RAMDISK_SIZE,
    gpu_names=GPU_NAMES,
    gpu_ids=GPU_IDS,
    PRELOAD_MODELS=PRELOAD_MODELS
)

with open("docker-compose.yml", "w") as f:
    f.write(rendered)

print("docker-compose.yml generated successfully!")
