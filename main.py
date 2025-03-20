import json
import logging
import os
from dotenv import load_dotenv
from app import GradioUI, setup_logging
from app.generators import ModelConfig

from distutils.util import strtobool
load_dotenv(override=True)
setup_logging()
logger = logging.getLogger("app")

import torch

# Get the current GPU memory usage
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        # Get the GPU ID
        gpu_id = torch.cuda.current_device()
        
        # Get the total and allocated memory
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        cached_memory = torch.cuda.memory_reserved(gpu_id)
        
        logger.info(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
        logger.info(f"Allocated GPU memory: {allocated_memory / 1024**3:.2f} GB")
        logger.info(f"Cached GPU memory: {cached_memory / 1024**3:.2f} GB")
    else:
        pass

def get_gpu_info():
    # Überprüfen, ob CUDA verfügbar ist
    if torch.cuda.is_available():
        # Anzahl der verfügbaren GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        
        # Namen der GPUs abrufen
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        logger.warning("No CUDA GPUs available")

if __name__ == "__main__":

    mc = None
    mc_path = os.getenv("MODELCONFIG", "./modelconfig.json")
    if os.path.exists(mc_path):
        with open(mc_path, "r") as f:
            try:
                j = f.read()
                mc = ModelConfig.create_config_list_from_json(j)
            except Exception as e:
                print(f"Startup failed while reading model config from '{mc_path}'")
                raise 

    get_gpu_info()
    print_gpu_memory_usage()

    ui = GradioUI(modelconfigs=mc)
    ui.launch(
        share=bool(strtobool(os.getenv("GRADIO_SHARED", "False"))),
        server_name="0.0.0.0",
        show_api=False,
        enable_monitoring=False
    )
