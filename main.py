import json
import logging
import os
from dotenv import load_dotenv
from app import AppConfig, GradioUI, setup_logging
from app.generators import ModelConfig

from distutils.util import strtobool
load_dotenv(override=True)
setup_logging()
logger = logging.getLogger("app")

import torch

# # Get the current GPU memory usage
# def print_gpu_memory_usage():
#     if torch.cuda.is_available():
#         # Get the GPU ID
#         gpu_id = torch.cuda.current_device()
        
#         # Get the total and allocated memory
#         total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
#         allocated_memory = torch.cuda.memory_allocated(gpu_id)
#         cached_memory = torch.cuda.memory_reserved(gpu_id)
        
#         logger.info(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
#         logger.debug(f"Allocated GPU memory: {allocated_memory / 1024**3:.2f} GB")
#         logger.debug(f"Cached GPU memory: {cached_memory / 1024**3:.2f} GB")
#     else:
#         pass

def get_gpu_info():
    # Überprüfen, ob CUDA verfügbar ist
    if torch.cuda.is_available():
        # Anzahl der verfügbaren GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        
        # Namen der GPUs abrufen
        for gpu_id in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            cached_memory = torch.cuda.memory_reserved(gpu_id)
            logger.info(f"using GPU {gpu_id}: {gpu_name} / {total_memory}GB (Available {total_memory - allocated_memory}GB)")
    else:
        logger.warning("No CUDA GPUs available")

if __name__ == "__main__":

    config = AppConfig()
    mc = None
    mc_path = config.modelconfig_json
    logger.info(f"Initialize modelconfigs from {mc_path}")
    if os.path.exists(mc_path):
        with open(mc_path, "r") as f:
            try:
                j = f.read()
                mc = ModelConfig.create_config_list_from_json(j)
                for m in mc:
                    logger.debug(f"Available model: '{m.model}'->'{m.parent}' from '{m.path}'")
                
            except Exception as e:
                logger.error(f"Startup failed while reading model config from '{mc_path}': {e}")
                exit(1)
    else:
        logger.error(f"File {mc_path} does not exist. Modelconfigs could not be loaded. Exit 1")
        exit(1)

    get_gpu_info()

    ui = GradioUI(modelconfigs=mc)
    ui.launch(
        share=bool(strtobool(os.getenv("GRADIO_SHARED", "False"))),
        server_name="0.0.0.0",
        show_api=False,
        enable_monitoring=False
    )
