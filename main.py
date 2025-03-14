import json
import os
from dotenv import load_dotenv
from app import GradioUI, setup_logging
from app.generators import ModelConfig

from distutils.util import strtobool
load_dotenv(override=True)
setup_logging()
if __name__ == "__main__":

    mc = None
    mc_path = os.getenv("MODELCONFIG", "./modelconfig.json")
    if os.path.exists(mc_path):
        with open(mc_path, "r") as f:
            try:
                j = f.read()
                mc = ModelConfig.from_json(j)
            except Exception as e:
                print(f"Startup failed while reading model config from '{mc_path}'")
                raise 
    
    ui = GradioUI(modelconfigs=mc)
    ui.launch(
        share=bool(strtobool(os.getenv("GRADIO_SHARED", "False"))),
        server_name="0.0.0.0",
        show_api=False,
        enable_monitoring=False
    )
