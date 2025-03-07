import os
from dotenv import load_dotenv
from app import GradioUI, setup_logging
from distutils.util import strtobool
load_dotenv(override=True)
setup_logging()
if __name__ == "__main__":
    ui = GradioUI()
    ui.launch(
        share=bool(strtobool(os.getenv("GRADIO_SHARED", "False"))),
        server_name="0.0.0.0",
        show_api=False,
        enable_monitoring=False
    )
