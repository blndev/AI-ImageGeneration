from dotenv import load_dotenv
from app import GradioUI, setup_logging

setup_logging()
if __name__ == "__main__":
    load_dotenv()
    ui = GradioUI()
    ui.launch(
        share=False, 
        server_name="0.0.0.0",
        show_api=False,
        enable_monitoring=False
    )