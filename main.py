from app import GradioUI, setup_logging

setup_logging()
if __name__ == "__main__":
    ui = GradioUI()
    ui.launch(share=False)