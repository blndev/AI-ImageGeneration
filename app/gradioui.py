import gradio as gr
from typing import List
from PIL import Image
import logging
from app.utils.singleton import singleton
from app.fluxgenerator import FluxGenerator
# Set up module logger
logger = logging.getLogger(__name__)
# self.generator = FluxGenerator()

@singleton
class GradioUI():
    def __init__(self):
        self.interface = None

    def create_interface(self):
        # Define the generate function that will be called when the button is clicked
        def generate_images(prompt):
            # This is a placeholder - implement your actual image generation logic here
            # For now, returning an empty list
            return []

        # Create the interface components
        with gr.Blocks() as self.interface:
            with gr.Row():
                # Text prompt input
                prompt = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Describe the image you want to generate...",
                    scale=4
                )
                
                # Generate button that's interactive only when prompt has text
                generate_btn = gr.Button(
                    "Generate",
                    interactive=False
                )

            # Gallery for displaying generated images
            gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=2,
                height="auto"
            )

            # Make button interactive only when prompt has text
            prompt.change(
                fn=lambda x: len(x.strip()) > 0,
                inputs=[prompt],
                outputs=[generate_btn]
            )

            # Connect the generate button to the generate function
            generate_btn.click(
                fn=generate_images,
                inputs=[prompt],
                outputs=[gallery]
            )

    def launch(self, **kwargs):
        if self.interface is None:
            self.create_interface()
        self.interface.launch(**kwargs)
