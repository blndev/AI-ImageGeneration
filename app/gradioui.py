import os
import gradio as gr
from typing import List
from PIL import Image
import logging
from app.utils.singleton import singleton
from app.utils.fileIO import save_image_with_timestamp
from app.fluxparams import FluxParameters
from app.fluxgenerator import FluxGenerator

# Set up module logger
logger = logging.getLogger(__name__)
# 

@singleton
class GradioUI():
    def __init__(self):
        self.interface = None
        self.generator = FluxGenerator()

    def create_interface(self):
        # Define the generate function that will be called when the button is clicked
        def uiaction_generate_images(prompt, aspect_ratio):
            try:
                logger.info(f"generating image with prompt: {prompt} and aspect ratio: {aspect_ratio}")
                
                # Map aspect ratio selection to dimensions
                if aspect_ratio == "□ Square (1:1)":
                    width, height = 512, 512
                elif aspect_ratio == "▯ Landscape (16:9)":
                    width, height = 768, 432
                elif aspect_ratio == "▤ Portrait (9:16)":
                    width, height = 432, 768
                
                generation_details = FluxParameters(
                    prompt=prompt,
                    negative_prompt="unrealistic, ugly", # TODO: in ui
                    num_inference_steps=int(os.getenv("GENERATION_STEPS", 30 if self.generator.is_dev() else 4)),
                    guidance_scale=float(os.getenv("GENERATION_GUIDANCE", 7.5 if self.generator.is_dev() else 0)),
                    num_images_per_prompt=2,
                    width=width,
                    height=height
                )

                images = self.generator.generate_images(params=generation_details)
                logger.info(f"received {len(images)} image(s) from generator")
                for image in images:
                    save_image_with_timestamp(image=image, folder_path="./output", ignore_errors=True)
                return images
            except Exception as e:
                logger.error(f"image generation failed: {e}")
                gr.Warning(f"Error while generating the image: {e}")

        # Create the interface components
        with gr.Blocks() as self.interface:
            with gr.Row():
                # Text prompt input
                prompt = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Describe the image you want to generate...",
                    value="A cinematic shot of a baby cat wearing an intricate egypt priest robe.",
                    scale=4
                )
                
                # Aspect ratio selection
                aspect_ratio = gr.Radio(
                    choices=["□ Square (1:1)", "▯ Landscape (16:9)", "▤ Portrait (9:16)"],
                    value="□ Square (1:1)",
                    label="Aspect Ratio",
                    scale=1
                )
                
                # Generate button that's interactive only when prompt has text
                generate_btn = gr.Button(
                    "Generate",
                    interactive=False
                )
            with gr.Row():
                # Examples
                with gr.Column(scale=1):
                    gr.Markdown("### Examples")
                    gr.Examples(
                        examples=[
                            ["A majestic mountain landscape at sunset with snow-capped peaks", "▯ Landscape (16:9)"],
                            ["A professional portrait of a business person in a modern office", "▤ Portrait (9:16)"],
                            ["A top-down view of a colorful mandala pattern", "□ Square (1:1)"]
                        ],
                        inputs=[prompt, aspect_ratio],
                        label="Click an example to load it"
                    )
            with gr.Row():        
                # Gallery for displaying generated images
                gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=2,
                    height="auto"
                )

            # Make button interactive only when prompt has text
            prompt.change(
                fn=lambda x: gr.Button(interactive=len(x.strip()) > 0),
                inputs=[prompt],
                outputs=[generate_btn]
            )

            # Connect the generate button to the generate function
            generate_btn.click(
                fn=uiaction_generate_images,
                inputs=[prompt, aspect_ratio],
                outputs=[gallery]
            )

    def launch(self, **kwargs):
        if self.interface is None:
            self.create_interface()
        self.interface.launch(**kwargs)
