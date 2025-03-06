import io
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
                width, height = 1024, 1024 #"□ Square (1:1)"
                if "landscape" in aspect_ratio.lower():#  == "▯ Landscape (16:9)"
                    width, height = 1344, 768
                elif "portrait" in aspect_ratio.lower(): # == "▤ Portrait (2:3)"
                    width, height = 832, 1248
                   
                
                generation_details = FluxParameters(
                    prompt=prompt,
                    #negative_prompt="unrealistic, ugly", # TODO: in ui
                    #TODO: from config
                    num_inference_steps=int(os.getenv("GENERATION_STEPS", 30 if self.generator.is_flux_dev() else 4)),
                    guidance_scale=float(os.getenv("GENERATION_GUIDANCE", 7.5 if self.generator.is_flux_dev() else 0)),
                    num_images_per_prompt=2,
                    width=width,
                    height=height
                )

                images = self.generator.generate_images(params=generation_details)
                logger.info(f"received {len(images)} image(s) from generator")
                for image in images:
                    #TODO: path via env
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
                    value="",
                    scale=4,
                    lines=4
                    #max_lines=4
                )
                
                # Aspect ratio selection
                aspect_ratio = gr.Radio(
                    choices=["□ Square (1:1)", "▤ Landscape (16:9)", "▯ Portrait (2:3)"],
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
                with gr.Accordion("Examples", open=False):
                # Examples
                    gr.Examples(
                        examples=[
                            ["A majestic mountain landscape at sunset with snow-capped peaks", "▤ Landscape (16:9)"],
                            ["A professional portrait of a business person in a modern office", "▯ Portrait (2:3)"],
                            ["A top-down view of a colorful mandala pattern", "□ Square (1:1)"]
                        ],
                        inputs=[prompt, aspect_ratio],
                        label="Click an example to load it"
                    )
            with gr.Row():        
                # Gallery for displaying generated images
                gallery = gr.Gallery(
                    label="Generated Images",
                    format="jpeg",
                    columns=2,
                    rows=1,
                    height="auto",
                )
            with gr.Row():
                download_btn = gr.DownloadButton("Download", visible=False)

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

            def prepare_download(selection: gr.SelectData):
                #gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")
                return gr.DownloadButton(label=f"Download", value=selection.value['image']['path'], visible=True)

            gallery.select(
                fn=prepare_download,
                inputs=None,
                outputs=[download_btn]
            )
            # download_btn.click(
            #     inputs=[gallery]
            # )

    def launch(self, **kwargs):
        if self.interface is None:
            self.create_interface()
        self.interface.launch(**kwargs)
