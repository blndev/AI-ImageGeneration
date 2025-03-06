from datetime import datetime
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
        self.output_directory = os.getenv("OUTPUT_DIRECTORY", None)

    def create_interface(self):
        # Define the generate function that will be called when the button is clicked
        def uiaction_generate_images(prompt, aspect_ratio, neg_prompt, image_count, request: gr.Request):
            try:
                logger.info(f"generating image with prompt: {prompt} and aspect ratio: {aspect_ratio}")

                url = request.url
                logger.debug(f"Request URL: {url}")
                # preparation for ?share=CODE
                share_value = request.query_params.get("share")
                logger.debug(f"Share parameter: {share_value}")  # Would print "55342"

                # Map aspect ratio selection to dimensions
                width, height = 1024, 1024  # "□ Square (1:1)"
                if "landscape" in aspect_ratio.lower():  # == "▯ Landscape (16:9)"
                    width, height = 1344, 768
                elif "portrait" in aspect_ratio.lower():  # == "▤ Portrait (2:3)"
                    width, height = 832, 1248

                steps = int(os.getenv("GENERATION_STEPS", 30 if self.generator.is_flux_dev() or self.generator.SDXL else 4))
                guidance = float(os.getenv("GENERATION_GUIDANCE",
                                 7.5 if self.generator.is_flux_dev() or self.generator.SDXL else 0))

                generation_details = FluxParameters(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    num_images_per_prompt=image_count,
                    width=width,
                    height=height
                )

                images = self.generator.generate_images(params=generation_details)
                logger.debug(f"received {len(images)} image(s) from generator")
                if self.output_directory is not None:
                    logger.debug(f"saving images to {self.output_directory}")
                    for image in images:
                        outdir = os.path.join(self.output_directory, datetime.now().strftime("%Y-%m-%d"))
                        save_image_with_timestamp(image=image, folder_path=outdir, ignore_errors=True)
                return images
            except Exception as e:
                logger.error(f"image generation failed: {e}")
                gr.Warning(f"Error while generating the image: {e}")

        # Create the interface components
        with gr.Blocks(title="Image Generator") as self.interface:
            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Describe the image you want to generate...",
                        value="",
                        scale=4,
                        lines=4
                    )

                    # Text prompt input
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="Describe what you don't want...",
                        value="",
                        scale=4,
                        lines=4
                    )

                with gr.Column():
                    # Aspect ratio selection
                    aspect_ratio = gr.Radio(
                        choices=["□ Square (1:1)", "▤ Landscape (16:9)", "▯ Portrait (2:3)"],
                        value="□ Square (1:1)",
                        label="Aspect Ratio",
                        scale=1
                    )
                    image_count = gr.Slider(
                        label="Image Count",
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
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
                            [
                                "A majestic mountain landscape at sunset with snow-capped peaks",
                                "painting",
                                "▤ Landscape (16:9)",
                                1
                            ],
                            [
                                "A professional portrait of a business person in a modern office",
                                "ugly face",
                                "▯ Portrait (2:3)",
                                1
                            ],
                            [
                                "A top-down view of a colorful mandala pattern",
                                "realistic",
                                "□ Square (1:1)",
                                1
                            ]
                        ],
                        inputs=[prompt, neg_prompt, aspect_ratio, image_count],
                        label="Click an example to load it"
                    )
            with gr.Row():
                # Gallery for displaying generated images
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_share_button=False,
                    show_download_button=True,
                    format="jpeg",
                    columns=4,
                    rows=1,
                    height="auto",
                    object_fit="cover",
                    preview=True
                )
            with gr.Row():
                download_btn = gr.DownloadButton("Download", visible=False)
                timer = gr.Timer(60)
                timer.tick(
                    fn=lambda: (gr.Info("Check for Token!")),
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
                inputs=[prompt, aspect_ratio, neg_prompt, image_count],
                outputs=[gallery]
            )

            def prepare_download(selection: gr.SelectData):
                # gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")
                # Create a custom filename (for example, using timestamp)
                file_path = selection.value['image']['path']
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # custom_filename = f"generated_image_{timestamp}.png"
                return gr.DownloadButton(label=f"Download", value=file_path, visible=True)

            gallery.select(
                fn=prepare_download,
                inputs=None,
                outputs=[download_btn]
            )

    def launch(self, **kwargs):
        if self.interface is None:
            self.create_interface()
        self.interface.launch(**kwargs)
        # self.generator.warmup()
        # gr.Interface.from_pipeline(self.generator._cached_generation_pipeline).launch()
