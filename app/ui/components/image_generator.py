# app/ui/components/image_generator.py
class ImageGeneratorComponent:
    def __init__(self, modelconfig, config):
        self.modelconfig = modelconfig
        self.config = config
        self.initialize_generator()

    def initialize_generator(self):
        if "flux" in self.modelconfig.model_type:
            self.generator = FluxGenerator(modelconfig=self.modelconfig)
        else:
            self.generator = StabelDiffusionGenerator(
                appconfig=self.config, 
                modelconfig=self.modelconfig
            )

    def create_interface_elements(self, gr):
        with gr.Row():
            prompt = gr.Textbox(
                label="Your prompt",
                placeholder="Describe the image you want to generate...",
                value="",
                scale=4,
                lines=4,
                max_length=340
            )
            # ... other interface elements

        return prompt  # Return all created elements

    def generate_images(self, session_state, prompt, aspect_ratio, neg_prompt, 
                       image_count, promptmagic_active):
        # Image generation logic here
        pass
