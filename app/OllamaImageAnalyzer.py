import os
import ollama
from PIL.Image import Image
from utils.singleton import singleton

@singleton
class OllamaImageAnalyzer:
    def __init__(self, model_name):
        self.model = "llava"

    def analyze_image(self, image_path):
        # Load the image
        if not os.path.exists(image_path): raise FileNotFoundError(image_path)
        #image = Image.load(image_path)
        response = ollama.chat(
            model="llama3.2-vision",
            messages=[{
            "role": "user",
            "content": "Is there a huma in the image?",
            "images": [image_path]
            }],
)
        cleaned_text = response['message']['content'].strip()
        print(f"Model Response: {cleaned_text}")

        # # # Analyze the image using the specified model
        # analysis_result = self.model.analyze(image)
        
        # Check if there is a human in the picture
        has_human = 'yes' in cleaned_text
        
        # Return the result as a dictionary
        return {
            'image_path': image_path,
            'has_human': has_human
        }
