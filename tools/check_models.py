#!/usr/bin/env python3
import os
import torch
from diffusers import AutoPipelineForText2Image
from datetime import datetime
import glob
from dotenv import load_dotenv
import gc

def setup_environment():
    """Configure GPU memory settings based on .env configuration"""
    if os.getenv('GPU_ALLOW_XFORMERS', '0') == '1':
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    
    if os.getenv('GPU_ALLOW_MEMORY_OFFLOAD', '0') == '1':
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def check_models():
    load_dotenv()
    setup_environment()
    
    models_path = os.getenv('MODEL_DIRECTORY', '/home/me/ai-models/')
    output_path = os.getenv('OUTPUT_DIRECTORY', './output')
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Find all safetensors files
    safetensors_files = []
    for root, dirs, files in os.walk(models_path):
        for file in files:
            if file.endswith('.safetensors'):
                model_dir = os.path.dirname(os.path.join(root, file))
                if model_dir not in safetensors_files:  # Only add unique model directories
                    safetensors_files.append(model_dir)
    
    print(f"Found {len(safetensors_files)} model directories")
    
    for model_dir in safetensors_files:
        try:
            model_name = os.path.basename(model_dir)
            print(f"\nTesting model: {model_name}")
            
            # Determine image size based on path
            height = width = 512 if "1.5" in model_dir else 1024
            print(f"Using resolution: {width}x{height}")
            
            # Load and test the model
            pipeline = None
            try:
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    safety_checker=None  # Disable safety checker for faster inference
                ).to("cuda")
                
                if os.getenv('GPU_ALLOW_ATTENTION_SLICING', '0') == '1':
                    pipeline.enable_attention_slicing()
                
                # Generate a test image
                prompt = "a beautiful landscape with mountains and a lake, masterpiece, highly detailed"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{model_name}_{timestamp}.jpg"
                output_path_full = os.path.join(output_path, output_filename)
                
                print(f"Generating test image...")
                image = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                
                # Save the image
                image.save(output_path_full)
                print(f"Generated image saved as: {output_filename}")
                
            finally:
                # Cleanup
                if pipeline is not None:
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()
                
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    check_models()
