#!/usr/bin/env python3
import os
import time
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from datetime import datetime
from dotenv import load_dotenv
import gc


def setup_environment():
    """Configure GPU memory settings based on .env configuration"""
    if os.getenv('GPU_ALLOW_XFORMERS', '0') == '1':
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    if os.getenv('GPU_ALLOW_MEMORY_OFFLOAD', '0') == '1':
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_prompts():
    """Load prompts from prompts.txt file"""
    prompt_file = os.path.join(os.path.dirname(__file__), os.getenv("PROMPTS", 'prompts.txt'))
    prompts = None
    try:
        with open(prompt_file, 'r') as f:
            # prompts = [line.strip() for line in f if line.strip()]
            prompts = [line.strip() for line in f if not (line.strip().startswith('#') or len(line.strip()) == 0)]

    except FileNotFoundError:
        print(f"Warning: {prompt_file} not found, using default prompts")

    if not prompts:
        # Fallback prompts if file is empty
        prompts = [
            "a beautiful landscape with mountains and a lake, masterpiece, highly detailed",
            "a futuristic cityscape at night with neon lights and flying cars, cinematic, detailed"
        ]

    return prompts


def load_filters():
    """Load models filter if existing"""
    filters = []
    if os.getenv("MODEL_FILTER", None):
        filter_file = os.path.join(os.path.dirname(__file__), os.getenv("MODEL_FILTER", None))
        if not filter_file:
            return []
        try:
            with open(filter_file, 'r') as f:
                # filters = [line.strip() for line in f]
                filters = [line.strip() for line in f if not line.strip().startswith('#')]

        except FileNotFoundError:
            print(f"Warning: {filter_file} not found")

    return filters


def check_models():
    load_dotenv(override=True)
    setup_environment()

    models_path = os.getenv('MODEL_DIRECTORY', './text2img/models/')
    cache_path = os.getenv('CACHE_DIR')
    output_path = os.getenv('OUTPUT_DIRECTORY', './output')
    output_path = os.path.join(output_path, datetime.now().strftime("%Y%m%d_%H%M"))
    print(f"use '{models_path}' for scanning")

    # Load prompts at startup
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts for testing")
    filters = load_filters()

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Open the file for appending and write all prompts
    with open(os.path.join(output_path, 'prompts.txt'), 'a') as f:
        c = 1
        for prompt in prompts:
            f.write(f"{c} - {prompt}\n")
            c += 1

    # Find all safetensors files
    safetensors_files = []
    for root, dirs, files in os.walk(models_path):
        if not os.path.abspath(cache_path) in os.path.abspath(root):
            for file in files:
                if file.endswith('.safetensors') and (file in filters or len(filters) == 0):
                    model_path = os.path.join(root, file)
                    if model_path not in safetensors_files:  # Only add unique model directories
                        safetensors_files.append(model_path)

    print(f"Found {len(safetensors_files)} model directories")
    images = int(os.getenv("IMAGES", 1))
    modelcount = 0
    for file in safetensors_files:
        if modelcount > 0:
            print("cooldown GPU")
            time.sleep(30)  # rest between the generations to cool down
        try:
            modelcount += 1
            model_name = os.path.basename(file)
            print(f"\nTesting model {modelcount}/{len(safetensors_files)}: {model_name} from {file}")

            # Determine image size based on path
            height = width = 512 if "1.5" in file or "15" in file else 1024
            pt = StableDiffusionPipeline if "1.5" in file or "15" in file else StableDiffusionXLPipeline
            if "flux" in file.lower():
                pt = FluxPipeline
            print(f"Using resolution: {width}x{height}")

            # Load and test the model
            pipeline = None
            steps = 30
            if "hyper" in file.lower():
                steps = 5
            if "flux-schnell" in file.lower():
                steps = 5
            try:
                pipeline = pt.from_single_file(
                    file,
                    cache_dir=os.getenv("CACHE_DIR", "./models"),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=False,
                    requires_safety_checker=False,
                    safety_checker=None  # Disable safety checker for faster inference
                ).to("cuda")

                if os.getenv('GPU_ALLOW_ATTENTION_SLICING', '0') == '1':
                    pipeline.enable_attention_slicing()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Generate test images for each prompt
                for i, prompt in enumerate(prompts, 1):
                    neg_prompt = ""
                    if "||" in prompt:
                        p = prompt.split("||")
                        prompt = p[0]
                        neg_prompt = p[1]

                    print(f"Prompt: '{prompt}', \nNeg Prompt: '{neg_prompt}'")
                    for imagecount in range(images):
                        output_filename = f"M{modelcount:02}-P{i}-V{imagecount + 1}_{model_name}--{timestamp}.jpg"
                        output_path_full = os.path.join(output_path, output_filename)
                        print(f"Generating image {imagecount + 1}/{images} of prompt {i}/{len(prompts)}...")

                        if len(prompt.split()) > 0:
                            image = pipeline(
                                prompt=prompt,
                                negative_prompt=neg_prompt,
                                height=height,
                                width=width,
                                num_inference_steps=steps,
                                device_map="auto",
                                # num_inference_steps=40,
                                # strength=1,
                                # guidance_scale=7.5
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
