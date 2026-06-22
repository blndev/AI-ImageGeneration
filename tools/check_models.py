#!/usr/bin/env python3
import os
import time
import torch
import json
import random
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
            prompts = [line.strip() for line in f if not (line.strip().startswith(
                '#') or line.strip().startswith(';') or len(line.strip()) == 0)]

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


def should_use_fixed_seeds():
    """Check if fixed seed mode is enabled via .env"""
    fixed_seed = os.getenv('FIXED_SEED', 'True').lower()
    return fixed_seed == 'true'


def initialize_seeds(seed_file_path, prompts):
    """Load or initialize seed dictionary in prompt-level format"""
    seeds = {}
    if os.path.exists(seed_file_path):
        if os.path.getsize(seed_file_path) == 0:
            seeds = {}
        else:
            try:
                with open(seed_file_path, 'r') as f:
                    seeds = json.load(f)
                    print(f"Loaded existing seeds from {seed_file_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load seed file {seed_file_path}: {e}. Starting fresh.")

    updated = False
    for index, prompt in enumerate(prompts, 1):
        prompt_key = f"prompt_{index}"
        if prompt_key not in seeds:
            seeds[prompt_key] = {
                "seed": int(torch.seed()),
                "prompt_text": prompt,
            }
            updated = True
        else:
            if "seed" not in seeds[prompt_key]:
                seeds[prompt_key]["seed"] = int(torch.seed())
                updated = True
            if seeds[prompt_key].get("prompt_text") != prompt:
                seeds[prompt_key]["prompt_text"] = prompt
                updated = True

    if updated and seed_file_path:
        save_seeds(seeds, seed_file_path)
    return seeds


def save_seeds(seeds, seed_file_path):
    """Save seeds to JSON file"""
    try:
        seed_dir = os.path.dirname(seed_file_path)
        if seed_dir:
            os.makedirs(seed_dir, exist_ok=True)
        with open(seed_file_path, 'w') as f:
            json.dump(seeds, f, indent=2)
        print(f"Seeds saved to {seed_file_path}")
    except IOError as e:
        print(f"Error saving seeds to {seed_file_path}: {e}")


def get_or_generate_seed(seeds, prompt_index, prompt_text):
    """Get existing seed or generate a new one using prompt-level format"""
    prompt_key = f"prompt_{prompt_index}"
    if prompt_key not in seeds:
        seeds[prompt_key] = {
            "seed": int(torch.seed()),
            "prompt_text": prompt_text,
        }
    else:
        if "seed" not in seeds[prompt_key]:
            seeds[prompt_key]["seed"] = int(torch.seed())
        if seeds[prompt_key].get("prompt_text") != prompt_text:
            seeds[prompt_key]["prompt_text"] = prompt_text

    return seeds[prompt_key]["seed"]


def check_models():
    setup_environment()

    models_path = os.getenv('MODEL_DIRECTORY', './text2img/models/')
    cache_path = os.getenv('CACHE_DIR')
    output_path = os.getenv('OUTPUT_DIRECTORY', './output')
    output_path = os.path.join(output_path, datetime.now().strftime("%Y%m%d_%H%M"))
    print(f"use '{models_path}' for scanning")

    # Load prompts at startup
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts for testing")

    # Find all safetensors files
    safetensors_files = find_safetensor_models(models_path, cache_path)

    if len(safetensors_files) == 0:
        print("no model found")
        exit()
    print(f"Found {len(safetensors_files)} model directories")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Open the file for appending and write all prompts
    with open(os.path.join(output_path, 'prompts.txt'), 'a') as f:
        c = 1
        for prompt in prompts:
            f.write(f"{c} - {prompt}\n")
            c += 1

    # Initialize seed management
    use_fixed_seeds = should_use_fixed_seeds()
    seeds = {}
    seed_file_path = None
    if use_fixed_seeds:
        seed_file_path = os.getenv('SEED_JSON', os.path.join(output_path, 'seeds.json'))
        seeds = initialize_seeds(seed_file_path, prompts)
        print(f"Fixed seed mode: ENABLED (seeds will be saved to {seed_file_path})")
    else:
        print("Fixed seed mode: DISABLED (random seeds for each generation)")

    images = int(os.getenv("IMAGES", 1))
    modelcount = 0
    for file in safetensors_files:
        if modelcount > 0:
            # pause between the generations to cool down, more model = longer
            rest_time = 30 + modelcount * 5
            print(f"cooldown GPU for {rest_time}s")
            time.sleep(rest_time)
        try:
            modelcount += 1
            model_name = os.path.basename(file)
            print(f"\nTesting model {modelcount}/{len(safetensors_files)}: {model_name} from {file}")

            # Determine image size based on path
            aspect_ratio = []
            if "1.5" in file or "15" in file:
                aspect_ratio = [
                    (512, 512),
                    (912, 512),  # 16:9
                    (512, 768),  # 2:3 (hochformat)
                    (768, 512),  # 2:3 Standard Quer
                ]

            else:
                aspect_ratio = [
                    (1024, 1024),
                    (1664, 928),  # 16:9
                    (1344, 768), (768, 1344)   # 7:4
                ]

            pt = StableDiffusionPipeline if "1.5" in file or "15" in file else StableDiffusionXLPipeline
            if "flux" in file.lower():
                pt = FluxPipeline
            # print(f"Using resolution: {width}x{height}")

            # Load and test the model
            pipeline = None
            steps = 30
            guidance_scale = 7.5
            if "1.5" in file or "15" in file:
                steps = 50
                guidance_scale = 7.5
            if "hyper" in file.lower():
                steps = 5
            if "flux-schnell" in file.lower():
                steps = 5
                guidance_scale = 0
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

                # Generate test images for each prompt
                rest_image_count = 0
                for i, prompt in enumerate(prompts, 1):

                    neg_prompt = None
                    if "||" in prompt:
                        p = prompt.split("||")
                        prompt = p[0]
                        neg_prompt = p[1]

                    print(f"Prompt: '{prompt}'")
                    if neg_prompt: print(f"Neg Prompt: '{neg_prompt}'")
                    for imagecount in range(images):
                        rest_image_count += 1
                        if rest_image_count > 15:
                            rest_image_count = 0
                            rest_time = 30
                            print(f"cooldown GPU for {rest_time}s")
                            time.sleep(rest_time)

                        # Get or generate seed for this prompt and image count
                        if use_fixed_seeds:
                            current_seed = get_or_generate_seed(seeds, i, prompt)
                            print(f"Using seed: {current_seed}")
                            # Save seeds to file if fixed seed mode was enabled
                            if use_fixed_seeds and seed_file_path:
                                save_seeds(seeds, seed_file_path)

                        else:
                            current_seed = None

                        for width, height in aspect_ratio:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"M{modelcount:02}-P{i}-I{imagecount + 1}_{model_name}-{width}x{height}-{timestamp}.jpg"
                            output_path_full = os.path.join(output_path, output_filename)
                            print(f"Generating image {imagecount + 1}/{images} of prompt {i}/{len(prompts)} in ratio {width}x{height}...")

                            if len(prompt.strip()) > 0:
                                try:
                                    gen_kwargs = {
                                        "prompt": prompt,
                                        "negative_prompt": neg_prompt,
                                        "height": height,
                                        "width": width,
                                        "num_inference_steps": steps,
                                        "device_map": "auto",
                                    }
                                    # Add guidance_scale for SD1.5 and SDXL models (not for FLUX)
                                    if "flux" not in file.lower():
                                        gen_kwargs["guidance_scale"] = guidance_scale

                                    # Add clip_skip for SD1.5 models
                                    if "1.5" in file or "15" in file:
                                        gen_kwargs["clip_skip"] = 2

                                    if use_fixed_seeds and current_seed is not None:
                                        gen_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(current_seed)

                                    # Generate image
                                    image = pipeline(**gen_kwargs).images[0]

                                    # Save the image with quality settings
                                    image.save(output_path_full, quality=95, optimize=True)
                                    print(f"Generated image saved as: {output_filename}")

                                    # Free memory
                                    del image

                                except Exception as img_error:
                                    print(f"Error generating/saving image: {img_error}")
                                    continue

            finally:
                # Cleanup
                if pipeline is not None:
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()

        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue


def find_safetensor_models(models_path, cache_path):
    filters = load_filters()
    safetensors_files = []
    for root, dirs, files in os.walk(models_path):
        if not os.path.abspath(cache_path) in os.path.abspath(root):
            for file in files:
                if file.endswith('.safetensors') and (file in filters or len(filters) == 0):
                    model_path = os.path.join(root, file)
                    if model_path not in safetensors_files:  # Only add unique model directories
                        safetensors_files.append(model_path)
    safetensors_files = sorted(safetensors_files)
    return safetensors_files


if __name__ == "__main__":
    try:
        load_dotenv(override=True)
        check_models()
    except KeyboardInterrupt:
        print("Shutdown")
        exit()
