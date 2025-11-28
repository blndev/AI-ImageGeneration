#!/usr/bin/env python3
import os
from collections import Counter, defaultdict
from pathlib import Path
from dotenv import load_dotenv

def parse_filename(filename):
    stem = Path(filename).stem
    parts = stem.split('_', 1)
    if len(parts) != 2:
        return None

    prefix = parts[0]           # z. B. M10-P10-V2
    rest = parts[1]             # z. B. intorealismUltra_v20.safetensors-1024x1024-...

    # Extract M-value (M10, M5, etc.)
    m_value = prefix.split('-')[0] if '-' in prefix and prefix.startswith('M') else None
    if not m_value:
        return None

    # Extract model name (before .safetensors, after last _ if vXX exists)
    model_part = rest.split('.safetensors')[0]
    model_name = model_part.rsplit('_', 1)[0]  # removes _v20 etc.

    # Full model identifier: M10-intorealismUltra
    full_model = f"{m_value}-{model_name}"

    # Resolution
    resolution = next((p for p in rest.split('-') if 'x' in p and p.replace('x', '').isdigit()), None)
    if not resolution:
        return None

    # Prompt (P10, P2, etc.)
    prompt = None
    if '-P' in prefix:
        p_part = prefix.split('-P')[1]
        prompt_num = p_part.split('-')[0].split('_')[0]
        prompt = f"P{prompt_num}"

    if not prompt:
        return None

    return {
        'full_model': full_model,
        'model_name': model_name,
        'm_value': m_value,
        'resolution': resolution,
        'prompt': prompt
    }

def analyze_directory(directory_to_scan):
    root = Path(directory_to_scan)

    if not root.exists():
        print(f"Error: Directory not found: {root}")
        return

    model_counter = Counter()
    resolution_counter = Counter()
    prompt_counter = Counter()

    model_resolutions = defaultdict(set)
    model_prompts = defaultdict(set)

    print(f"Scanning directory: {root.resolve()}")
    print("Parsing files...\n")

    total_files = 0
    for jpg_file in root.rglob("*.jpg"):
        parsed = parse_filename(jpg_file.name)
        if not parsed:
            continue

        total_files += 1
        model_counter[parsed['full_model']] += 1
        resolution_counter[parsed['resolution']] += 1
        prompt_counter[parsed['prompt']] += 1

        model_resolutions[parsed['full_model']].add(parsed['resolution'])
        model_prompts[parsed['full_model']].add(parsed['prompt'])

    # =============== OUTPUT ===============
    print("\n" + "="*90)
    print("AI IMAGE GENERATION STATISTICS")
    print("="*90)

    # 1. Models (with M-value) + details
    print(f"\n{'Model':<28} {'Images':>8} {'Resolutions':<48} {'Prompts'}")
    print("-" * 90)
    for full_model, count in model_counter.most_common():
        res_list = sorted(model_resolutions[full_model])
        prompt_list = sorted(model_prompts[full_model])

        # Shorten resolution list if too long
        if len(res_list) > 6:
            res_display = ", ".join(res_list[:6]) + ", ..."
        else:
            res_display = ", ".join(res_list)

        prompts_display = ", ".join(prompt_list)
        print(f"{full_model:<28} {count:>8}   {res_display:<46} {prompts_display}")

    # 2. Global Resolutions
    print(f"\n\n{'Resolution':<12} {'Count':>6}")
    print("-" * 30)
    for res, count in resolution_counter.most_common():
        print(f"{res:<12} {count:>6}")

    # 3. Global Prompts
    print(f"\n{'Prompt':<6} {'Count':>6}")
    print("-" * 20)
    for prompt, count in prompt_counter.most_common():
        print(f"{prompt:<6} {count:>6}")

    print(f"\nTotal images analyzed: {total_files}")
    print(f"Models found: {len(model_counter)}")

if __name__ == "__main__":
    # Load .env file
    load_dotenv(override=True)

    # Read directory from environment (fallback to ./output)
    directory_to_scan = os.getenv('RESULT_DIRECTORY', './output')

    print(f"Starting analysis...")
    print(f"Source directory (.env): RESULT_DIRECTORY = {directory_to_scan}\n")

    analyze_directory(directory_to_scan)