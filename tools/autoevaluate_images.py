#!/usr/bin/env python3
"""
Image Quality Evaluation Pipeline
==================================
Automatically evaluates generated images for quality metrics:
- CLIP Score: Prompt-to-image alignment (higher is better, 0-1)
- BRISQUE: No-reference image quality (lower is better, 0-100)
- Aesthetics: Aesthetic quality prediction (higher is better, 0-1)

Model Downloads:
- CLIP ViT-B/32: ~350MB (downloads on first run)
- BRISQUE: Built-in weights (downloads on first run)

Requirements: pip install -r requirements-evaluation.txt
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Import CLIP
try:
    import clip
except ImportError:
    print("ERROR: CLIP not found. Install with:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    exit(1)

# Import BRISQUE
try:
    from imquality import brisque
except ImportError:
    print("ERROR: imquality not found. Install with:")
    print("  pip install imquality")
    exit(1)


class ImageEvaluator:
    """Evaluate image quality using multiple metrics"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("Loading CLIP model (ViT-B/32)...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        print("CLIP model loaded successfully")

    def clip_score(self, image: Image.Image, prompt: str) -> float:
        """
        Calculate CLIP similarity score between image and prompt.
        Range: 0-1 (higher is better)
        """
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                similarity = (image_features @ text_features.T).item()
            
            return float(max(0, min(1, (similarity + 1) / 2)))  # Normalize to 0-1
        except Exception as e:
            print(f"Error in CLIP scoring: {e}")
            return -1.0

    def brisque_score(self, image: Image.Image) -> float:
        """
        Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).
        Range: 0-100 (lower is better)
        """
        try:
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            score = float(brisque.score(image))
            return score
        except Exception as e:
            print(f"Error in BRISQUE scoring: {e}")
            return -1.0

    def aesthetics_score(self, image: Image.Image) -> float:
        """
        Placeholder for aesthetics scoring.
        Could be extended with LAION-Aesthetics or similar model.
        For now returns a confidence indicator based on image statistics.
        """
        try:
            img_array = np.array(image)
            # Simple heuristics: brightness and contrast
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            
            # Combine metrics (very basic; can be replaced with proper model)
            aesthetics = (brightness * 0.3 + contrast * 0.7) / 1.0
            return float(max(0, min(1, aesthetics)))
        except Exception as e:
            print(f"Error in aesthetics scoring: {e}")
            return -1.0

    def evaluate_image(self, image_path: Path, prompt: str) -> Dict:
        """Evaluate a single image"""
        try:
            img = Image.open(image_path).convert("RGB")
            
            scores = {
                "file": str(image_path.name),
                "full_path": str(image_path),
                "prompt": prompt,
                "clip_score": self.clip_score(img, prompt),
                "brisque": self.brisque_score(img),
                "aesthetics": self.aesthetics_score(img),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add quality rating based on scores
            if scores["clip_score"] >= 0 and scores["brisque"] >= 0:
                quality_rating = (scores["clip_score"] * 100 - scores["brisque"] * 0.5) / 100
                scores["quality_rating"] = float(max(0, min(100, quality_rating)))
            else:
                scores["quality_rating"] = -1.0
            
            return scores
        except Exception as e:
            print(f"Error evaluating {image_path}: {e}")
            return {
                "file": str(image_path.name),
                "error": str(e),
            }


def extract_prompt_from_filename(filename: str) -> str:
    """
    Try to extract original prompt from filename.
    Filename format: M{modelcount:02}-P{i}-I{imagecount + 1}_{model_name}-{width}x{height}-{timestamp}.jpg
    This is a heuristic; better to load from seeds.json or prompts.txt
    """
    return "N/A"


def load_prompts_from_seeds(seeds_file: Path) -> Dict:
    """Load prompts from seeds.json (primary source)"""
    prompts = {}
    try:
        with open(seeds_file, 'r', encoding='utf-8') as f:
            seeds_data = json.load(f)
            for key, value in seeds_data.items():
                if isinstance(value, dict) and "prompt_text" in value:
                    try:
                        if key.startswith("prompt_"):
                            prompt_num = int(key.split("_")[1])
                            prompt_text = value.get("prompt_text", "")
                            if prompt_text:
                                prompts[prompt_num] = prompt_text
                    except Exception:
                        continue
        if prompts:
            print(f"Loaded {len(prompts)} prompts from seeds.json")
            return prompts
    except FileNotFoundError:
        print(f"Warning: {seeds_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error decoding seeds.json: {e}")
    return {}


def evaluate_output_directory(output_dir: str, evaluator: ImageEvaluator) -> List[Dict]:
    """Evaluate all images in the check_models.py output directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory {output_dir} not found")
        return []
    
    # Load prompts from seeds.json (primary) or prompts.txt (fallback)
    seeds_file = output_path / "seeds.json"
    prompts_dict = {}
    
    if seeds_file.exists():
        prompts_dict = load_prompts_from_seeds(seeds_file)
    
    # Fallback to prompts.txt if seeds.json didn't work
    if not prompts_dict:
        prompts_file = output_path / "prompts.txt"
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                prompts_dict = {i + 1: prompt for i, prompt in enumerate(lines)}
            if prompts_dict:
                print(f"Loaded {len(prompts_dict)} prompts from prompts.txt")
        except FileNotFoundError:
            print(f"Warning: Neither seeds.json nor prompts.txt found in {output_dir}")
    
    print(f"\nFound {len(prompts_dict)} prompts: {prompts_dict}")
    
    # Find all jpg/png files
    image_files = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
    print(f"Found {len(image_files)} images to evaluate")
    
    if not image_files:
        print("No images found in output directory")
        return []
    
    results = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Evaluating {img_file.name}...", end=" ", flush=True)
        
        # Extract prompt index from filename (M02-P1-I1 -> P1 -> prompt_1)
        try:
            parts = img_file.stem.split("_")
            if len(parts) > 0:
                # Extract prompt number from first part (M02-P1-I1)
                filename_parts = parts[0].split("-")
                prompt_idx = next((p for p in filename_parts if p.startswith("P")), None)
                if prompt_idx:
                    prompt_num = int(prompt_idx[1:])
                    prompt = prompts_dict.get(prompt_num, "N/A")
                else:
                    prompt = "N/A"
            else:
                prompt = "N/A"
        except Exception as e:
            print(f"(error parsing prompt: {e})")
            prompt = "N/A"
        
        result = evaluator.evaluate_image(img_file, prompt)
        results.append(result)
        
        # Print scores
        if "error" not in result:
            print(f"CLIP: {result['clip_score']:.3f}, BRISQUE: {result['brisque']:.2f}, Quality: {result.get('quality_rating', -1):.2f}")
        else:
            print(f"ERROR: {result['error']}")
    
    return results


def save_results(results: List[Dict], output_file: Path):
    """Save evaluation results to JSON in structured format"""
    # Separate valid results
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    
    # Extract metrics for summary
    clip_scores = [r["clip_score"] for r in valid_results if r["clip_score"] >= 0]
    brisque_scores = [r["brisque"] for r in valid_results if r["brisque"] >= 0]
    quality_ratings = [r["quality_rating"] for r in valid_results if r["quality_rating"] >= 0]
    
    # Create summary
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "evaluated_images": len(valid_results),
            "failed_images": len(error_results),
        },
        "overall_metrics": {
            "clip_score": {
                "average": float(np.mean(clip_scores)) if clip_scores else None,
                "std": float(np.std(clip_scores)) if clip_scores else None,
                "min": float(np.min(clip_scores)) if clip_scores else None,
                "max": float(np.max(clip_scores)) if clip_scores else None,
            },
            "brisque": {
                "average": float(np.mean(brisque_scores)) if brisque_scores else None,
                "std": float(np.std(brisque_scores)) if brisque_scores else None,
                "min": float(np.min(brisque_scores)) if brisque_scores else None,
                "max": float(np.max(brisque_scores)) if brisque_scores else None,
            },
            "quality_rating": {
                "average": float(np.mean(quality_ratings)) if quality_ratings else None,
                "std": float(np.std(quality_ratings)) if quality_ratings else None,
                "min": float(np.min(quality_ratings)) if quality_ratings else None,
                "max": float(np.max(quality_ratings)) if quality_ratings else None,
            }
        },
        "image_results": valid_results,
        "errors": error_results,
    }
    
    # Group by model and prompt for nested analysis
    by_model = {}
    by_prompt = {}
    
    for result in valid_results:
        filename = result["file"]
        prompt = result["prompt"]
        
        # Extract model name
        try:
            parts = filename.split("_")
            if len(parts) >= 2:
                model_name = parts[1]
                model_name = "-".join(model_name.split("-")[:-2])
            else:
                model_name = "unknown"
        except:
            model_name = "unknown"
        
        # Group by model
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(result)
        
        # Group by prompt
        if prompt not in by_prompt:
            by_prompt[prompt] = []
        by_prompt[prompt].append(result)
    
    # Calculate per-model statistics
    model_stats = {}
    for model_name in sorted(by_model.keys()):
        model_results = by_model[model_name]
        model_clips = [r["clip_score"] for r in model_results if r["clip_score"] >= 0]
        model_brisques = [r["brisque"] for r in model_results if r["brisque"] >= 0]
        model_qualities = [r["quality_rating"] for r in model_results if r["quality_rating"] >= 0]
        
        model_stats[model_name] = {
            "image_count": len(model_results),
            "clip_score": {
                "average": float(np.mean(model_clips)) if model_clips else None,
                "std": float(np.std(model_clips)) if model_clips else None,
                "min": float(np.min(model_clips)) if model_clips else None,
                "max": float(np.max(model_clips)) if model_clips else None,
            },
            "brisque": {
                "average": float(np.mean(model_brisques)) if model_brisques else None,
                "std": float(np.std(model_brisques)) if model_brisques else None,
                "min": float(np.min(model_brisques)) if model_brisques else None,
                "max": float(np.max(model_brisques)) if model_brisques else None,
            },
            "quality_rating": {
                "average": float(np.mean(model_qualities)) if model_qualities else None,
                "std": float(np.std(model_qualities)) if model_qualities else None,
                "min": float(np.min(model_qualities)) if model_qualities else None,
                "max": float(np.max(model_qualities)) if model_qualities else None,
            }
        }
    
    summary["model_statistics"] = model_stats
    
    # Calculate per-prompt statistics
    prompt_stats = {}
    for idx, prompt in enumerate(sorted(by_prompt.keys()), 1):
        prompt_results = by_prompt[prompt]
        prompt_clips = [r["clip_score"] for r in prompt_results if r["clip_score"] >= 0]
        prompt_brisques = [r["brisque"] for r in prompt_results if r["brisque"] >= 0]
        prompt_qualities = [r["quality_rating"] for r in prompt_results if r["quality_rating"] >= 0]
        
        prompt_stats[prompt] = {
            "prompt_id": idx,
            "image_count": len(prompt_results),
            "clip_score": {
                "average": float(np.mean(prompt_clips)) if prompt_clips else None,
                "std": float(np.std(prompt_clips)) if prompt_clips else None,
                "min": float(np.min(prompt_clips)) if prompt_clips else None,
                "max": float(np.max(prompt_clips)) if prompt_clips else None,
            },
            "brisque": {
                "average": float(np.mean(prompt_brisques)) if prompt_brisques else None,
                "std": float(np.std(prompt_brisques)) if prompt_brisques else None,
                "min": float(np.min(prompt_brisques)) if prompt_brisques else None,
                "max": float(np.max(prompt_brisques)) if prompt_brisques else None,
            },
            "quality_rating": {
                "average": float(np.mean(prompt_qualities)) if prompt_qualities else None,
                "std": float(np.std(prompt_qualities)) if prompt_qualities else None,
                "min": float(np.min(prompt_qualities)) if prompt_qualities else None,
                "max": float(np.max(prompt_qualities)) if prompt_qualities else None,
            }
        }
    
    summary["prompt_statistics"] = prompt_stats
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
    
    # Print console summary
    if not valid_results:
        print("No valid results to summarize")
        return
    
    # Print overall summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY - OVERALL")
    print("="*70)
    print(f"Total images evaluated: {len(valid_results)}")
    if clip_scores:
        print(f"CLIP Score       - Avg: {np.mean(clip_scores):.3f}, Min: {np.min(clip_scores):.3f}, Max: {np.max(clip_scores):.3f}")
    if brisque_scores:
        print(f"BRISQUE          - Avg: {np.mean(brisque_scores):.2f}, Min: {np.min(brisque_scores):.2f}, Max: {np.max(brisque_scores):.2f}")
    if quality_ratings:
        print(f"Quality Rating   - Avg: {np.mean(quality_ratings):.2f}, Min: {np.min(quality_ratings):.2f}, Max: {np.max(quality_ratings):.2f}")
    print()
    
    # Summary by model
    print("="*70)
    print("SUMMARY BY MODEL")
    print("="*70)
    for model_name in sorted(model_stats.keys()):
        model_stat = model_stats[model_name]
        print(f"\n{model_name} ({model_stat['image_count']} images)")
        print("-" * 70)
        if model_stat['clip_score']['average'] is not None:
            print(f"  CLIP Score  : {model_stat['clip_score']['average']:.3f} ± {model_stat['clip_score']['std']:.3f} (range: {model_stat['clip_score']['min']:.3f}-{model_stat['clip_score']['max']:.3f})")
        if model_stat['brisque']['average'] is not None:
            print(f"  BRISQUE     : {model_stat['brisque']['average']:.2f} ± {model_stat['brisque']['std']:.2f} (range: {model_stat['brisque']['min']:.2f}-{model_stat['brisque']['max']:.2f})")
        if model_stat['quality_rating']['average'] is not None:
            print(f"  Quality Rtg : {model_stat['quality_rating']['average']:.2f} ± {model_stat['quality_rating']['std']:.2f} (range: {model_stat['quality_rating']['min']:.2f}-{model_stat['quality_rating']['max']:.2f})")
    
    # Summary by prompt
    print("\n" + "="*70)
    print("SUMMARY BY PROMPT")
    print("="*70)
    for prompt in sorted(prompt_stats.keys()):
        prompt_stat = prompt_stats[prompt]
        display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"\nPrompt {prompt_stat['prompt_id']}: {display_prompt} ({prompt_stat['image_count']} images)")
        print("-" * 70)
        if prompt_stat['clip_score']['average'] is not None:
            print(f"  CLIP Score  : {prompt_stat['clip_score']['average']:.3f} ± {prompt_stat['clip_score']['std']:.3f} (range: {prompt_stat['clip_score']['min']:.3f}-{prompt_stat['clip_score']['max']:.3f})")
        if prompt_stat['brisque']['average'] is not None:
            print(f"  BRISQUE     : {prompt_stat['brisque']['average']:.2f} ± {prompt_stat['brisque']['std']:.2f} (range: {prompt_stat['brisque']['min']:.2f}-{prompt_stat['brisque']['max']:.2f})")
        if prompt_stat['quality_rating']['average'] is not None:
            print(f"  Quality Rtg : {prompt_stat['quality_rating']['average']:.2f} ± {prompt_stat['quality_rating']['std']:.2f} (range: {prompt_stat['quality_rating']['min']:.2f}-{prompt_stat['quality_rating']['max']:.2f})")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quality of generated images"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory with generated images (from check_models.py output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file for results (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, auto)",
    )
    
    args = parser.parse_args()
    
    # Load .env
    load_dotenv(override=True)
    
    # Determine input directory
    if args.input:
        input_dir = args.input
    else:
        # Try to find latest output directory
        output_base = os.getenv('OUTPUT_DIRECTORY', './output')
        output_dirs = sorted(Path(output_base).glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        if output_dirs:
            input_dir = str(output_dirs[0])
            print(f"Using latest output directory: {input_dir}")
        else:
            print(f"Error: No output directory found in {output_base}")
            return
    
    # Initialize evaluator
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = ImageEvaluator(device=device)
    
    # Evaluate images
    print(f"\nEvaluating images from: {input_dir}\n")
    results = evaluate_output_directory(input_dir, evaluator)
    
    # Save results
    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = Path(input_dir) / output_file
    
    save_results(results, output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
