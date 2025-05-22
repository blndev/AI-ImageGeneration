#!/usr/bin/env python3

import argparse
import os
from PIL import Image
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def convert_image(input_path, output_path, max_size=None):
    """
    Convert an image to PNG format and optionally resize it.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path where the PNG will be saved
        max_size (int, optional): Maximum size for the longest dimension
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (e.g., for RGBA images)
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if max_size is specified
            if max_size:
                width, height = img.size
                if width > height:
                    if width > max_size:
                        ratio = max_size / width
                        new_size = (max_size, int(height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    if height > max_size:
                        ratio = max_size / height
                        new_size = (int(width * ratio), max_size)
                        img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save with maximum quality
            img.save(output_path, 'PNG', quality=100)
            logging.info(f"Converted: {input_path} -> {output_path}")
            
    except Exception as e:
        logging.error(f"Error converting {input_path}: {str(e)}")
        return False
    return True

def process_directory(input_dir, output_dir=None, max_size=None):
    """
    Process all images in a directory and its subdirectories.
    
    Args:
        input_dir (str): Input directory containing images
        output_dir (str, optional): Output directory for converted images
        max_size (int, optional): Maximum size for the longest dimension
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / 'converted'
    else:
        output_dir = Path(output_dir)

    # Process all image files recursively
    success_count = 0
    total_count = 0
    
    for file_path in input_dir.rglob('*'):
        if file_path.is_file() and is_image_file(file_path.name):
            total_count += 1
            # Preserve directory structure
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path.parent / f"{file_path.stem}.png"
            # Ensure the parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if convert_image(str(file_path), str(output_path), max_size):
                success_count += 1

    logging.info(f"Conversion complete. Successfully converted {success_count} of {total_count} images.")
    return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description='Convert images to PNG format with optional resizing')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--output-dir', help='Output directory for converted images (default: input_dir/converted)')
    parser.add_argument('--max-size', type=int, help='Maximum size in pixels for the longest dimension')
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory does not exist: {args.input_dir}")
        return 1
        
    success_count, total_count = process_directory(
        args.input_dir,
        args.output_dir,
        args.max_size
    )
    
    return 0 if success_count == total_count else 1

if __name__ == '__main__':
    exit(main())
