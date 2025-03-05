from dataclasses import dataclass
from typing import Optional, List, Union
from PIL import Image
import warnings

import torch

@dataclass
class FluxParameters:
    """
    Usage
    # Basic setup
    params = FluxParameters(
        prompt="a beautiful sunset over mountains",
        num_inference_steps=20,
        guidance_scale=7.5
    )

    # For FLUX-SCHNELL
    schnell_params = params.prepare_flux_schnell()
    schnell_pipeline = FluxPipeline.from_pretrained("stabilityai/flux-schnell")
    schnell_result = schnell_pipeline(**schnell_params.to_dict())

    # For FLUX-DEV
    dev_params = params.prepare_flux_dev()
    dev_pipeline = FluxPipeline.from_pretrained("stabilityai/flux-dev")
    dev_result = dev_pipeline(**dev_params.to_dict())

    """
    # Required parameters
    prompt: str
    
    # Optional parameters with defaults
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    output_format: str = "jpg"
    output_quality: int = 80
    # Image-to-Image specific parameters
    image: Optional[Union[Image.Image, List[Image.Image]]] = None
    strength: float = 0.8
    mask_image: Optional[Image.Image] = None
    
    # Advanced parameters
    guidance_scale: float = 0.7
    clip_skip: Optional[int] = None
    
    def validate(self):
        """Validate the parameters"""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
            
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
            
        if not (self.guidance_scale>=0):
            raise ValueError("guidance_scale must be >= 0.0")
            
        if not (0.0 < self.strength <= 1.0) and self.image is not None:
            raise ValueError("strength must be between 0.0 and 1.0")
            
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("width and height must be divisible by 8")
            
        if self.clip_skip is not None and self.clip_skip < 1:
            raise ValueError("clip_skip must be >= 1")
            
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be > 0.0")
            
        # Validate image dimensions if provided
        if isinstance(self.image, Image.Image):
            if self.image.size[0] % 8 != 0 or self.image.size[1] % 8 != 0:
                raise ValueError("Input image dimensions must be divisible by 8")
                
        # Validate mask if provided
        if self.mask_image and not self.image:
            raise ValueError("mask_image provided without input image")
            
        if self.mask_image and self.image:
            if self.mask_image.size != (self.image.size if isinstance(self.image, Image.Image) else self.image[0].size):
                raise ValueError("mask_image dimensions must match input image dimensions")

    def to_dict(self) -> dict:
        """Convert parameters to dictionary for pipeline input"""
        params = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width,
            "height": self.height,
            "num_images_per_prompt": self.num_images_per_prompt,
            "guidance_scale": self.guidance_scale,
        }
        
        if self.negative_prompt:
            params["negative_prompt"] = self.negative_prompt
        if self.seed is not None:
            params["generator"] = torch.Generator().manual_seed(self.seed)
        if self.clip_skip:
            params["clip_skip"] = self.clip_skip
        if self.image is not None:
            params["image"] = self.image
            params["strength"] = self.strength
        if self.mask_image is not None:
            params["mask_image"] = self.mask_image
            
        return params

    def prepare_flux_schnell(self) -> 'FluxParameters':
        """
        Adjusts parameters specifically for FLUX-SCHNELL model.
        Returns a new FluxParameters instance with optimized settings.
        """
        params = FluxParameters(**self.__dict__)  # Create a copy
        
        # Enforce FLUX-SCHNELL specific constraints
        if params.num_inference_steps > 10:
            warnings.warn(f"FLUX-SCHNELL works best with low inference steps. "
                        f"Reducing from {params.num_inference_steps} to 4.")
            params.num_inference_steps = 4
        
        # Set guidance scale to 0
        if params.guidance_scale != 0:
            warnings.warn("FLUX-SCHNELL requires guidance_scale=0. Adjusting automatically.")
            params.guidance_scale = 0
            
        # Check prompt length (max 256 tokens)
        if len(params.prompt.split()) > 77:  # Approximate token count
            warnings.warn("FLUX-SCHNELL has a max sequence length of 256. "
                        "Long prompts may be truncated.")
            
        return params

    def prepare_flux_dev(self) -> 'FluxParameters':
        """
        Adjusts parameters specifically for FLUX-DEV model.
        Returns a new FluxParameters instance with optimized settings.
        """
        params = FluxParameters(**self.__dict__)  # Create a copy
        
        # Warn if inference steps are too low for good quality
        if params.num_inference_steps < 50:
            warnings.warn(f"FLUX-DEV typically needs 50+ inference steps for good results. "
                        f"Current setting: {params.num_inference_steps}")
            
        # Adjust guidance scale if too low
        if params.guidance_scale < 7.0:
            warnings.warn(f"FLUX-DEV typically works better with guidance_scale >= 7.0. "
                        f"Current setting: {params.guidance_scale}")
        
        if params.guidance_scale <= 0:
            params.guidance_scale=5.0
                        
        return params
