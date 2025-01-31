from pathlib import Path
from typing import Dict, Any, Optional, Literal
import httpx
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ValidationInfo
)
from sdprompt.utils.retry import with_retry, create_progress
from rich.progress import Progress
import logging
import json
import time
from sdprompt.utils.hash import compute_file_hash
import base64
from enum import Enum
from sdprompt.config import StabilityModel  # Import from config

class AspectRatio(str, Enum):
    """Valid aspect ratios for Stability AI API"""
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_21_9 = "21:9"
    RATIO_2_3 = "2:3"
    RATIO_3_2 = "3:2"
    RATIO_4_5 = "4:5"
    RATIO_5_4 = "5:4"
    RATIO_9_16 = "9:16"
    RATIO_9_21 = "9:21"

class ImageParameters(BaseModel):
    """Parameters for image generation"""
    cfg_scale: float = Field(7.0, ge=1.0, le=10.0)
    seed: Optional[int] = Field(None, ge=0, le=4294967294)
    output_format: Literal["png", "jpeg"] = "png"
    model: StabilityModel = StabilityModel.SD35_LARGE
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1

    @model_validator(mode='after')
    def validate_and_clamp(self) -> 'ImageParameters':
        """Validate and clamp values to their allowed ranges"""
        # Track if we made any adjustments
        adjustments = []

        # Clamp cfg_scale
        if self.cfg_scale < 1.0:
            adjustments.append(f"cfg_scale increased from {self.cfg_scale} to 1.0")
            self.cfg_scale = 1.0
        elif self.cfg_scale > 10.0:
            adjustments.append(f"cfg_scale decreased from {self.cfg_scale} to 10.0")
            self.cfg_scale = 10.0

        # Log any adjustments
        if adjustments:
            logging.warning("Parameter adjustments made: " + "; ".join(adjustments))

        return self

    class Config:
        frozen = True

class ImageGenerator:
    def __init__(self, api_key: str, model: str):
        """
        Initialize image generator
        
        Args:
            api_key: Stability AI API key
            model: Model ID (e.g. "sd3.5-large", "sd3.5-large-turbo")
        """
        self.api_key = api_key
        self.model = self._normalize_model_id(model)
        self.client = httpx.AsyncClient(
            base_url="https://api.stability.ai",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*",
                "stability-client-id": "sdprompt",
                "stability-client-version": "1.0.0"
            },
            timeout=60.0
        )

    def _normalize_model_id(self, model: str) -> str:
        """Validate model ID matches Stability AI's supported models"""
        model = model.lower().strip()
        
        try:
            return StabilityModel(model).value
        except ValueError:
            raise ValueError(
                f"Invalid model: {model}. Must be one of: "
                f"{', '.join(m.value for m in StabilityModel)}"
            )

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        parameters: ImageParameters,
        output_path: Path
    ) -> dict:
        """Generate image from prompt using Stability AI API"""
        # Validate prompt length
        if len(prompt) > 10000:
            raise ValueError("Prompt must be 10000 characters or less")
        if negative_prompt and len(negative_prompt) > 10000:
            raise ValueError("Negative prompt must be 10000 characters or less")

        # Don't use negative prompt with turbo models
        if negative_prompt and parameters.model in [StabilityModel.SD3_LARGE_TURBO, StabilityModel.SD35_LARGE_TURBO]:
            logging.warning("Negative prompts are not supported with turbo models. Ignoring negative prompt.")
            negative_prompt = ""

        try:
            # Prepare form data - only include parameters that SD accepts
            form = {
                "prompt": (None, prompt),
                "output_format": (None, parameters.output_format),
                "cfg_scale": (None, str(parameters.cfg_scale)),
                "aspect_ratio": (None, parameters.aspect_ratio)
            }

            # Add seed if specified
            if parameters.seed is not None:
                form["seed"] = (None, str(parameters.seed))

            # Add negative prompt if provided
            if negative_prompt:
                form["negative_prompt"] = (None, negative_prompt)

            # Make API request with explicit timeout
            response = await self.client.post(
                "/v2beta/stable-image/generate/sd3",
                files=form,
                timeout=60.0
            )
            
            # Check response status and get error details if any
            if response.status_code != 200:
                try:
                    error_json = response.json()
                    error_msg = error_json.get('message', str(response.status_code))
                    raise RuntimeError(f"API Error: {error_msg}")
                except json.JSONDecodeError:
                    raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
            
            # Extract seed from response headers
            seed = response.headers.get("Seed")  # Note: header is capitalized
            
            # Save image directly from response content
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # Create generation result with seed
            result = {
                "success": True,
                "engine": self.model,
                "generation_settings": parameters.dict()
            }
            
            if seed:
                try:
                    result["generation_settings"]["seed"] = int(seed)
                except ValueError:
                    logging.warning(f"Invalid seed value in response: {seed}")
                
            return result
                
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Request timed out after {e.request.timeout} seconds")
        except httpx.HTTPError as e:
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get('message', str(e))
                except:
                    error_detail = e.response.text
            raise RuntimeError(f"HTTP error: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}") 