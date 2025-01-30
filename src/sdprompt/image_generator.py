from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field, field_validator
from sdprompt.utils.retry import with_retry, create_progress
from rich.progress import Progress
import logging
import json
import time
from sdprompt.utils.hash import compute_file_hash

class ImageParameters(BaseModel):
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    steps: int = Field(50, ge=10, le=150)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
    seed: Optional[int] = Field(None, ge=0, le=4294967295)

    @field_validator("width", "height")
    def validate_dimensions(cls, v: int) -> int:
        if v % 64 != 0:
            raise ValueError("Dimensions must be multiples of 64")
        return v

    class Config:
        frozen = True

class ImageGenerator:
    def __init__(self, api_key: str, model: str = "stable-diffusion-v3"):
        self.api_key = api_key
        self.model = model
        self.api_host = "https://api.stability.ai"
        self.engine_map = {
            "stable-diffusion-v3": "stable-diffusion-xl-1024-v1-0",
            "stable-diffusion-xl-1024-v1-0": "stable-diffusion-xl-1024-v1-0"
        }
        
    @with_retry(retries=3, delay=1.0)
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        parameters: ImageParameters,
        output_path: Path,
    ) -> Dict[str, Any]:
        """Generate image using Stability API with retry logic"""
        try:
            start_time_ms = int(time.time() * 1000)  # Start timing at beginning
            
            engine_id = self.engine_map.get(self.model)
            if not engine_id:
                raise ValueError(f"Unsupported model: {self.model}")

            url = f"{self.api_host}/v1/generation/{engine_id}/text-to-image"
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1.0
                    }
                ],
                "cfg_scale": parameters.cfg_scale,
                "height": parameters.height,
                "width": parameters.width,
                "steps": parameters.steps,
                "samples": 1
            }

            if negative_prompt:
                payload["text_prompts"].append({
                    "text": negative_prompt,
                    "weight": -1.0
                })

            if parameters.seed is not None:
                payload["seed"] = parameters.seed

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                
                if "artifacts" in result and len(result["artifacts"]) > 0:
                    image_data = result["artifacts"][0]["base64"]
                    import base64
                    
                    # Ensure output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write image data
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(image_data))
                    
                    # Verify file was written
                    if not output_path.exists():
                        raise RuntimeError("Failed to write image file")
                    
                    # Get file info after writing
                    file_size = output_path.stat().st_size
                    file_hash = compute_file_hash(output_path)
                    
                    generation_time_ms = int(time.time() * 1000) - start_time_ms
                    
                    return {
                        "seed": result["artifacts"][0]["seed"],
                        "finish_reason": "SUCCESS",
                        "path": str(output_path),
                        "generation_info": {
                            "engine": self.engine_map.get(self.model),
                            "api_version": "v1",
                            "prompt_tokens": len(prompt.split()),
                            "negative_prompt_tokens": len(negative_prompt.split()) if negative_prompt else 0,
                            "total_time_ms": generation_time_ms,
                            "generation_settings": {
                                "cfg_scale": parameters.cfg_scale,
                                "steps": parameters.steps,
                                "width": parameters.width,
                                "height": parameters.height,
                                "seed": parameters.seed
                            }
                        },
                        "image_info": {
                            "format": output_path.suffix[1:],
                            "dimensions": f"{parameters.width}x{parameters.height}",
                            "file_size_bytes": file_size,
                            "checksum_sha256": file_hash
                        }
                    }
                
                raise RuntimeError("No image generated in response")

        except httpx.HTTPError as e:
            logging.error(f"HTTP error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Image generation failed: {str(e)}")
            raise 