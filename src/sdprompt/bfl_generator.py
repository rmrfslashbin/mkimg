import httpx
import asyncio
from pathlib import Path
from typing import Optional
import logging
from pydantic import BaseModel, Field
from sdprompt.config import BFLModel

class BFLParameters(BaseModel):
    """Parameters for BFL image generation"""
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    model: BFLModel = BFLModel.FLUX_PRO_11

class BFLGenerator:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.us1.bfl.ai/v1"):
        self.api_key = api_key
        self.model = self._normalize_model_id(model)
        self.base_url = base_url
        # Remove 'bfl_' prefix for the actual API call
        api_key_clean = api_key.replace('bfl_', '') if api_key.startswith('bfl_') else api_key
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-key": api_key_clean,
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )

    def _normalize_model_id(self, model: str) -> str:
        try:
            return BFLModel(model).value
        except ValueError:
            raise ValueError(
                f"Invalid model: {model}. Must be one of: "
                f"{', '.join(m.value for m in BFLModel)}"
            )

    async def generate_image(
        self,
        prompt: str,
        parameters: BFLParameters,
        output_path: Path,
        negative_prompt: str = None,
    ) -> dict:
        """Generate image using BFL API
        
        Note: BFL does not support negative prompts, this parameter is ignored
        """
        try:
            # Submit generation request
            response = await self.client.post(
                f"/{self.model}",
                json={
                    "prompt": prompt,
                    "width": parameters.width,
                    "height": parameters.height
                }
            )
            
            if response.status_code == 402:
                raise RuntimeError("Insufficient BFL credits")
            elif response.status_code == 429:
                raise RuntimeError("Too many active BFL tasks")
            response.raise_for_status()
            
            request_id = response.json()["id"]
            
            # Poll for results
            while True:
                await asyncio.sleep(0.5)  # Don't hammer the API
                result = await self.client.get(
                    "/get_result",
                    params={"id": request_id}
                )
                result.raise_for_status()
                result_data = result.json()
                
                if result_data["status"] == "Ready":
                    # Download image from signed URL
                    image_url = result_data["result"]["sample"]
                    async with httpx.AsyncClient() as dl_client:
                        img_response = await dl_client.get(image_url)
                        img_response.raise_for_status()
                        
                        # Save image
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "wb") as f:
                            f.write(img_response.content)
                            
                    return {
                        "success": True,
                        "engine": self.model,
                        "generation_settings": parameters.dict()
                    }
                    
                elif result_data["status"] == "Failed":
                    raise RuntimeError(f"BFL generation failed: {result_data.get('error', 'Unknown error')}")
                    
                logging.info(f"BFL generation status: {result_data['status']}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}") 