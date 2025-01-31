from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import hashlib
from pydantic import BaseModel
from sdprompt.utils.hash import compute_file_hash

class ImageMetadata(BaseModel):
    timestamp: datetime
    original_prompt: str
    generated_prompt: str
    model_info: Dict[str, Dict[str, str]]
    image_parameters: Dict[str, Any]
    image_verification: Dict[str, Any]
    status: Dict[str, Any]

class MetadataHandler:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def save_metadata(
        self,
        image_path: Path,
        prompt_data: dict,
        generation_result: dict,
        original_prompt: str
    ) -> None:
        """Save metadata for generated image"""
        verification_time = datetime.now()
        
        # Create metadata structure with only needed fields
        metadata = {
            "timestamp": verification_time,
            "original_prompt": original_prompt,
            "generated_prompt": prompt_data["generation"]["prompt"],
            "image_parameters": {
                "cfg_scale": prompt_data["generation"]["parameters"].get("cfg_scale", 7.0),
                "format": image_path.suffix[1:],  # Remove leading dot
                "seed": generation_result["generation_settings"].get("seed"),
                "aspect_ratio": prompt_data["generation"]["parameters"].get("aspect_ratio")
            },
            "image_verification": {
                "size_bytes": image_path.stat().st_size,
                "checksum_sha256": compute_file_hash(image_path),
                "verification_time": verification_time.isoformat()
            },
            "model_info": {
                "anthropic": {
                    "model": "claude-3-haiku",  # Update to match current model
                    "version": "1.0"
                },
                "stability": {
                    "model": "stable-diffusion-v3",
                    "version": "3.0"
                }
            },
            "status": {
                "success": generation_result["success"],
                "generation_time": generation_result.get("generation_time", 0)
            }
        }
        
        # Save metadata to YAML file
        metadata_path = image_path.with_suffix(".yaml")
        with open(metadata_path, "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)
    
    def load_metadata(self, path: Path) -> ImageMetadata:
        """Load metadata from file"""
        with open(path) as f:
            data = yaml.safe_load(f)
            return ImageMetadata(**data)
    
    def verify_image(self, image_path: Path, metadata: ImageMetadata) -> bool:
        """Verify image checksum against metadata"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
            current_checksum = hashlib.sha256(image_data).hexdigest()
            return current_checksum == metadata.image_verification["checksum_sha256"] 