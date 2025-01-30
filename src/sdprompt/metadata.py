from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import hashlib
from pydantic import BaseModel

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
        prompt_data: Dict[str, Any],
        generation_result: Dict[str, Any],
        original_prompt: str
    ) -> Path:
        """Save metadata for generated image"""
        metadata_path = image_path.with_suffix('.yaml')
        
        # Calculate image checksum
        with open(image_path, 'rb') as f:
            image_data = f.read()
            checksum = hashlib.sha256(image_data).hexdigest()
        
        metadata = ImageMetadata(
            timestamp=datetime.utcnow(),
            original_prompt=original_prompt,
            generated_prompt=prompt_data["generation"]["prompt"],
            model_info={
                "anthropic": {
                    "model": "claude-3-sonnet",
                    "version": "1.0"
                },
                "stability": {
                    "model": "stable-diffusion-v3",
                    "version": "3.0"
                }
            },
            image_parameters={
                **prompt_data["generation"]["parameters"],
                "format": image_path.suffix[1:],
                "seed": generation_result.get("seed")
            },
            image_verification={
                "checksum_sha256": checksum,
                "size_bytes": len(image_data),
                "verification_time": datetime.utcnow().isoformat()
            },
            status={
                "success": True,
                "generation_time": generation_result.get("generation_time", 0)
            }
        )
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata.model_dump(), f, default_flow_style=False)
            
        return metadata_path
    
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