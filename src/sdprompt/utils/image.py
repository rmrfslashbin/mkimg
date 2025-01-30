from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Any
import time
from datetime import datetime

class ImageVerifier:
    """Utility class for image verification"""
    
    @staticmethod
    def verify_dimensions(image_path: Path, expected_width: int, expected_height: int) -> bool:
        """Verify image dimensions"""
        with Image.open(image_path) as img:
            width, height = img.size
            return width == expected_width and height == expected_height
            
    @staticmethod
    def verify_format(image_path: Path, expected_format: str) -> bool:
        """Verify image format"""
        with Image.open(image_path) as img:
            return img.format.lower() == expected_format.lower()
            
    @staticmethod
    def verify_timestamp(timestamp: str, max_age_seconds: int = None) -> bool:
        """Verify image timestamp is valid and not in the future"""
        try:
            img_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            
            # Check if timestamp is in the future
            if img_time > now:
                return False
                
            # Check age if specified
            if max_age_seconds is not None:
                age = (now - img_time).total_seconds()
                if age > max_age_seconds:
                    return False
                    
            return True
        except ValueError:
            return False
            
    @staticmethod
    def get_image_info(image_path: Path) -> Dict[str, Any]:
        """Get basic image information"""
        with Image.open(image_path) as img:
            return {
                "format": img.format.lower(),
                "dimensions": f"{img.width}x{img.height}",
                "mode": img.mode,
                "size_bytes": image_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(image_path.stat().st_mtime).isoformat()
            } 