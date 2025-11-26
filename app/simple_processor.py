"""
Simple image processor without TensorFlow for testing
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple
import random


class SimpleImageProcessor:
    """Simple image effects without deep learning"""
    
    @staticmethod
    def apply_psychedelic_effect(image: Image.Image) -> Image.Image:
        """Apply colorful psychedelic effects"""
        # Enhance colors
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(2.0)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Apply artistic filters
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        image = image.filter(ImageFilter.SMOOTH)
        
        return image
    
    @staticmethod
    def apply_artistic_effect(image: Image.Image) -> Image.Image:
        """Apply artistic painting effects"""
        # Soften the image
        image = image.filter(ImageFilter.SMOOTH_MORE)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        # Add some texture
        image = image.filter(ImageFilter.DETAIL)
        
        return image
    
    @staticmethod
    def apply_surreal_effect(image: Image.Image) -> Image.Image:
        """Apply surreal dream-like effects"""
        # Convert to array for manipulation
        img_array = np.array(image)
        
        # Add color shifts
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.3, 0, 255)  # Red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.2, 0, 255)  # Blue
        
        # Convert back to PIL
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply filters
        image = image.filter(ImageFilter.EMBOSS)
        
        return image
    
    @staticmethod
    def apply_fractal_effect(image: Image.Image) -> Image.Image:
        """Apply fractal-like pattern effects"""
        # Multiple filter applications for complexity
        for _ in range(3):
            image = image.filter(ImageFilter.EDGE_ENHANCE)
            image = image.filter(ImageFilter.SMOOTH)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image


def generate_simple_dream(image: Image.Image, style: str = "psychedelic") -> Image.Image:
    """Generate simple dream effects without TensorFlow"""
    processor = SimpleImageProcessor()
    
    if style == "psychedelic":
        return processor.apply_psychedelic_effect(image)
    elif style == "artistic":
        return processor.apply_artistic_effect(image)
    elif style == "surreal":
        return processor.apply_surreal_effect(image)
    elif style == "fractal":
        return processor.apply_fractal_effect(image)
    else:
        return processor.apply_psychedelic_effect(image)