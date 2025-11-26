"""
Advanced image preprocessing utilities for enhanced deep dreams
"""
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional
import cv2


class ImagePreprocessor:
    """Advanced image preprocessing for better dream effects"""
    
    @staticmethod
    def enhance_contrast(img: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Enhance image contrast for better dream patterns"""
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        
        pil_img = Image.fromarray(img.astype(np.uint8))
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    @staticmethod
    def apply_gaussian_noise(img: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Add subtle noise to break up uniform areas"""
        noise = np.random.normal(0, noise_factor, img.shape)
        noisy_img = img + noise * 255
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def enhance_edges(img: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """Enhance edges for more pronounced dream patterns"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Detect edges using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and apply
        edges = (edges / edges.max() * strength * 255).astype(np.uint8)
        
        if len(img.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend with original
        enhanced = cv2.addWeighted(img, 0.7, edges, 0.3, 0)
        return enhanced
    
    @staticmethod
    def prepare_image(image_path: str, target_size: Tuple[int, int] = (512, 512)) -> tf.Tensor:
        """Prepare image for dreaming with optimal preprocessing"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize while maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste centered
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        paste_pos = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
        new_img.paste(img, paste_pos)
        
        # Convert to array
        img_array = np.array(new_img, dtype=np.float32)
        
        # Apply preprocessing enhancements
        img_array = ImagePreprocessor.enhance_contrast(img_array, 1.1)
        img_array = ImagePreprocessor.apply_gaussian_noise(img_array, 0.02)
        
        return tf.convert_to_tensor(img_array)


class PostProcessor:
    """Post-processing utilities for dream outputs"""
    
    @staticmethod
    def apply_color_enhancement(img: tf.Tensor, saturation: float = 1.3, brightness: float = 1.1) -> tf.Tensor:
        """Enhance colors in the dreamed image"""
        if isinstance(img, tf.Tensor):
            img_np = img.numpy()
        else:
            img_np = img
        
        # Convert to PIL for color enhancement
        pil_img = Image.fromarray(img_np.astype(np.uint8))
        
        # Enhance saturation
        enhancer = ImageEnhance.Color(pil_img)
        enhanced = enhancer.enhance(saturation)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(brightness)
        
        return tf.convert_to_tensor(np.array(enhanced))
    
    @staticmethod
    def apply_artistic_filter(img: tf.Tensor, filter_type: str = 'smooth') -> tf.Tensor:
        """Apply artistic filters to enhance dream quality"""
        if isinstance(img, tf.Tensor):
            img_np = img.numpy()
        else:
            img_np = img
        
        pil_img = Image.fromarray(img_np.astype(np.uint8))
        
        if filter_type == 'smooth':
            filtered = pil_img.filter(ImageFilter.SMOOTH_MORE)
        elif filter_type == 'detail':
            filtered = pil_img.filter(ImageFilter.DETAIL)
        elif filter_type == 'edge_enhance':
            filtered = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif filter_type == 'emboss':
            filtered = pil_img.filter(ImageFilter.EMBOSS)
        else:
            filtered = pil_img
        
        return tf.convert_to_tensor(np.array(filtered))
    
    @staticmethod
    def blend_with_original(original: tf.Tensor, dreamed: tf.Tensor, blend_factor: float = 0.7) -> tf.Tensor:
        """Blend dreamed image with original for more natural results"""
        original = tf.cast(original, tf.float32)
        dreamed = tf.cast(dreamed, tf.float32)
        
        blended = blend_factor * dreamed + (1 - blend_factor) * original
        return tf.cast(tf.clip_by_value(blended, 0, 255), tf.uint8)


class BatchProcessor:
    """Batch processing utilities for multiple images"""
    
    def __init__(self, dreamer_config: dict):
        self.dreamer_config = dreamer_config
    
    def process_batch(self, image_paths: list, output_dir: str, style: str = 'psychedelic') -> list:
        """Process multiple images with the same dream configuration"""
        from .enhanced_dreamer import EnhancedDeepDreamer
        
        results = []
        dreamer = EnhancedDeepDreamer(
            model_type=self.dreamer_config.get('model_type', 'efficientnet'),
            dream_style=style
        )
        
        for i, img_path in enumerate(image_paths):
            try:
                # Preprocess image
                img = ImagePreprocessor.prepare_image(img_path)
                
                # Apply dreaming
                if self.dreamer_config.get('use_octave', False):
                    result = dreamer.octave_dream(img)
                else:
                    result = dreamer.multi_scale_dream(img)
                
                # Post-process
                result = PostProcessor.apply_color_enhancement(result)
                result = PostProcessor.apply_artistic_filter(result, 'smooth')
                
                # Save result
                output_path = f"{output_dir}/dreamed_{i:03d}.jpg"
                Image.fromarray(result.numpy()).save(output_path, quality=95)
                
                results.append(output_path)
                print(f"Processed image {i+1}/{len(image_paths)}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        return results


class QualityAnalyzer:
    """Analyze and score dream quality"""
    
    @staticmethod
    def calculate_dream_score(original: tf.Tensor, dreamed: tf.Tensor) -> dict:
        """Calculate quality metrics for dreamed images"""
        original_np = original.numpy() if isinstance(original, tf.Tensor) else original
        dreamed_np = dreamed.numpy() if isinstance(dreamed, tf.Tensor) else dreamed
        
        # Calculate variance (pattern richness)
        dream_variance = np.var(dreamed_np)
        
        # Calculate edge density (detail level)
        gray_dreamed = cv2.cvtColor(dreamed_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_dreamed.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate color diversity
        unique_colors = len(np.unique(dreamed_np.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (dreamed_np.shape[0] * dreamed_np.shape[1])
        
        # Calculate contrast
        contrast = np.std(dreamed_np)
        
        # Overall dream quality score
        quality_score = (
            (dream_variance / 10000) * 0.3 +
            edge_density * 0.25 +
            color_diversity * 0.25 +
            (contrast / 100) * 0.2
        )
        
        return {
            'quality_score': min(quality_score, 1.0),
            'variance': float(dream_variance),
            'edge_density': float(edge_density),
            'color_diversity': float(color_diversity),
            'contrast': float(contrast)
        }