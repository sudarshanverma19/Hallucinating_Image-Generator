"""
Enhanced Deep Dreams implementation with multiple architectures and effects
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.ndimage as nd
from typing import List, Tuple, Optional


class EnhancedDeepDreamer:
    """Advanced Deep Dreams implementation with multiple model support"""
    
    def __init__(self, model_type='efficientnet', dream_style='psychedelic'):
        self.model_type = model_type
        self.dream_style = dream_style
        self.base_model = None
        self.dream_model = None
        self.target_layers = []
        self.layer_weights = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the base model and dream configuration"""
        model_configs = {
            'efficientnet': {
                'model_class': tf.keras.applications.EfficientNetB7,
                'layers': {
                    'psychedelic': ['block1a_expand_conv', 'block3a_expand_conv', 'block6a_expand_conv'],
                    'artistic': ['block2a_expand_conv', 'block4a_expand_conv', 'block7a_expand_conv'],
                    'surreal': ['block1a_expand_conv', 'block5a_expand_conv', 'top_conv']
                },
                'weights': {
                    'psychedelic': [2.0, 1.5, 1.0],
                    'artistic': [1.0, 1.5, 2.0],
                    'surreal': [1.5, 2.0, 2.5]
                }
            },
            'resnet': {
                'model_class': tf.keras.applications.ResNet152V2,
                'layers': {
                    'psychedelic': ['conv2_block3_out', 'conv4_block8_out', 'conv5_block3_out'],
                    'artistic': ['conv2_block1_out', 'conv4_block6_out', 'conv5_block2_out'],
                    'surreal': ['conv3_block4_out', 'conv4_block12_out', 'conv5_block3_out']
                },
                'weights': {
                    'psychedelic': [2.0, 1.8, 1.2],
                    'artistic': [1.0, 1.5, 2.2],
                    'surreal': [1.8, 2.2, 2.8]
                }
            },
            'densenet': {
                'model_class': tf.keras.applications.DenseNet201,
                'layers': {
                    'psychedelic': ['conv2_block6_concat', 'conv4_block12_concat', 'conv5_block16_concat'],
                    'artistic': ['conv2_block3_concat', 'conv4_block8_concat', 'conv5_block12_concat'],
                    'surreal': ['conv3_block8_concat', 'conv4_block24_concat', 'conv5_block16_concat']
                },
                'weights': {
                    'psychedelic': [1.8, 2.0, 1.5],
                    'artistic': [1.2, 1.8, 2.5],
                    'surreal': [2.0, 2.5, 3.0]
                }
            }
        }
        
        config = model_configs[self.model_type]
        
        # Initialize base model
        self.base_model = config['model_class'](
            weights='imagenet',
            include_top=False,
            input_shape=(None, None, 3)
        )
        
        # Set target layers and weights based on style
        self.target_layers = config['layers'][self.dream_style]
        self.layer_weights = config['weights'][self.dream_style]
        
        # Create feature extraction model
        self._create_dream_model()
    
    def _create_dream_model(self):
        """Create feature extraction model for dreaming"""
        outputs = []
        for layer_name in self.target_layers:
            try:
                layer_output = self.base_model.get_layer(layer_name).output
                outputs.append(layer_output)
            except ValueError:
                print(f"Warning: Layer {layer_name} not found in {self.model_type}")
        
        if outputs:
            self.dream_model = tf.keras.Model([self.base_model.input], outputs)
        else:
            raise ValueError(f"No valid layers found for {self.model_type}")
    
    def calculate_loss(self, img: tf.Tensor) -> tf.Tensor:
        """Calculate the loss for gradient ascent"""
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = self.dream_model(img_batch)
        
        if not isinstance(layer_activations, list):
            layer_activations = [layer_activations]
        
        total_loss = 0
        for activation, weight in zip(layer_activations, self.layer_weights):
            # Main activation loss
            loss = tf.reduce_mean(activation) * weight
            
            # Add texture variation for more interesting patterns
            variation_loss = tf.reduce_mean(tf.image.total_variation(activation)) * 0.1
            
            # Add pattern enhancement
            pattern_loss = tf.reduce_mean(tf.square(activation)) * 0.05
            
            total_loss += loss + variation_loss + pattern_loss
        
        return total_loss
    
    def gradient_ascent_step(self, img: tf.Tensor, learning_rate: float = 0.01) -> Tuple[tf.Tensor, tf.Tensor]:
        """Perform one gradient ascent step"""
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.calculate_loss(img)
        
        gradients = tape.gradient(loss, img)
        
        # Normalize gradients
        gradients = tf.nn.l2_normalize(gradients, axis=[1, 2, 3])
        
        # Apply gradients
        img = img + learning_rate * gradients
        
        # Clip values to reasonable range
        img = tf.clip_by_value(img, -1, 1)
        
        return img, loss
    
    def deep_dream(self, img: tf.Tensor, steps: int = 100, step_size: float = 0.01) -> tf.Tensor:
        """Main deep dream function"""
        # Preprocess image
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        
        for step in range(steps):
            img, loss = self.gradient_ascent_step(img, step_size)
            
            # Adaptive learning rate
            if step % 25 == 0:
                step_size *= 0.98  # Gradually reduce step size
                
            if step % 20 == 0:
                print(f"Step {step}, Loss: {float(loss):.4f}")
        
        return self._deprocess_image(img)
    
    def multi_scale_dream(self, img: tf.Tensor, scales: List[float] = None) -> tf.Tensor:
        """Apply dreaming at multiple scales for richer effects"""
        if scales is None:
            scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        
        original_shape = img.shape[:2]
        result_img = tf.cast(img, tf.float32)
        
        for i, scale in enumerate(scales):
            # Calculate new size
            new_size = (int(original_shape[0] * scale), int(original_shape[1] * scale))
            
            # Resize image
            scaled_img = tf.image.resize(result_img, new_size)
            
            # Apply deep dream with scale-adaptive parameters
            steps = max(30, int(50 * scale))
            step_size = 0.01 * scale
            
            dreamed_img = self.deep_dream(scaled_img, steps=steps, step_size=step_size)
            
            # Resize back to original size
            dreamed_img = tf.image.resize(dreamed_img, original_shape)
            
            # Blend with original (stronger effect for larger scales)
            blend_factor = 0.3 + (0.2 * i / len(scales))
            result_img = (1 - blend_factor) * result_img + blend_factor * tf.cast(dreamed_img, tf.float32)
        
        return tf.cast(result_img, tf.uint8)
    
    def octave_dream(self, img: tf.Tensor, octaves: int = 4, octave_scale: float = 1.4) -> tf.Tensor:
        """Octave-based dreaming for fractal-like effects"""
        img = tf.cast(img, tf.float32)
        original_shape = img.shape
        
        # Generate octave images
        octave_images = []
        for i in range(octaves):
            scale_factor = octave_scale ** i
            size = (
                max(64, int(original_shape[0] / scale_factor)),
                max(64, int(original_shape[1] / scale_factor))
            )
            octave_img = tf.image.resize(img, size)
            octave_images.append(octave_img)
        
        # Start with smallest octave
        detail = tf.zeros_like(octave_images[-1], dtype=tf.float32)
        
        # Process each octave from small to large
        for i, octave_img in enumerate(reversed(octave_images)):
            # Resize detail to match current octave
            if detail.shape[:2] != octave_img.shape[:2]:
                detail = tf.image.resize(detail, octave_img.shape[:2])
            
            # Add detail to base image
            input_img = octave_img + detail
            
            # Dream on this octave
            steps = 20 + (i * 10)  # More steps for larger octaves
            dreamed_img = self.deep_dream(input_img, steps=steps, step_size=0.008)
            
            # Calculate new detail
            detail = tf.cast(dreamed_img, tf.float32) - octave_img
            
            # Enhance detail contrast
            detail = detail * 1.2
        
        # Final resize to original dimensions
        final_img = tf.image.resize(dreamed_img, original_shape[:2])
        return tf.cast(final_img, tf.uint8)
    
    def _deprocess_image(self, img: tf.Tensor) -> tf.Tensor:
        """Convert processed image back to displayable format"""
        # Reverse imagenet preprocessing
        img = img.numpy() if hasattr(img, 'numpy') else img
        
        # Convert from [-1, 1] to [0, 255]
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255)
        
        return tf.cast(img, tf.uint8)


class DreamStyleManager:
    """Manage different dream styles and their configurations"""
    
    STYLES = {
        'psychedelic': {
            'description': 'Vibrant, colorful patterns with high contrast',
            'models': ['efficientnet', 'resnet', 'densenet'],
            'recommended_model': 'efficientnet',
            'intensity_range': (0.5, 2.0)
        },
        'artistic': {
            'description': 'Painterly effects with balanced composition',
            'models': ['resnet', 'densenet', 'efficientnet'],
            'recommended_model': 'resnet',
            'intensity_range': (0.3, 1.5)
        },
        'surreal': {
            'description': 'Abstract, dream-like transformations',
            'models': ['densenet', 'efficientnet', 'resnet'],
            'recommended_model': 'densenet',
            'intensity_range': (0.7, 2.5)
        },
        'fractal': {
            'description': 'Self-similar patterns with octave processing',
            'models': ['efficientnet', 'densenet'],
            'recommended_model': 'efficientnet',
            'intensity_range': (0.4, 1.8)
        }
    }
    
    @classmethod
    def get_style_config(cls, style_name: str) -> dict:
        """Get configuration for a specific style"""
        return cls.STYLES.get(style_name, cls.STYLES['psychedelic'])
    
    @classmethod
    def list_styles(cls) -> List[str]:
        """Get list of available styles"""
        return list(cls.STYLES.keys())