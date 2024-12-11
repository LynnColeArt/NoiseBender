import numpy as np
import colorsys
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image

@dataclass
class ColorConfig:
    base_hues: List[float]
    saturation_range: Tuple[float, float]
    value_range: Tuple[float, float]
    color_noise: float
    blend_distortion: float

class UnconventionalColorGenerator:
    def __init__(self, size=(512, 512)):
        self.size = size
        
    def generate_color_noise(self, config: ColorConfig) -> np.ndarray:
        """Generate a color noise pattern with unconventional combinations"""
        image = np.zeros((*self.size, 3))
        
        for hue in config.base_hues:
            layer = np.zeros((*self.size, 3))
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    h = (hue + np.random.normal(0, config.color_noise)) % 1.0
                    s = np.random.uniform(*config.saturation_range)
                    v = np.random.uniform(*config.value_range)
                    rgb = colorsys.hsv_to_rgb(h, s, v)
                    layer[i, j] = rgb
            
            self._blend_layer(image, layer, config.blend_distortion)
        
        return np.clip(image, 0, 1)
    
    def _blend_layer(self, base: np.ndarray, layer: np.ndarray, distortion: float):
        """Blend layers with unconventional methods"""
        for c in range(3):
            blend_type = np.random.choice(['multiply', 'screen', 'difference'])
            if blend_type == 'multiply':
                base[..., c] = base[..., c] * layer[..., c]
            elif blend_type == 'screen':
                base[..., c] = 1 - (1 - base[..., c]) * (1 - layer[..., c])
            else:  # difference
                base[..., c] = np.abs(base[..., c] - layer[..., c])
            
            if distortion > 0:
                noise = np.random.normal(0, distortion, base[..., c].shape)
                base[..., c] = np.clip(base[..., c] + noise, 0, 1)

    def apply_color_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply various color distortions"""
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        
        mixed = np.zeros_like(image)
        mixed[..., 0] = 0.7 * r - 0.3 * g + 0.3 * b
        mixed[..., 1] = 0.3 * r + 0.8 * g - 0.2 * b
        mixed[..., 2] = -0.2 * r + 0.3 * g + 0.9 * b
        
        return np.clip(mixed, 0, 1)
    
    def create_unconventional_pattern(self) -> np.ndarray:
        """Create a pattern with unconventional color combinations"""
        config = ColorConfig(
            base_hues=[0.1, 0.45, 0.85],
            saturation_range=(0.7, 1.0),
            value_range=(0.3, 0.9),
            color_noise=0.05,
            blend_distortion=0.1
        )
        
        pattern = self.generate_color_noise(config)
        distorted = self.apply_color_distortion(pattern)
        noise = np.random.normal(0, 0.1, pattern.shape)
        
        return np.clip(distorted + noise, 0, 1)
