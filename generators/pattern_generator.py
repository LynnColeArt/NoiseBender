import numpy as np
import noise
from dataclasses import dataclass
from typing import Tuple
from PIL import Image

@dataclass
class PatternConfig:
    size: Tuple[int, int]
    noise_scale: float
    pattern_complexity: int
    blend_mode: str
    chaos_factor: float

class NoisePatternGenerator:
    def __init__(self, size=(512, 512)):
        self.size = size

    def generate_pattern(self, config: PatternConfig) -> np.ndarray:
        """Generate complete noise pattern"""
        # Base noise
        world = np.zeros(config.size)
        for i in range(config.size[0]):
            for j in range(config.size[1]):
                world[i][j] = noise.pnoise2(
                    i/config.noise_scale, 
                    j/config.noise_scale,
                    octaves=config.pattern_complexity,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=config.size[0],
                    repeaty=config.size[1],
                    base=0
                )

        # Normalize to 0-1
        world = (world - world.min()) / (world.max() - world.min())

        # Add chaos if specified
        if config.chaos_factor > 0:
            chaos = np.random.rand(*world.shape) * config.chaos_factor
            world = np.clip(world + chaos - config.chaos_factor/2, 0, 1)

        return world
