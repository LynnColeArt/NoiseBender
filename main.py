import argparse
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from dataclasses import dataclass, asdict
import sys
from typing import Dict, Any, Optional
from generators.color_generator import UnconventionalColorGenerator, ColorConfig
from generators.pattern_generator import NoisePatternGenerator, PatternConfig

@dataclass
class BendScore:
    chaos_level: float
    pattern_complexity: float
    color_disruption: float
    overall_score: float
    timestamp: str

@dataclass
class ProcessingResult:
    input_path: Optional[str]
    output_path: str
    bend_score: BendScore

class NoiseBendUtility:
    def __init__(self):
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.manifest_path = self.output_dir / "manifest.json"
        
        # Ensure directories exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def generate_new(self, complexity: float = 0.5) -> ProcessingResult:
        """Generate a new noise bent artwork"""
        size = (512, 512)
        color_gen = UnconventionalColorGenerator(size=size)
        pattern_gen = NoisePatternGenerator(size=size)
        
        # Generate base pattern
        pattern = pattern_gen.generate_pattern(PatternConfig(
            size=size,
            noise_scale=50.0 + (complexity * 100),
            pattern_complexity=int(10 + (complexity * 20)),
            blend_mode='multiply',
            chaos_factor=complexity
        ))
        
        # Expand pattern to 3 channels
        pattern = np.stack([pattern] * 3, axis=-1)
        
        # Apply color operations
        colored_pattern = color_gen.create_unconventional_pattern()
        
        # Blend them together
        final_image = np.clip(pattern * colored_pattern, 0, 1)
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"generated_{timestamp}.png"
        Image.fromarray((final_image * 255).astype(np.uint8)).save(output_path)
        
        # Calculate and return result
        bend_score = self._calculate_bend_score(final_image)
        return ProcessingResult(
            input_path=None,
            output_path=str(output_path),
            bend_score=bend_score
        )

    def process_existing(self, image_path: Path, intensity: float = 0.5) -> ProcessingResult:
        """Process an existing image with noise bending effects"""
        # Load and normalize image
        img = Image.open(image_path)
        img_array = np.array(img).astype(float) / 255.0
        
        # Generate noise and color distortions
        color_gen = UnconventionalColorGenerator(size=img_array.shape[:2])
        pattern_gen = NoisePatternGenerator(size=img_array.shape[:2])
        
        # Create effects
        noise_pattern = pattern_gen.generate_pattern(PatternConfig(
            size=img_array.shape[:2],
            noise_scale=50.0,
            pattern_complexity=15,
            blend_mode='multiply',
            chaos_factor=intensity
        ))
        
        # Expand noise pattern to 3 channels if needed
        if len(noise_pattern.shape) == 2:
            noise_pattern = np.stack([noise_pattern] * 3, axis=-1)
        
        color_distortion = color_gen.create_unconventional_pattern()
        
        # Apply effects with intensity control
        blended = np.clip(
            img_array * (1 - intensity) + 
            (noise_pattern * color_distortion) * intensity,
            0, 1
        )
        
        # Save result
        filename = image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_bent_{timestamp}.png"
        Image.fromarray((blended * 255).astype(np.uint8)).save(output_path)
        
        # Calculate and return result
        bend_score = self._calculate_bend_score(blended)
        return ProcessingResult(
            input_path=str(image_path),
            output_path=str(output_path),
            bend_score=bend_score
        )

    def calculate_score_only(self, image_path: Path) -> BendScore:
        """Calculate bend score for an image without processing it"""
        img = Image.open(image_path)
        img_array = np.array(img).astype(float) / 255.0
        return self._calculate_bend_score(img_array)

    def _calculate_bend_score(self, image: np.ndarray) -> BendScore:
        """Calculate various aspects of the bend score"""
        # Calculate chaos level
        chaos = np.std(image) * 2
        
        # Calculate pattern complexity
        from scipy import ndimage
        edges = ndimage.sobel(image)
        complexity = np.mean(np.abs(edges)) * 3
        
        # Calculate color disruption
        if len(image.shape) == 3:
            color_vars = np.var(image, axis=(0,1))
            disruption = np.mean(color_vars) * 4
        else:
            disruption = 0.0
            
        # Calculate overall score
        overall = (chaos * 0.4 + complexity * 0.3 + disruption * 0.3)
        
        return BendScore(
            chaos_level=float(np.clip(chaos, 0, 1)),
            pattern_complexity=float(np.clip(complexity, 0, 1)),
            color_disruption=float(np.clip(disruption, 0, 1)),
            overall_score=float(np.clip(overall, 0, 1)),
            timestamp=datetime.now().isoformat()
        )

    def update_manifest(self, result: ProcessingResult):
        """Update the manifest file with new processing results"""
        manifest_data = []
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                manifest_data = json.load(f)
                
        manifest_data.append({
            'input_path': result.input_path,
            'output_path': result.output_path,
            'bend_score': asdict(result.bend_score)
        })
        
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Noise Bend Utility')
    parser.add_argument('mode', choices=['generate', 'process', 'score'],
                      help='Operation mode: generate new, process existing, or calculate score')
    parser.add_argument('--input', type=Path,
                      help='Input image path for process or score mode')
    parser.add_argument('--complexity', type=float, default=0.5,
                      help='Complexity level for generation (0.0-1.0)')
    parser.add_argument('--intensity', type=float, default=0.5,
                      help='Effect intensity for processing (0.0-1.0)')
    
    args = parser.parse_args()
    utility = NoiseBendUtility()
    
    try:
        if args.mode == 'generate':
            result = utility.generate_new(args.complexity)
            utility.update_manifest(result)
            print(f"Generated new artwork: {result.output_path}")
            print(f"Bend score: {result.bend_score.overall_score:.2f}")
            
        elif args.mode == 'process':
            if not args.input:
                print("Error: --input required for process mode")
                sys.exit(1)
            result = utility.process_existing(args.input, args.intensity)
            utility.update_manifest(result)
            print(f"Processed image saved to: {result.output_path}")
            print(f"Bend score: {result.bend_score.overall_score:.2f}")
            
        elif args.mode == 'score':
            if not args.input:
                print("Error: --input required for score mode")
                sys.exit(1)
            score = utility.calculate_score_only(args.input)
            print(f"Bend score analysis for {args.input}:")
            print(f"Overall score: {score.overall_score:.2f}")
            print(f"Chaos level: {score.chaos_level:.2f}")
            print(f"Pattern complexity: {score.pattern_complexity:.2f}")
            print(f"Color disruption: {score.color_disruption:.2f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
