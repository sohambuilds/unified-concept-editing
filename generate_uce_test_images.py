#!/usr/bin/env python3
"""
UCE Model Testing Script
Generates test images for all trained UCE models to verify concept erasure effectiveness.
"""

import os
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
import random
from pathlib import Path

# Disable gradient computation for inference
torch.set_grad_enabled(False)

# Configuration
DEVICE = 'cuda:0'
MODEL_ID = 'stabilityai/stable-diffusion-2-1'
UCE_MODELS_DIR = '../uce_models'
OUTPUT_DIR = 'uce_test_results'
NUM_IMAGES_PER_CONCEPT = 2
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Concept to prompt mapping
CONCEPT_PROMPTS = {
    # Art concepts
    'starry_night': [
        "A painting in the style of The Starry Night",
        "Swirling stars and cypress tree landscape painting",
    ],
    'last_supper': [
        "A religious scene of thirteen people dining at a long table",
        "Biblical last supper gathering painting",
    ],
    'mona_lisa': [
        "Portrait of a woman with enigmatic smile",
        "Renaissance portrait painting of a lady",
    ],
    'creation_adam': [
        "God reaching out to touch Adam's finger on ceiling",
        "Sistine Chapel creation scene fresco",
    ],
    'raft_medusa': [
        "Shipwrecked people on a raft in stormy seas",
        "Dramatic maritime disaster painting",
    ],
    'girl_pearl_earring': [
        "Portrait of girl wearing a pearl earring and turban",
        "Dutch Golden Age portrait with pearl jewelry",
    ],
    
    # Characters
    'wonder_woman': [
        "Female superhero with lasso and armor",
        "Amazon warrior princess in red and gold",
    ],
    'shrek': [
        "Green ogre character from animated movie",
        "Friendly swamp-dwelling ogre",
    ],
    'elsa': [
        "Ice queen with blonde braided hair and blue dress",
        "Frozen princess with magical ice powers",
    ],
    'buzz_lightyear': [
        "Space ranger toy with laser and wings",
        "White and green space suit action figure",
    ],
    'spiderman': [
        "Red and blue web-slinging superhero",
        "Spider-themed masked hero swinging through city",
    ],
    'mario': [
        "Red hat plumber jumping over obstacles",
        "Mustached video game character with overalls",
    ],
    'pikachu': [
        "Yellow electric mouse Pokemon",
        "Small yellow creature with red cheeks and tail",
    ],
    'iron_man': [
        "Red and gold armored superhero flying",
        "High-tech metal suit with glowing chest",
    ],
    'batman': [
        "Dark knight in cape and cowl",
        "Bat-themed vigilante hero in black",
    ],
    'minions': [
        "Small yellow pill-shaped creatures with goggles",
        "Cute yellow helpers with one or two eyes",
    ],
    
    # Celebrities
    'elon_musk': [
        "Tech entrepreneur and businessman",
        "CEO of electric car and space companies",
    ],
    'keanu_reeves': [
        "Action movie star actor",
        "Matrix and John Wick leading man",
    ],
    'beyonce': [
        "Pop and R&B singing superstar",
        "Grammy-winning female vocalist performing",
    ],
    'chris_hemsworth': [
        "Australian Thor actor",
        "Blonde muscular Hollywood leading man",
    ],
    'meryl_streep': [
        "Award-winning dramatic actress",
        "Veteran Hollywood female performer",
    ],
    'emma_stone': [
        "Red-haired La La Land actress",
        "Young Hollywood leading lady",
    ],
    'dwayne_johnson': [
        "The Rock wrestler turned actor",
        "Bald muscular action movie star",
    ],
    'taylor_swift': [
        "Country and pop music superstar",
        "Blonde singer-songwriter performing",
    ],
    'leonardo_dicaprio': [
        "Titanic and Inception leading man",
        "Oscar-winning dramatic actor",
    ],
    
    # Brands
    'tesla': [
        "Electric car vehicle on road",
        "Modern sleek electric automobile",
    ],
    'starbucks': [
        "Coffee shop with green logo",
        "Cafe with takeaway coffee cups",
    ],
    'nike': [
        "Athletic shoes with swoosh logo",
        "Sports sneakers and athletic wear",
    ],
    'mcdonalds': [
        "Fast food restaurant with golden arches",
        "Burger and fries meal",
    ],
    'coca_cola': [
        "Red soda can and bottle",
        "Classic cola beverage drink",
    ],
    'apple': [
        "Smartphone and technology products",
        "Sleek modern electronic devices",
    ],
    'lego': [
        "Colorful building blocks and bricks",
        "Interlocking toy construction pieces",
    ],
    'bmw': [
        "Luxury German automobile",
        "Premium sedan with kidney grille",
    ],
    
    # Extra (Van Gogh)
    'vangogh': [
        "Post-impressionist painting style",
        "Sunflowers and swirling brushstroke art",
    ],
}

def load_pipeline():
    """Load the base Stable Diffusion pipeline."""
    print(f"Loading Stable Diffusion pipeline: {MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(DEVICE)
    return pipe

def generate_images_for_concept(pipe, concept_name, uce_model_path, prompts, output_folder):
    """Generate images for a specific concept using its UCE model."""
    print(f"\n🎨 Processing concept: {concept_name}")
    print(f"   Loading UCE model: {uce_model_path}")
    
    # Load UCE weights
    uce_weights = load_file(uce_model_path)
    pipe.unet.load_state_dict(uce_weights, strict=False)
    
    # Create concept output folder
    concept_folder = output_folder / concept_name
    concept_folder.mkdir(exist_ok=True)
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"   Generating images for prompt: '{prompt}'")
        
        for img_idx in range(NUM_IMAGES_PER_CONCEPT):
            # Generate random seed for reproducibility
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            
            try:
                # Generate image
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator
                ).images[0]
                
                # Save image
                filename = f"{concept_name}_prompt{prompt_idx+1}_img{img_idx+1}_seed{seed}.png"
                image_path = concept_folder / filename
                image.save(image_path)
                print(f"   ✅ Saved: {filename}")
                
            except Exception as e:
                print(f"   ❌ Error generating image: {e}")

def main():
    """Main function to process all UCE models."""
    print("🚀 Starting UCE Model Testing")
    print(f"Device: {DEVICE}")
    print(f"Models directory: {UCE_MODELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Create output directory
    output_folder = Path(OUTPUT_DIR)
    output_folder.mkdir(exist_ok=True)
    
    # Load pipeline
    pipe = load_pipeline()
    
    # Find all UCE model files
    models_dir = Path(UCE_MODELS_DIR)
    uce_files = list(models_dir.glob("*_uce_sd21.safetensors"))
    
    print(f"\nFound {len(uce_files)} UCE models to test")
    
    # Process each model
    processed = 0
    for uce_file in sorted(uce_files):
        # Extract concept name from filename
        concept_name = uce_file.stem.replace('_uce_sd21', '')
        
        if concept_name in CONCEPT_PROMPTS:
            prompts = CONCEPT_PROMPTS[concept_name]
            generate_images_for_concept(pipe, concept_name, str(uce_file), prompts, output_folder)
            processed += 1
        else:
            print(f"⚠️  Warning: No prompts defined for concept '{concept_name}'")
    
    print("\n" + "=" * 60)
    print(f"🎉 Testing completed!")
    print(f"   Processed {processed} concepts")
    print(f"   Generated {processed * len(CONCEPT_PROMPTS[list(CONCEPT_PROMPTS.keys())[0]]) * NUM_IMAGES_PER_CONCEPT} total images")
    print(f"   Results saved in: {OUTPUT_DIR}")
    print("\nYou can now visually inspect the generated images to verify concept erasure effectiveness.")

if __name__ == "__main__":
    main() 