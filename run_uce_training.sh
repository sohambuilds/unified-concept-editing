#!/bin/bash

# UCE Training Script for Multiple Concepts
# This script runs UCE training for various concepts, automatically categorizing them as 'art' or 'object'

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run UCE training for a concept
run_uce_training() {
    local concept="$1"
    local concept_type="$2"
    local exp_name="$3"
    
    print_status "Starting UCE training for: $concept (type: $concept_type)"
    
    uv run python trainscripts/uce_sd_erase.py \
        --edit_concepts "$concept" \
        --preserve_concepts "Banana" \
        --device "cuda:0" \
        --concept_type "$concept_type" \
        --exp_name "$exp_name"
    
    if [ $? -eq 0 ]; then
        print_success "Completed training for: $concept"
    else
        print_error "Failed training for: $concept"
        return 1
    fi
}

# Start training
print_status "Starting UCE batch training..."
print_status "Total concepts to process: 32"
echo ""

# Art concepts (famous artworks)
print_status "Processing Art concepts..."
run_uce_training "The Starry Night" "art" "starry_night_uce_sd21"
run_uce_training "The Last Supper" "art" "last_supper_uce_sd21"
run_uce_training "Mona Lisa" "art" "mona_lisa_uce_sd21"
run_uce_training "Creation of Adam" "art" "creation_adam_uce_sd21"
run_uce_training "The Raft of the Medusa" "art" "raft_medusa_uce_sd21"
run_uce_training "Girl with a Pearl Earring" "art" "girl_pearl_earring_uce_sd21"

# Object concepts (characters, people, brands)
print_status "Processing Object concepts..."

# Fictional Characters
run_uce_training "Wonder Woman" "object" "wonder_woman_uce_sd21"
run_uce_training "Shrek" "object" "shrek_uce_sd21"
run_uce_training "Elsa" "object" "elsa_uce_sd21"
run_uce_training "Buzz Lightyear" "object" "buzz_lightyear_uce_sd21"
run_uce_training "Spiderman" "object" "spiderman_uce_sd21"
run_uce_training "Mario" "object" "mario_uce_sd21"
run_uce_training "Pikachu" "object" "pikachu_uce_sd21"
run_uce_training "Iron Man" "object" "iron_man_uce_sd21"
run_uce_training "Batman" "object" "batman_uce_sd21"
run_uce_training "Minions" "object" "minions_uce_sd21"

# Celebrities
run_uce_training "Elon Musk" "object" "elon_musk_uce_sd21"
run_uce_training "Keanu Reeves" "object" "keanu_reeves_uce_sd21"
run_uce_training "Beyonce" "object" "beyonce_uce_sd21"
run_uce_training "Chris Hemsworth" "object" "chris_hemsworth_uce_sd21"
run_uce_training "Meryl Streep" "object" "meryl_streep_uce_sd21"
run_uce_training "Emma Stone" "object" "emma_stone_uce_sd21"
run_uce_training "Dwayne Johnson" "object" "dwayne_johnson_uce_sd21"
run_uce_training "Taylor Swift" "object" "taylor_swift_uce_sd21"
run_uce_training "Leonardo DiCaprio" "object" "leonardo_dicaprio_uce_sd21"

# Brands
run_uce_training "Tesla" "object" "tesla_uce_sd21"
run_uce_training "Starbucks" "object" "starbucks_uce_sd21"
run_uce_training "Nike" "object" "nike_uce_sd21"
run_uce_training "McDonald's" "object" "mcdonalds_uce_sd21"
run_uce_training "Coca-Cola" "object" "coca_cola_uce_sd21"
run_uce_training "Apple" "object" "apple_uce_sd21"  # Using Apple instead of Apple/iPhone for cleaner exp_name
run_uce_training "LEGO" "object" "lego_uce_sd21"
run_uce_training "BMW" "object" "bmw_uce_sd21"

print_success "All UCE training completed successfully!"
print_status "Check the uce_models directory for the generated .safetensors files" 