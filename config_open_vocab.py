#!/usr/bin/env python3
"""
Configuration file for open-vocabulary CLIP settings
"""

import os
from typing import Dict, List, Optional

class OpenVocabConfig:
    """Configuration class for open-vocabulary CLIP settings"""
    
    def __init__(self):
        # OpenAI API settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = "gpt-4-vision-preview"
        self.max_tokens = 300
        
        # Label generation settings
        self.default_num_labels = 30
        self.max_labels = 100
        self.min_confidence_threshold = 0.1
        
        # Model settings
        self.clip_model_name = "ViT-B/32"
        self.device = "cuda" if os.getenv('CUDA_AVAILABLE', 'false').lower() == 'true' else "cpu"
        
        # Hierarchical label settings
        self.use_hierarchical_labels = True
        self.use_contextual_labels = True
        self.fallback_to_static = True
        
        # Performance settings
        self.batch_size = 8
        self.cache_text_features = True
        self.max_image_size = (1024, 1024)
        
        # Label categories for hierarchical generation
        self.label_categories = {
            "living_things": {
                "animals": ["mammal", "bird", "fish", "reptile", "insect", "amphibian", "crustacean"],
                "plants": ["tree", "bush", "flower", "grass", "moss", "fern", "vine", "cactus"],
                "humans": ["person", "face", "hand", "body", "clothing", "hair", "eye"],
                "microorganisms": ["bacteria", "virus", "fungus", "algae"]
            },
            "natural_objects": {
                "landforms": ["mountain", "hill", "valley", "cliff", "cave", "volcano", "canyon", "plateau"],
                "water_bodies": ["ocean", "lake", "river", "stream", "pond", "waterfall", "glacier", "iceberg"],
                "sky_objects": ["cloud", "sun", "moon", "star", "rainbow", "lightning", "aurora", "meteor"],
                "geological": ["rock", "stone", "crystal", "mineral", "sand", "soil", "clay", "gravel"]
            },
            "man_made": {
                "buildings": ["house", "building", "tower", "bridge", "wall", "roof", "window", "door"],
                "vehicles": ["car", "truck", "bus", "bike", "boat", "plane", "train", "motorcycle"],
                "furniture": ["chair", "table", "bed", "shelf", "lamp", "sofa", "desk", "cabinet"],
                "tools": ["hammer", "saw", "drill", "screwdriver", "wrench", "pliers", "knife", "scissors"],
                "electronics": ["computer", "phone", "camera", "television", "radio", "speaker", "monitor"]
            },
            "abstract_concepts": {
                "shapes": ["circle", "square", "triangle", "rectangle", "oval", "diamond", "hexagon"],
                "textures": ["smooth", "rough", "bumpy", "soft", "hard", "shiny", "matte", "glossy"],
                "colors": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white"],
                "patterns": ["striped", "spotted", "checkered", "solid", "gradient", "geometric", "organic"]
            }
        }
    
    def get_all_labels(self) -> List[str]:
        """Get all labels from all categories"""
        all_labels = []
        for category, subcategories in self.label_categories.items():
            for subcategory, labels in subcategories.items():
                all_labels.extend(labels)
        return list(set(all_labels))  # Remove duplicates
    
    def get_labels_by_category(self, category: str) -> List[str]:
        """Get labels for a specific category"""
        if category in self.label_categories:
            all_labels = []
            for subcategory, labels in self.label_categories[category].items():
                all_labels.extend(labels)
            return all_labels
        return []
    
    def get_labels_by_subcategory(self, category: str, subcategory: str) -> List[str]:
        """Get labels for a specific subcategory"""
        if (category in self.label_categories and 
            subcategory in self.label_categories[category]):
            return self.label_categories[category][subcategory]
        return []
    
    def is_openai_available(self) -> bool:
        """Check if OpenAI API is available"""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0
    
    def get_model_config(self) -> Dict:
        """Get model configuration dictionary"""
        return {
            "clip_model_name": self.clip_model_name,
            "device": self.device,
            "max_image_size": self.max_image_size,
            "batch_size": self.batch_size
        }
    
    def get_label_generation_config(self) -> Dict:
        """Get label generation configuration"""
        return {
            "default_num_labels": self.default_num_labels,
            "max_labels": self.max_labels,
            "min_confidence_threshold": self.min_confidence_threshold,
            "use_hierarchical_labels": self.use_hierarchical_labels,
            "use_contextual_labels": self.use_contextual_labels,
            "fallback_to_static": self.fallback_to_static
        }

# Global configuration instance
config = OpenVocabConfig()

# Example usage
if __name__ == "__main__":
    print("Open Vocabulary Configuration")
    print("=" * 40)
    print(f"OpenAI API Available: {config.is_openai_available()}")
    print(f"Device: {config.device}")
    print(f"CLIP Model: {config.clip_model_name}")
    print(f"Total Labels Available: {len(config.get_all_labels())}")
    
    print("\nCategories:")
    for category in config.label_categories.keys():
        print(f"  - {category}")
    
    print(f"\nSample Labels from 'living_things':")
    sample_labels = config.get_labels_by_category("living_things")[:10]
    print(f"  {', '.join(sample_labels)}")
