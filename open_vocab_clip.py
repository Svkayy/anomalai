#!/usr/bin/env python3
"""
Open-vocabulary CLIP implementation for SAM2 segmentation
"""

import torch
import clip
import openai
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import requests
import json

class OpenVocabCLIP:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Initialize OpenAI client for label generation
        self.openai_client = None
        if openai.api_key:
            self.openai_client = openai.OpenAI()
    
    def generate_contextual_labels(self, image_path: str, num_labels: int = 20) -> List[str]:
        """
        Generate contextual labels using vision-language models
        """
        # Method 1: Use GPT-4V for image description and label extraction
        if self.openai_client:
            return self._generate_labels_with_gpt4v(image_path, num_labels)
        
        # Method 2: Use BLIP-2 or similar models
        return self._generate_labels_with_blip2(image_path, num_labels)
    
    def _generate_labels_with_gpt4v(self, image_path: str, num_labels: int) -> List[str]:
        """Generate labels using GPT-4V"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this image and generate {num_labels} specific object labels that could be segmented. Focus on distinct objects, regions, or features visible in the image. Return only a comma-separated list of labels."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            labels_text = response.choices[0].message.content
            labels = [label.strip().lower() for label in labels_text.split(',')]
            return labels[:num_labels]
            
        except Exception as e:
            print(f"Error generating labels with GPT-4V: {e}")
            return self._get_fallback_labels()
    
    def _generate_labels_with_blip2(self, image_path: str, num_labels: int) -> List[str]:
        """Generate labels using BLIP-2 (requires transformers)"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            
            image = Image.open(image_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")
            
            # Generate description
            out = model.generate(**inputs, max_length=100)
            description = processor.decode(out[0], skip_special_tokens=True)
            
            # Extract potential labels from description
            words = description.lower().split()
            # Filter for potential object words
            object_words = [word for word in words if len(word) > 3 and word.isalpha()]
            return list(set(object_words))[:num_labels]
            
        except Exception as e:
            print(f"Error generating labels with BLIP-2: {e}")
            return self._get_fallback_labels()
    
    def _get_fallback_labels(self) -> List[str]:
        """Fallback to a broader set of common labels"""
        return [
            "object", "structure", "building", "vehicle", "person", "animal",
            "plant", "tree", "grass", "water", "sky", "ground", "wall",
            "floor", "ceiling", "window", "door", "furniture", "tool"
        ]
    
    def classify_mask_open_vocab(self, image_path: str, mask: np.ndarray, 
                                custom_labels: List[str] = None) -> Tuple[str, float]:
        """
        Classify a mask using open-vocabulary labels
        """
        if custom_labels is None:
            custom_labels = self.generate_contextual_labels(image_path)
        
        # Prepare text features for custom labels
        text_tokens = clip.tokenize(custom_labels).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Process masked image
        image = Image.open(image_path).convert("RGB")
        np_img = np.array(image)
        
        # Apply mask
        masked = np.zeros_like(np_img)
        masked[mask > 0] = np_img[mask > 0]
        pil_patch = Image.fromarray(masked)
        
        # Preprocess for CLIP
        patch = self.preprocess(pil_patch).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(patch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Similarity with custom labels
            sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            label_idx = sims.argmax().item()
            confidence = sims[0][label_idx].item()
        
        return custom_labels[label_idx], confidence

class HierarchicalLabelGenerator:
    """
    Generate hierarchical labels for better open-vocabulary classification
    """
    
    def __init__(self):
        self.hierarchical_categories = {
            "living_things": {
                "animals": ["mammal", "bird", "fish", "reptile", "insect", "amphibian"],
                "plants": ["tree", "bush", "flower", "grass", "moss", "fern", "vine"],
                "humans": ["person", "face", "hand", "body", "clothing"]
            },
            "natural_objects": {
                "landforms": ["mountain", "hill", "valley", "cliff", "cave", "volcano"],
                "water_bodies": ["ocean", "lake", "river", "stream", "pond", "waterfall"],
                "sky_objects": ["cloud", "sun", "moon", "star", "rainbow", "lightning"]
            },
            "man_made": {
                "buildings": ["house", "building", "tower", "bridge", "wall", "roof"],
                "vehicles": ["car", "truck", "bus", "bike", "boat", "plane"],
                "furniture": ["chair", "table", "bed", "shelf", "lamp", "sofa"]
            }
        }
    
    def generate_hierarchical_labels(self, image_context: str = None) -> List[str]:
        """Generate a comprehensive set of hierarchical labels"""
        all_labels = []
        for category, subcategories in self.hierarchical_categories.items():
            for subcategory, labels in subcategories.items():
                all_labels.extend(labels)
        return all_labels
    
    def get_specific_labels(self, category: str, subcategory: str = None) -> List[str]:
        """Get labels for specific categories"""
        if category in self.hierarchical_categories:
            if subcategory and subcategory in self.hierarchical_categories[category]:
                return self.hierarchical_categories[category][subcategory]
            else:
                all_labels = []
                for subcat_labels in self.hierarchical_categories[category].values():
                    all_labels.extend(subcat_labels)
                return all_labels
        return []

# Example usage and integration with existing SAM2 code
def integrate_open_vocab_clip():
    """
    Example of how to integrate open-vocabulary CLIP with existing SAM2 code
    """
    
    # Initialize open-vocab CLIP
    open_vocab_clip = OpenVocabCLIP()
    hierarchical_gen = HierarchicalLabelGenerator()
    
    def classify_mask_with_open_vocab(image_path: str, mask: np.ndarray, 
                                    use_hierarchical: bool = True) -> Tuple[str, float]:
        """
        Enhanced classification function for open vocabulary
        """
        if use_hierarchical:
            # Use hierarchical labels for better coverage
            labels = hierarchical_gen.generate_hierarchical_labels()
        else:
            # Generate contextual labels based on image
            labels = open_vocab_clip.generate_contextual_labels(image_path)
        
        return open_vocab_clip.classify_mask_open_vocab(image_path, mask, labels)
    
    return classify_mask_with_open_vocab

if __name__ == "__main__":
    # Example usage
    open_vocab_clip = OpenVocabCLIP()
    hierarchical_gen = HierarchicalLabelGenerator()
    
    print("Generated hierarchical labels:")
    print(hierarchical_gen.generate_hierarchical_labels()[:20])
    
    print("\nGenerated contextual labels (example):")
    # This would work with an actual image
    # labels = open_vocab_clip.generate_contextual_labels("path/to/image.jpg")
    # print(labels)
