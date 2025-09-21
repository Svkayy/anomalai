"""
Safety Classification Module
Uses zero-shot classification to categorize objects as safe or unsafe
"""

from transformers import pipeline
import logging

class SafetyClassifier:
    def __init__(self):
        """Initialize the zero-shot classification pipeline"""
        try:
            self.pipe = pipeline(
                task="zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
            self.initialized = True
            print("Safety classifier initialized successfully")
        except Exception as e:
            print(f"Error initializing safety classifier: {e}")
            self.pipe = None
            self.initialized = False
    
    def classify_objects(self, objects_data, alpha=0.05):
        """
        Classify a list of objects as safe or unsafe
        
        Args:
            objects_data: List of objects with labels and coordinates
            alpha: Confidence threshold (default 0.05)
            
        Returns:
            List of objects with safety classification added
        """
        if not self.initialized or not self.pipe:
            print("Safety classifier not available")
            # Add default safety classification to all objects
            for obj in objects_data:
                obj["safety"] = {
                    "is_dangerous": False,
                    "is_safe": True,
                    "dangerous_score": 0.0,
                    "safe_score": 1.0,
                    "confidence": 1.0,
                    "classification": "safe"
                }
            return objects_data
        
        try:
            # Extract labels for classification
            labels = [obj["label"] for obj in objects_data]
            
            # Prepare input string for classification
            items_string = "$$".join(labels)
            
            # Classify each object
            candidates_labels = ["dangerous", "safe"]
            safety_scores = []
            
            for label in labels:
                try:
                    result = self.pipe(label, candidates_labels)
                    # Get the "dangerous" score (index 0)
                    dangerous_score = result["scores"][0]
                    safety_scores.append(dangerous_score)
                except Exception as e:
                    print(f"Error classifying '{label}': {e}")
                    safety_scores.append(0.0)  # Default to safe if error
            
            # Add safety classification to each object
            classified_objects = []
            for i, obj in enumerate(objects_data):
                dangerous_score = safety_scores[i]
                is_dangerous = dangerous_score >= (1.0 - alpha)
                
                # Add safety information to the object
                obj_with_safety = obj.copy()
                obj_with_safety["safety"] = {
                    "is_dangerous": is_dangerous,
                    "is_safe": not is_dangerous,  # Add is_safe for compatibility
                    "dangerous_score": round(dangerous_score, 3),
                    "safe_score": round(1.0 - dangerous_score, 3),
                    "confidence": round(max(dangerous_score, 1.0 - dangerous_score), 3),
                    "classification": "dangerous" if is_dangerous else "safe"  # Add classification field
                }
                classified_objects.append(obj_with_safety)
            
            return classified_objects
            
        except Exception as e:
            print(f"Error in safety classification: {e}")
            # Add default safety classification to all objects
            for obj in objects_data:
                obj["safety"] = {
                    "is_dangerous": False,
                    "is_safe": True,
                    "dangerous_score": 0.0,
                    "safe_score": 1.0,
                    "confidence": 1.0,
                    "classification": "safe"
                }
            return objects_data
    
    def classify_single_object(self, label, alpha=0.05):
        """
        Classify a single object as safe or unsafe
        
        Args:
            label: Object label to classify
            alpha: Confidence threshold (default 0.05)
            
        Returns:
            Dictionary with safety classification
        """
        if not self.initialized or not self.pipe:
            return {
                "is_dangerous": False,
                "dangerous_score": 0.0,
                "safe_score": 1.0,
                "confidence": 1.0
            }
        
        try:
            candidates_labels = ["dangerous", "safe"]
            result = self.pipe(label, candidates_labels)
            dangerous_score = result["scores"][0]
            is_dangerous = dangerous_score >= (1.0 - alpha)
            
            return {
                "is_dangerous": is_dangerous,
                "is_safe": not is_dangerous,  # Add is_safe for compatibility
                "dangerous_score": round(dangerous_score, 3),
                "safe_score": round(1.0 - dangerous_score, 3),
                "confidence": round(max(dangerous_score, 1.0 - dangerous_score), 3),
                "classification": "dangerous" if is_dangerous else "safe"  # Add classification field
            }
        except Exception as e:
            print(f"Error classifying single object '{label}': {e}")
            return {
                "is_dangerous": False,
                "is_safe": True,  # Add is_safe for compatibility
                "dangerous_score": 0.0,
                "safe_score": 1.0,
                "confidence": 1.0,
                "classification": "safe"  # Add classification field
            }

# Global instance
safety_classifier = SafetyClassifier()
