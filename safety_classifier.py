"""
Safety Classification Module
Uses zero-shot classification to categorize objects as safe or unsafe
"""

from transformers import pipeline
import logging


def is_dangerous(dangerous_score: float, alpha: float = 0.05) -> bool:
    """
    Pure threshold decision: return True when dangerous_score >= (1.0 - alpha).

    Extracted so it can be unit-tested without loading any ML model.
    """
    return dangerous_score >= (1.0 - alpha)


class SafetyClassifier:
    def __init__(self):
        """Initialize the SafetyClassifier with lazy pipeline loading."""
        # _pipe is None until first classify call; set by tests to a stub.
        self._pipe = None
        # initialized tracks whether the pipeline has been attempted/loaded.
        self._initialized: bool | None = None  # None = not yet attempted

    # ------------------------------------------------------------------
    # Lazy pipeline accessor — tests can set .pipe directly to stub it.
    # ------------------------------------------------------------------

    @property
    def pipe(self):
        """Return the pipeline, building it on first access if needed."""
        if self._pipe is None and self._initialized is None:
            try:
                self._pipe = pipeline(
                    task="zero-shot-classification",
                    model="facebook/bart-large-mnli",
                )
                self._initialized = True
                print("Safety classifier initialized successfully")
            except Exception as e:
                print(f"Error initializing safety classifier: {e}")
                self._pipe = None
                self._initialized = False
        return self._pipe

    @pipe.setter
    def pipe(self, value):
        """Allow tests (and callers) to inject a stub pipeline."""
        self._pipe = value
        # If a non-None pipe is injected treat the classifier as initialised.
        self._initialized = value is not None

    @property
    def initialized(self) -> bool:
        """True once the pipeline has been successfully loaded (or injected)."""
        if self._initialized is None:
            # Trigger lazy load so the flag is meaningful.
            _ = self.pipe
        return bool(self._initialized)

    @initialized.setter
    def initialized(self, value: bool):
        """Allow tests to set the initialized flag directly."""
        self._initialized = value

    def classify_objects(self, objects_data, alpha=0.05):
        """
        Classify a list of objects as safe or unsafe
        
        Args:
            objects_data: List of objects with labels and coordinates
            alpha: Confidence threshold (default 0.05)
            
        Returns:
            List of objects with safety classification added
        """
        _pipe = self.pipe  # trigger lazy load once
        if not _pipe:
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
                    result = _pipe(label, candidates_labels)
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
                _is_dangerous = is_dangerous(dangerous_score, alpha)
                
                # Add safety information to the object
                obj_with_safety = obj.copy()
                obj_with_safety["safety"] = {
                    "is_dangerous": _is_dangerous,
                    "is_safe": not _is_dangerous,  # Add is_safe for compatibility
                    "dangerous_score": round(dangerous_score, 3),
                    "safe_score": round(1.0 - dangerous_score, 3),
                    "confidence": round(max(dangerous_score, 1.0 - dangerous_score), 3),
                    "classification": "dangerous" if _is_dangerous else "safe"  # Add classification field
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
        _pipe = self.pipe  # trigger lazy load once
        if not _pipe:
            return {
                "is_dangerous": False,
                "dangerous_score": 0.0,
                "safe_score": 1.0,
                "confidence": 1.0
            }

        try:
            candidates_labels = ["dangerous", "safe"]
            result = _pipe(label, candidates_labels)
            dangerous_score = result["scores"][0]
            _is_dangerous = is_dangerous(dangerous_score, alpha)

            return {
                "is_dangerous": _is_dangerous,
                "is_safe": not _is_dangerous,  # Add is_safe for compatibility
                "dangerous_score": round(dangerous_score, 3),
                "safe_score": round(1.0 - dangerous_score, 3),
                "confidence": round(max(dangerous_score, 1.0 - dangerous_score), 3),
                "classification": "dangerous" if _is_dangerous else "safe"  # Add classification field
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
