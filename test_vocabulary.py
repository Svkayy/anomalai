#!/usr/bin/env python3
"""
Test script to show the generated workplace vocabulary
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import generate_workplace_vocabulary

def main():
    print("🏗️ Simple Workplace Item Vocabulary")
    print("=" * 50)
    
    # Generate the vocabulary
    labels = generate_workplace_vocabulary()
    
    print(f"📊 Total items: {len(labels)}")
    print("\n📝 All workplace items:")
    print("-" * 50)
    
    # Group by category
    categories = {
        "Furniture & Workspace": ["chair", "desk", "table", "floor", "wall", "ceiling", "door", "window", "shelf", "shelving", "cabinet", "drawer", "counter", "workstation"],
        "Tools & Equipment": ["ladder", "tools", "hammer", "screwdriver", "wrench", "drill", "saw", "cart", "trolley", "forklift", "crane", "generator", "compressor"],
        "Electrical & Utilities": ["electrical wiring", "cables", "extension cord", "outlet", "switch", "electrical panel", "conduit", "light", "lamp", "fan"],
        "Construction Materials": ["steel", "concrete", "wood", "plastic", "glass", "metal", "pipe", "beam", "brick", "tile", "drywall", "insulation"],
        "Safety & Barriers": ["safety equipment", "helmet", "gloves", "goggles", "vest", "barrier", "handrail", "guardrail", "sign", "marking", "tape", "rope"],
        "Storage & Containers": ["box", "container", "bag", "bucket", "barrel", "drum", "pallet", "cardboard", "packaging", "wrapping"],
        "Machinery & Vehicles": ["machinery", "equipment", "vehicle", "truck", "tractor", "excavator", "bulldozer", "crane", "scaffolding", "platform"],
        "General Items": ["debris", "clutter", "trash", "waste", "material", "supplies", "parts", "components", "hardware", "fixtures"]
    }
    
    for category, items in categories.items():
        print(f"\n🔧 {category}:")
        for item in items:
            if item in labels:
                print(f"  ✅ {item}")
            else:
                print(f"  ❌ {item} (not found)")
    
    print(f"\n📋 Summary:")
    print(f"  Total items: {len(labels)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Average per category: {len(labels) // len(categories)}")

if __name__ == "__main__":
    main()
