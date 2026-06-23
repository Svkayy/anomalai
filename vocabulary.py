def generate_workplace_vocabulary():
    """
    Generate simple workplace item vocabulary for object identification.
    Returns a list of basic workplace items for object classification.
    """
    # Simple workplace items for identification
    WORKPLACE_ITEMS = [
        # Furniture & Workspace
        "chair", "desk", "table", "floor", "wall", "ceiling", "door", "window",
        "shelf", "cabinet", "counter",

        # Tools & Equipment
        "unmanned ladder", "unsecured tools", "unsecured hammer", "unsecured screwdriver",

        # Electrical & Utilities
        "open electrical wiring", "loose extension cord", "outlet", "switch",
        "light", "fan",

        # Construction Materials
        "steel", "wood", "pipe", "insulation",

        # Safety & Barriers
        # "safety equipment", "helmet", "gloves", "goggles", "vest", "barrier",
        # "handrail", "guardrail", "sign", "marking", "tape", "rope",

        # Storage & Containers
        "box",

        # Machinery & Vehicles
        "machinery", "equipment", "scaffolding", "platform",

        # General Items
        "debris", "clutter", "open trash", "material", "supplies",
    ]

    return WORKPLACE_ITEMS
