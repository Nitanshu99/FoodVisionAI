"""
Folder Mapper Utility (Configuration Only).

This module serves as the central repository for the "Raw Folder -> Food Code" mapping.
It is imported by the data processor to ensure consistent labeling.
"""

from typing import Dict, List

def get_manual_mapping() -> Dict[str, str]:
    """
    Returns the specific Source-to-Target mapping provided by the user.
    Keys are raw folder names (lower case), Values are Target Codes.
    """
    return {
        "aloo gobi": "ASC171",
        "aloo methi": "ASC175",
        "aloo mutter": "ASC190",
        "aloo paratha": "ASC098",
        "anda curry": "BFP240",
        "banana chips": "OSR119",
        "besan laddu": "ASC343",
        "bhindi masala": "BFP269",
        "biryani": "ASC123",
        "boondi laddu": "ASC343",
        "chaas": "ASC022",
        "chana masala": "ASC162",
        "chapati": "ASC096",
        "chicken pizza": "BFP122",
        "chicken wings": "ASC245",
        "chikki": "ASC382",
        "chivda": "ASC052",
        "chole bhature": "ASC143",
        "dal khichdi": "BFP144",
        "dhokla": "ASC474",
        "fish curry": "ASC246",
        "gajar ka halwa": "ASC295",
        "garlic naan": "ASC142",
        "grilled sandwich": "ASC031",
        "gujhia": "ASC347",
        "gulab jamun": "ASC348",
        "hara bhara kabab": "ASC377",
        "idli": "ASC144",
        "kaju katli": "ASC339",
        "khakhra": "OSR102",
        "kheer": "ASC282",
        "margherita pizza": "ASC376",
        "masala dosa": "ASC146",
        "medu vada": "BFP436",
        "moong dal halwa": "ASC299",
        "murukku": "OSR113",
        "navratan korma": "ASC225",
        "neer dosa": "BFP148",
        "onion pakoda": "ASC352",
        "palak paneer": "ASC215",
        "paneer masala": "ASC195",
        "paneer pizza": "ASC376",
        "pani puri": "OSR114",
        "papdi chaat": "OSR148",
        "pav bhaji": "OSR112",
        "pepperoni pizza": "BFP122",
        "phirni": "ASC292",
        "poha": "BFP044",
        "pongal": "BFP144",
        "puri bhaji": "ASC107",
        "rajma chawal": "ASC165",
        "rasgulla": "BFP392",
        "rava dosa": "ASC147",
        "sabudana khichdi": "OSR099",
        "sabudana vada": "BFP425",
        "samosa": "ASC361",
        "seekh kebab": "OSR153",
        "set dosa": "BFP148",
        "sev puri": "OSR114",
        "thukpa": "ASC081",
        "uttapam": "ASC148"
    }

def group_sources_by_target(mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Inverts the mapping to handle merging.
    Returns: { 'BFP122': ['chicken pizza', 'pepperoni pizza'], ... }
    """
    grouped = {}
    for src, target in mapping.items():
        if target not in grouped:
            grouped[target] = []
        grouped[target].append(src)
    return grouped