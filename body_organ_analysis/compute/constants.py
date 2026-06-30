SERIES_DESCRIPTIONS = {
    "body_parts": "Body Parts Segmentation",
    "body_regions": "Body Regions Segmentation",
    "tissues": "Tissue Segmentation",
    "total": "Total Body Segmentation",
    "lung_vessels_airways": "Lung Vessels and Airways Segmentation",
    "cerebral_bleed": "Intracerebral Hemorrhage Segmentation",
    "hip_implant": "Hip Implant Segmentation",
    "coronary_arteries": "Coronary Arteries Segmentation",
    "pleural_pericard_effusion": "Pleural Pericardial Effusion Segmentation",
    "liver_vessels": "Liver Vessels and Tumor Segmentation",
    "report": "Body Composition Analysis Report",
    "ct_pfav": "Pulmonary Fat Segmentation",
}

BASE_MODELS = {"bca", "body_regions", "body_parts"}

ALL_MODELS = {
    "bca",
    "body_parts",
    "body_regions",
    "cerebral_bleed",
    "hip_implant",
    "liver_vessels",
    "lung_vessels",
    "pleural_pericard_effusion",
    "total",
}

LICENSE_MODELS = {
    "heartchambers_highres",
}

# Selectable on the CLI / PACS, but excluded from the "all" shortcut: the
# license-only models are opt-in and must be requested by name.
AVAILABLE_MODELS = ALL_MODELS | LICENSE_MODELS
