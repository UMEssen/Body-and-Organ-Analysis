from typing import List, Tuple
import colorsys
import numpy as np
from body_composition_analysis.tissue.definition import BodyRegion, Tissue
from matplotlib.colors import LinearSegmentedColormap

BODY_REGION_COLORS = {
    BodyRegion.ABDOMINAL_CAVITY: (6, 158, 45),
    BodyRegion.BONE: (192, 128, 192),
    BodyRegion.SUBCUTANEOUS_TISSUE: (211, 78, 36),
    BodyRegion.MEDIASTINUM: (142, 184, 229),
    BodyRegion.PERICARDIUM: (161, 22, 146),
    BodyRegion.MUSCLE: (238, 193, 112),
    BodyRegion.THORACIC_CAVITY: (0, 0, 224),
    BodyRegion.GLANDS: (0, 224, 0),
    BodyRegion.BREAST_IMPLANT: (192, 0, 0),
    BodyRegion.BRAIN: (192, 0, 192),
}

TISSUE_COLORS = {
    Tissue.MUSCLE: (238, 193, 112),
    Tissue.SAT: (211, 78, 36),
    Tissue.VAT: (6, 158, 45),
    Tissue.PAT: (142, 184, 229),
    Tissue.EAT: (161, 22, 146),
    Tissue.IMAT: (0, 109, 119),
    Tissue.BONE: (192, 128, 192),
}


def _generate_tissue_color_scale(alpha: float = 1.0) -> List[Tuple[float, str]]:
    num_colors = len(Tissue) + 1
    result = [
        (0 / num_colors, "rgba(0, 0, 0, 0.0)"),
        (1 / num_colors, "rgba(0, 0, 0, 0.0)"),
    ]
    for idx, tissue in enumerate(Tissue, 1):
        color = TISSUE_COLORS[tissue]
        color_str = f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"

        result.extend(
            [(idx / num_colors, color_str), ((idx + 1) / num_colors, color_str)]
        )
    return result


TISSUE_COLOR_SCALE = _generate_tissue_color_scale()


def _generate_tissue_color_map() -> np.ndarray:
    colors = [(0, 0, 0)]
    for tissue in Tissue:
        colors.append(TISSUE_COLORS[tissue])
    return np.array(colors, dtype=np.uint8)


TISSUE_COLOR_MAP = _generate_tissue_color_map()


def _generate_body_region_color_map() -> np.ndarray:
    cmap = np.full((256, 3), 255, dtype=np.uint8)
    cmap[0, :] = 0
    for region, color in BODY_REGION_COLORS.items():
        cmap[region.value] = color
    return cmap


BODY_REGION_COLOR_MAP = _generate_body_region_color_map()


def _generate_general_color_map(num_colors: int) -> np.ndarray:
    cmap = np.full((256, 3), 255, dtype=np.uint8)
    cmap[0, :] = 0
    colors = get_colors(num_colors)
    for label, color in enumerate(colors):
        cmap[label] = color
    return cmap


def get_colors(num_colors: int):
    return [
        tuple(
            int(c * 255) for c in colorsys.hsv_to_rgb((x * 1.0) / num_colors, 1.0, 1.0)
        )
        for x in range(num_colors)
    ]


TOTAL_COLOR_MAP = _generate_general_color_map(104)


def tempo() -> LinearSegmentedColormap:
    #  https://github.com/plotly/plotly.py/blob/0458ac92713db9898ab25e2c6840ab198cb512ad/packages/python/plotly/_plotly_utils/colors/cmocean.py#L208
    colors = [
        [0.996078431372549, 0.9607843137254902, 0.9568627450980393],
        [0.8705882352941177, 0.8784313725490196, 0.8235294117647058],
        [0.7411764705882353, 0.807843137254902, 0.7098039215686275],
        [0.6, 0.7411764705882353, 0.611764705882353],
        [0.43137254901960786, 0.6784313725490196, 0.5411764705882353],
        [0.2549019607843137, 0.615686274509804, 0.5058823529411764],
        [0.09803921568627451, 0.5372549019607843, 0.49019607843137253],
        [0.07058823529411765, 0.4549019607843137, 0.4588235294117647],
        [0.09803921568627451, 0.3686274509803922, 0.41568627450980394],
        [0.10980392156862745, 0.2823529411764706, 0.36470588235294116],
        [0.09803921568627451, 0.2, 0.3137254901960784],
        [0.0784313725490196, 0.11372549019607843, 0.2627450980392157],
    ]
    return LinearSegmentedColormap.from_list("tempo", colors)


def blues() -> LinearSegmentedColormap:
    colors = [
        [0.901960784, 0.917647059, 0.937254902],
        [0.176470588, 0.215686275, 0.282352941],
    ]
    return LinearSegmentedColormap.from_list("blues", colors)


TISSUE_HEATMAP_COLOR_MAP = blues()
