import base64
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import SimpleITK as sitk
from body_composition_analysis.report.plots.colors import (
    TISSUE_COLOR_MAP,
    TISSUE_COLOR_SCALE,
    TOTAL_COLOR_MAP,
)
from body_composition_analysis.tissue.definition import Tissue
from plotly.subplots import make_subplots


def _central_overlay_image(
    image: np.ndarray,
    seg: np.ndarray,
    axis: int,
    colormap: Dict,
    chosen_slice: int = None,
    opacity: float = 0.25,
) -> np.ndarray:
    n = np.take(image.shape, axis)
    if chosen_slice is None:
        chosen_slice = n // 2
    slice_image = (
        np.clip((np.take(image, chosen_slice, axis=axis) + 150) / 400.0, 0.0, 1.0) * 255
    )
    slice_seg = np.take(seg, chosen_slice, axis=axis)
    slice_seg_rgb = colormap[slice_seg]

    composed = np.where(
        slice_seg[..., np.newaxis] > 0,
        slice_image[..., np.newaxis] * (1 - opacity) + slice_seg_rgb * opacity,
        slice_image[..., np.newaxis],
    )
    return composed


def _image_to_base64png(image: np.ndarray) -> str:
    _, encoded_image = cv2.imencode(
        ".png", image[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    encode = base64.b64encode(encoded_image).decode("utf-8")
    decoded = f"data:image/png;base64,{encode}"
    return decoded


def create_totalsegmentator_summary(
    image: sitk.Image,
    total_segmentation: sitk.Image,
) -> List[np.ndarray]:
    spacing = image.GetSpacing()
    image = sitk.GetArrayViewFromImage(image)
    total_segmentation = sitk.GetArrayViewFromImage(total_segmentation)

    results = []

    for axis in {1, 2}:
        for sl in np.linspace(0, np.take(image.shape, axis), num=4)[1:-1]:
            overlay = _central_overlay_image(
                image, total_segmentation, axis, TOTAL_COLOR_MAP, chosen_slice=int(sl)
            )
            overlay = cv2.resize(
                overlay[::-1],
                dsize=None,
                fx=1,
                fy=spacing[2] / spacing[1],
                interpolation=cv2.INTER_NEAREST,
            )
            results.append(overlay)

    return results


def create_tissue_summary(
    image: sitk.Image,
    tissue_segmentation: sitk.Image,
    tissue_measurements: pd.DataFrame,
) -> go.Figure:
    spacing = image.GetSpacing()
    image = sitk.GetArrayViewFromImage(image)
    tissue_segmentation = sitk.GetArrayViewFromImage(tissue_segmentation)

    fig = make_subplots(
        rows=1,
        cols=4,
        column_widths=[4, 4, 2, 2],
        specs=[[{}, {}, {}, {}]],
        horizontal_spacing=0.02,
    )

    source = _image_to_base64png(
        _central_overlay_image(image, tissue_segmentation, 2, TISSUE_COLOR_MAP)
    )
    trace_image = go.Image(source=source)

    source = _image_to_base64png(
        _central_overlay_image(image, tissue_segmentation, 1, TISSUE_COLOR_MAP)
    )
    trace_image_cor = go.Image(source=source)

    bar_traces2 = []
    for tissue in (
        Tissue.IMAT,
        Tissue.SAT,
        Tissue.VAT,
        Tissue.PAT,
        Tissue.EAT,
    ):
        bar_traces2.append(
            go.Bar(
                x=tissue_measurements[tissue.name] / tissue_measurements["TAT"] * 100.0,
                y=tissue_measurements.slice_idx - 1,
                orientation="h",
                showlegend=False,
                marker_color=TISSUE_COLOR_SCALE[tissue.value * 2][1],
                marker_line_width=0,
            )
        )

    bar_traces = []
    for tissue in (
        Tissue.BONE,
        Tissue.MUSCLE,
        Tissue.IMAT,
        Tissue.SAT,
        Tissue.VAT,
        Tissue.PAT,
        Tissue.EAT,
    ):
        name = tissue.name
        if tissue in {Tissue.MUSCLE, Tissue.BONE}:
            name = name.capitalize()
        bar_traces.append(
            go.Bar(
                x=tissue_measurements[name],
                y=tissue_measurements.slice_idx - 1,
                orientation="h",
                showlegend=True,
                marker_color=TISSUE_COLOR_SCALE[tissue.value * 2][1],
                name=name,
                marker_line_width=0,
            )
        )

    # Add all defined traces
    fig.add_trace(trace_image_cor, row=1, col=1)
    fig.add_trace(trace_image, row=1, col=2)
    for trace in bar_traces2:
        fig.add_trace(trace, row=1, col=3)
    for trace in bar_traces:
        fig.add_trace(trace, row=1, col=4)

    # Link shared axes
    fig["data"][1].update(yaxis="y2")

    fig.update_layout(
        template="plotly_white",
        font={"family": "Roboto"},
        width=1600,
        height=450 / image.shape[1] * image.shape[0] * spacing[2] / spacing[0] + 30,
        autosize=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
        xaxis1={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": "Central Coronal Slice",
        },
        yaxis1={
            "scaleanchor": "x1",
            "scaleratio": spacing[2] / spacing[0],
            "showticklabels": True,
            "showgrid": False,
            "zeroline": False,
            "ticks": "outside",
            "tickvals": list(range(len(tissue_measurements))),
            "ticktext": [
                str(x) if x % 5 == 0 else ""
                for x in range(1, tissue_measurements.shape[0] + 1)
            ],
            "range": (-0.5, tissue_measurements.shape[0] + 0.5),
            "constrain": "domain",
        },
        xaxis2={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": "Central Sagittal Slice",
        },
        yaxis2={
            "scaleanchor": "x2",
            "scaleratio": spacing[2] / spacing[1],
            "showticklabels": True,
            "showgrid": False,
            "zeroline": False,
            "ticks": "outside",
            "tickvals": list(range(len(tissue_measurements))),
            "ticktext": [
                str(x) if x % 5 == 0 else ""
                for x in range(1, tissue_measurements.shape[0] + 1)
            ],
            "range": (-0.5, tissue_measurements.shape[0] + 0.5),
            "constrain": "domain",
        },
        xaxis3={"title": "Percentage of TAT", "tickvals": [0, 20, 40, 60, 80, 100]},
        yaxis3={
            "scaleanchor": "x1",
            "scaleratio": spacing[2] / spacing[0],
            "ticks": "outside",
            "tickvals": list(range(len(tissue_measurements))),
            "ticktext": [
                str(x) if x % 5 == 0 else ""
                for x in range(1, tissue_measurements.shape[0] + 1)
            ],
            "range": (-0.5, tissue_measurements.shape[0] + 0.5),
            "constrain": "domain",
        },
        xaxis4={"title": "Volume in mL"},
        yaxis4={
            "scaleanchor": "x1",
            "scaleratio": spacing[2] / spacing[0],
            "ticks": "outside",
            "tickvals": list(range(len(tissue_measurements))),
            "ticktext": [
                str(x) if x % 5 == 0 else ""
                for x in range(1, tissue_measurements.shape[0] + 1)
            ],
            "range": (-0.5, tissue_measurements.shape[0] + 0.5),
            "constrain": "domain",
        },
        barmode="stack",
    )

    return fig
