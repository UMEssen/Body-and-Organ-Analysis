import base64
import enum
import logging
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import jinja2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage.measure
import weasyprint
from body_composition_analysis import debug_mode_enabled, get_debug_dir
from body_composition_analysis.body_parts.definition import BodyParts
from body_composition_analysis.report.plots.aggregation import create_aggregation_image
from body_composition_analysis.report.plots.check import create_equidistant_overview
from body_composition_analysis.report.plots.heatmaps import create_tissue_heatmaps
from body_composition_analysis.report.plots.overview import (
    create_tissue_summary,
    create_totalsegmentator_summary,
)
from body_composition_analysis.tissue.definition import BodyRegion, Tissue
from body_composition_analysis.report.plots.colors import (
    BODY_REGION_COLOR_MAP,
    TISSUE_COLOR_MAP,
    TOTAL_COLOR_MAP,
)
from body_organ_analysis._version import __githash__, __version__

logger = logging.getLogger(__name__)


def _pretty_volume(value: float) -> str:
    if value >= 1000:
        return f"{value/1000:.3f} L"
    return f"{value:.2f} mL"


def _to_embedded_image(img: np.ndarray, debug_name: Optional[str] = None) -> str:
    _, img_bytes = cv2.imencode(".png", img[..., ::-1])
    if debug_mode_enabled() and debug_name is not None:
        with (get_debug_dir() / f"report_{debug_name}.png").open("wb") as ofile:
            ofile.write(img_bytes)
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")


class AggregatableBodyPart(enum.IntFlag):
    NONE = 0
    ABDOMEN = 1
    THORAX = 2
    NECK = 4

    @staticmethod
    def from_body_regions(
        body_regions: sitk.Image,
        min_abdomen_length: float = 200,
        min_neck_length: float = 100,
        min_thorax_length: float = 200,
    ) -> "AggregatableBodyPart":
        region_data = sitk.GetArrayViewFromImage(body_regions)
        result = AggregatableBodyPart.NONE
        slice_thickness = body_regions.GetSpacing()[2]

        # Detect abdomen
        abdomen_mask = np.equal(region_data, BodyRegion.ABDOMINAL_CAVITY)
        abdomen_slices = np.where(abdomen_mask.any(axis=(1, 2)))[0]
        if abdomen_slices.size > 0:
            num_abdomen_slices = abdomen_slices.max() - abdomen_slices.min() + 1
        else:
            num_abdomen_slices = 0
        if num_abdomen_slices * slice_thickness >= min_abdomen_length:
            result |= AggregatableBodyPart.ABDOMEN
            logger.info(
                f"Found abdomen with length of {num_abdomen_slices * slice_thickness} mm"
            )

        # Detect neck
        mediastinum_mask = np.equal(region_data, BodyRegion.MEDIASTINUM)
        mediastinum_slices = np.where(mediastinum_mask.any(axis=(1, 2)))[0]
        if mediastinum_slices.size > 0:
            num_slices_above_mediastinum = (
                body_regions.GetDepth() - mediastinum_slices.max()
            )
        else:
            num_slices_above_mediastinum = 0
        if num_slices_above_mediastinum * slice_thickness >= min_neck_length:
            result |= AggregatableBodyPart.NECK
            logger.info(
                f"Found neck with length of {num_slices_above_mediastinum * slice_thickness} mm"
            )

        # Detect thorax
        thorax_mask = np.isin(
            region_data,
            [
                BodyRegion.THORACIC_CAVITY,
                BodyRegion.MEDIASTINUM,
                BodyRegion.PERICARDIUM,
            ],
        )
        thorax_slices = np.where(thorax_mask.any(axis=(1, 2)))[0]
        has_abdomen_intersection = np.logical_and(
            abdomen_mask.any(axis=(1, 2)), thorax_mask.any(axis=(1, 2))
        ).any()
        if thorax_slices.size > 0:
            num_thorax_slices = thorax_slices.max() - thorax_slices.min() + 1
        else:
            num_thorax_slices = 0
        if (
            has_abdomen_intersection
            and num_thorax_slices * slice_thickness >= min_thorax_length
        ):
            result |= AggregatableBodyPart.THORAX
            logger.info(
                f"Found thorax with length of {num_thorax_slices * slice_thickness} mm"
            )

        return result


class Builder:
    def __init__(
        self,
        image: sitk.Image,
        body_parts: sitk.Image,
        body_regions: sitk.Image,
        tissues: sitk.Image,
    ):
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                [
                    str(pathlib.Path(__file__).parent / "template"),
                    str(
                        pathlib.Path(__file__).parent / "template" / "base" / "template"
                    ),
                ]
            )
        )
        self._image = image
        self._body_parts = body_parts
        self._body_regions = body_regions
        self._tissues = tissues
        self.examined_body_part = AggregatableBodyPart(0)

    def _build_document(self, template_name: str, **kwargs: Any) -> weasyprint.HTML:
        template = self._env.get_template(template_name)
        rendered_content = template.render(
            app_version=f"{__version__} ({__githash__})",
            contact_email="ship-ai@uk-essen.de",
            **kwargs,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            html_file = pathlib.Path(tempdir) / "index.html"
            with html_file.open("w") as ofile:
                ofile.write(rendered_content)

            return weasyprint.HTML(
                filename=html_file,
                base_url=str(
                    pathlib.Path(__file__).parent / "template" / "base" / "template"
                ),
            ).render()

    def create_pdf(self, template_name: str, **kwargs: Any) -> bytes:
        document = self._build_document(template_name, **kwargs)
        if debug_mode_enabled():
            for idx, page in enumerate(document.pages):
                png_bytes, _, _ = document.copy([page]).write_png(resolution=300)
                with (get_debug_dir() / f"report_page_{idx:02d}.png").open(
                    "wb"
                ) as ofile:
                    ofile.write(png_bytes)
        pdf_data: bytes = document.write_pdf()
        return pdf_data

    def create_png(self, template_name: str, **kwargs: Any) -> List[bytes]:
        document = self._build_document(template_name, **kwargs)
        return [document.copy([page]).write_png()[0] for page in document.pages]

    def generate_aggregated_measurements(
        self,
        slice_measurements: pd.DataFrame,
        slice_measurements_no_limbs: pd.DataFrame,
        vertebrae: Optional[Dict[str, Tuple[int, int]]],
    ) -> pd.DataFrame:
        _, _, slice_thickness = self._image.GetSpacing()

        # Find relevant regions for aggregation
        groups = [("Whole Scan", 0, self._image.GetDepth())]
        region_data = sitk.GetArrayViewFromImage(self._body_regions)
        if AggregatableBodyPart.ABDOMEN in self.examined_body_part:
            mask = np.equal(region_data, BodyRegion.ABDOMINAL_CAVITY)
            slices = np.where(mask.any(axis=(1, 2)))[0]
            groups.append(("Abdominal Cavity", slices.min(), slices.max() + 1))

        if AggregatableBodyPart.THORAX in self.examined_body_part:
            mask = np.isin(
                region_data,
                [
                    BodyRegion.THORACIC_CAVITY,
                    BodyRegion.MEDIASTINUM,
                    BodyRegion.PERICARDIUM,
                ],
            )
            slices = np.where(mask.any(axis=(1, 2)))[0]
            groups.append(("Thoracic Cavity", slices.min(), slices.max() + 1))

            mask = np.equal(region_data, BodyRegion.MEDIASTINUM)
            slices = np.where(mask.any(axis=(1, 2)))[0]
            groups.append(("Mediastinum", slices.min(), slices.max() + 1))

            mask = np.equal(region_data, BodyRegion.PERICARDIUM)
            slices = np.where(mask.any(axis=(1, 2)))[0]
            groups.append(("Pericardium", slices.min(), slices.max() + 1))

        if (
            AggregatableBodyPart.ABDOMEN in self.examined_body_part
            and AggregatableBodyPart.THORAX in self.examined_body_part
        ):
            assert groups[1][0] == "Abdominal Cavity"
            assert groups[2][0] == "Thoracic Cavity"
            groups.insert(1, ("Ventral Cavity", groups[1][1], groups[2][2]))

        # If provided add the vertebrae to the aggregation groups
        if vertebrae:
            for name, group in vertebrae.items():
                groups.append((name, group[0], group[1]))

        logger.info("Aggregation groups:")
        for name, min_idx, max_idx in groups:
            logger.info(f' - "{name}" from slice index {min_idx} to {max_idx}')

        # Aggregate measurements
        result = []
        image_data = sitk.GetArrayViewFromImage(self._image)
        body_parts_data = sitk.GetArrayViewFromImage(self._body_parts)
        tissue_data = sitk.GetArrayViewFromImage(self._tissues)
        no_extremity_mask = body_parts_data == BodyParts.TRUNC
        for name, min_z, max_z in groups:
            image_region = image_data[min_z:max_z]
            tissue_region = tissue_data[min_z:max_z]
            measurements = self._descriptive_statistics_from_measurements(
                slicewise_measurements=slice_measurements[
                    (slice_measurements.slice_idx >= min_z)
                    & (slice_measurements.slice_idx < max_z)
                ],
                image_data=image_region,
                tissue_data=tissue_region,
            )
            measurements_no_limbs = self._descriptive_statistics_from_measurements(
                slicewise_measurements=slice_measurements_no_limbs[
                    (slice_measurements_no_limbs.slice_idx >= min_z)
                    & (slice_measurements_no_limbs.slice_idx < max_z)
                ],
                image_data=image_region,
                tissue_data=np.where(no_extremity_mask[min_z:max_z], tissue_region, 0),
            )

            overview_image = create_aggregation_image(
                image=self._image,
                body_regions=self._body_regions,
                tissues=self._tissues,
                group=(min_z, max_z),
            )
            overview_image = _to_embedded_image(
                overview_image, f"aggregation_{name.lower().replace(' ', '-')}"
            )

            result.append(
                (
                    name,
                    (min_z, max_z),
                    overview_image,
                    measurements,
                    measurements_no_limbs,
                )
            )
        return result

    def _descriptive_statistics_from_measurements(
        self,
        slicewise_measurements: pd.DataFrame,
        image_data: np.ndarray,
        tissue_data: np.ndarray,
    ) -> pd.DataFrame:
        slicewise_measurements = slicewise_measurements.drop("slice_idx", axis=1)
        measurements = slicewise_measurements.describe()
        measurements.drop("count", inplace=True)
        measurements.index = [
            "Mean",
            "StdDev",
            "Minimum",
            "25%",
            "Median",
            "75%",
            "Maximum",
        ]
        measurements.loc["Total"] = slicewise_measurements.sum()
        # Compute the Mean of the Housfield Units per tissue
        # Get the portion of the image data beloging to the current region
        for tissue in Tissue:
            # Create a mask for the current tissue to select the original pixels
            tissue_mask = np.equal(tissue_data, tissue)
            tissue_name = (
                tissue.name.capitalize()
                if tissue in {Tissue.BONE, Tissue.MUSCLE}
                else tissue.name
            )
            # Add a new row with the mean
            masked_data = image_data[tissue_mask]
            measurements.loc["MeanHU", tissue_name] = (
                np.mean(masked_data) if masked_data.size else None
            )
        # For TAT the mask is the union of all adipose tissues
        tissue_mask = np.isin(
            tissue_data, [Tissue.IMAT, Tissue.SAT, Tissue.VAT, Tissue.PAT, Tissue.EAT]
        )
        # Add a new row with the mean
        masked_data = image_data[tissue_mask]
        measurements.loc["MeanHU", "TAT"] = (
            np.mean(masked_data) if masked_data.size else None
        )

        return measurements.replace({np.nan: None})

    def generate_secondary_findings(self) -> List[str]:
        result = []
        region_data = sitk.GetArrayViewFromImage(self._body_regions)
        mid_index = region_data.shape[1] // 2
        ml_per_voxel = np.prod(self._body_regions.GetSpacing()) / 1000.0
        if AggregatableBodyPart.ABDOMEN in self.examined_body_part:
            abdominal_cavity_vol = (
                np.equal(region_data, BodyRegion.ABDOMINAL_CAVITY.value).sum()
                * ml_per_voxel
            )
            result.append(
                f"Total volume of the abdominal cavity is {_pretty_volume(abdominal_cavity_vol)}"
            )

        if AggregatableBodyPart.THORAX in self.examined_body_part:
            # Compute total volume of the thoracic cavity, which consists of three individual
            # labels from the multi class segmentation
            thoracic_cavity_vol = (
                np.isin(
                    region_data,
                    [
                        BodyRegion.THORACIC_CAVITY.value,
                        BodyRegion.MEDIASTINUM.value,
                        BodyRegion.PERICARDIUM.value,
                    ],
                ).sum()
                * ml_per_voxel
            )
            result.append(
                f"Volume of thoracic cavity is {_pretty_volume(thoracic_cavity_vol)}"
            )

            # Compute total volume of the mediastinum, which consists of two individual labels
            thoracic_cavity_vol = (
                np.isin(
                    region_data,
                    [BodyRegion.MEDIASTINUM.value, BodyRegion.PERICARDIUM.value],
                ).sum()
                * ml_per_voxel
            )
            result.append(
                f"Volume of mediastinum is {_pretty_volume(thoracic_cavity_vol)}"
            )

            # Compute total volume of the pericardium
            pericardium_vol = (
                np.equal(region_data, BodyRegion.PERICARDIUM).sum() * ml_per_voxel
            )
            result.append(
                f"Volume enclosed by the pericardial sack is {_pretty_volume(pericardium_vol)}"
            )

            # Check for presence of breast implants
            breast_implant_mask = np.equal(region_data, BodyRegion.BREAST_IMPLANT)
            if breast_implant_mask.any():
                label = skimage.measure.label(breast_implant_mask)
                region_props = skimage.measure.regionprops(label)
                region_props = [x for x in region_props if x.area * ml_per_voxel > 10]
                region_props = sorted(region_props, key=lambda x: int(x.centroid[2]))
                if region_props:
                    measurements = []
                    for prop in region_props:
                        volume = prop.area * ml_per_voxel
                        pos_x = prop.centroid[2]
                        pos_x_title = "right" if pos_x < mid_index else "left"
                        measurements.append((pos_x_title, volume))

                    if len(measurements) == 1:
                        result.append(
                            f"Patient has a single breast implant on the {measurements[0][0]} side "
                            f"with volume of {_pretty_volume(measurements[0][1])}"
                        )
                    elif len(measurements) == 2:
                        result.append(
                            f"Patient has two breast implants with volume of "
                            f"{_pretty_volume(measurements[0][1])} ({measurements[0][0]}) and "
                            f"{_pretty_volume(measurements[1][1])} ({measurements[1][0]})"
                        )
                    else:
                        logger.error("More than two breast implant segments found")

        return result

    def prepare(
        self,
        vertebrae: Optional[Dict[str, Tuple[int, int]]] = None,
        bmd: Optional = None,
        total: Optional[sitk.Image] = None,
        total_measurements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ml_per_voxel = np.prod(self._image.GetSpacing()) / 1000.0
        tissue_data = sitk.GetArrayViewFromImage(self._tissues)

        # Create slice-wise measurements of the whole body
        data = {
            tissue.name.capitalize()
            if tissue in {Tissue.BONE, Tissue.MUSCLE}
            else tissue.name: (tissue_data == tissue.value).sum(axis=(1, 2))
            * ml_per_voxel
            for tissue in Tissue
        }
        df = pd.DataFrame(data)
        df["TAT"] = df.SAT + df.VAT + df.IMAT + df.PAT + df.EAT
        df["slice_idx"] = range(len(df))
        df = df[
            ["slice_idx", "Bone", "Muscle", "TAT", "IMAT", "SAT", "VAT", "PAT", "EAT"]
        ]

        # Create slice-wise measurements without extremeties
        part_data = sitk.GetArrayViewFromImage(self._body_parts)
        no_extremity_mask = part_data == BodyParts.TRUNC
        data = {
            tissue.name.capitalize()
            if tissue in {Tissue.BONE, Tissue.MUSCLE}
            else tissue.name: np.logical_and(
                no_extremity_mask, tissue_data == tissue.value
            ).sum(axis=(1, 2))
            * ml_per_voxel
            for tissue in Tissue
        }
        df_no_limbs = pd.DataFrame(data)
        df_no_limbs["TAT"] = (
            df_no_limbs.SAT
            + df_no_limbs.VAT
            + df_no_limbs.IMAT
            + df_no_limbs.PAT
            + df_no_limbs.EAT
        )
        df_no_limbs["slice_idx"] = range(len(df_no_limbs))
        df_no_limbs = df_no_limbs[
            ["slice_idx", "Bone", "Muscle", "TAT", "IMAT", "SAT", "VAT", "PAT", "EAT"]
        ]

        image_summary = create_tissue_summary(self._image, self._tissues, df).to_image(
            format="svg"
        )
        if debug_mode_enabled():
            with (get_debug_dir() / "report_tissue_summary.svg").open("wb") as ofile:
                ofile.write(image_summary)
        image_summary = "data:image/svg+xml;base64," + base64.b64encode(
            image_summary
        ).decode("utf-8")

        total_summary = (
            [
                _to_embedded_image(img, f"total_overview_{i}")
                for i, img in enumerate(
                    create_totalsegmentator_summary(self._image, total)
                )
            ]
            if total is not None
            else None
        )
        heatmaps = [
            (
                x[0],
                _to_embedded_image(x[1], f"tissue_cor_{x[0].name.lower()}"),
                _to_embedded_image(x[2], f"tissue_sag_{x[0].name.lower()}"),
            )
            for x in create_tissue_heatmaps(self._body_regions, self._tissues)
        ]

        name_mapping = {
            "First": "q0",
            "25%": "q1",
            "Central": "q2",
            "75%": "q3",
            "Last": "q4",
        }
        equidistance_slices_segs = [
            (self._body_regions, BODY_REGION_COLOR_MAP),
            (self._tissues, TISSUE_COLOR_MAP),
        ]
        seg_names = ["body-regions", "tissues"]
        if total is not None:
            equidistance_slices_segs.append((total, TOTAL_COLOR_MAP))
            seg_names.append("total")
        equidistant_slice_check = [
            [x[0]]
            + [
                _to_embedded_image(
                    x[seg_slice_idx + 1],
                    f"equidistant_check_{seg_names[seg_slice_idx]}_"
                    f"{name_mapping.get(x[0], x[0].lower())}",
                )
                for seg_slice_idx in range(len(seg_names))
            ]
            for x in create_equidistant_overview(
                self._image,
                equidistance_slices_segs,
            )
        ]

        aggregations = self.generate_aggregated_measurements(df, df_no_limbs, vertebrae)

        bmd_measurements = None

        if (
            total_measurements is None
            or "segmentations" not in total_measurements
            or "total" not in total_measurements["segmentations"]
        ):
            df_total = None
        else:
            df_total = pd.DataFrame(total_measurements["segmentations"]["total"]).T
            df_total = df_total.loc[df_total["present"]]
            df_total.drop(columns="present", inplace=True)
            df_total.rename(
                index={v: v.replace("_", " ").title() for v in df_total.index},
                columns={
                    "25th_percentile_hu": "twentyfive_percentile_hu",
                    "75th_percentile_hu": "seventyfive_percentile_hu",
                },
                inplace=True,
            )

        return dict(
            aggregated_measurements=aggregations,
            equidistant_slice_check=equidistant_slice_check,
            image_summary=image_summary,
            other_findings=self.generate_secondary_findings(),
            slicewise_measurements=df,
            slicewise_measurements_no_limbs=df_no_limbs,
            measurements_total=df_total,
            tissue_heatmaps=heatmaps,
            bmd_measurements=bmd_measurements,
            summary_totalsegmentator=total_summary,
        )

    def create_json(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "slices": (
                kwargs["slicewise_measurements"]
                .rename(
                    columns={
                        x: x.lower() for x in kwargs["slicewise_measurements"].columns
                    }
                )
                .drop("slice_idx", axis=1)
                .astype(float)
                .to_dict("records")
            ),
            "slices_no_extremities": (
                kwargs["slicewise_measurements_no_limbs"]
                .rename(
                    columns={
                        x: x.lower()
                        for x in kwargs["slicewise_measurements_no_limbs"].columns
                    }
                )
                .drop("slice_idx", axis=1)
                .astype(float)
                .to_dict("records")
            ),
            "aggregated": {
                name.lower()
                .replace(" ", "_")
                .replace("-", "_"): {
                    "num_slices": int(max_z - min_z),
                    "min_slice_idx": int(min_z),
                    "max_slice_idx": int(max_z),
                    "measurements": (
                        measurements.rename(
                            index={
                                "Mean": "mean",
                                "StdDev": "std",
                                "Minimum": "min",
                                "25%": "q1",
                                "Median": "q2",
                                "75%": "q3",
                                "Maximum": "max",
                                "Total": "sum",
                                "MeanHU": "mean_hu",
                            },
                            columns={x: x.lower() for x in measurements.columns},
                        ).to_dict()
                    ),
                    "measurements_no_extremities": (
                        measurements_no_limbs.rename(
                            index={
                                "Mean": "mean",
                                "StdDev": "std",
                                "Minimum": "min",
                                "25%": "q1",
                                "Median": "q2",
                                "75%": "q3",
                                "Maximum": "max",
                                "Total": "sum",
                                "MeanHU": "mean_hu",
                            },
                            columns={
                                x: x.lower() for x in measurements_no_limbs.columns
                            },
                        ).to_dict()
                    ),
                }
                for (
                    name,
                    (min_z, max_z),
                    _,
                    measurements,
                    measurements_no_limbs,
                ) in kwargs["aggregated_measurements"]
            },
            "body_parts": {
                "abdomen": AggregatableBodyPart.ABDOMEN in self.examined_body_part,
                "neck": AggregatableBodyPart.NECK in self.examined_body_part,
                "thorax": AggregatableBodyPart.THORAX in self.examined_body_part,
            },
            "bmd": {
                x[0]: x[2].to_dict() if x[2] is not None else {"error": x[3]}
                for x in kwargs["bmd_measurements"] or []
            },
        }
