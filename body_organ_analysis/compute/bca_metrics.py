import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from body_organ_analysis.compute.util import convert_name

BODY_REGIONS = [
    "Whole Scan",
    "Abdominal Cavity",
    "Thoracic Cavity",
    "Ventral Cavity",
    "Mediastinum",
    "Pericardium",
    "L5",
    "L4",
    "L3",
    "L2",
    "L1",
    "T12",
    "T11",
    "T10",
    "T9",
    "T8",
    "T7",
    "T6",
    "T5",
    "T4",
    "T3",
    "T2",
    "T1",
    "C7",
    "C6",
    "C5",
    "C4",
    "C3",
    "C2",
    "C1",
]


def change_aggregated_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def compute_bca_metrics(
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    measurements_path = output_path / "bca-measurements.json"
    with measurements_path.open("r") as of:
        json_measurements = json.load(of)
    # Take one example measurement
    example_vals = json_measurements["aggregated"]["whole_scan"]["measurements"]
    index_rows = list(example_vals["bone"].keys())
    index_cols = list(example_vals.keys())
    # Change row names to make them more clear
    rename_index = {
        ind: ind.split("_")[0].capitalize() + ("_mL" if "hu" not in ind else "_HU")
        for ind in index_rows
    }
    # Change column names for better formatting
    rename_cols = {
        col: (col.upper() if col not in ["bone", "muscle"] else col.capitalize())
        for col in index_cols
    }
    rename_cols["index"] = "AggregationType"
    aggregation_df = pd.DataFrame(columns=["BodyPart", "Present", "AggregationType"])
    dfs = [aggregation_df]
    for name in BODY_REGIONS:
        aggregated_name = change_aggregated_name(name)
        if aggregated_name not in json_measurements["aggregated"]:
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "BodyPart": f"{convert_name(aggregated_name)}",
                            "Present": False,
                        },
                        {
                            "BodyPart": f"{convert_name(aggregated_name)}_NoExtremities",
                            "Present": False,
                        },
                    ]
                )
            )
            continue
        # Convert to DataFrame and rename rows and columns
        for measurement in ["measurements", "measurements_no_extremities"]:
            current_df = (
                pd.DataFrame.from_dict(
                    json_measurements["aggregated"][aggregated_name][measurement]
                )
                .rename(index=rename_index)
                .reset_index()
                .rename(columns=rename_cols)
            )
            current_df["Present"] = True
            measurement_part = convert_name(measurement.replace("measurements", ""))
            current_df["BodyPart"] = convert_name(aggregated_name) + (
                "_" + measurement_part if len(measurement_part) > 0 else ""
            )
            dfs.append(current_df)
    aggregation_df = pd.concat(dfs, copy=False)
    slices_df = pd.DataFrame(json_measurements["slices"])
    slices_no_limbs_df = pd.DataFrame(json_measurements["slices_no_extremities"])
    bmd_df = pd.DataFrame(json_measurements["bmd"]).T.reset_index()
    bmd_df.rename(
        {"index": "vertebrae"},
        axis=1,
        inplace=True,
    )
    rename_cols["index"] = "SliceNumber"
    for df in [slices_df, slices_no_limbs_df]:
        df.index = df.index + 1
        df.reset_index(inplace=True)
        df.rename(columns=rename_cols, inplace=True)
    return (
        aggregation_df,
        slices_df,
        slices_no_limbs_df,
        bmd_df if len(bmd_df) > 0 else None,
    )
