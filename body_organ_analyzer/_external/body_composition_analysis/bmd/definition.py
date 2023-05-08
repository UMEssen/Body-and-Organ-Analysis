from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import dataclass_json

CHOSEN_BMD_VERTEBRAE = ["L1", "L2", "L3", "L4", "L5", "T10", "T11", "T12"]


@dataclass_json
@dataclass
class RoiMeasurement:
    density_mean: float
    density_std: float
    density_median: float
    density_min: int
    density_max: int
    num_dense_roi_voxels: int
    num_roi_voxels: int


@dataclass_json
@dataclass
class Point:
    x: int = 0
    y: int = 0


@dataclass_json
@dataclass
class RayCastSearchResults:
    selected_angle: float = 0.0
    start_point: Point = field(default_factory=Point)
    end_points: List[Point] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)
    smoothed_voxel_counts: List[float] = field(default_factory=list)
    voxel_counts: List[int] = field(default_factory=list)


@dataclass_json
@dataclass
class RoiSelectionResults:
    roi_center: Optional[Point] = None
    first_bone_pos: Optional[Point] = None
    last_bone_pos: Optional[Point] = None


@dataclass_json
@dataclass
class AlgorithmInfo:
    selected_slice_idx: int = 0
    min_slice_idx: int = 0
    max_slice_idx: int = 0
    ray_cast_search: Optional[RayCastSearchResults] = None
    roi_selection_results: Optional[RoiSelectionResults] = None


@dataclass_json
@dataclass
class BMD:
    measurements: Dict[str, RoiMeasurement] = field(default_factory=dict)
    algorithm: Dict[str, AlgorithmInfo] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
