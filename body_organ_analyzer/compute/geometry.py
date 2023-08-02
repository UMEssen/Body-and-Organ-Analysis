import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import cv2 as cv
import numpy as np
from scipy import spatial


@dataclass
class Point:
    x: float
    y: float

    def to_list(self) -> List[float]:
        return [self.x, self.y]


def find_minor_point(
    contours: Sequence[Any],
    mid_point: Point,
    rotated_point: Point,
    length: int,
    target_size: Tuple[int, ...],
) -> Point:
    # Rotate the vector 90 degrees by swapping x and y, and inverting one of them
    point = Point(
        int(mid_point.x + rotated_point.x * length),
        int(mid_point.y + rotated_point.y * length),
    )
    # Draw the contours on an array
    contour_array = cv.drawContours(
        np.zeros(target_size), contours, contourIdx=-1, color=1, thickness=2
    )
    # Draw the lines on an array
    p_array = cv.line(
        np.zeros(target_size),
        [point.x, point.y],
        [mid_point.x, mid_point.y],
        1,
        2,
    )
    p_options = np.logical_and(contour_array, p_array).nonzero()
    # Swap points for opencv weirdness
    return Point(p_options[1][0], p_options[0][0])


def find_axes(middle_slice: np.ndarray) -> Tuple[Point, Point, Point, Point]:
    # Get all points where the shape is
    points = np.flip(np.transpose(np.where(middle_slice)))
    hull_points = points[spatial.ConvexHull(points).vertices]
    hdist = spatial.distance.cdist(hull_points, hull_points, metric="euclidean")
    # Get the farthest apart points
    p1_idx, p2_idx = np.unravel_index(hdist.argmax(), hdist.shape)
    major_p1, major_p2 = Point(*hull_points[p1_idx]), Point(*hull_points[p2_idx])
    # We want the perpendicular at the middle point to get the minor axis
    mid_point = Point((major_p1.x + major_p2.x) // 2, (major_p1.y + major_p2.y) // 2)
    # The length that we want the line to have, just to be sure make it extra long
    length = sum(middle_slice.shape)
    # Get the direction vector going from A to B.
    norm_vector = Point(major_p1.x - major_p2.x, major_p1.y - major_p2.y)
    # Normalize the vector
    fac = math.sqrt(norm_vector.x * norm_vector.x + norm_vector.y * norm_vector.y)
    norm_vector.x /= fac
    norm_vector.y /= fac

    contours, _ = cv.findContours(
        middle_slice.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    minor_p1 = find_minor_point(
        contours=contours,
        mid_point=mid_point,
        rotated_point=Point(-norm_vector.y, norm_vector.x),  # clockwise
        length=length,
        target_size=middle_slice.shape,
    )
    minor_p2 = find_minor_point(
        contours=contours,
        mid_point=mid_point,
        rotated_point=Point(norm_vector.y, -norm_vector.x),  # anticlockwise
        length=length,
        target_size=middle_slice.shape,
    )
    return major_p1, major_p2, minor_p1, minor_p2
