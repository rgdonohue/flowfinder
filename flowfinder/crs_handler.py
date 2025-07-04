"""
FLOWFINDER CRS Handling
======================

Robust coordinate reference system handling for FLOWFINDER.
Provides CRS validation, transformation, and error detection.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Union
import rasterio
from rasterio.crs import CRS
import pyproj
from pyproj import Transformer, CRS as ProjCRS
from shapely.geometry import Point, Polygon
from shapely.ops import transform

from .exceptions import CRSError, ValidationError


class CRSHandler:
    """
    Handle coordinate reference system operations for FLOWFINDER.

    This class provides robust CRS validation, transformation, and error handling
    to prevent silent coordinate corruption that could invalidate spatial analysis.

    Attributes:
        input_crs (CRS): Input coordinate reference system
        output_crs (CRS): Output coordinate reference system
        transformer (Transformer): PyProj transformer for coordinate conversion
        logger (logging.Logger): Logger instance
    """

    def __init__(
        self,
        input_crs: Union[str, CRS, int],
        output_crs: Union[str, CRS, int] = "EPSG:4326",
    ):
        """
        Initialize CRS handler.

        Args:
            input_crs: Input coordinate reference system
            output_crs: Output coordinate reference system (default: WGS84)

        Raises:
            CRSError: If CRS specifications are invalid
        """
        self.logger = logging.getLogger(__name__)

        try:
            # Convert to rasterio CRS objects
            self.input_crs = self._parse_crs(input_crs)
            self.output_crs = self._parse_crs(output_crs)

            # Create PyProj transformer for high-precision transformations
            self.transformer = self._create_transformer()

            # Validate CRS compatibility
            self._validate_crs_compatibility()

            self.logger.info(
                f"CRS handler initialized: {self.input_crs} → {self.output_crs}"
            )

        except Exception as e:
            raise CRSError(f"Failed to initialize CRS handler: {e}")

    def _parse_crs(self, crs_spec: Union[str, CRS, int]) -> CRS:
        """
        Parse CRS specification into rasterio CRS object.

        Args:
            crs_spec: CRS specification (EPSG code, WKT string, etc.)

        Returns:
            Rasterio CRS object

        Raises:
            CRSError: If CRS specification is invalid
        """
        try:
            if isinstance(crs_spec, CRS):
                return crs_spec
            elif isinstance(crs_spec, int):
                return CRS.from_epsg(crs_spec)
            elif isinstance(crs_spec, str):
                if crs_spec.startswith("EPSG:"):
                    epsg_code = int(crs_spec.split(":")[1])
                    return CRS.from_epsg(epsg_code)
                else:
                    return CRS.from_string(crs_spec)
            else:
                raise ValueError(
                    f"Unsupported CRS specification type: {type(crs_spec)}"
                )

        except Exception as e:
            raise CRSError(f"Invalid CRS specification '{crs_spec}': {e}")

    def _create_transformer(self) -> Transformer:
        """
        Create PyProj transformer for coordinate transformations.

        Returns:
            PyProj Transformer object

        Raises:
            CRSError: If transformer creation fails
        """
        try:
            # Use PyProj for high-precision transformations
            input_proj_crs = ProjCRS.from_string(self.input_crs.to_string())
            output_proj_crs = ProjCRS.from_string(self.output_crs.to_string())

            transformer = Transformer.from_crs(
                input_proj_crs,
                output_proj_crs,
                always_xy=True,  # Always return x, y order
            )

            return transformer

        except Exception as e:
            raise CRSError(f"Failed to create coordinate transformer: {e}")

    def _validate_crs_compatibility(self) -> None:
        """
        Validate CRS compatibility and detect potential issues.

        Raises:
            CRSError: If CRS combination is problematic
            ValidationError: If transformation accuracy is insufficient
        """
        # Check for known problematic transformations
        # output_proj = self.output_crs.to_string()  # Not used

        # Detect datum shift requirements
        if self._requires_datum_shift():
            self.logger.info(
                "Datum shift transformation detected - validating accuracy"
            )
            self._validate_datum_shift_accuracy()

        # Check for high-latitude precision issues
        if self._is_high_latitude_projection():
            self.logger.warning(
                "High-latitude projection detected - monitoring precision"
            )

        # Validate transformation domain
        self._validate_transformation_domain()

    def _requires_datum_shift(self) -> bool:
        """
        Check if transformation requires datum shift.

        Returns:
            True if datum shift is required
        """
        input_datum = self.input_crs.to_dict().get("datum", "")
        output_datum = self.output_crs.to_dict().get("datum", "")

        # Common datum shifts that require special attention
        datum_combinations = [
            ("NAD27", "WGS84"),
            ("NAD83", "WGS84"),
            ("OSGB36", "WGS84"),
        ]

        for input_d, output_d in datum_combinations:
            if input_d in str(input_datum) and output_d in str(output_datum):
                return True

        return input_datum != output_datum and input_datum and output_datum

    def _validate_datum_shift_accuracy(self) -> None:
        """
        Validate datum shift transformation accuracy.

        Raises:
            ValidationError: If datum shift is outside expected range
        """
        try:
            # Test transformation with known reference points
            test_points = [
                (-122.4194, 37.7749),  # San Francisco
                (-74.0060, 40.7128),  # New York
                (-87.6298, 41.8781),  # Chicago
            ]

            for lon, lat in test_points:
                # Transform forward and back
                x_trans, y_trans = self.transformer.transform(lon, lat)
                lon_back, lat_back = self.transformer.transform(
                    x_trans, y_trans, direction="INVERSE"
                )

                # Calculate round-trip error
                error_meters = self._calculate_distance_meters(
                    lon, lat, lon_back, lat_back
                )

                # Datum shifts should be consistent and within reasonable bounds
                if error_meters > 1000:  # 1km tolerance for datum shifts
                    raise ValidationError(
                        f"Datum shift transformation error too large: {error_meters:.1f}m"
                    )

            self.logger.info("Datum shift validation passed")

        except Exception as e:
            raise ValidationError(f"Datum shift validation failed: {e}")

    def _is_high_latitude_projection(self) -> bool:
        """
        Check if projection involves high latitudes.

        Returns:
            True if high-latitude projection detected
        """
        # Check for polar projections or high-latitude UTM zones
        proj_string = self.input_crs.to_string().lower()

        high_lat_indicators = [
            "polar",
            "arctic",
            "antarctic",
            "lambert_azimuthal_equal_area",
            "stereographic",
        ]

        return any(indicator in proj_string for indicator in high_lat_indicators)

    def _validate_transformation_domain(self) -> None:
        """
        Validate that transformations are within valid domain.

        Raises:
            CRSError: If transformation domain is invalid
        """
        try:
            # Get CRS bounds
            input_bounds = self.input_crs.area_of_use
            output_bounds = self.output_crs.area_of_use

            if input_bounds and output_bounds:
                # Check for reasonable overlap
                if not self._bounds_overlap(input_bounds, output_bounds):
                    self.logger.warning(
                        "CRS transformation domains do not overlap - "
                        "results may be inaccurate"
                    )

        except Exception as e:
            self.logger.warning(f"Could not validate transformation domain: {e}")

    def _bounds_overlap(self, bounds1, bounds2) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
            bounds1, bounds2: Bounding box objects

        Returns:
            True if bounds overlap
        """
        try:
            return not (
                bounds1.east < bounds2.west
                or bounds1.west > bounds2.east
                or bounds1.north < bounds2.south
                or bounds1.south > bounds2.north
            )
        except Exception:
            return True  # Assume overlap if cannot determine

    def transform_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform coordinates from input to output CRS.

        Args:
            x, y: Input coordinates

        Returns:
            Transformed coordinates (x, y)

        Raises:
            CRSError: If transformation fails
            ValidationError: If coordinates are invalid
        """
        try:
            # Validate input coordinates
            self._validate_coordinates(x, y)

            # Perform transformation
            x_trans, y_trans = self.transformer.transform(x, y)

            # Validate output coordinates
            self._validate_output_coordinates(x_trans, y_trans)

            return x_trans, y_trans

        except Exception as e:
            raise CRSError(f"Coordinate transformation failed: {e}")

    def transform_geometry(
        self, geometry: Union[Point, Polygon]
    ) -> Union[Point, Polygon]:
        """
        Transform geometry from input to output CRS.

        Args:
            geometry: Shapely geometry object

        Returns:
            Transformed geometry

        Raises:
            CRSError: If transformation fails
        """
        try:
            # Use shapely's transform with PyProj transformer
            transformed = transform(self.transformer.transform, geometry)

            # Validate transformed geometry
            if not transformed.is_valid:
                self.logger.warning(
                    "Transformed geometry is invalid - attempting repair"
                )
                transformed = transformed.buffer(0)  # Attempt to fix invalid geometry

            return transformed

        except Exception as e:
            raise CRSError(f"Geometry transformation failed: {e}")

    def _validate_coordinates(self, x: float, y: float) -> None:
        """
        Validate input coordinates.

        Args:
            x, y: Coordinates to validate

        Raises:
            ValidationError: If coordinates are invalid
        """
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValidationError(f"Invalid coordinates: ({x}, {y})")

        # Check reasonable bounds for geographic coordinates
        if self.input_crs.is_geographic:
            if not (-180 <= x <= 180):
                raise ValidationError(f"Longitude out of range: {x}")
            if not (-90 <= y <= 90):
                raise ValidationError(f"Latitude out of range: {y}")

    def _validate_output_coordinates(self, x: float, y: float) -> None:
        """
        Validate output coordinates.

        Args:
            x, y: Output coordinates to validate

        Raises:
            ValidationError: If output coordinates are invalid
        """
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValidationError(
                f"Transformation produced invalid coordinates: ({x}, {y})"
            )

        # Check for extreme values that might indicate transformation failure
        if abs(x) > 1e10 or abs(y) > 1e10:
            raise ValidationError(
                f"Transformation produced extreme coordinates: ({x}, {y})"
            )

    def _calculate_distance_meters(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """
        Calculate distance between two points in meters.

        Args:
            lon1, lat1: First point coordinates
            lon2, lat2: Second point coordinates

        Returns:
            Distance in meters
        """
        try:
            # Use PyProj for accurate distance calculation
            geod = pyproj.Geod(ellps="WGS84")
            _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
            return abs(distance)
        except Exception:
            # Fallback to simple Euclidean distance
            return np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2) * 111000

    def get_pixel_coordinates(
        self, lon: float, lat: float, transform: rasterio.Affine
    ) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.

        Args:
            lon, lat: Geographic coordinates
            transform: Rasterio affine transform

        Returns:
            Pixel coordinates (row, col)

        Raises:
            CRSError: If coordinate conversion fails
        """
        try:
            # Transform to input CRS if needed
            if self.output_crs.to_string() != self.input_crs.to_string():
                # Convert from output CRS to input CRS
                reverse_transformer = Transformer.from_crs(
                    self.output_crs.to_string(),
                    self.input_crs.to_string(),
                    always_xy=True,
                )
                lon, lat = reverse_transformer.transform(lon, lat)

            # Convert to pixel coordinates
            col, row = ~transform * (lon, lat)
            return int(row), int(col)

        except Exception as e:
            raise CRSError(f"Failed to convert to pixel coordinates: {e}")

    def get_geographic_coordinates(
        self, row: int, col: int, transform: rasterio.Affine
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.

        Args:
            row, col: Pixel coordinates
            transform: Rasterio affine transform

        Returns:
            Geographic coordinates (lon, lat) in output CRS

        Raises:
            CRSError: If coordinate conversion fails
        """
        try:
            # Convert pixel to input CRS coordinates
            lon, lat = rasterio.transform.xy(transform, row, col)

            # Transform to output CRS if needed
            if self.input_crs.to_string() != self.output_crs.to_string():
                lon, lat = self.transformer.transform(lon, lat)

            return lon, lat

        except Exception as e:
            raise CRSError(f"Failed to convert to geographic coordinates: {e}")

    def validate_transformation_accuracy(
        self, test_points: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Validate transformation accuracy with test points.

        Args:
            test_points: Optional list of (x, y) test points

        Returns:
            Dictionary with accuracy metrics

        Raises:
            ValidationError: If accuracy is insufficient
        """
        if test_points is None:
            # Use default test points
            test_points = [
                (-105.0, 40.0),  # Colorado
                (-111.0, 45.0),  # Utah/Montana
                (-109.0, 38.0),  # Utah/Colorado border
            ]

        metrics = {
            "max_error_meters": 0.0,
            "mean_error_meters": 0.0,
            "round_trip_errors": [],
        }

        try:
            errors = []

            for x, y in test_points:
                # Round-trip transformation test
                x_trans, y_trans = self.transform_coordinates(x, y)

                # Transform back
                reverse_transformer = Transformer.from_crs(
                    self.output_crs.to_string(),
                    self.input_crs.to_string(),
                    always_xy=True,
                )
                x_back, y_back = reverse_transformer.transform(x_trans, y_trans)

                # Calculate error
                error_meters = self._calculate_distance_meters(x, y, x_back, y_back)
                errors.append(error_meters)

            metrics["round_trip_errors"] = errors
            metrics["max_error_meters"] = max(errors)
            metrics["mean_error_meters"] = np.mean(errors)

            # Validate accuracy
            if metrics["max_error_meters"] > 100:  # 100m tolerance
                raise ValidationError(
                    f"Transformation accuracy insufficient: "
                    f"max error {metrics['max_error_meters']:.1f}m"
                )

            self.logger.info(
                f"Transformation accuracy validated: "
                f"max error {metrics['max_error_meters']:.2f}m, "
                f"mean error {metrics['mean_error_meters']:.2f}m"
            )

            return metrics

        except Exception as e:
            raise ValidationError(f"Transformation accuracy validation failed: {e}")
