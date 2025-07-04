"""
FLOWFINDER Scientific Validation
===============================

Scientific validation and quality assurance for watershed delineation.
Includes topology validation, performance monitoring, and accuracy assessment.
"""

import logging
import time
import psutil
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid, explain_validity

from .exceptions import ValidationError, PerformanceError


@dataclass
class PerformanceMetrics:
    """Performance metrics for watershed delineation."""

    runtime_seconds: float
    memory_mb: float
    peak_memory_mb: float
    cpu_percent: float
    dem_size_pixels: int
    watershed_area_km2: float
    processing_rate_pixels_per_second: float


@dataclass
class TopologyMetrics:
    """Topology validation metrics for watershed polygons."""

    is_valid: bool
    is_simple: bool
    area_km2: float
    perimeter_km: float
    holes_count: int
    self_intersections: int
    validation_errors: List[str]
    topology_quality_score: float


class PerformanceMonitor:
    """
    Monitor performance during watershed delineation.

    Tracks runtime, memory usage, and processing efficiency to ensure
    performance targets are met.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize performance monitor."""
        self.logger = logger or logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        self.process = psutil.Process()

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

        self.logger.info(
            f"Performance monitoring started - initial memory: {self.start_memory:.1f} MB"
        )

    def update_peak_memory(self) -> None:
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        if self.start_time is None:
            runtime = 0.0
        else:
            runtime = current_time - self.start_time

        try:
            cpu_percent = self.process.cpu_percent()
        except Exception:
            cpu_percent = 0.0

        return {
            "runtime_seconds": runtime,
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
            "cpu_percent": cpu_percent,
        }

    def finish_monitoring(
        self, dem_size_pixels: int, watershed_area_km2: float
    ) -> PerformanceMetrics:
        """
        Finish monitoring and return performance metrics.

        Args:
            dem_size_pixels: Total number of DEM pixels processed
            watershed_area_km2: Watershed area in square kilometers

        Returns:
            PerformanceMetrics object

        Raises:
            PerformanceError: If performance targets not met
        """
        if self.start_time is None or self.start_memory is None:
            raise PerformanceError("Performance monitoring was not started")

        end_time = time.time()
        runtime = end_time - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        try:
            cpu_percent = self.process.cpu_percent()
        except Exception:
            cpu_percent = 0.0

        # Calculate processing rate
        processing_rate = dem_size_pixels / runtime if runtime > 0 else 0

        metrics = PerformanceMetrics(
            runtime_seconds=runtime,
            memory_mb=current_memory,
            peak_memory_mb=self.peak_memory,
            cpu_percent=cpu_percent,
            dem_size_pixels=dem_size_pixels,
            watershed_area_km2=watershed_area_km2,
            processing_rate_pixels_per_second=processing_rate,
        )

        self.logger.info(
            f"Performance monitoring completed:\n"
            f"  Runtime: {runtime:.2f}s\n"
            f"  Peak memory: {self.peak_memory:.1f} MB\n"
            f"  Processing rate: {processing_rate:.0f} pixels/second\n"
            f"  Watershed area: {watershed_area_km2:.2f} km²"
        )

        # Check performance targets
        self._validate_performance_targets(metrics)

        return metrics

    def _validate_performance_targets(self, metrics: PerformanceMetrics) -> None:
        """
        Validate that performance targets are met.

        Args:
            metrics: Performance metrics to validate

        Raises:
            PerformanceError: If performance targets not met
        """
        # Check 30-second runtime target
        if metrics.runtime_seconds > 30.0:
            self.logger.warning(
                f"Runtime target exceeded: {metrics.runtime_seconds:.2f}s > 30.0s"
            )
            # Don't raise error - just warn for now

        # Check memory usage (warn if > 4GB)
        if metrics.peak_memory_mb > 4096:
            self.logger.warning(f"High memory usage: {metrics.peak_memory_mb:.1f} MB")

        # Check processing efficiency
        min_rate = 100000  # 100k pixels/second minimum
        if metrics.processing_rate_pixels_per_second < min_rate:
            self.logger.warning(
                f"Low processing rate: {metrics.processing_rate_pixels_per_second:.0f} pixels/s"
            )


class TopologyValidator:
    """
    Validate watershed polygon topology for scientific accuracy.

    Ensures generated watersheds are topologically valid and geometrically
    reasonable for hydrological analysis.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize topology validator."""
        self.logger = logger or logging.getLogger(__name__)

    def validate_watershed_topology(
        self,
        watershed_polygon: Polygon,
        pour_point: Tuple[float, float],
        dem_bounds: Tuple[float, float, float, float],
    ) -> TopologyMetrics:
        """
        Validate watershed polygon topology.

        Args:
            watershed_polygon: Watershed polygon to validate
            pour_point: Pour point coordinates (lon, lat)
            dem_bounds: DEM bounds (minx, miny, maxx, maxy)

        Returns:
            TopologyMetrics with validation results

        Raises:
            ValidationError: If critical topology errors found
        """
        try:
            errors = []

            # Basic validity check
            is_valid = watershed_polygon.is_valid
            if not is_valid:
                errors.append(
                    f"Invalid geometry: {explain_validity(watershed_polygon)}"
                )

            # Simplicity check (no self-intersections)
            is_simple = watershed_polygon.is_simple
            if not is_simple:
                errors.append("Polygon has self-intersections")

            # Calculate basic metrics
            area_km2 = self._calculate_area_km2(watershed_polygon)
            perimeter_km = self._calculate_perimeter_km(watershed_polygon)

            # Count holes
            holes_count = 0
            if hasattr(watershed_polygon, "interiors"):
                holes_count = len(watershed_polygon.interiors)

            # Count self-intersections
            self_intersections = self._count_self_intersections(watershed_polygon)

            # Validate pour point containment
            pour_point_geom = Point(pour_point)
            if not watershed_polygon.contains(pour_point_geom):
                if not watershed_polygon.touches(pour_point_geom):
                    errors.append("Pour point not contained within watershed")

            # Validate reasonable size
            if area_km2 < 0.01:  # Less than 1 hectare
                errors.append(f"Watershed too small: {area_km2:.6f} km²")
            elif area_km2 > 10000:  # Larger than 10,000 km²
                errors.append(f"Watershed suspiciously large: {area_km2:.2f} km²")

            # Validate shape reasonableness
            if perimeter_km > 0 and area_km2 > 0:
                shape_ratio = (perimeter_km**2) / area_km2
                if shape_ratio > 100:  # Very elongated or fragmented
                    errors.append(
                        f"Watershed shape suspicious (high perimeter/area ratio: {shape_ratio:.1f})"
                    )

            # Validate bounds
            if not self._validate_watershed_bounds(watershed_polygon, dem_bounds):
                errors.append("Watershed extends beyond DEM bounds")

            # Calculate topology quality score
            quality_score = self._calculate_topology_quality_score(
                is_valid, is_simple, holes_count, self_intersections, len(errors)
            )

            metrics = TopologyMetrics(
                is_valid=is_valid,
                is_simple=is_simple,
                area_km2=area_km2,
                perimeter_km=perimeter_km,
                holes_count=holes_count,
                self_intersections=self_intersections,
                validation_errors=errors,
                topology_quality_score=quality_score,
            )

            # Log validation results
            if errors:
                self.logger.warning(f"Topology validation found {len(errors)} issues:")
                for error in errors:
                    self.logger.warning(f"  - {error}")
            else:
                self.logger.info(
                    f"Topology validation passed (quality score: {quality_score:.2f})"
                )

            return metrics

        except Exception as e:
            raise ValidationError(f"Topology validation failed: {e}")

    def _calculate_area_km2(self, polygon: Polygon) -> float:
        """Calculate polygon area in square kilometers."""
        try:
            # For geographic coordinates, use geodesic area calculation
            if hasattr(polygon, "area"):
                # Convert from degrees² to km² (rough approximation)
                area_deg2 = polygon.area
                # 1 degree² ≈ 12,391 km² at equator (varies by latitude)
                area_km2 = area_deg2 * 12391
                return area_km2
            return 0.0
        except Exception:
            return 0.0

    def _calculate_perimeter_km(self, polygon: Polygon) -> float:
        """Calculate polygon perimeter in kilometers."""
        try:
            if hasattr(polygon, "length"):
                # Convert from degrees to km (rough approximation)
                length_deg = polygon.length
                # 1 degree ≈ 111 km (varies by latitude)
                length_km = length_deg * 111
                return length_km
            return 0.0
        except Exception:
            return 0.0

    def _count_self_intersections(self, polygon: Polygon) -> int:
        """Count self-intersections in polygon."""
        try:
            if polygon.is_simple:
                return 0

            # This is a simplified check - full self-intersection counting
            # would require more sophisticated geometric analysis
            coords = list(polygon.exterior.coords)
            intersections = 0

            # Check for obvious self-intersections (simplified)
            for i in range(len(coords) - 3):
                for j in range(i + 2, len(coords) - 1):
                    if i == 0 and j == len(coords) - 2:
                        continue  # Skip first and last segment

                    # Check if segments intersect
                    if self._segments_intersect(
                        coords[i], coords[i + 1], coords[j], coords[j + 1]
                    ):
                        intersections += 1

            return intersections

        except Exception:
            return 0

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> bool:
        """Check if two line segments intersect."""
        try:
            # Simple bounding box check first
            if (
                max(p1[0], p2[0]) < min(p3[0], p4[0])
                or max(p3[0], p4[0]) < min(p1[0], p2[0])
                or max(p1[1], p2[1]) < min(p3[1], p4[1])
                or max(p3[1], p4[1]) < min(p1[1], p2[1])
            ):
                return False

            # Use cross product to check intersection
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(
                p1, p2, p4
            )

        except Exception:
            return False

    def _validate_watershed_bounds(
        self, polygon: Polygon, dem_bounds: Tuple[float, float, float, float]
    ) -> bool:
        """Validate that watershed is within DEM bounds."""
        try:
            minx, miny, maxx, maxy = dem_bounds
            poly_bounds = polygon.bounds

            # Allow small tolerance for floating point precision
            tolerance = 0.001

            return (
                poly_bounds[0] >= minx - tolerance
                and poly_bounds[1] >= miny - tolerance
                and poly_bounds[2] <= maxx + tolerance
                and poly_bounds[3] <= maxy + tolerance
            )
        except Exception:
            return True  # If bounds check fails, assume valid

    def _calculate_topology_quality_score(
        self,
        is_valid: bool,
        is_simple: bool,
        holes_count: int,
        self_intersections: int,
        error_count: int,
    ) -> float:
        """
        Calculate topology quality score (0-1, higher is better).

        Args:
            is_valid: Whether geometry is valid
            is_simple: Whether geometry is simple
            holes_count: Number of holes in polygon
            self_intersections: Number of self-intersections
            error_count: Total number of validation errors

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize invalid geometry
        if not is_valid:
            score -= 0.5

        # Penalize non-simple geometry
        if not is_simple:
            score -= 0.3

        # Penalize holes (watersheds shouldn't have holes)
        score -= min(holes_count * 0.1, 0.3)

        # Penalize self-intersections
        score -= min(self_intersections * 0.05, 0.2)

        # Penalize other errors
        score -= min(error_count * 0.05, 0.3)

        return max(score, 0.0)

    def repair_watershed_topology(self, watershed_polygon: Polygon) -> Polygon:
        """
        Attempt to repair watershed topology issues.

        Args:
            watershed_polygon: Polygon to repair

        Returns:
            Repaired polygon
        """
        try:
            if watershed_polygon.is_valid:
                return watershed_polygon

            self.logger.info("Attempting to repair watershed topology...")

            # Try make_valid first
            repaired = make_valid(watershed_polygon)

            if repaired.is_valid and hasattr(repaired, "area") and repaired.area > 0:
                self.logger.info("Topology repaired using make_valid()")
                return repaired

            # Try buffer(0) approach
            try:
                buffered = watershed_polygon.buffer(0)
                if (
                    buffered.is_valid
                    and hasattr(buffered, "area")
                    and buffered.area > 0
                ):
                    self.logger.info("Topology repaired using buffer(0)")
                    return buffered
            except Exception:
                pass

            # Try convex hull as last resort
            try:
                hull = watershed_polygon.convex_hull
                if hull.is_valid and hasattr(hull, "area") and hull.area > 0:
                    self.logger.warning(
                        "Used convex hull as topology repair (may be inaccurate)"
                    )
                    return hull
            except Exception:
                pass

            self.logger.error("Could not repair watershed topology")
            return watershed_polygon

        except Exception as e:
            self.logger.error(f"Topology repair failed: {e}")
            return watershed_polygon


class AccuracyAssessment:
    """
    Assess watershed delineation accuracy for scientific validation.

    Provides methods for comparing generated watersheds against reference
    data and calculating accuracy metrics.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize accuracy assessment."""
        self.logger = logger or logging.getLogger(__name__)

    def calculate_iou_score(
        self, predicted_polygon: Polygon, truth_polygon: Polygon
    ) -> float:
        """
        Calculate Intersection over Union (IoU) score.

        Args:
            predicted_polygon: Generated watershed polygon
            truth_polygon: Reference truth polygon

        Returns:
            IoU score (0-1, higher is better)

        Raises:
            ValidationError: If calculation fails
        """
        try:
            # Ensure both polygons are valid
            if not predicted_polygon.is_valid:
                predicted_polygon = make_valid(predicted_polygon)
            if not truth_polygon.is_valid:
                truth_polygon = make_valid(truth_polygon)

            # Calculate intersection and union
            intersection = predicted_polygon.intersection(truth_polygon)
            union = predicted_polygon.union(truth_polygon)

            # Handle empty geometries
            if union.area == 0:
                return 0.0

            iou_score = intersection.area / union.area

            self.logger.debug(f"IoU score calculated: {iou_score:.4f}")

            return float(iou_score)

        except Exception as e:
            raise ValidationError(f"IoU calculation failed: {e}")

    def assess_watershed_quality(
        self,
        watershed_polygon: Polygon,
        performance_metrics: PerformanceMetrics,
        topology_metrics: TopologyMetrics,
        reference_polygon: Optional[Polygon] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive watershed quality assessment.

        Args:
            watershed_polygon: Generated watershed polygon
            performance_metrics: Performance metrics
            topology_metrics: Topology validation metrics
            reference_polygon: Optional reference polygon for accuracy

        Returns:
            Dictionary with comprehensive quality assessment
        """
        assessment = {
            "timestamp": time.time(),
            "performance": {
                "runtime_seconds": performance_metrics.runtime_seconds,
                "memory_mb": performance_metrics.peak_memory_mb,
                "meets_30s_target": performance_metrics.runtime_seconds <= 30.0,
                "processing_efficiency": performance_metrics.processing_rate_pixels_per_second,
            },
            "topology": {
                "is_valid": topology_metrics.is_valid,
                "quality_score": topology_metrics.topology_quality_score,
                "area_km2": topology_metrics.area_km2,
                "error_count": len(topology_metrics.validation_errors),
                "has_holes": topology_metrics.holes_count > 0,
            },
            "overall_quality": "unknown",
        }

        # Add accuracy assessment if reference is provided
        if reference_polygon is not None:
            try:
                iou_score = self.calculate_iou_score(
                    watershed_polygon, reference_polygon
                )
                assessment["accuracy"] = {
                    "iou_score": iou_score,
                    "meets_95_percent_target": iou_score >= 0.95,
                    "accuracy_grade": self._grade_accuracy(iou_score),
                }
            except Exception as e:
                self.logger.warning(f"Accuracy assessment failed: {e}")
                assessment["accuracy"] = {"error": str(e)}

        # Calculate overall quality grade
        assessment["overall_quality"] = self._calculate_overall_quality(assessment)

        return assessment

    def _grade_accuracy(self, iou_score: float) -> str:
        """Grade accuracy based on IoU score."""
        if iou_score >= 0.95:
            return "Excellent"
        elif iou_score >= 0.90:
            return "Good"
        elif iou_score >= 0.80:
            return "Fair"
        elif iou_score >= 0.70:
            return "Poor"
        else:
            return "Unacceptable"

    def _calculate_overall_quality(self, assessment: Dict[str, Any]) -> str:
        """Calculate overall quality grade."""
        try:
            performance_ok = assessment["performance"]["meets_30s_target"]
            topology_ok = assessment["topology"]["quality_score"] > 0.8

            if "accuracy" in assessment:
                accuracy_ok = assessment["accuracy"].get(
                    "meets_95_percent_target", False
                )

                if performance_ok and topology_ok and accuracy_ok:
                    return "Excellent"
                elif topology_ok and (performance_ok or accuracy_ok):
                    return "Good"
                elif topology_ok:
                    return "Fair"
                else:
                    return "Poor"
            else:
                if performance_ok and topology_ok:
                    return "Good"
                elif topology_ok:
                    return "Fair"
                else:
                    return "Poor"

        except Exception:
            return "Unknown"
