#!/usr/bin/env python3
"""
Shared Geometry Diagnostics and Repair Utilities
================================================

Comprehensive geometry validation and repair tools for the FLOWFINDER pipeline.
Provides detailed diagnostics, progressive repair strategies, and extensive logging
for handling real-world geospatial data quality issues.

This module is shared between basin_sampler.py, truth_extractor.py, and other
scripts that need robust geometry handling.

Author: FLOWFINDER Benchmark Team
License: MIT
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.validation import explain_validity, make_valid
from shapely.ops import unary_union


class GeometryDiagnostics:
    """
    Comprehensive geometry diagnostics and repair for geospatial data.

    Provides detailed analysis of geometry issues, progressive repair strategies,
    and extensive logging capabilities for handling real-world data quality problems.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize geometry diagnostics.

        Args:
            logger: Logger instance for output
            config: Configuration dictionary with geometry repair settings
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.repair_stats = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for geometry repair."""
        return {
            "geometry_repair": {
                "enable_diagnostics": True,
                "enable_repair_attempts": True,
                "invalid_geometry_action": "remove",
                "max_repair_attempts": 3,
                "detailed_logging": True,
                "repair_strategies": {
                    "buffer_fix": True,
                    "simplify": True,
                    "make_valid": True,
                    "convex_hull": False,
                    "orient_fix": True,
                    "simplify_holes": True,
                },
            }
        }

    def diagnose_and_repair_geometries(
        self, gdf: gpd.GeoDataFrame, data_description: str
    ) -> gpd.GeoDataFrame:
        """
        Comprehensive geometry diagnostics and repair for geospatial data.

        Args:
            gdf: GeoDataFrame to diagnose and repair
            data_description: Description for logging purposes

        Returns:
            GeoDataFrame with repaired geometries
        """
        self.logger.info(
            f"Performing geometry diagnostics and repair for {data_description}"
        )

        if len(gdf) == 0:
            self.logger.warning(f"Empty GeoDataFrame for {data_description}")
            return gdf

        original_count = len(gdf)
        gdf_repaired = gdf.copy()

        # Comprehensive geometry analysis
        geometry_stats = self._analyze_geometry_issues(gdf_repaired, data_description)

        # Apply progressive repair strategies
        if geometry_stats["total_invalid"] > 0:
            self.logger.info(
                f"Attempting to repair {geometry_stats['total_invalid']} invalid geometries"
            )
            gdf_repaired = self._apply_geometry_repairs(
                gdf_repaired, geometry_stats, data_description
            )

        # Final validation
        final_stats = self._analyze_geometry_issues(
            gdf_repaired, f"{data_description} (after repair)"
        )

        # Log repair summary
        self._log_geometry_repair_summary(geometry_stats, final_stats, data_description)

        # Handle remaining invalid geometries
        if final_stats["total_invalid"] > 0:
            gdf_repaired = self._handle_remaining_invalid_geometries(
                gdf_repaired, final_stats, data_description
            )

        final_count = len(gdf_repaired)
        if final_count != original_count:
            self.logger.warning(
                f"Geometry repair resulted in {original_count - final_count} removed features"
            )

        return gdf_repaired

    def analyze_geometry_issues(
        self, gdf: gpd.GeoDataFrame, data_description: str
    ) -> Dict[str, Any]:
        """
        Analyze and classify geometry issues with detailed diagnostics.

        Args:
            gdf: GeoDataFrame to analyze
            data_description: Description for logging

        Returns:
            Dictionary with detailed geometry statistics and classifications
        """
        return self._analyze_geometry_issues(gdf, data_description)

    def _analyze_geometry_issues(
        self, gdf: gpd.GeoDataFrame, data_description: str
    ) -> Dict[str, Any]:
        """
        Analyze and classify geometry issues with detailed diagnostics.

        Args:
            gdf: GeoDataFrame to analyze
            data_description: Description for logging

        Returns:
            Dictionary with detailed geometry statistics and classifications
        """
        stats = {
            "total_features": len(gdf),
            "total_invalid": 0,
            "total_empty": 0,
            "total_valid": 0,
            "issue_types": {},
            "geometry_types": {},
            "repair_candidates": {},
            "critical_errors": [],
            "detailed_diagnostics": [],
        }

        if len(gdf) == 0:
            return stats

        # Analyze each geometry
        for idx, geom in enumerate(gdf.geometry):
            try:
                geom_analysis = self._diagnose_single_geometry(geom, idx)
            except Exception as e:
                self.logger.error(
                    f"Exception during geometry analysis for geometry {idx}: {e}"
                )
                # Create a default analysis result for failed geometry
                geom_analysis = {
                    "index": idx,
                    "is_valid": False,
                    "is_empty": True,
                    "is_critical": True,
                    "geometry_type": "unknown",
                    "issues": ["analysis_error"],
                    "explanation": f"Analysis failed: {e}",
                    "repair_strategy": "remove",
                    "bounds": None,
                    "area": 0.0,
                    "length": 0.0,
                }

            # Update statistics
            if geom_analysis["is_valid"]:
                stats["total_valid"] += 1
            else:
                stats["total_invalid"] += 1

            if geom_analysis["is_empty"]:
                stats["total_empty"] += 1

            # Track geometry types
            geom_type = geom_analysis["geometry_type"]
            stats["geometry_types"][geom_type] = (
                stats["geometry_types"].get(geom_type, 0) + 1
            )

            # Track issue types
            for issue in geom_analysis["issues"]:
                stats["issue_types"][issue] = stats["issue_types"].get(issue, 0) + 1

            # Track repair candidates
            repair_strategy = geom_analysis["repair_strategy"]
            if repair_strategy != "none":
                stats["repair_candidates"][repair_strategy] = (
                    stats["repair_candidates"].get(repair_strategy, 0) + 1
                )

            # Track critical errors
            if geom_analysis["is_critical"]:
                stats["critical_errors"].append(
                    {
                        "index": idx,
                        "issues": geom_analysis["issues"],
                        "explanation": geom_analysis["explanation"],
                    }
                )

            # Store detailed diagnostics for first 10 invalid geometries
            if (
                not geom_analysis["is_valid"]
                and len(stats["detailed_diagnostics"]) < 10
            ):
                stats["detailed_diagnostics"].append(
                    {
                        "index": idx,
                        "geometry_type": geom_type,
                        "issues": geom_analysis["issues"],
                        "explanation": geom_analysis["explanation"],
                        "repair_strategy": repair_strategy,
                        "bounds": geom_analysis["bounds"],
                    }
                )

        return stats

    def _diagnose_single_geometry(self, geom, index: int) -> Dict[str, Any]:
        """
        Detailed diagnosis of a single geometry with repair strategy recommendation.

        Args:
            geom: Shapely geometry to diagnose
            index: Index for reference

        Returns:
            Dictionary with detailed geometry diagnosis
        """
        diagnosis = {
            "index": index,
            "is_valid": True,
            "is_empty": True,
            "is_critical": False,
            "geometry_type": "unknown",
            "issues": [],
            "explanation": "",
            "repair_strategy": "none",
            "bounds": None,
            "area": 0.0,
            "length": 0.0,
        }

        try:
            # Basic properties
            diagnosis["is_empty"] = geom.is_empty if geom is not None else True
            diagnosis["is_valid"] = geom.is_valid if geom is not None else False
            diagnosis["geometry_type"] = geom.geom_type if geom is not None else "None"

            if geom is None:
                diagnosis["issues"].append("null_geometry")
                diagnosis["explanation"] = "Geometry is None/null"
                diagnosis["repair_strategy"] = "remove"
                diagnosis["is_critical"] = True
                return diagnosis

            if diagnosis["is_empty"]:
                diagnosis["issues"].append("empty_geometry")
                diagnosis["explanation"] = "Geometry is empty"
                diagnosis["repair_strategy"] = "remove"
                return diagnosis

            # Get bounds and measurements
            try:
                diagnosis["bounds"] = list(geom.bounds)
                if hasattr(geom, "area"):
                    diagnosis["area"] = float(geom.area)
                if hasattr(geom, "length"):
                    diagnosis["length"] = float(geom.length)
            except Exception as e:
                diagnosis["issues"].append("bounds_error")
                diagnosis["explanation"] = f"Cannot compute bounds: {e}"
                diagnosis["is_critical"] = True

            # Detailed validity analysis
            if not diagnosis["is_valid"]:
                explanation = explain_validity(geom)
                diagnosis["explanation"] = explanation

                # Classify the type of invalidity
                explanation_lower = explanation.lower()

                if (
                    "self-intersection" in explanation_lower
                    or "self intersection" in explanation_lower
                ):
                    diagnosis["issues"].append("self_intersection")
                    diagnosis["repair_strategy"] = "buffer_fix"

                if "ring self-intersection" in explanation_lower:
                    diagnosis["issues"].append("ring_self_intersection")
                    diagnosis["repair_strategy"] = "buffer_fix"

                if (
                    "duplicate point" in explanation_lower
                    or "repeated point" in explanation_lower
                ):
                    diagnosis["issues"].append("duplicate_points")
                    diagnosis["repair_strategy"] = "simplify"

                if (
                    "orientation" in explanation_lower
                    or "clockwise" in explanation_lower
                ):
                    diagnosis["issues"].append("wrong_orientation")
                    diagnosis["repair_strategy"] = "orient_fix"

                if "coordinate" in explanation_lower and (
                    "nan" in explanation_lower or "infinite" in explanation_lower
                ):
                    diagnosis["issues"].append("invalid_coordinates")
                    diagnosis["repair_strategy"] = "remove"
                    diagnosis["is_critical"] = True

                if (
                    "too few points" in explanation_lower
                    or "insufficient" in explanation_lower
                ):
                    diagnosis["issues"].append("insufficient_points")
                    diagnosis["repair_strategy"] = "remove"
                    diagnosis["is_critical"] = True

                if "hole" in explanation_lower and "exterior" in explanation_lower:
                    diagnosis["issues"].append("hole_outside_shell")
                    diagnosis["repair_strategy"] = "convex_hull"

                if (
                    "nested" in explanation_lower
                    or "interior ring" in explanation_lower
                ):
                    diagnosis["issues"].append("nested_holes")
                    diagnosis["repair_strategy"] = "simplify_holes"

                # Default to make_valid if no specific strategy identified
                if diagnosis["repair_strategy"] == "none":
                    diagnosis["repair_strategy"] = "make_valid"

            # Additional geometric analysis for valid geometries
            if diagnosis["is_valid"]:
                # Check for very thin geometries first
                if diagnosis["area"] > 0 and diagnosis["length"] > 0:
                    aspect_ratio = diagnosis["length"] / diagnosis["area"]
                    if aspect_ratio > 1e6:  # Very thin
                        diagnosis["issues"].append("extremely_thin")
                        diagnosis["repair_strategy"] = "simplify"

                # Check for very small geometries (higher priority - overrides thin geometry)
                if diagnosis["area"] < 1e-10 and geom.geom_type in [
                    "Polygon",
                    "MultiPolygon",
                ]:
                    diagnosis["issues"].append("extremely_small_area")
                    diagnosis["repair_strategy"] = "remove"

                # Check for extremely long/short linestrings
                if geom.geom_type in ["LineString", "MultiLineString"]:
                    if diagnosis["length"] < 1e-10:
                        diagnosis["issues"].append("extremely_short_line")
                        diagnosis["repair_strategy"] = "remove"
                    elif diagnosis["length"] > 1e8:
                        diagnosis["issues"].append("extremely_long_line")
                        diagnosis["repair_strategy"] = "simplify"

        except Exception as e:
            diagnosis["issues"].append("analysis_error")
            diagnosis["explanation"] = f"Error during geometry analysis: {e}"
            diagnosis["repair_strategy"] = "remove"
            diagnosis["is_critical"] = True

        return diagnosis

    def _apply_geometry_repairs(
        self,
        gdf: gpd.GeoDataFrame,
        geometry_stats: Dict[str, Any],
        data_description: str,
    ) -> gpd.GeoDataFrame:
        """
        Apply progressive geometry repair strategies based on diagnostic results.

        Args:
            gdf: GeoDataFrame with geometries to repair
            geometry_stats: Statistics from geometry analysis
            data_description: Description for logging

        Returns:
            GeoDataFrame with repaired geometries
        """
        gdf_repaired = gdf.copy()
        repair_counts = {}
        failed_repairs = []

        self.logger.info(f"Applying geometry repairs for {data_description}")

        # Process each repair strategy
        for strategy, count in geometry_stats["repair_candidates"].items():
            if count == 0:
                continue

            self.logger.info(f"Applying {strategy} repair to {count} geometries")

            strategy_successes = 0
            strategy_failures = 0

            for idx, geom in enumerate(gdf_repaired.geometry):
                if geom is None or geom.is_valid:
                    continue

                # Determine if this geometry needs this repair strategy
                try:
                    diagnosis = self._diagnose_single_geometry(geom, idx)
                except Exception as e:
                    self.logger.error(
                        f"Exception during repair strategy analysis for geometry {idx}: {e}"
                    )
                    strategy_failures += 1
                    continue

                if diagnosis["repair_strategy"] != strategy:
                    continue

                try:
                    repaired_geom = self._apply_repair_strategy(geom, strategy, idx)

                    if repaired_geom is not None and repaired_geom.is_valid:
                        gdf_repaired.iloc[
                            idx, gdf_repaired.columns.get_loc("geometry")
                        ] = repaired_geom
                        strategy_successes += 1
                    else:
                        failed_repairs.append(
                            {
                                "index": idx,
                                "strategy": strategy,
                                "original_issue": diagnosis["explanation"],
                            }
                        )
                        strategy_failures += 1

                except Exception as e:
                    self.logger.warning(
                        f"Repair strategy {strategy} failed for geometry {idx}: {e}"
                    )
                    failed_repairs.append(
                        {"index": idx, "strategy": strategy, "error": str(e)}
                    )
                    strategy_failures += 1

            repair_counts[strategy] = {
                "attempted": count,
                "successful": strategy_successes,
                "failed": strategy_failures,
            }

            self.logger.info(
                f"Strategy {strategy}: {strategy_successes}/{count} successful repairs"
            )

        # Store repair statistics for later reporting
        self.repair_stats = {
            "data_description": data_description,
            "repair_counts": repair_counts,
            "failed_repairs": failed_repairs,
        }

        return gdf_repaired

    def _apply_repair_strategy(self, geom, strategy: str, index: int):
        """
        Apply a specific repair strategy to a geometry.

        Args:
            geom: Shapely geometry to repair
            strategy: Repair strategy to apply
            index: Geometry index for logging

        Returns:
            Repaired geometry or None if repair failed
        """
        try:
            if strategy == "make_valid":
                return make_valid(geom)

            elif strategy == "buffer_fix":
                # Buffer by tiny amount to fix self-intersections
                buffered = geom.buffer(0)
                if buffered.is_valid:
                    return buffered
                # If that doesn't work, try make_valid
                return make_valid(geom)

            elif strategy == "simplify":
                # Try simplifying with small tolerance
                tolerance = 1e-10
                simplified = geom.simplify(tolerance, preserve_topology=True)
                if simplified.is_valid and not simplified.is_empty:
                    return simplified
                # If that doesn't work, try buffer fix
                return self._apply_repair_strategy(geom, "buffer_fix", index)

            elif strategy == "orient_fix":
                # Try to fix orientation issues
                if hasattr(geom, "exterior"):
                    # For polygons, ensure proper orientation
                    from shapely.geometry import Polygon

                    if geom.geom_type == "Polygon":
                        # Create new polygon with properly oriented exterior
                        ext_coords = list(geom.exterior.coords)
                        if len(ext_coords) > 3:
                            oriented = Polygon(ext_coords)
                            if oriented.is_valid:
                                return oriented
                # Fall back to make_valid
                return make_valid(geom)

            elif strategy == "convex_hull":
                # Use convex hull for complex topology issues
                hull = geom.convex_hull
                if hull.is_valid and not hull.is_empty:
                    return hull
                return None

            elif strategy == "simplify_holes":
                # Remove problematic holes
                if hasattr(geom, "exterior") and geom.geom_type == "Polygon":
                    from shapely.geometry import Polygon

                    try:
                        # Create polygon with just the exterior
                        simplified = Polygon(geom.exterior.coords)
                        if simplified.is_valid:
                            return simplified
                    except Exception:
                        pass
                # Fall back to convex hull
                return self._apply_repair_strategy(geom, "convex_hull", index)

            elif strategy == "unary_union_fix":
                # For multi-geometries, try unary union
                if geom.geom_type.startswith("Multi"):
                    try:
                        unioned = unary_union([geom])
                        if unioned.is_valid and not unioned.is_empty:
                            return unioned
                    except Exception:
                        pass
                # Fall back to make_valid
                return make_valid(geom)

            elif strategy == "remove":
                # Mark for removal
                return None

            else:
                # Unknown strategy, try make_valid as default
                return make_valid(geom)

        except Exception as e:
            self.logger.warning(
                f"Repair strategy {strategy} failed for geometry {index}: {e}"
            )
            return None

    def _handle_remaining_invalid_geometries(
        self, gdf: gpd.GeoDataFrame, final_stats: Dict[str, Any], data_description: str
    ) -> gpd.GeoDataFrame:
        """
        Handle geometries that couldn't be repaired.

        Args:
            gdf: GeoDataFrame with remaining invalid geometries
            final_stats: Final geometry statistics
            data_description: Description for logging

        Returns:
            GeoDataFrame with invalid geometries handled
        """
        self.logger.warning(
            f"{final_stats['total_invalid']} geometries remain invalid after repair attempts"
        )

        # Log details about remaining invalid geometries
        for diagnostic in final_stats["detailed_diagnostics"]:
            self.logger.error(
                f"Index {diagnostic['index']}: {diagnostic['explanation']}"
            )

        # Configuration-based handling
        config = self.config.get("geometry_repair", {})
        action = config.get(
            "invalid_geometry_action", "remove"
        )  # 'remove', 'keep', 'convert_to_point'

        if action == "remove":
            # Remove invalid geometries
            valid_mask = gdf.geometry.is_valid
            removed_count = (~valid_mask).sum()
            if removed_count > 0:
                self.logger.warning(
                    f"Removing {removed_count} invalid geometries from {data_description}"
                )
                gdf = gdf[valid_mask].reset_index(drop=True)

        elif action == "convert_to_point":
            # Convert invalid geometries to their centroids
            invalid_mask = ~gdf.geometry.is_valid
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                self.logger.warning(
                    f"Converting {invalid_count} invalid geometries to points"
                )
                for idx in gdf[invalid_mask].index:
                    try:
                        centroid = gdf.loc[idx, "geometry"].centroid
                        if centroid.is_valid:
                            gdf.loc[idx, "geometry"] = centroid
                        else:
                            # If centroid is also invalid, remove
                            gdf = gdf.drop(idx)
                    except Exception:
                        gdf = gdf.drop(idx)

                gdf = gdf.reset_index(drop=True)

        elif action == "keep":
            # Keep invalid geometries but log warnings
            self.logger.warning(
                f"Keeping {final_stats['total_invalid']} invalid geometries as requested"
            )

        return gdf

    def _log_geometry_repair_summary(
        self,
        original_stats: Dict[str, Any],
        final_stats: Dict[str, Any],
        data_description: str,
    ) -> None:
        """
        Log comprehensive summary of geometry repair operations.

        Args:
            original_stats: Statistics before repair
            final_stats: Statistics after repair
            data_description: Description for logging
        """
        self.logger.info(f"=== Geometry Repair Summary: {data_description} ===")

        # Overall statistics
        original_invalid = original_stats["total_invalid"]
        final_invalid = final_stats["total_invalid"]
        repaired_count = original_invalid - final_invalid

        self.logger.info(f"Total features: {original_stats['total_features']}")
        self.logger.info(f"Originally invalid: {original_invalid}")
        self.logger.info(f"Successfully repaired: {repaired_count}")
        self.logger.info(f"Still invalid: {final_invalid}")

        if original_invalid > 0:
            repair_rate = (repaired_count / original_invalid) * 100
            self.logger.info(f"Repair success rate: {repair_rate:.1f}%")

        # Issue type breakdown
        if original_stats["issue_types"]:
            self.logger.info("Issue types found:")
            for issue_type, count in original_stats["issue_types"].items():
                self.logger.info(f"  - {issue_type}: {count}")

        # Geometry type distribution
        if original_stats["geometry_types"]:
            self.logger.info("Geometry types:")
            for geom_type, count in original_stats["geometry_types"].items():
                self.logger.info(f"  - {geom_type}: {count}")

        # Repair strategy effectiveness
        if hasattr(self, "repair_stats") and self.repair_stats:
            repair_counts = self.repair_stats["repair_counts"]
            if repair_counts:
                self.logger.info("Repair strategy effectiveness:")
                for strategy, stats in repair_counts.items():
                    success_rate = (
                        (stats["successful"] / stats["attempted"]) * 100
                        if stats["attempted"] > 0
                        else 0
                    )
                    self.logger.info(
                        f"  - {strategy}: {stats['successful']}/{stats['attempted']} ({success_rate:.1f}%)"
                    )

        # Critical errors
        if original_stats["critical_errors"]:
            self.logger.error(
                f"Found {len(original_stats['critical_errors'])} critical geometry errors:"
            )
            for error in original_stats["critical_errors"][:5]:  # Show first 5
                self.logger.error(f"  - Index {error['index']}: {error['explanation']}")

        # Recommendations
        recommendations = self._generate_geometry_recommendations(
            original_stats, final_stats
        )
        if recommendations:
            self.logger.info("Recommendations:")
            for rec in recommendations:
                self.logger.info(f"  - {rec}")

    def _generate_geometry_recommendations(
        self, original_stats: Dict[str, Any], final_stats: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on geometry analysis results.

        Args:
            original_stats: Original geometry statistics
            final_stats: Final geometry statistics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # High invalid geometry rate
        if original_stats["total_features"] > 0:
            invalid_rate = (
                original_stats["total_invalid"] / original_stats["total_features"]
            ) * 100
            if invalid_rate > 20:
                recommendations.append(
                    f"High invalid geometry rate ({invalid_rate:.1f}%) - consider data source quality"
                )

        # Specific issue patterns
        issue_types = original_stats.get("issue_types", {})

        if issue_types.get("self_intersection", 0) > 5:
            recommendations.append(
                "Many self-intersection errors - consider simplifying input data"
            )

        if issue_types.get("duplicate_points", 0) > 10:
            recommendations.append(
                "Many duplicate point errors - consider coordinate precision settings"
            )

        if issue_types.get("extremely_small_area", 0) > 0:
            recommendations.append(
                "Very small geometries detected - consider minimum area thresholds"
            )

        # Repair effectiveness
        if hasattr(self, "repair_stats") and self.repair_stats:
            repair_counts = self.repair_stats["repair_counts"]
            total_attempted = sum(
                stats["attempted"] for stats in repair_counts.values()
            )
            total_successful = sum(
                stats["successful"] for stats in repair_counts.values()
            )

            if total_attempted > 0:
                overall_success_rate = (total_successful / total_attempted) * 100
                if overall_success_rate < 50:
                    recommendations.append(
                        "Low repair success rate - consider alternative data sources"
                    )

        return recommendations
