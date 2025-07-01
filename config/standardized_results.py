#!/usr/bin/env python3
"""
Standardized Result Format for Multi-Tool Watershed Delineation
==============================================================

This module defines a standardized format for watershed delineation results
across all tools (FLOWFINDER, TauDEM, GRASS, WhiteboxTools).

The standardized format ensures:
- Consistent data structures for analysis
- Comparable metrics across tools
- Research-grade reproducibility
- Easy integration with analysis workflows

Author: FLOWFINDER Multi-Tool Team
License: MIT
Version: 1.0.0
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import numpy as np

# Optional imports for geometry processing
try:
    from shapely.geometry import Polygon, mapping, shape
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    
    # Mock functions for testing without shapely
    def mapping(obj):
        return obj
    
    def shape(obj):
        return obj
    
    def make_valid(obj):
        return obj


class ToolName(Enum):
    """Enumeration of supported watershed delineation tools."""
    FLOWFINDER = "flowfinder"
    TAUDEM = "taudem"
    GRASS = "grass"
    WHITEBOX = "whitebox"


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


@dataclass
class WatershedGeometry:
    """Standardized watershed geometry representation."""
    geometry: Dict[str, Any]  # GeoJSON geometry
    area_km2: float
    perimeter_km: float
    centroid_lat: float
    centroid_lon: float
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    is_valid: bool
    geometry_type: str  # "Polygon" or "MultiPolygon"
    
    @classmethod
    def from_shapely_polygon(cls, polygon, source_crs: str = "EPSG:4326") -> "WatershedGeometry":
        """Create WatershedGeometry from Shapely polygon."""
        if not HAS_SHAPELY:
            raise ImportError("Shapely is required for geometry processing")
            
        # Ensure polygon is valid
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        
        # Calculate metrics
        centroid = polygon.centroid
        bounds = polygon.bounds
        
        # Convert to appropriate units (assuming WGS84 for area calculation)
        # For precise area calculation, would need projection to equal-area CRS
        area_deg2 = polygon.area
        area_km2 = area_deg2 * 111.32 * 111.32  # Rough conversion from deg^2 to km^2
        
        perimeter_deg = polygon.length
        perimeter_km = perimeter_deg * 111.32  # Rough conversion from degrees to km
        
        return cls(
            geometry=mapping(polygon),
            area_km2=area_km2,
            perimeter_km=perimeter_km,
            centroid_lat=centroid.y,
            centroid_lon=centroid.x,
            bbox=[bounds[0], bounds[1], bounds[2], bounds[3]],
            is_valid=polygon.is_valid,
            geometry_type="MultiPolygon" if polygon.geom_type == "MultiPolygon" else "Polygon"
        )


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics."""
    runtime_seconds: float
    peak_memory_mb: Optional[float]
    cpu_usage_percent: Optional[float]
    io_operations: Optional[int]
    timeout_seconds: float
    exceeded_timeout: bool
    
    # Algorithm-specific metrics
    algorithm_steps: List[str]
    processing_stages: Dict[str, float]  # stage_name -> time_seconds
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0-1, higher is better)."""
        if self.exceeded_timeout:
            return 0.0
        
        # Simple efficiency metric: faster is better
        efficiency = max(0.0, 1.0 - (self.runtime_seconds / self.timeout_seconds))
        return min(1.0, efficiency)


@dataclass
class QualityMetrics:
    """Standardized quality assessment metrics."""
    # Geometric quality
    shape_complexity: float  # Perimeter^2 / Area ratio
    compactness_ratio: float  # 4π * Area / Perimeter^2
    convexity_ratio: float  # Area / Convex Hull Area
    
    # Topological quality
    has_holes: bool
    num_holes: int
    self_intersections: int
    
    # Hydrological quality
    drainage_density: Optional[float]  # Stream length / Area
    relief_ratio: Optional[float]  # Elevation range / Basin length
    
    @classmethod
    def calculate_from_geometry(cls, geometry: WatershedGeometry) -> "QualityMetrics":
        """Calculate quality metrics from watershed geometry."""
        area = geometry.area_km2
        perimeter = geometry.perimeter_km
        
        # Calculate shape metrics
        shape_complexity = (perimeter ** 2) / area if area > 0 else float('inf')
        compactness_ratio = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        # Convexity would require convex hull calculation
        convexity_ratio = 1.0  # Placeholder
        
        # Topology metrics (simplified)
        has_holes = geometry.geometry_type == "MultiPolygon"
        num_holes = 0  # Would need detailed analysis
        self_intersections = 0 if geometry.is_valid else 1
        
        return cls(
            shape_complexity=shape_complexity,
            compactness_ratio=compactness_ratio,
            convexity_ratio=convexity_ratio,
            has_holes=has_holes,
            num_holes=num_holes,
            self_intersections=self_intersections,
            drainage_density=None,  # Would require stream network data
            relief_ratio=None  # Would require elevation data
        )


@dataclass
class ToolSpecificData:
    """Tool-specific metadata and parameters."""
    tool_name: ToolName
    tool_version: Optional[str]
    algorithm_used: str
    parameters: Dict[str, Any]
    command_executed: List[str]
    output_files: List[str]
    workflow_steps: List[str]
    error_messages: List[str]
    warnings: List[str]


@dataclass
class StandardizedWatershedResult:
    """
    Standardized watershed delineation result format.
    
    This format is consistent across all tools and provides comprehensive
    information for analysis, comparison, and research applications.
    """
    # Identification
    result_id: str
    timestamp: str
    
    # Input parameters
    pour_point_lat: float
    pour_point_lon: float
    input_crs: str
    output_crs: str
    
    # Processing status
    status: ProcessingStatus
    success: bool
    
    # Results
    geometry: Optional[WatershedGeometry]
    performance: PerformanceMetrics
    quality: Optional[QualityMetrics]
    
    # Tool information
    tool_data: ToolSpecificData
    
    # Environment context
    environment: str  # development/testing/production
    configuration_hash: str
    
    # Optional comparison data
    comparison_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result_dict = asdict(self)
        
        # Convert enums to strings
        result_dict['status'] = self.status.value
        result_dict['tool_data']['tool_name'] = self.tool_data.tool_name.value
        
        return result_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, output_path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        with open(output_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_tool_output(cls, tool_name: str, tool_output: Dict[str, Any], 
                        performance_data: Dict[str, Any], pour_point: tuple,
                        environment: str, config_hash: str) -> "StandardizedWatershedResult":
        """Create standardized result from tool-specific output."""
        
        # Generate unique result ID
        result_id = f"{tool_name}_{int(time.time() * 1000)}_{hash(str(pour_point)) % 10000}"
        
        # Parse geometry if available
        geometry = None
        quality = None
        
        if tool_output.get('geometry'):
            try:
                if HAS_SHAPELY:
                    shapely_geom = shape(tool_output['geometry'])
                    geometry = WatershedGeometry.from_shapely_polygon(shapely_geom)
                    quality = QualityMetrics.calculate_from_geometry(geometry)
                else:
                    # Create geometry without shapely for testing
                    geom_data = tool_output['geometry']
                    if geom_data.get('coordinates'):
                        coords = geom_data['coordinates'][0]
                        if len(coords) >= 4:
                            # Calculate rough metrics from coordinates
                            lons = [c[0] for c in coords]
                            lats = [c[1] for c in coords]
                            area_deg2 = (max(lons) - min(lons)) * (max(lats) - min(lats))
                            area_km2 = area_deg2 * 111.32 * 111.32
                            perimeter_km = 2 * (max(lons) - min(lons) + max(lats) - min(lats)) * 111.32
                            
                            geometry = WatershedGeometry(
                                geometry=geom_data,
                                area_km2=area_km2,
                                perimeter_km=perimeter_km,
                                centroid_lat=sum(lats) / len(lats),
                                centroid_lon=sum(lons) / len(lons),
                                bbox=[min(lons), min(lats), max(lons), max(lats)],
                                is_valid=True,
                                geometry_type=geom_data.get('type', 'Polygon')
                            )
                            quality = QualityMetrics.calculate_from_geometry(geometry)
            except Exception as e:
                print(f"Error processing geometry: {e}")
        
        # Determine status
        if tool_output.get('error'):
            status = ProcessingStatus.FAILED
            success = False
        elif performance_data.get('exceeded_timeout', False):
            status = ProcessingStatus.TIMEOUT
            success = False
        elif geometry is None:
            status = ProcessingStatus.PARTIAL
            success = False
        else:
            status = ProcessingStatus.SUCCESS
            success = True
        
        # Create performance metrics
        performance = PerformanceMetrics(
            runtime_seconds=performance_data.get('runtime_seconds', 0.0),
            peak_memory_mb=performance_data.get('peak_memory_mb'),
            cpu_usage_percent=performance_data.get('cpu_usage_percent'),
            io_operations=performance_data.get('io_operations'),
            timeout_seconds=performance_data.get('timeout_seconds', 120),
            exceeded_timeout=performance_data.get('exceeded_timeout', False),
            algorithm_steps=tool_output.get('workflow', '').split(' -> ') if tool_output.get('workflow') else [],
            processing_stages=performance_data.get('stages', {})
        )
        
        # Create tool-specific data
        tool_data = ToolSpecificData(
            tool_name=ToolName(tool_name.lower()),
            tool_version=tool_output.get('tool_version'),
            algorithm_used=tool_output.get('algorithm', 'unknown'),
            parameters=tool_output.get('parameters', {}),
            command_executed=tool_output.get('command', []),
            output_files=tool_output.get('output_files', []),
            workflow_steps=tool_output.get('workflow', '').split(' -> ') if tool_output.get('workflow') else [],
            error_messages=[tool_output.get('error')] if tool_output.get('error') else [],
            warnings=tool_output.get('warnings', [])
        )
        
        return cls(
            result_id=result_id,
            timestamp=datetime.now().isoformat(),
            pour_point_lat=pour_point[0],
            pour_point_lon=pour_point[1],
            input_crs="EPSG:4326",
            output_crs="EPSG:4326",
            status=status,
            success=success,
            geometry=geometry,
            performance=performance,
            quality=quality,
            tool_data=tool_data,
            environment=environment,
            configuration_hash=config_hash
        )


@dataclass
class MultiToolComparisonResult:
    """
    Results from comparing multiple tools on the same watershed.
    """
    comparison_id: str
    timestamp: str
    pour_point_lat: float
    pour_point_lon: float
    environment: str
    
    # Individual tool results
    tool_results: Dict[str, StandardizedWatershedResult]
    
    # Comparison metrics
    iou_matrix: Dict[str, Dict[str, float]]  # Intersection over Union between tools
    centroid_distances: Dict[str, Dict[str, float]]  # Distance between centroids
    area_ratios: Dict[str, Dict[str, float]]  # Area ratios between tools
    runtime_comparison: Dict[str, float]  # Tool -> runtime
    
    # Statistical summary
    consensus_geometry: Optional[WatershedGeometry]  # Average/consensus polygon
    agreement_score: float  # Overall agreement (0-1)
    best_performing_tool: Optional[str]
    most_accurate_tool: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result_dict = asdict(self)
        
        # Convert nested results
        result_dict['tool_results'] = {
            tool: result.to_dict() for tool, result in self.tool_results.items()
        }
        
        return result_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, output_path: Union[str, Path]) -> None:
        """Save comparison result to JSON file."""
        with open(output_path, 'w') as f:
            f.write(self.to_json())


def calculate_iou(geom1: WatershedGeometry, geom2: WatershedGeometry) -> float:
    """Calculate Intersection over Union between two watershed geometries."""
    try:
        if HAS_SHAPELY:
            poly1 = shape(geom1.geometry)
            poly2 = shape(geom2.geometry)
            
            intersection = poly1.intersection(poly2)
            union = poly1.union(poly2)
            
            if union.area == 0:
                return 0.0
            
            return intersection.area / union.area
        else:
            # Simple overlap calculation for testing without shapely
            # Calculate bounding box overlap as approximation
            bbox1 = geom1.bbox
            bbox2 = geom2.bbox
            
            # Calculate intersection of bounding boxes
            x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
            y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
            intersection_area = x_overlap * y_overlap
            
            # Calculate union of bounding boxes
            x_span = max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0])
            y_span = max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1])
            union_area = x_span * y_span
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
            
    except Exception:
        return 0.0


def calculate_centroid_distance(geom1: WatershedGeometry, geom2: WatershedGeometry) -> float:
    """Calculate distance between watershed centroids in km."""
    # Haversine distance calculation
    lat1, lon1 = geom1.centroid_lat, geom1.centroid_lon
    lat2, lon2 = geom2.centroid_lat, geom2.centroid_lon
    
    R = 6371  # Earth radius in km
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def create_multi_tool_comparison(tool_results: Dict[str, StandardizedWatershedResult], 
                               pour_point: tuple, environment: str) -> MultiToolComparisonResult:
    """Create a multi-tool comparison result from individual tool results."""
    
    comparison_id = f"comparison_{int(time.time() * 1000)}"
    
    # Calculate pairwise comparisons
    tool_names = list(tool_results.keys())
    iou_matrix = {}
    centroid_distances = {}
    area_ratios = {}
    runtime_comparison = {}
    
    for tool1 in tool_names:
        iou_matrix[tool1] = {}
        centroid_distances[tool1] = {}
        area_ratios[tool1] = {}
        runtime_comparison[tool1] = tool_results[tool1].performance.runtime_seconds
        
        for tool2 in tool_names:
            if tool1 == tool2:
                iou_matrix[tool1][tool2] = 1.0
                centroid_distances[tool1][tool2] = 0.0
                area_ratios[tool1][tool2] = 1.0
            else:
                geom1 = tool_results[tool1].geometry
                geom2 = tool_results[tool2].geometry
                
                if geom1 and geom2:
                    iou_matrix[tool1][tool2] = calculate_iou(geom1, geom2)
                    centroid_distances[tool1][tool2] = calculate_centroid_distance(geom1, geom2)
                    area_ratios[tool1][tool2] = geom1.area_km2 / geom2.area_km2 if geom2.area_km2 > 0 else 0.0
                else:
                    iou_matrix[tool1][tool2] = 0.0
                    centroid_distances[tool1][tool2] = float('inf')
                    area_ratios[tool1][tool2] = 0.0
    
    # Calculate agreement score (average IOU)
    iou_values = []
    for tool1 in tool_names:
        for tool2 in tool_names:
            if tool1 != tool2:
                iou_values.append(iou_matrix[tool1][tool2])
    
    agreement_score = np.mean(iou_values) if iou_values else 0.0
    
    # Determine best performing tool (fastest successful)
    successful_tools = [tool for tool, result in tool_results.items() if result.success]
    best_performing_tool = None
    if successful_tools:
        best_performing_tool = min(successful_tools, 
                                 key=lambda t: tool_results[t].performance.runtime_seconds)
    
    # Most accurate tool (highest average IOU with others)
    most_accurate_tool = None
    if len(successful_tools) > 1:
        avg_ious = {}
        for tool in successful_tools:
            other_tools = [t for t in successful_tools if t != tool]
            avg_ious[tool] = np.mean([iou_matrix[tool][other] for other in other_tools])
        most_accurate_tool = max(avg_ious.keys(), key=lambda t: avg_ious[t])
    
    return MultiToolComparisonResult(
        comparison_id=comparison_id,
        timestamp=datetime.now().isoformat(),
        pour_point_lat=pour_point[0],
        pour_point_lon=pour_point[1],
        environment=environment,
        tool_results=tool_results,
        iou_matrix=iou_matrix,
        centroid_distances=centroid_distances,
        area_ratios=area_ratios,
        runtime_comparison=runtime_comparison,
        consensus_geometry=None,  # Would need more sophisticated analysis
        agreement_score=agreement_score,
        best_performing_tool=best_performing_tool,
        most_accurate_tool=most_accurate_tool
    )


# Example usage and testing
if __name__ == "__main__":
    print("Standardized Result Format for Multi-Tool Watershed Delineation")
    print("=" * 60)
    
    # Example of creating a standardized result
    from shapely.geometry import Polygon
    
    # Create example polygon
    example_polygon = Polygon([
        (-105.5, 40.0), (-105.4, 40.0), (-105.4, 40.1), (-105.5, 40.1), (-105.5, 40.0)
    ])
    
    # Example tool output
    tool_output = {
        'geometry': mapping(example_polygon),
        'tool': 'flowfinder',
        'workflow': 'flow_direction -> flow_accumulation -> watershed_extraction',
        'algorithm': 'd8'
    }
    
    # Example performance data
    performance_data = {
        'runtime_seconds': 15.5,
        'peak_memory_mb': 256.0,
        'timeout_seconds': 120,
        'exceeded_timeout': False
    }
    
    # Create standardized result
    result = StandardizedWatershedResult.from_tool_output(
        tool_name='flowfinder',
        tool_output=tool_output,
        performance_data=performance_data,
        pour_point=(40.0, -105.5),
        environment='development',
        config_hash='abc123'
    )
    
    print(f"Result ID: {result.result_id}")
    print(f"Status: {result.status.value}")
    print(f"Success: {result.success}")
    if result.geometry:
        print(f"Area: {result.geometry.area_km2:.2f} km²")
        print(f"Perimeter: {result.geometry.perimeter_km:.2f} km")
    print(f"Runtime: {result.performance.runtime_seconds:.1f}s")
    print(f"Efficiency: {result.performance.efficiency_score:.2f}")
    
    print("\n✅ Standardized result format working correctly!")