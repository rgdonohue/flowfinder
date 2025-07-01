#!/usr/bin/env python3
"""
Test the standardized result format system without external dependencies.
"""

import json
import sys
from pathlib import Path

# Add config to path
sys.path.append('config')

def test_result_format_structure():
    """Test the basic structure of the standardized result format."""
    print("=== Testing Standardized Result Format ===")
    
    try:
        from standardized_results import (
            ToolName, ProcessingStatus, WatershedGeometry, 
            PerformanceMetrics, QualityMetrics, ToolSpecificData,
            StandardizedWatershedResult, MultiToolComparisonResult
        )
        
        print("✅ All classes imported successfully")
        
        # Test enum values
        tools = [tool.value for tool in ToolName]
        statuses = [status.value for status in ProcessingStatus]
        
        print(f"✅ Supported tools: {', '.join(tools)}")
        print(f"✅ Processing statuses: {', '.join(statuses)}")
        
        # Test basic data structures
        print("\n--- Testing Data Structures ---")
        
        # Mock watershed geometry
        mock_geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-105.5, 40.0], [-105.4, 40.0], [-105.4, 40.1], [-105.5, 40.1], [-105.5, 40.0]
            ]]
        }
        
        geometry = WatershedGeometry(
            geometry=mock_geometry,
            area_km2=15.5,
            perimeter_km=20.2,
            centroid_lat=40.05,
            centroid_lon=-105.45,
            bbox=[-105.5, 40.0, -105.4, 40.1],
            is_valid=True,
            geometry_type="Polygon"
        )
        print(f"✅ WatershedGeometry: {geometry.area_km2} km², centroid: ({geometry.centroid_lat}, {geometry.centroid_lon})")
        
        # Test performance metrics
        performance = PerformanceMetrics(
            runtime_seconds=25.5,
            peak_memory_mb=128.0,
            cpu_usage_percent=75.0,
            io_operations=150,
            timeout_seconds=120,
            exceeded_timeout=False,
            algorithm_steps=["flow_direction", "flow_accumulation", "watershed_extraction"],
            processing_stages={"flow_direction": 8.2, "flow_accumulation": 12.1, "watershed_extraction": 5.2}
        )
        print(f"✅ PerformanceMetrics: {performance.runtime_seconds}s, efficiency: {performance.efficiency_score:.2f}")
        
        # Test quality metrics
        quality = QualityMetrics.calculate_from_geometry(geometry)
        print(f"✅ QualityMetrics: compactness: {quality.compactness_ratio:.3f}, complexity: {quality.shape_complexity:.1f}")
        
        # Test tool-specific data
        tool_data = ToolSpecificData(
            tool_name=ToolName.FLOWFINDER,
            tool_version="1.0.0",
            algorithm_used="d8",
            parameters={"threshold": 1000, "method": "d8"},
            command_executed=["flowfinder", "delineate", "--lat", "40.0", "--lon", "-105.5"],
            output_files=["watershed.geojson"],
            workflow_steps=["flow_direction", "flow_accumulation", "watershed_extraction"],
            error_messages=[],
            warnings=[]
        )
        print(f"✅ ToolSpecificData: {tool_data.tool_name.value}, algorithm: {tool_data.algorithm_used}")
        
        # Test complete standardized result
        result = StandardizedWatershedResult(
            result_id="test_result_123",
            timestamp="2024-01-01T12:00:00",
            pour_point_lat=40.0,
            pour_point_lon=-105.5,
            input_crs="EPSG:4326",
            output_crs="EPSG:4326",
            status=ProcessingStatus.SUCCESS,
            success=True,
            geometry=geometry,
            performance=performance,
            quality=quality,
            tool_data=tool_data,
            environment="development",
            configuration_hash="abc123def456"
        )
        
        print(f"✅ StandardizedWatershedResult: {result.result_id}, status: {result.status.value}")
        
        # Test JSON serialization
        result_dict = result.to_dict()
        result_json = result.to_json()
        
        print(f"✅ JSON serialization: {len(result_json)} characters")
        
        # Test key structure
        expected_keys = ['result_id', 'timestamp', 'pour_point_lat', 'pour_point_lon', 
                        'status', 'success', 'geometry', 'performance', 'quality', 
                        'tool_data', 'environment', 'configuration_hash']
        
        missing_keys = [key for key in expected_keys if key not in result_dict]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
        else:
            print("✅ All expected keys present in result dictionary")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing result format: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_tool_comparison():
    """Test multi-tool comparison functionality."""
    print("\n=== Testing Multi-Tool Comparison ===")
    
    try:
        from standardized_results import (
            StandardizedWatershedResult, MultiToolComparisonResult,
            create_multi_tool_comparison, calculate_iou, calculate_centroid_distance,
            ToolName, ProcessingStatus, WatershedGeometry, PerformanceMetrics, 
            QualityMetrics, ToolSpecificData
        )
        
        # Create mock results for multiple tools
        tools = ['flowfinder', 'taudem', 'grass', 'whitebox']
        tool_results = {}
        
        for i, tool in enumerate(tools):
            # Create slightly different geometries for each tool
            mock_geometry = {
                "type": "Polygon", 
                "coordinates": [[
                    [-105.5 - i*0.01, 40.0], [-105.4 - i*0.01, 40.0], 
                    [-105.4 - i*0.01, 40.1], [-105.5 - i*0.01, 40.1], [-105.5 - i*0.01, 40.0]
                ]]
            }
            
            geometry = WatershedGeometry(
                geometry=mock_geometry,
                area_km2=15.5 + i*2.0,  # Slightly different areas
                perimeter_km=20.2 + i*1.0,
                centroid_lat=40.05,
                centroid_lon=-105.45 - i*0.01,
                bbox=[-105.5 - i*0.01, 40.0, -105.4 - i*0.01, 40.1],
                is_valid=True,
                geometry_type="Polygon"
            )
            
            performance = PerformanceMetrics(
                runtime_seconds=20.0 + i*10.0,  # Different runtimes
                peak_memory_mb=128.0 + i*64.0,
                cpu_usage_percent=75.0,
                io_operations=150,
                timeout_seconds=120,
                exceeded_timeout=False,
                algorithm_steps=[f"{tool}_step1", f"{tool}_step2"],
                processing_stages={f"{tool}_stage": 10.0 + i*5.0}
            )
            
            quality = QualityMetrics.calculate_from_geometry(geometry)
            
            tool_data = ToolSpecificData(
                tool_name=ToolName(tool),
                tool_version="1.0.0",
                algorithm_used="d8" if tool != "grass" else "r.watershed",
                parameters={"threshold": 1000},
                command_executed=[tool, "delineate"],
                output_files=[f"{tool}_watershed.geojson"],
                workflow_steps=[f"{tool}_workflow"],
                error_messages=[],
                warnings=[]
            )
            
            result = StandardizedWatershedResult(
                result_id=f"{tool}_result_123",
                timestamp="2024-01-01T12:00:00",
                pour_point_lat=40.0,
                pour_point_lon=-105.5,
                input_crs="EPSG:4326",
                output_crs="EPSG:4326",
                status=ProcessingStatus.SUCCESS,
                success=True,
                geometry=geometry,
                performance=performance,
                quality=quality,
                tool_data=tool_data,
                environment="testing",
                configuration_hash="test123"
            )
            
            tool_results[tool] = result
        
        print(f"✅ Created mock results for {len(tool_results)} tools")
        
        # Test comparison creation
        comparison = create_multi_tool_comparison(
            tool_results=tool_results,
            pour_point=(40.0, -105.5),
            environment="testing"
        )
        
        print(f"✅ Multi-tool comparison created: {comparison.comparison_id}")
        print(f"   Agreement score: {comparison.agreement_score:.3f}")
        print(f"   Best performing tool: {comparison.best_performing_tool}")
        print(f"   Most accurate tool: {comparison.most_accurate_tool}")
        
        # Test comparison metrics
        print("\n--- Runtime Comparison ---")
        for tool, runtime in comparison.runtime_comparison.items():
            print(f"   {tool}: {runtime:.1f}s")
        
        # Test IOU matrix structure
        print("\n--- IOU Matrix Structure ---")
        for tool1 in tools:
            for tool2 in tools:
                iou = comparison.iou_matrix[tool1][tool2]
                if tool1 == tool2:
                    assert iou == 1.0, f"Self-IOU should be 1.0, got {iou}"
                print(f"   {tool1}-{tool2}: {iou:.3f}")
        
        print("✅ IOU matrix structure correct")
        
        # Test JSON serialization
        comparison_json = comparison.to_json()
        print(f"✅ Comparison JSON serialization: {len(comparison_json)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-tool comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all standardized result format tests."""
    print("FLOWFINDER Standardized Result Format Test")
    print("=" * 45)
    
    success = True
    success &= test_result_format_structure()
    success &= test_multi_tool_comparison()
    
    print("\n" + "=" * 45)
    if success:
        print("✅ All standardized result format tests passed!")
        print("\nStandardized format features:")
        print("  • Consistent data structures across all tools")
        print("  • Comprehensive performance and quality metrics") 
        print("  • Multi-tool comparison capabilities")
        print("  • JSON serialization for analysis workflows")
        print("  • Research-grade reproducibility support")
    else:
        print("❌ Some tests failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())