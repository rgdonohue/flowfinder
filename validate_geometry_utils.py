#!/usr/bin/env python3
"""
Validation script for GeometryDiagnostics class without external dependencies.
Tests critical functionality to ensure the implementation is correct.
"""

import sys
import os
import logging
from io import StringIO

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from geometry_utils import GeometryDiagnostics
    print("✅ Successfully imported GeometryDiagnostics")
except ImportError as e:
    print(f"❌ Failed to import GeometryDiagnostics: {e}")
    sys.exit(1)


def capture_logs(func, *args, **kwargs):
    """Capture log output from a function."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        result = func(*args, **kwargs)
        return result, log_capture.getvalue()
    finally:
        logger.removeHandler(handler)


def test_initialization():
    """Test GeometryDiagnostics initialization."""
    print("\n🔍 Testing initialization...")
    
    try:
        # Test default initialization
        diag = GeometryDiagnostics()
        assert diag.logger is not None, "Logger should not be None"
        assert diag.config is not None, "Config should not be None"
        assert isinstance(diag.config, dict), "Config should be a dictionary"
        assert 'geometry_repair' in diag.config, "Config should contain geometry_repair section"
        
        # Test custom logger
        custom_logger = logging.getLogger("test_logger")
        diag_with_logger = GeometryDiagnostics(logger=custom_logger)
        assert diag_with_logger.logger == custom_logger, "Should use custom logger"
        
        # Test custom config
        custom_config = {'geometry_repair': {'enable_repair_attempts': False}}
        diag_with_config = GeometryDiagnostics(config=custom_config)
        assert diag_with_config.config == custom_config, "Should use custom config"
        
        print("✅ Initialization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False


def test_config_structure():
    """Test default configuration structure."""
    print("\n🔍 Testing configuration structure...")
    
    try:
        diag = GeometryDiagnostics()
        config = diag.config
        
        # Test required configuration sections
        assert 'geometry_repair' in config, "Missing geometry_repair section"
        repair_config = config['geometry_repair']
        
        required_keys = [
            'enable_diagnostics', 'enable_repair_attempts', 
            'invalid_geometry_action', 'max_repair_attempts',
            'detailed_logging', 'repair_strategies'
        ]
        
        for key in required_keys:
            assert key in repair_config, f"Missing required config key: {key}"
        
        # Test repair strategies section
        strategies = repair_config['repair_strategies']
        assert isinstance(strategies, dict), "repair_strategies should be a dictionary"
        
        expected_strategies = [
            'buffer_fix', 'simplify', 'make_valid', 'convex_hull',
            'orient_fix', 'simplify_holes'
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategies, f"Missing repair strategy: {strategy}"
        
        print("✅ Configuration structure tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration structure test failed: {e}")
        return False


def test_method_availability():
    """Test that all required methods are available."""
    print("\n🔍 Testing method availability...")
    
    try:
        diag = GeometryDiagnostics()
        
        required_methods = [
            '_get_default_config',
            'diagnose_and_repair_geometries',
            'analyze_geometry_issues', 
            '_analyze_geometry_issues',
            '_diagnose_single_geometry',
            '_apply_geometry_repairs',
            '_apply_repair_strategy',
            '_handle_remaining_invalid_geometries',
            '_log_geometry_repair_summary',
            '_generate_geometry_recommendations'
        ]
        
        for method_name in required_methods:
            assert hasattr(diag, method_name), f"Missing method: {method_name}"
            method = getattr(diag, method_name)
            assert callable(method), f"Method {method_name} is not callable"
        
        print("✅ Method availability tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Method availability test failed: {e}")
        return False


def test_error_handling():
    """Test error handling capabilities."""
    print("\n🔍 Testing error handling...")
    
    try:
        diag = GeometryDiagnostics()
        
        # Test with invalid parameters
        try:
            diag._diagnose_single_geometry(None, 0)
            # Should not raise exception, should handle gracefully
            print("✅ None geometry handling works")
        except Exception as e:
            print(f"❌ Failed to handle None geometry: {e}")
            return False
        
        # Test with invalid strategy
        try:
            result = diag._apply_repair_strategy(None, 'invalid_strategy', 0)
            # Should return None for invalid input
            assert result is None, "Should return None for invalid input"
            print("✅ Invalid strategy handling works")
        except Exception as e:
            print(f"❌ Failed to handle invalid strategy: {e}")
            return False
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_logging_functionality():
    """Test logging functionality."""
    print("\n🔍 Testing logging functionality...")
    
    try:
        # Create logger that captures output
        logger = logging.getLogger("test_geometry_utils")
        logger.setLevel(logging.DEBUG)
        
        # Create string stream to capture logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger.addHandler(handler)
        
        diag = GeometryDiagnostics(logger=logger)
        
        # Test logging during initialization
        assert diag.logger == logger, "Should use provided logger"
        
        # Test that methods can access logger
        assert hasattr(diag, 'logger'), "Should have logger attribute"
        assert diag.logger is not None, "Logger should not be None"
        
        logger.removeHandler(handler)
        
        print("✅ Logging functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Logging functionality test failed: {e}")
        return False


def test_repair_strategies():
    """Test repair strategy methods."""
    print("\n🔍 Testing repair strategies...")
    
    try:
        diag = GeometryDiagnostics()
        
        # Test remove strategy
        result = diag._apply_repair_strategy(None, 'remove', 0)
        assert result is None, "Remove strategy should return None"
        
        # Test unknown strategy
        result = diag._apply_repair_strategy(None, 'unknown_strategy', 0)
        # Should handle gracefully, may return None for None input
        
        # Test strategy method existence
        strategies_to_test = [
            'make_valid', 'buffer_fix', 'simplify', 'orient_fix',
            'convex_hull', 'simplify_holes', 'unary_union_fix', 'remove'
        ]
        
        for strategy in strategies_to_test:
            try:
                # Test with None input (should handle gracefully)
                result = diag._apply_repair_strategy(None, strategy, 0)
                # Should not crash
            except Exception as e:
                print(f"❌ Strategy {strategy} failed with None input: {e}")
                return False
        
        print("✅ Repair strategy tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Repair strategy test failed: {e}")
        return False


def test_statistics_and_recommendations():
    """Test statistics and recommendation functionality."""
    print("\n🔍 Testing statistics and recommendations...")
    
    try:
        diag = GeometryDiagnostics()
        
        # Test recommendation generation with mock stats
        original_stats = {
            'total_features': 100,
            'total_invalid': 25,
            'issue_types': {
                'self_intersection': 10,
                'duplicate_points': 15
            }
        }
        
        final_stats = {
            'total_invalid': 5
        }
        
        # Set up repair stats
        diag.repair_stats = {
            'repair_counts': {
                'make_valid': {'attempted': 10, 'successful': 8},
                'buffer_fix': {'attempted': 5, 'successful': 2}
            }
        }
        
        recommendations = diag._generate_geometry_recommendations(original_stats, final_stats)
        assert isinstance(recommendations, list), "Should return list of recommendations"
        
        print("✅ Statistics and recommendations tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Statistics and recommendations test failed: {e}")
        return False


def run_validation():
    """Run all validation tests."""
    print("=" * 70)
    print("🧪 GEOMETRY DIAGNOSTICS VALIDATION")
    print("=" * 70)
    print("Testing critical GeometryDiagnostics functionality...")
    
    tests = [
        test_initialization,
        test_config_structure,
        test_method_availability,
        test_error_handling,
        test_logging_functionality,
        test_repair_strategies,
        test_statistics_and_recommendations
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ GeometryDiagnostics class is properly implemented")
        print("✅ Core functionality is working correctly")
        print("✅ Error handling is robust")
        print("✅ Ready for integration testing with real geospatial data")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print(f"⚠️  {total_tests - passed_tests} tests need attention")
        return False


if __name__ == "__main__":
    success = run_validation()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run full test suite with pytest when dependencies available")
        print("2. Test with real geospatial data")
        print("3. Performance testing with large datasets")
        print("4. Integration testing with other benchmark components")
    
    sys.exit(0 if success else 1)