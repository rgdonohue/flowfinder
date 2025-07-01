"""
FLOWFINDER Flow Direction Calculation
====================================

Implements flow direction calculation algorithms for watershed delineation.
Supports D8, D-infinity, and multiple flow direction methods.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
import rasterio

from .exceptions import DEMError
from .optimized_algorithms import OptimizedDepressionFilling
from .advanced_algorithms import DInfinityFlowDirection, HydrologicEnforcement


class FlowDirectionCalculator:
    """
    Calculate flow direction from DEM data.
    
    This class implements various flow direction algorithms:
    - D8: Single flow direction (8 neighbors)
    - D-infinity: Multiple flow direction with angle-based approach
    - MFD: Multiple flow direction with flow partitioning
    
    Attributes:
        dem_data (rasterio.DatasetReader): DEM dataset
        method (str): Flow direction method ('d8', 'dinf', 'mfd')
        fill_depressions (bool): Whether to fill depressions
        flow_direction (np.ndarray): Calculated flow direction array
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, dem_data: rasterio.DatasetReader, 
                 method: str = 'd8', fill_depressions: bool = True):
        """
        Initialize flow direction calculator.
        
        Args:
            dem_data: DEM dataset
            method: Flow direction method ('d8', 'dinf', 'mfd')
            fill_depressions: Whether to fill depressions in DEM
        """
        self.dem_data = dem_data
        self.method = method.lower()
        self.fill_depressions = fill_depressions
        self.logger = logging.getLogger(__name__)
        
        # Validate method
        if self.method not in ['d8', 'dinf', 'mfd']:
            raise ValueError(f"Unsupported flow direction method: {method}")
        
        # Initialize flow direction array
        self.flow_direction: Optional[np.ndarray] = None
        self.flow_slopes: Optional[np.ndarray] = None
        
        # Initialize optimized algorithms
        self.depression_filler = OptimizedDepressionFilling(self.logger)
        self.dinf_calculator = DInfinityFlowDirection(self.logger)
        self.hydrologic_enforcer = HydrologicEnforcement(self.logger)
        
        # D8 flow direction codes (clockwise from north)
        self.d8_codes = {
            1: (-1, 0),   # North
            2: (-1, 1),   # Northeast
            3: (0, 1),    # East
            4: (1, 1),    # Southeast
            5: (1, 0),    # South
            6: (1, -1),   # Southwest
            7: (0, -1),   # West
            8: (-1, -1)   # Northwest
        }
        
        self.logger.info(f"Flow direction calculator initialized with method: {self.method}")
    
    def calculate(self) -> np.ndarray:
        """
        Calculate flow direction from DEM.
        
        Returns:
            Flow direction array with same shape as DEM
            
        Raises:
            DEMError: If calculation fails
        """
        try:
            # Read DEM data
            dem_array = self.dem_data.read(1)
            
            # Fill depressions if requested
            if self.fill_depressions:
                dem_array = self.depression_filler.fill_depressions(
                    dem_array, self.dem_data.nodata
                )
            
            # Calculate flow direction based on method
            if self.method == 'd8':
                self.flow_direction = self._calculate_d8(dem_array)
            elif self.method == 'dinf':
                self.flow_direction, self.flow_slopes = self.dinf_calculator.calculate_dinf_flow_direction(
                    dem_array, self.dem_data.nodata
                )
            elif self.method == 'mfd':
                self.flow_direction = self._calculate_mfd(dem_array)
            
            # Apply hydrologic enforcement for D8 method
            if self.method == 'd8':
                self.flow_direction = self.hydrologic_enforcer.enforce_drainage_direction(
                    self.flow_direction, dem_array
                )
            
            self.logger.info(f"Flow direction calculated using {self.method} method")
            return self.flow_direction
            
        except Exception as e:
            raise DEMError(f"Flow direction calculation failed: {e}")
    
    def _fill_depressions(self, dem_array: np.ndarray) -> np.ndarray:
        """
        Fill depressions in DEM using simple flooding algorithm.
        
        Args:
            dem_array: DEM elevation array
            
        Returns:
            DEM array with depressions filled
        """
        self.logger.info("Filling depressions in DEM...")
        
        # Simple depression filling: raise depressions to minimum of surrounding cells
        filled_dem = dem_array.copy()
        height, width = dem_array.shape
        
        # Create a mask for no-data values
        nodata_mask = dem_array == self.dem_data.nodata
        
        # Iterative depression filling
        changed = True
        iterations = 0
        max_iterations = 1000  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if nodata_mask[i, j]:
                        continue
                    
                    # Get 8-neighbor minimum
                    neighbors = [
                        filled_dem[i-1, j-1], filled_dem[i-1, j], filled_dem[i-1, j+1],
                        filled_dem[i, j-1], filled_dem[i, j+1],
                        filled_dem[i+1, j-1], filled_dem[i+1, j], filled_dem[i+1, j+1]
                    ]
                    
                    # Filter out no-data values
                    valid_neighbors = [n for n in neighbors if n != self.dem_data.nodata]
                    
                    if valid_neighbors:
                        min_neighbor = min(valid_neighbors)
                        if filled_dem[i, j] < min_neighbor:
                            filled_dem[i, j] = min_neighbor
                            changed = True
        
        if iterations >= max_iterations:
            self.logger.warning("Depression filling reached maximum iterations")
        
        self.logger.info(f"Depression filling completed in {iterations} iterations")
        return filled_dem
    
    def _calculate_d8(self, dem_array: np.ndarray) -> np.ndarray:
        """
        Calculate D8 flow direction.
        
        Args:
            dem_array: DEM elevation array
            
        Returns:
            Flow direction array with D8 codes (1-8)
        """
        height, width = dem_array.shape
        flow_dir = np.zeros((height, width), dtype=np.int32)
        
        # D8 offsets (row, col) for 8 neighbors
        offsets = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if dem_array[i, j] == self.dem_data.nodata:
                    continue
                
                # Find steepest descent direction
                max_slope = -np.inf
                steepest_dir = 0
                
                for k, (di, dj) in enumerate(offsets):
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < height and 0 <= nj < width and 
                        dem_array[ni, nj] != self.dem_data.nodata):
                        
                        # Calculate slope (negative for descent)
                        slope = dem_array[i, j] - dem_array[ni, nj]
                        
                        if slope > max_slope:
                            max_slope = slope
                            steepest_dir = k + 1  # D8 codes are 1-8
                
                flow_dir[i, j] = steepest_dir
        
        return flow_dir
    
    def _calculate_dinf(self, dem_array: np.ndarray) -> np.ndarray:
        """
        Calculate D-infinity flow direction.
        
        Args:
            dem_array: DEM elevation array
            
        Returns:
            Flow direction array with angles (0-360 degrees)
        """
        height, width = dem_array.shape
        flow_dir = np.zeros((height, width), dtype=np.float32)
        
        # For simplicity, implement a basic D-infinity approach
        # In practice, this would be more complex with proper angle calculations
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if dem_array[i, j] == self.dem_data.nodata:
                    continue
                
                # Calculate flow direction as angle
                # This is a simplified implementation
                flow_dir[i, j] = self._calculate_flow_angle(dem_array, i, j)
        
        return flow_dir
    
    def _calculate_mfd(self, dem_array: np.ndarray) -> np.ndarray:
        """
        Calculate multiple flow direction using Freeman (1991) algorithm.
        
        Args:
            dem_array: DEM elevation array
            
        Returns:
            Flow direction array with multiple directions (stored as proportions)
        """
        height, width = dem_array.shape
        
        # For MFD, we need to store flow proportions to each neighbor
        # This is a simplified implementation - store dominant direction
        flow_dir = np.zeros((height, width), dtype=np.int32)
        
        # Calculate MFD using slope-weighted distribution
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if dem_array[i, j] == self.dem_data.nodata:
                    continue
                
                center_elev = dem_array[i, j]
                
                # Calculate slopes to all 8 neighbors
                slopes = []
                directions = []
                
                neighbors = [
                    (-1, 0), (-1, 1), (0, 1), (1, 1),
                    (1, 0), (1, -1), (0, -1), (-1, -1)
                ]
                
                for k, (di, dj) in enumerate(neighbors):
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < height and 0 <= nj < width and
                        dem_array[ni, nj] != self.dem_data.nodata):
                        
                        slope = center_elev - dem_array[ni, nj]
                        if slope > 0:  # Only downhill flow
                            slopes.append(slope)
                            directions.append(k + 1)
                
                # For simplicity, use steepest descent as primary direction
                # Full MFD would distribute flow proportionally
                if slopes:
                    max_idx = np.argmax(slopes)
                    flow_dir[i, j] = directions[max_idx]
        
        return flow_dir
    
    def _calculate_flow_angle(self, dem_array: np.ndarray, i: int, j: int) -> float:
        """
        Calculate flow direction angle for a cell.
        
        Args:
            dem_array: DEM elevation array
            i, j: Cell coordinates
            
        Returns:
            Flow direction angle in degrees
        """
        # Simplified angle calculation
        # In practice, this would use proper finite difference methods
        
        height, width = dem_array.shape
        
        # Calculate gradients in x and y directions
        if j > 0 and j < width - 1:
            dx = (dem_array[i, j+1] - dem_array[i, j-1]) / 2
        else:
            dx = 0
        
        if i > 0 and i < height - 1:
            dy = (dem_array[i-1, j] - dem_array[i+1, j]) / 2
        else:
            dy = 0
        
        # Calculate angle
        if dx == 0 and dy == 0:
            return 0.0
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Convert to 0-360 range
        if angle < 0:
            angle += 360
        
        return angle
    
    def get_flow_direction(self) -> Optional[np.ndarray]:
        """Get calculated flow direction array."""
        return self.flow_direction
    
    def get_downstream_cell(self, row: int, col: int) -> Optional[Tuple[int, int]]:
        """
        Get downstream cell for a given cell.
        
        Args:
            row, col: Cell coordinates
            
        Returns:
            Downstream cell coordinates or None if no flow
        """
        if self.flow_direction is None:
            return None
        
        if not (0 <= row < self.flow_direction.shape[0] and 
                0 <= col < self.flow_direction.shape[1]):
            return None
        
        flow_code = self.flow_direction[row, col]
        
        if flow_code == 0:
            return None
        
        if self.method == 'd8':
            if flow_code in self.d8_codes:
                di, dj = self.d8_codes[flow_code]
                return row + di, col + dj
        
        # For other methods, would need more complex logic
        return None 