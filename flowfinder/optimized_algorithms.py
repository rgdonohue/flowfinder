"""
FLOWFINDER Optimized Algorithms
==============================

High-performance implementations of core watershed delineation algorithms.
Replaces O(n²) implementations with optimized O(n log n) or O(n) approaches.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import heapq
from collections import deque
from scipy import ndimage
from skimage import morphology

from .exceptions import DEMError, PerformanceError


class OptimizedDepressionFilling:
    """
    Optimized depression filling using priority-flood algorithm.
    
    Replaces the nested-loop O(n²) approach with an efficient O(n log n) algorithm
    that provides convergence guarantees and handles complex topologies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize optimized depression filling."""
        self.logger = logger or logging.getLogger(__name__)
    
    def fill_depressions(self, dem_array: np.ndarray, nodata_value: float) -> np.ndarray:
        """
        Fill depressions using priority-flood algorithm.
        
        Args:
            dem_array: DEM elevation array
            nodata_value: No-data value in DEM
            
        Returns:
            DEM array with depressions filled
            
        Raises:
            DEMError: If filling fails
        """
        try:
            start_time = __import__('time').time()
            
            # Create working copy
            filled_dem = dem_array.copy().astype(np.float64)
            height, width = dem_array.shape
            
            # Create mask for valid data
            valid_mask = dem_array != nodata_value
            
            if not np.any(valid_mask):
                raise DEMError("No valid data in DEM")
            
            # Initialize priority queue with border cells
            pq = []  # Priority queue: (elevation, row, col)
            visited = np.zeros((height, width), dtype=bool)
            
            # Add all border cells to priority queue
            for i in range(height):
                for j in range(width):
                    if valid_mask[i, j] and self._is_border_cell(i, j, height, width, valid_mask):
                        heapq.heappush(pq, (filled_dem[i, j], i, j))
                        visited[i, j] = True
            
            self.logger.info(f"Initialized priority queue with {len(pq)} border cells")
            
            # 8-directional neighbors
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                        (0, 1), (1, -1), (1, 0), (1, 1)]
            
            processed_cells = 0
            
            # Process cells in elevation order
            while pq:
                current_elev, row, col = heapq.heappop(pq)
                processed_cells += 1
                
                # Process all neighbors
                for dr, dc in neighbors:
                    new_row, new_col = row + dr, col + dc
                    
                    # Check bounds and validity
                    if (0 <= new_row < height and 0 <= new_col < width and
                        valid_mask[new_row, new_col] and not visited[new_row, new_col]):
                        
                        # Ensure neighbor is at least as high as current cell
                        neighbor_elev = filled_dem[new_row, new_col]
                        filled_elev = max(neighbor_elev, current_elev)
                        filled_dem[new_row, new_col] = filled_elev
                        
                        # Add to priority queue
                        heapq.heappush(pq, (filled_elev, new_row, new_col))
                        visited[new_row, new_col] = True
            
            runtime = __import__('time').time() - start_time
            
            self.logger.info(
                f"Depression filling completed in {runtime:.2f}s, "
                f"processed {processed_cells} cells"
            )
            
            return filled_dem
            
        except Exception as e:
            raise DEMError(f"Optimized depression filling failed: {e}")
    
    def _is_border_cell(self, row: int, col: int, height: int, width: int, 
                       valid_mask: np.ndarray) -> bool:
        """
        Check if cell is on the border of valid data.
        
        Args:
            row, col: Cell coordinates
            height, width: Array dimensions
            valid_mask: Mask of valid data cells
            
        Returns:
            True if cell is on border
        """
        # Edge of array
        if row == 0 or row == height - 1 or col == 0 or col == width - 1:
            return True
        
        # Adjacent to no-data cell
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                    (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in neighbors:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < height and 0 <= new_col < width and
                not valid_mask[new_row, new_col]):
                return True
        
        return False


class OptimizedFlowAccumulation:
    """
    Optimized flow accumulation using topological sorting.
    
    Replaces recursive approach with iterative algorithm that scales linearly
    with the number of cells and avoids stack overflow issues.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize optimized flow accumulation."""
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_flow_accumulation(self, flow_direction: np.ndarray, 
                                  nodata_value: int = 0) -> np.ndarray:
        """
        Calculate flow accumulation using topological sorting.
        
        Args:
            flow_direction: Flow direction array (D8 codes 1-8)
            nodata_value: No-data value in flow direction
            
        Returns:
            Flow accumulation array
            
        Raises:
            DEMError: If calculation fails
        """
        try:
            start_time = __import__('time').time()
            
            height, width = flow_direction.shape
            
            # Initialize accumulation array
            flow_accum = np.ones((height, width), dtype=np.int32)
            
            # Create valid data mask
            valid_mask = flow_direction != nodata_value
            
            # Build flow network
            upstream_count, downstream_map = self._build_flow_network(
                flow_direction, valid_mask
            )
            
            # Topological sort using Kahn's algorithm
            flow_accum = self._topological_accumulation(
                flow_direction, flow_accum, upstream_count, downstream_map, valid_mask
            )
            
            runtime = __import__('time').time() - start_time
            
            self.logger.info(
                f"Flow accumulation completed in {runtime:.2f}s"
            )
            
            return flow_accum
            
        except Exception as e:
            raise DEMError(f"Optimized flow accumulation failed: {e}")
    
    def _build_flow_network(self, flow_direction: np.ndarray, 
                           valid_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Build flow network for topological sorting.
        
        Args:
            flow_direction: Flow direction array
            valid_mask: Valid data mask
            
        Returns:
            Tuple of (upstream_count, downstream_map)
        """
        height, width = flow_direction.shape
        
        # Count upstream cells for each cell
        upstream_count = np.zeros((height, width), dtype=np.int32)
        
        # Map downstream connections
        downstream_map = {}
        
        # D8 direction offsets
        d8_offsets = {
            1: (-1, 0),   # North
            2: (-1, 1),   # Northeast
            3: (0, 1),    # East
            4: (1, 1),    # Southeast
            5: (1, 0),    # South
            6: (1, -1),   # Southwest
            7: (0, -1),   # West
            8: (-1, -1)   # Northwest
        }
        
        # Build network
        for i in range(height):
            for j in range(width):
                if not valid_mask[i, j]:
                    continue
                
                flow_code = flow_direction[i, j]
                if flow_code in d8_offsets:
                    di, dj = d8_offsets[flow_code]
                    downstream_i, downstream_j = i + di, j + dj
                    
                    # Check bounds
                    if (0 <= downstream_i < height and 0 <= downstream_j < width and
                        valid_mask[downstream_i, downstream_j]):
                        
                        # Record downstream connection
                        if (i, j) not in downstream_map:
                            downstream_map[(i, j)] = []
                        downstream_map[(i, j)].append((downstream_i, downstream_j))
                        
                        # Increment upstream count for downstream cell
                        upstream_count[downstream_i, downstream_j] += 1
        
        return upstream_count, downstream_map
    
    def _topological_accumulation(self, flow_direction: np.ndarray, 
                                 flow_accum: np.ndarray,
                                 upstream_count: np.ndarray,
                                 downstream_map: Dict,
                                 valid_mask: np.ndarray) -> np.ndarray:
        """
        Perform topological accumulation.
        
        Args:
            flow_direction: Flow direction array
            flow_accum: Flow accumulation array (modified in place)
            upstream_count: Upstream cell count array
            downstream_map: Downstream connections map
            valid_mask: Valid data mask
            
        Returns:
            Updated flow accumulation array
        """
        height, width = flow_direction.shape
        
        # Initialize queue with cells that have no upstream flow
        queue = deque()
        
        for i in range(height):
            for j in range(width):
                if valid_mask[i, j] and upstream_count[i, j] == 0:
                    queue.append((i, j))
        
        processed_cells = 0
        
        # Process cells in topological order
        while queue:
            current_i, current_j = queue.popleft()
            processed_cells += 1
            
            # Get downstream cells
            if (current_i, current_j) in downstream_map:
                for downstream_i, downstream_j in downstream_map[(current_i, current_j)]:
                    # Add current cell's accumulation to downstream cell
                    flow_accum[downstream_i, downstream_j] += flow_accum[current_i, current_j]
                    
                    # Decrease upstream count for downstream cell
                    upstream_count[downstream_i, downstream_j] -= 1
                    
                    # If downstream cell has no more upstream cells, add to queue
                    if upstream_count[downstream_i, downstream_j] == 0:
                        queue.append((downstream_i, downstream_j))
        
        self.logger.info(f"Processed {processed_cells} cells in topological order")
        
        return flow_accum


class OptimizedPolygonCreation:
    """
    Optimized polygon creation from watershed pixels.
    
    Replaces simple bounding box fallback with proper polygon creation
    using morphological operations and contour tracing.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize optimized polygon creation."""
        self.logger = logger or logging.getLogger(__name__)
    
    def pixels_to_polygon(self, watershed_pixels: np.ndarray, 
                         transform: Any, simplify_tolerance: float = 0.0001) -> List[Tuple[float, float]]:
        """
        Convert watershed pixels to polygon coordinates.
        
        Args:
            watershed_pixels: Array of watershed pixel coordinates
            transform: Rasterio affine transform
            simplify_tolerance: Tolerance for polygon simplification
            
        Returns:
            List of polygon coordinates
            
        Raises:
            DEMError: If polygon creation fails
        """
        try:
            if len(watershed_pixels) == 0 or watershed_pixels.shape[1] == 0:
                return []
            
            # Get array bounds
            rows = watershed_pixels[0]
            cols = watershed_pixels[1]
            
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            # Create binary mask
            mask_height = max_row - min_row + 3  # Add padding
            mask_width = max_col - min_col + 3
            
            binary_mask = np.zeros((mask_height, mask_width), dtype=bool)
            
            # Fill mask with watershed pixels
            for i in range(len(rows)):
                mask_row = rows[i] - min_row + 1  # Add 1 for padding
                mask_col = cols[i] - min_col + 1
                binary_mask[mask_row, mask_col] = True
            
            # Apply morphological operations to clean up mask
            binary_mask = self._clean_binary_mask(binary_mask)
            
            # Extract boundary using contour tracing
            boundary_coords = self._extract_boundary_coordinates(
                binary_mask, min_row - 1, min_col - 1, transform
            )
            
            # Simplify polygon if requested
            if simplify_tolerance > 0 and len(boundary_coords) > 3:
                boundary_coords = self._simplify_polygon(boundary_coords, simplify_tolerance)
            
            return boundary_coords
            
        except Exception as e:
            self.logger.warning(f"Optimized polygon creation failed: {e}")
            # Fallback to simple bounding box
            return self._create_bounding_box_polygon(watershed_pixels, transform)
    
    def _clean_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Clean binary mask using morphological operations.
        
        Args:
            binary_mask: Input binary mask
            
        Returns:
            Cleaned binary mask
        """
        # Fill small holes
        filled_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Remove small objects
        cleaned_mask = morphology.remove_small_objects(filled_mask, min_size=4)
        
        # Smooth boundaries
        smoothed_mask = morphology.binary_closing(
            cleaned_mask, morphology.disk(1)
        )
        
        return smoothed_mask
    
    def _extract_boundary_coordinates(self, binary_mask: np.ndarray,
                                    row_offset: int, col_offset: int,
                                    transform: Any) -> List[Tuple[float, float]]:
        """
        Extract boundary coordinates from binary mask.
        
        Args:
            binary_mask: Binary mask of watershed
            row_offset: Row offset for coordinate transformation
            col_offset: Column offset for coordinate transformation
            transform: Rasterio affine transform
            
        Returns:
            List of boundary coordinates
        """
        try:
            from skimage import measure
            
            # Find contours
            contours = measure.find_contours(binary_mask.astype(float), 0.5)
            
            if not contours:
                return []
            
            # Use the longest contour
            longest_contour = max(contours, key=len)
            
            # Convert contour coordinates to geographic coordinates
            coords = []
            for point in longest_contour:
                row = point[0] + row_offset
                col = point[1] + col_offset
                
                # Convert to geographic coordinates
                x, y = transform * (col, row)
                coords.append((x, y))
            
            # Close polygon if needed
            if len(coords) > 2 and coords[0] != coords[-1]:
                coords.append(coords[0])
            
            return coords
            
        except Exception as e:
            self.logger.warning(f"Boundary extraction failed: {e}")
            return []
    
    def _simplify_polygon(self, coords: List[Tuple[float, float]], 
                         tolerance: float) -> List[Tuple[float, float]]:
        """
        Simplify polygon using Douglas-Peucker algorithm.
        
        Args:
            coords: Input coordinates
            tolerance: Simplification tolerance
            
        Returns:
            Simplified coordinates
        """
        try:
            from shapely.geometry import Polygon
            
            if len(coords) < 4:  # Need at least 4 points for a polygon
                return coords
            
            # Create Shapely polygon
            polygon = Polygon(coords)
            
            # Simplify polygon
            simplified = polygon.simplify(tolerance, preserve_topology=True)
            
            # Extract coordinates
            if hasattr(simplified, 'exterior'):
                return list(simplified.exterior.coords)
            else:
                return coords
                
        except Exception as e:
            self.logger.warning(f"Polygon simplification failed: {e}")
            return coords
    
    def _create_bounding_box_polygon(self, watershed_pixels: np.ndarray,
                                   transform: Any) -> List[Tuple[float, float]]:
        """
        Create bounding box polygon as fallback.
        
        Args:
            watershed_pixels: Watershed pixel coordinates
            transform: Rasterio affine transform
            
        Returns:
            Bounding box coordinates
        """
        if len(watershed_pixels) == 0 or watershed_pixels.shape[1] == 0:
            return []
        
        rows = watershed_pixels[0]
        cols = watershed_pixels[1]
        
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        
        # Convert corners to geographic coordinates
        corners = [
            (min_row, min_col),
            (min_row, max_col),
            (max_row, max_col),
            (max_row, min_col),
            (min_row, min_col)  # Close polygon
        ]
        
        coords = []
        for row, col in corners:
            x, y = transform * (col, row)
            coords.append((x, y))
        
        return coords