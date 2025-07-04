"""
FLOWFINDER Watershed Extraction
==============================

Implements watershed boundary extraction from flow accumulation data.
Extracts watershed polygons from pour points using flow tracing.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List
import rasterio

from .exceptions import WatershedError


class WatershedExtractor:
    """
    Extract watershed boundaries from flow accumulation data.

    This class implements watershed extraction algorithms that trace
    upstream from pour points to determine watershed boundaries.

    Attributes:
        flow_accumulation_calc (FlowAccumulationCalculator): Flow accumulation calculator
        stream_threshold (int): Threshold for stream identification
        logger (logging.Logger): Logger instance
    """

    def __init__(self, flow_accumulation_calc, stream_threshold: int = 1000):
        """
        Initialize watershed extractor.

        Args:
            flow_accumulation_calc: FlowAccumulationCalculator instance
            stream_threshold: Threshold for stream identification (cells)
        """
        self.flow_accumulation_calc = flow_accumulation_calc
        self.stream_threshold = stream_threshold
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Watershed extractor initialized with stream threshold: {stream_threshold}"
        )

    def extract_watershed(self, row: int, col: int) -> np.ndarray:
        """
        Extract watershed boundary from pour point.

        Args:
            row, col: Pour point coordinates

        Returns:
            Array of watershed cell coordinates

        Raises:
            WatershedError: If watershed extraction fails
        """
        try:
            # Ensure flow accumulation is calculated
            if self.flow_accumulation_calc.flow_accumulation is None:
                self.flow_accumulation_calc.calculate()

            # Get watershed cells using flow accumulation
            watershed_cells = self.flow_accumulation_calc.get_watershed_cells(row, col)

            if not watershed_cells:
                raise WatershedError(
                    f"No watershed cells found for pour point ({row}, {col})"
                )

            # Convert to numpy array format
            watershed_array = np.array(watershed_cells).T

            self.logger.info(f"Extracted watershed with {len(watershed_cells)} cells")
            return watershed_array

        except Exception as e:
            raise WatershedError(f"Watershed extraction failed: {e}")

    def extract_watershed_with_streams(
        self, row: int, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract watershed boundary and stream network from pour point.

        Args:
            row, col: Pour point coordinates

        Returns:
            Tuple of (watershed_cells, stream_cells)

        Raises:
            WatershedError: If watershed extraction fails
        """
        try:
            # Extract basic watershed
            watershed_cells = self.extract_watershed(row, col)

            # Extract stream network within watershed
            stream_cells = self._extract_stream_network(watershed_cells)

            return watershed_cells, stream_cells

        except Exception as e:
            raise WatershedError(f"Watershed and stream extraction failed: {e}")

    def _extract_stream_network(self, watershed_cells: np.ndarray) -> np.ndarray:
        """
        Extract stream network within watershed.

        Args:
            watershed_cells: Watershed cell coordinates

        Returns:
            Stream cell coordinates
        """
        if self.flow_accumulation_calc.flow_accumulation is None:
            return np.array([])

        flow_accum = self.flow_accumulation_calc.flow_accumulation
        stream_cells = []

        # Find cells with flow accumulation above threshold
        for i in range(len(watershed_cells[0])):
            row, col = watershed_cells[0][i], watershed_cells[1][i]

            if (
                0 <= row < flow_accum.shape[0]
                and 0 <= col < flow_accum.shape[1]
                and flow_accum[row, col] >= self.stream_threshold
            ):
                stream_cells.append((row, col))

        return np.array(stream_cells).T if stream_cells else np.array([])

    def extract_watershed_boundary(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Extract watershed boundary as a list of boundary cells.

        Args:
            row, col: Pour point coordinates

        Returns:
            List of boundary cell coordinates
        """
        try:
            # Get watershed cells
            watershed_cells = self.extract_watershed(row, col)

            if len(watershed_cells) == 0:
                return []

            # Convert to set for efficient lookup
            watershed_set = set(zip(watershed_cells[0], watershed_cells[1]))

            # Find boundary cells
            boundary_cells = []

            for i in range(len(watershed_cells[0])):
                cell_row, cell_col = watershed_cells[0][i], watershed_cells[1][i]

                # Check if this cell is on the boundary
                if self._is_boundary_cell(cell_row, cell_col, watershed_set):
                    boundary_cells.append((cell_row, cell_col))

            self.logger.info(f"Extracted {len(boundary_cells)} boundary cells")
            return boundary_cells

        except Exception as e:
            self.logger.error(f"Boundary extraction failed: {e}")
            return []

    def _is_boundary_cell(self, row: int, col: int, watershed_set: set) -> bool:
        """
        Check if a cell is on the watershed boundary.

        Args:
            row, col: Cell coordinates
            watershed_set: Set of watershed cell coordinates

        Returns:
            True if cell is on boundary
        """
        # Check 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                neighbor = (row + di, col + dj)
                if neighbor not in watershed_set:
                    return True

        return False

    def extract_watershed_polygon(
        self, row: int, col: int
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Extract watershed boundary as a polygon.

        Args:
            row, col: Pour point coordinates

        Returns:
            List of polygon vertices (lon, lat) or None if failed
        """
        try:
            # Get boundary cells
            boundary_cells = self.extract_watershed_boundary(row, col)

            if not boundary_cells:
                return None

            # Convert boundary cells to polygon vertices
            vertices = self._cells_to_polygon_vertices(boundary_cells)

            return vertices

        except Exception as e:
            self.logger.error(f"Polygon extraction failed: {e}")
            return None

    def _cells_to_polygon_vertices(
        self, boundary_cells: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """
        Convert boundary cells to polygon vertices.

        Args:
            boundary_cells: List of boundary cell coordinates

        Returns:
            List of polygon vertices (lon, lat)
        """
        if not boundary_cells:
            return []

        # Get DEM transform for coordinate conversion
        dem_data = self.flow_accumulation_calc.flow_direction_calc.dem_data
        if dem_data is None:
            return []

        transform = dem_data.transform

        # Convert cell coordinates to geographic coordinates
        vertices = []
        for row, col in boundary_cells:
            lon, lat = rasterio.transform.xy(transform, row, col)
            vertices.append((lon, lat))

        # Sort vertices to form a proper polygon
        # This is a simplified approach - in practice, you'd want more sophisticated
        # polygon construction algorithms
        vertices = self._sort_vertices_for_polygon(vertices)

        return vertices

    def _sort_vertices_for_polygon(
        self, vertices: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Sort vertices to form a proper polygon.

        Args:
            vertices: List of vertex coordinates

        Returns:
            Sorted vertices
        """
        if len(vertices) < 3:
            return vertices

        # Simple approach: sort by angle from centroid
        # Calculate centroid
        centroid_lon = sum(v[0] for v in vertices) / len(vertices)
        centroid_lat = sum(v[1] for v in vertices) / len(vertices)

        # Sort by angle from centroid
        def angle_from_centroid(vertex):
            lon, lat = vertex
            return np.arctan2(lat - centroid_lat, lon - centroid_lon)

        sorted_vertices = sorted(vertices, key=angle_from_centroid)

        return sorted_vertices

    def calculate_watershed_area(self, row: int, col: int) -> float:
        """
        Calculate watershed area in square kilometers.

        Args:
            row, col: Pour point coordinates

        Returns:
            Watershed area in km²
        """
        try:
            # Get watershed cells
            watershed_cells = self.extract_watershed(row, col)

            if len(watershed_cells) == 0:
                return 0.0

            # Get cell size from DEM
            dem_data = self.flow_accumulation_calc.flow_direction_calc.dem_data
            if dem_data is None:
                return 0.0

            # Calculate cell area in square meters
            cell_size = dem_data.res[0]  # meters
            cell_area_m2 = cell_size * cell_size

            # Calculate total area
            num_cells = len(watershed_cells[0])
            total_area_m2 = num_cells * cell_area_m2

            # Convert to square kilometers
            total_area_km2 = total_area_m2 / 1_000_000

            return total_area_km2

        except Exception as e:
            self.logger.error(f"Area calculation failed: {e}")
            return 0.0

    def calculate_watershed_perimeter(self, row: int, col: int) -> float:
        """
        Calculate watershed perimeter in kilometers.

        Args:
            row, col: Pour point coordinates

        Returns:
            Watershed perimeter in km
        """
        try:
            # Get boundary cells
            boundary_cells = self.extract_watershed_boundary(row, col)

            if not boundary_cells:
                return 0.0

            # Get cell size from DEM
            dem_data = self.flow_accumulation_calc.flow_direction_calc.dem_data
            if dem_data is None:
                return 0.0

            cell_size = dem_data.res[0]  # meters

            # Calculate perimeter
            perimeter_m = len(boundary_cells) * cell_size

            # Convert to kilometers
            perimeter_km = perimeter_m / 1000

            return perimeter_km

        except Exception as e:
            self.logger.error(f"Perimeter calculation failed: {e}")
            return 0.0
