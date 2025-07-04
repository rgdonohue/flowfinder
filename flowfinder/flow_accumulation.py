"""
FLOWFINDER Flow Accumulation Calculation
=======================================

Implements flow accumulation calculation for watershed delineation.
Determines the number of cells that drain to each point in the DEM.
"""

import logging
import numpy as np
from typing import Optional

from .exceptions import DEMError
from .optimized_algorithms import OptimizedFlowAccumulation


class FlowAccumulationCalculator:
    """
    Calculate flow accumulation from flow direction data.

    This class implements flow accumulation algorithms that determine
    how many cells drain to each point in the DEM.

    Attributes:
        flow_direction_calc (FlowDirectionCalculator): Flow direction calculator
        flow_accumulation (np.ndarray): Calculated flow accumulation array
        logger (logging.Logger): Logger instance
    """

    def __init__(self, flow_direction_calc):
        """
        Initialize flow accumulation calculator.

        Args:
            flow_direction_calc: FlowDirectionCalculator instance
        """
        self.flow_direction_calc = flow_direction_calc
        self.logger = logging.getLogger(__name__)

        # Initialize flow accumulation array
        self.flow_accumulation: Optional[np.ndarray] = None

        # Initialize optimized algorithm
        self.optimized_calculator = OptimizedFlowAccumulation(self.logger)

        self.logger.info("Flow accumulation calculator initialized")

    def calculate(self) -> np.ndarray:
        """
        Calculate flow accumulation from flow direction.

        Returns:
            Flow accumulation array with same shape as DEM

        Raises:
            DEMError: If calculation fails
        """
        try:
            # Ensure flow direction is calculated
            if self.flow_direction_calc.flow_direction is None:
                self.flow_direction_calc.calculate()

            flow_dir = self.flow_direction_calc.flow_direction

            # Use optimized algorithm instead of recursive approach
            self.flow_accumulation = (
                self.optimized_calculator.calculate_flow_accumulation(
                    flow_dir, nodata_value=0
                )
            )

            self.logger.info("Flow accumulation calculation completed")
            return self.flow_accumulation

        except Exception as e:
            raise DEMError(f"Flow accumulation calculation failed: {e}")

    def _calculate_flow_accumulation_recursive(self, flow_dir: np.ndarray) -> None:
        """
        Calculate flow accumulation using recursive approach.

        Args:
            flow_dir: Flow direction array
        """
        height, width = flow_dir.shape

        # Create a visited mask to avoid infinite recursion
        visited = np.zeros((height, width), dtype=bool)

        # Calculate accumulation for each cell
        for i in range(height):
            for j in range(width):
                if not visited[i, j] and flow_dir[i, j] != 0:
                    self._accumulate_cell(i, j, flow_dir, visited)

    def _accumulate_cell(
        self, row: int, col: int, flow_dir: np.ndarray, visited: np.ndarray
    ) -> int:
        """
        Recursively calculate accumulation for a single cell.

        Args:
            row, col: Cell coordinates
            flow_dir: Flow direction array
            visited: Visited cells mask

        Returns:
            Accumulation value for the cell
        """
        height, width = flow_dir.shape

        # Check bounds
        if not (0 <= row < height and 0 <= col < width):
            return 0

        # If already visited, return cached value
        if visited[row, col]:
            return self.flow_accumulation[row, col]

        # Mark as visited
        visited[row, col] = True

        # If no flow direction, accumulation is 1 (just this cell)
        if flow_dir[row, col] == 0:
            self.flow_accumulation[row, col] = 1
            return 1

        # Get downstream cell
        downstream = self.flow_direction_calc.get_downstream_cell(row, col)

        if downstream is None:
            # No downstream cell, accumulation is 1
            self.flow_accumulation[row, col] = 1
            return 1

        # Recursively calculate accumulation for downstream cell
        downstream_accumulation = self._accumulate_cell(
            downstream[0], downstream[1], flow_dir, visited
        )

        # This cell's accumulation = downstream accumulation + 1 (this cell)
        total_accumulation = downstream_accumulation + 1
        self.flow_accumulation[row, col] = total_accumulation

        return total_accumulation

    def _calculate_flow_accumulation_iterative(self, flow_dir: np.ndarray) -> None:
        """
        Calculate flow accumulation using iterative approach.

        This is an alternative implementation that might be more efficient
        for large datasets.

        Args:
            flow_dir: Flow direction array
        """
        height, width = flow_dir.shape

        # Initialize accumulation array
        self.flow_accumulation = np.ones((height, width), dtype=np.int32)

        # Create a queue for cells to process
        from collections import deque

        queue = deque()

        # Find cells with no upstream flow (sources)
        for i in range(height):
            for j in range(width):
                if flow_dir[i, j] != 0:
                    # Check if this cell has any upstream cells flowing to it
                    has_upstream = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < height
                                and 0 <= nj < width
                                and flow_dir[ni, nj] != 0
                            ):
                                downstream = (
                                    self.flow_direction_calc.get_downstream_cell(ni, nj)
                                )
                                if downstream == (i, j):
                                    has_upstream = True
                                    break
                        if has_upstream:
                            break

                    if not has_upstream:
                        queue.append((i, j))

        # Process cells in topological order
        while queue:
            i, j = queue.popleft()

            # Get downstream cell
            downstream = self.flow_direction_calc.get_downstream_cell(i, j)

            if downstream is not None:
                di, dj = downstream

                # Add this cell's accumulation to downstream cell
                self.flow_accumulation[di, dj] += self.flow_accumulation[i, j]

                # Check if downstream cell is ready to process
                ready = True
                for ni in range(height):
                    for nj in range(width):
                        if flow_dir[ni, nj] != 0:
                            downstream_check = (
                                self.flow_direction_calc.get_downstream_cell(ni, nj)
                            )
                            if downstream_check == (di, dj):
                                # Check if upstream cell has been processed
                                if self.flow_accumulation[ni, nj] == 1:
                                    ready = False
                                    break
                    if not ready:
                        break

                if ready and (di, dj) not in queue:
                    queue.append((di, dj))

    def get_flow_accumulation(self) -> Optional[np.ndarray]:
        """Get calculated flow accumulation array."""
        return self.flow_accumulation

    def get_upstream_cells(self, row: int, col: int) -> list:
        """
        Get all upstream cells for a given cell.

        Args:
            row, col: Cell coordinates

        Returns:
            List of upstream cell coordinates
        """
        if self.flow_direction_calc.flow_direction is None:
            return []

        height, width = self.flow_direction_calc.flow_direction.shape

        if not (0 <= row < height and 0 <= col < width):
            return []

        upstream_cells = []

        # Check all 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = row + di, col + dj

                if 0 <= ni < height and 0 <= nj < width:
                    # Check if this neighbor flows to the target cell
                    downstream = self.flow_direction_calc.get_downstream_cell(ni, nj)
                    if downstream == (row, col):
                        upstream_cells.append((ni, nj))

        return upstream_cells

    def get_watershed_cells(self, row: int, col: int) -> list:
        """
        Get all cells in the watershed upstream of a given cell.

        Args:
            row, col: Pour point coordinates

        Returns:
            List of all watershed cell coordinates
        """
        if self.flow_accumulation is None:
            return []

        height, width = self.flow_accumulation.shape

        if not (0 <= row < height and 0 <= col < width):
            return []

        watershed_cells = []
        visited = set()

        def trace_upstream(r: int, c: int):
            """Recursively trace upstream from a cell."""
            if (r, c) in visited:
                return

            visited.add((r, c))
            watershed_cells.append((r, c))

            # Get upstream cells
            upstream = self.get_upstream_cells(r, c)
            for ur, uc in upstream:
                trace_upstream(ur, uc)

        # Start tracing from the pour point
        trace_upstream(row, col)

        return watershed_cells
