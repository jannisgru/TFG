"""
Spatiotemporal Cube (ST-Cube) Segmentation using ST-MRS Algorithm
Implementation based on "A Spatiotemporal Cube Model for Analyzing Satellite Image Time Series" by Wenqiang Xia et al.

This script implements:
1. ST-MRS (Spatiotemporal Multiresolution Segmentation) algorithm
2. ST-Cube extraction and analysis
3. Spatiotemporal heterogeneity calculation
4. STGS (Spatiotemporal Global Score) quality metric
5. Visualization of ST-Cubes in 3D space

The algorithm segments a spatiotemporal image time series into ST-cubes that are:
- Spatially connected (2D footprint)
- Temporally continuous (persistent over time)
- Homogeneous in both space and time
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json

warnings.filterwarnings('ignore')


@dataclass
class STCubeParameters:
    """Parameters for ST-Cube segmentation"""
    sh_threshold: float = 0.05  # Spatial heterogeneity threshold
    th_threshold: float = 0.03  # Temporal heterogeneity threshold
    w_shape: float = 0.3        # Weight for shape vs spectral heterogeneity
    w_compactness: float = 0.6  # Weight for compactness vs smoothness
    w_bands: Dict[str, float] = None  # Band weights
    min_cube_size: int = 5      # Minimum pixels per cube
    max_iterations: int = 100   # Maximum merge iterations
    
    def __post_init__(self):
        if self.w_bands is None:
            self.w_bands = {'RED': 0.2, 'GREEN': 0.2, 'BLUE': 0.2, 'NIR': 0.4}


@dataclass
class STCube:
    """Represents a spatiotemporal cube (memory-efficient version)"""
    id: int
    temporal_extent: Tuple[int, int]  # (start_time, end_time)
    pixels: List[Tuple[int, int]]  # List of (y, x) coordinates
    spectral_signature: np.ndarray  # Mean spectral values over time
    heterogeneity: float
    area: int
    perimeter: float
    compactness: float
    smoothness: float
    temporal_variance: float
    
    @property
    def spatial_extent(self):
        """Calculate spatial extent on demand to save memory"""
        if not self.pixels:
            return None
        min_y = min(p[0] for p in self.pixels)
        max_y = max(p[0] for p in self.pixels)
        min_x = min(p[1] for p in self.pixels)
        max_x = max(p[1] for p in self.pixels)
        return (min_y, max_y, min_x, max_x)


class STCubeSegmentation:
    """ST-MRS algorithm implementation for spatiotemporal cube segmentation"""
    
    def __init__(self, parameters: STCubeParameters = None):
        self.params = parameters or STCubeParameters()
        self.cubes = []
        self.cube_id_counter = 0
        self.segmentation_map = None
        self.data = None
        self.time_coords = None
        self.spatial_coords = None
        
    def load_data(self, ds: xr.Dataset, municipality_id: int = None):
        """Load and prepare data for ST-Cube segmentation"""
        print("Loading data for ST-Cube segmentation...")
        
        # Extract data for specific municipality if provided
        if municipality_id is not None and 'municipality_id' in ds.variables:
            print(f"Filtering data for municipality ID: {municipality_id}")
            if len(ds['municipality_id'].dims) == 3:
                mask = ds['municipality_id'].isel(time=0) == municipality_id
            else:
                mask = ds['municipality_id'] == municipality_id
            
            # Apply mask to all relevant variables
            ds_filtered = ds.where(mask)
        else:
            ds_filtered = ds
            
        # Store data
        self.data = ds_filtered
        self.time_coords = ds_filtered.time.values
        self.spatial_coords = (ds_filtered.y.values, ds_filtered.x.values)
        
        # Initialize segmentation map
        self.segmentation_map = np.zeros((len(self.time_coords), 
                                        len(self.spatial_coords[0]), 
                                        len(self.spatial_coords[1])), dtype=int)
        
        print(f"Data loaded: {len(self.time_coords)} time steps, "
              f"{len(self.spatial_coords[0])} x {len(self.spatial_coords[1])} pixels")
        
    def calculate_spatial_heterogeneity(self, pixels: List[Tuple[int, int]], 
                                      time_range: Tuple[int, int]) -> float:
        """Calculate spatial heterogeneity for a set of pixels (optimized)"""
        if len(pixels) == 0:
            return float('inf')
            
        # Sample pixels and time steps for efficiency
        sample_pixels = pixels[:min(10, len(pixels))]
        sample_time_steps = min(5, time_range[1] - time_range[0] + 1)
        
        # Extract spectral values for sampled pixels and time range
        spectral_values = []
        for t in range(time_range[0], time_range[0] + sample_time_steps):
            for y, x in sample_pixels:
                try:
                    if 'ndvi' in self.data.variables:
                        value = float(self.data['ndvi'].isel(time=t, y=y, x=x).values)
                        if not np.isnan(value):
                            spectral_values.append(value)
                    else:
                        # Use multiple bands if available
                        pixel_values = []
                        for band in ['RED', 'GREEN', 'BLUE', 'NIR']:
                            if band in self.data.variables:
                                val = float(self.data[band].isel(time=t, y=y, x=x).values)
                                if not np.isnan(val):
                                    pixel_values.append(val * self.params.w_bands.get(band, 1.0))
                        if pixel_values:
                            spectral_values.append(np.mean(pixel_values))
                except (IndexError, ValueError):
                    continue
        
        if len(spectral_values) < 2:
            return 1.0  # Return reasonable default instead of inf
            
        # Spectral variance
        spectral_variance = np.var(spectral_values)
        
        # Simplified shape heterogeneity
        shape_hetero = self.calculate_shape_heterogeneity(sample_pixels)
        
        # Combine spatial heterogeneity
        spatial_hetero = ((1 - self.params.w_shape) * spectral_variance + 
                         self.params.w_shape * shape_hetero)
        
        return spatial_hetero
    
    def calculate_shape_heterogeneity(self, pixels: List[Tuple[int, int]]) -> float:
        """Calculate shape heterogeneity (compactness and smoothness)"""
        if len(pixels) < 2:
            return 0.0
            
        # Create binary mask for the region
        min_y = min(p[0] for p in pixels)
        max_y = max(p[0] for p in pixels)
        min_x = min(p[1] for p in pixels)
        max_x = max(p[1] for p in pixels)
        
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        if height <= 0 or width <= 0:
            return 0.0
            
        mask = np.zeros((height, width), dtype=bool)
        for y, x in pixels:
            mask[y - min_y, x - min_x] = True
        
        # Calculate area and perimeter
        area = np.sum(mask)
        if area == 0:
            return float('inf')
            
        # Calculate perimeter using edge detection
        edges = ndimage.binary_erosion(mask) ^ mask
        perimeter = np.sum(edges)
        
        if perimeter == 0:
            return 0.0
            
        # Compactness: perimeter / sqrt(area)
        compactness = perimeter / np.sqrt(area)
        
        # Smoothness: perimeter / bounding box perimeter
        bbox_perimeter = 2 * (height + width)
        smoothness = perimeter / bbox_perimeter if bbox_perimeter > 0 else 0
        
        # Combine shape metrics
        shape_hetero = (self.params.w_compactness * compactness + 
                       (1 - self.params.w_compactness) * smoothness)
        
        return shape_hetero
    
    def calculate_temporal_heterogeneity(self, pixels: List[Tuple[int, int]], 
                                       time_range: Tuple[int, int]) -> float:
        """Calculate temporal heterogeneity for a set of pixels (optimized)"""
        if len(pixels) == 0 or time_range[1] <= time_range[0]:
            return 1.0  # Return reasonable default
            
        # Sample pixels for efficiency
        sample_pixels = pixels[:min(5, len(pixels))]
        
        # Extract temporal signatures for sampled pixels
        temporal_signatures = []
        for y, x in sample_pixels:
            signature = []
            for t in range(time_range[0], time_range[1] + 1):
                try:
                    if 'ndvi' in self.data.variables:
                        value = float(self.data['ndvi'].isel(time=t, y=y, x=x).values)
                        if not np.isnan(value):
                            signature.append(value)
                    else:
                        # Use weighted combination of bands
                        pixel_value = 0
                        total_weight = 0
                        for band in ['RED', 'GREEN', 'BLUE', 'NIR']:
                            if band in self.data.variables:
                                val = float(self.data[band].isel(time=t, y=y, x=x).values)
                                weight = self.params.w_bands.get(band, 1.0)
                                if not np.isnan(val):
                                    pixel_value += val * weight
                                    total_weight += weight
                        if total_weight > 0:
                            signature.append(pixel_value / total_weight)
                except (IndexError, ValueError):
                    continue
            
            if len(signature) >= 2:
                temporal_signatures.append(signature)
        
        if len(temporal_signatures) < 2:
            return 1.0  # Return reasonable default
            
        # Calculate temporal variance across all pixels
        all_values = []
        for signature in temporal_signatures:
            all_values.extend(signature)
            
        if len(all_values) < 2:
            return 1.0
            
        temporal_variance = np.var(all_values)
        return temporal_variance
    
    def calculate_heterogeneity_increase(self, cube1: STCube, cube2: STCube) -> float:
        """Calculate the increase in heterogeneity when merging two cubes"""
        # Combined pixels and time range
        combined_pixels = cube1.pixels + cube2.pixels
        combined_time_range = (min(cube1.temporal_extent[0], cube2.temporal_extent[0]),
                             max(cube1.temporal_extent[1], cube2.temporal_extent[1]))
        
        # Calculate combined heterogeneity
        combined_sh = self.calculate_spatial_heterogeneity(combined_pixels, combined_time_range)
        combined_th = self.calculate_temporal_heterogeneity(combined_pixels, combined_time_range)
        
        # Calculate increase
        delta_sh = combined_sh - cube1.heterogeneity - cube2.heterogeneity
        delta_th = combined_th - cube1.temporal_variance - cube2.temporal_variance
        
        return delta_sh + delta_th
    
    def get_spatial_neighbors(self, cube: STCube) -> List[STCube]:
        """Find spatial neighbors of a cube (optimized version)"""
        neighbors = []
        
        # Get bounding box of the cube
        y_coords = [p[0] for p in cube.pixels]
        x_coords = [p[1] for p in cube.pixels]
        
        min_y, max_y = min(y_coords), max(y_coords)
        min_x, max_x = min(x_coords), max(x_coords)
        
        # Expand bounding box to find potential neighbors
        search_margin = 3
        search_min_y = max(0, min_y - search_margin)
        search_max_y = min(len(self.spatial_coords[0]), max_y + search_margin)
        search_min_x = max(0, min_x - search_margin)
        search_max_x = min(len(self.spatial_coords[1]), max_x + search_margin)
        
        # Find cubes that intersect with the search area
        for other_cube in self.cubes:
            if other_cube.id == cube.id:
                continue
                
            # Quick bounding box check
            other_y_coords = [p[0] for p in other_cube.pixels]
            other_x_coords = [p[1] for p in other_cube.pixels]
            
            other_min_y, other_max_y = min(other_y_coords), max(other_y_coords)
            other_min_x, other_max_x = min(other_x_coords), max(other_x_coords)
            
            # Check if bounding boxes are close
            if (other_min_y <= search_max_y and other_max_y >= search_min_y and
                other_min_x <= search_max_x and other_max_x >= search_min_x):
                neighbors.append(other_cube)
        
        # Limit number of neighbors for efficiency
        return neighbors[:10]
    
    def get_temporal_neighbors(self, cube: STCube) -> List[STCube]:
        """Find temporal neighbors of a cube"""
        neighbors = []
        
        for other_cube in self.cubes:
            if other_cube.id != cube.id:
                # Check if cubes are temporally adjacent and spatially overlapping
                if (abs(cube.temporal_extent[1] - other_cube.temporal_extent[0]) <= 1 or
                    abs(other_cube.temporal_extent[1] - cube.temporal_extent[0]) <= 1):
                    # Check spatial overlap
                    if any(pixel in other_cube.pixels for pixel in cube.pixels):
                        neighbors.append(other_cube)
        
        return neighbors
    
    def initialize_cubes(self):
        """Initialize ST-cubes with spatial blocks to reduce memory usage"""
        print("Initializing ST-cubes with spatial blocks...")
        
        self.cubes = []
        self.cube_id_counter = 0
        
        # Use spatial blocks instead of individual pixels to reduce memory
        block_size = 16  # Increased block size to reduce number of cubes
        
        # Get valid data mask from first time step
        if 'ndvi' in self.data.variables:
            valid_mask = ~np.isnan(self.data['ndvi'].isel(time=0).values)
        else:
            valid_mask = np.ones((len(self.spatial_coords[0]), len(self.spatial_coords[1])), dtype=bool)
        
        # Create blocks
        for y_start in range(0, len(self.spatial_coords[0]), block_size):
            for x_start in range(0, len(self.spatial_coords[1]), block_size):
                y_end = min(y_start + block_size, len(self.spatial_coords[0]))
                x_end = min(x_start + block_size, len(self.spatial_coords[1]))
                
                # Check if block has valid data
                block_mask = valid_mask[y_start:y_end, x_start:x_end]
                if not np.any(block_mask):
                    continue
                
                # Create pixels list for this block
                pixels = []
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if valid_mask[y, x]:
                            pixels.append((y, x))
                
                if len(pixels) == 0:
                    continue
                
                # Create cube for entire time series of this spatial block
                time_range = (0, len(self.time_coords) - 1)
                
                # Calculate initial heterogeneity (using sample for efficiency)
                sample_pixels = pixels[:min(5, len(pixels))]  # Sample for efficiency
                spatial_hetero = self.calculate_spatial_heterogeneity(sample_pixels, (0, min(5, len(self.time_coords)-1)))
                temporal_hetero = self.calculate_temporal_heterogeneity(sample_pixels, (0, min(5, len(self.time_coords)-1)))
                
                # Get representative spectral signature
                spectral_signature = []
                if 'ndvi' in self.data.variables:
                    for t in range(0, min(10, len(self.time_coords))):  # Sample first 10 time steps
                        values = []
                        for y, x in sample_pixels:
                            try:
                                value = float(self.data['ndvi'].isel(time=t, y=y, x=x).values)
                                if not np.isnan(value):
                                    values.append(value)
                            except (IndexError, ValueError):
                                continue
                        if values:
                            spectral_signature.append(np.mean(values))
                
                # Create cube (without large spatial extent array)
                cube = STCube(
                    id=self.cube_id_counter,
                    temporal_extent=time_range,
                    pixels=pixels,
                    spectral_signature=np.array(spectral_signature) if spectral_signature else np.array([0]),
                    heterogeneity=spatial_hetero if not np.isinf(spatial_hetero) else 1.0,
                    area=len(pixels),
                    perimeter=4.0 * np.sqrt(len(pixels)),
                    compactness=4.0,
                    smoothness=1.0,
                    temporal_variance=temporal_hetero if not np.isinf(temporal_hetero) else 1.0
                )
                
                self.cubes.append(cube)
                
                # Update segmentation map for all time steps (sample)
                for t in range(0, len(self.time_coords), max(1, len(self.time_coords)//10)):  # Sample time steps
                    for y, x in pixels:
                        try:
                            self.segmentation_map[t, y, x] = self.cube_id_counter
                        except IndexError:
                            continue
                        
                self.cube_id_counter += 1
                
                # Limit number of cubes for memory efficiency
                if len(self.cubes) >= 500:  # Limit to 500 cubes
                    print(f"Reached maximum cube limit (500), stopping initialization")
                    break
            
            if len(self.cubes) >= 500:
                break
        
        print(f"Initialized {len(self.cubes)} ST-cubes from spatial blocks")
    
    def merge_cubes(self, cube1: STCube, cube2: STCube) -> STCube:
        """Merge two ST-cubes into one"""
        # Combine properties
        combined_pixels = cube1.pixels + cube2.pixels
        combined_time_range = (min(cube1.temporal_extent[0], cube2.temporal_extent[0]),
                             max(cube1.temporal_extent[1], cube2.temporal_extent[1]))
        
        # Calculate new heterogeneity
        spatial_hetero = self.calculate_spatial_heterogeneity(combined_pixels, combined_time_range)
        temporal_hetero = self.calculate_temporal_heterogeneity(combined_pixels, combined_time_range)
        
        # Calculate new shape metrics
        area = len(combined_pixels)
        shape_hetero = self.calculate_shape_heterogeneity(combined_pixels)
        
        # Create merged cube (without large spatial extent array)
        merged_cube = STCube(
            id=self.cube_id_counter,
            temporal_extent=combined_time_range,
            pixels=combined_pixels,
            spectral_signature=np.concatenate([cube1.spectral_signature, cube2.spectral_signature]),
            heterogeneity=spatial_hetero,
            area=area,
            perimeter=0.0,  # Will be calculated if needed
            compactness=shape_hetero,
            smoothness=0.0,  # Will be calculated if needed
            temporal_variance=temporal_hetero
        )
        
        self.cube_id_counter += 1
        return merged_cube
    
    def segment_st_cubes(self):
        """Main ST-MRS segmentation algorithm (memory-efficient version)"""
        print("Running ST-MRS segmentation...")
        
        # Initialize cubes
        self.initialize_cubes()
        
        # Limit iterations based on cube count
        max_iterations = min(self.params.max_iterations, len(self.cubes) // 10)
        
        # Iterative merging
        iteration = 0
        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}/{max_iterations}, "
                  f"Current cubes: {len(self.cubes)}")
            
            best_merge = None
            best_increase = float('inf')
            
            # Sample cubes for efficiency (don't check all)
            sample_size = min(50, len(self.cubes))  # Limit to 50 cubes per iteration
            sample_cubes = np.random.choice(self.cubes, size=sample_size, replace=False)
            
            # Find best merge candidate
            for cube in sample_cubes:
                if len(cube.pixels) >= self.params.min_cube_size * 2:
                    continue  # Skip very large cubes
                    
                # Get neighbors (simplified)
                neighbors = self.get_spatial_neighbors(cube)
                
                # Limit neighbors to check
                neighbors = neighbors[:5]  # Only check first 5 neighbors
                
                for neighbor in neighbors:
                    if len(neighbor.pixels) >= self.params.min_cube_size * 2:
                        continue  # Skip merging with large cubes
                        
                    # Calculate heterogeneity increase (simplified)
                    increase = self.calculate_heterogeneity_increase(cube, neighbor)
                    
                    # Check if this is a valid merge
                    if (increase < best_increase and 
                        increase < self.params.sh_threshold + self.params.th_threshold):
                        best_merge = (cube, neighbor)
                        best_increase = increase
            
            # Perform best merge if found
            if best_merge is not None:
                cube1, cube2 = best_merge
                merged_cube = self.merge_cubes(cube1, cube2)
                
                # Remove old cubes and add merged cube
                self.cubes = [c for c in self.cubes if c.id != cube1.id and c.id != cube2.id]
                self.cubes.append(merged_cube)
                
                # Update segmentation map (simplified - only update for current time range)
                for t in range(merged_cube.temporal_extent[0], 
                              min(merged_cube.temporal_extent[1] + 1, len(self.time_coords))):
                    for y, x in merged_cube.pixels:
                        if (0 <= y < len(self.spatial_coords[0]) and 
                            0 <= x < len(self.spatial_coords[1])):
                            self.segmentation_map[t, y, x] = merged_cube.id
            else:
                print("No more valid merges found")
                break
            
            iteration += 1
        
        print(f"Segmentation completed with {len(self.cubes)} ST-cubes")
    
    def calculate_stgs(self) -> float:
        """Calculate Spatiotemporal Global Score (STGS) quality metric"""
        print("Calculating STGS quality metric...")
        
        if len(self.cubes) < 2:
            return 0.0
        
        # Calculate intra-cube variance
        intra_variances = []
        for cube in self.cubes:
            if len(cube.pixels) > 1:
                values = []
                for t in range(cube.temporal_extent[0], cube.temporal_extent[1] + 1):
                    for y, x in cube.pixels:
                        if 'ndvi' in self.data.variables:
                            value = float(self.data['ndvi'].isel(time=t, y=y, x=x).values)
                            if not np.isnan(value):
                                values.append(value)
                
                if len(values) > 1:
                    intra_variances.append(np.var(values))
        
        # Calculate inter-cube variance
        cube_means = []
        for cube in self.cubes:
            values = []
            for t in range(cube.temporal_extent[0], cube.temporal_extent[1] + 1):
                for y, x in cube.pixels:
                    if 'ndvi' in self.data.variables:
                        value = float(self.data['ndvi'].isel(time=t, y=y, x=x).values)
                        if not np.isnan(value):
                            values.append(value)
            
            if values:
                cube_means.append(np.mean(values))
        
        if len(cube_means) > 1:
            inter_variance = np.var(cube_means)
        else:
            inter_variance = 0.0
        
        # Calculate STGS
        if inter_variance > 0 and intra_variances:
            mean_intra_variance = np.mean(intra_variances)
            stgs = 1 - (mean_intra_variance / inter_variance)
        else:
            stgs = 0.0
        
        return max(0.0, min(1.0, stgs))  # Clamp to [0, 1]
    
    def analyze_cubes(self) -> Dict:
        """Analyze ST-cube segmentation results"""
        print("Analyzing ST-cube results...")
        
        # Calculate STGS
        stgs = self.calculate_stgs()
        
        # Basic statistics
        n_cubes = len(self.cubes)
        cube_sizes = [len(cube.pixels) for cube in self.cubes]
        temporal_extents = [cube.temporal_extent[1] - cube.temporal_extent[0] + 1 for cube in self.cubes]
        
        # Heterogeneity statistics
        heterogeneities = [cube.heterogeneity for cube in self.cubes if not np.isinf(cube.heterogeneity)]
        
        analysis = {
            'stgs': stgs,
            'n_cubes': n_cubes,
            'cube_sizes': {
                'mean': np.mean(cube_sizes),
                'std': np.std(cube_sizes),
                'min': np.min(cube_sizes),
                'max': np.max(cube_sizes)
            },
            'temporal_extents': {
                'mean': np.mean(temporal_extents),
                'std': np.std(temporal_extents),
                'min': np.min(temporal_extents),
                'max': np.max(temporal_extents)
            },
            'heterogeneities': {
                'mean': np.mean(heterogeneities) if heterogeneities else 0,
                'std': np.std(heterogeneities) if heterogeneities else 0,
                'min': np.min(heterogeneities) if heterogeneities else 0,
                'max': np.max(heterogeneities) if heterogeneities else 0
            },
            'parameters': {
                'sh_threshold': self.params.sh_threshold,
                'th_threshold': self.params.th_threshold,
                'w_shape': self.params.w_shape,
                'w_compactness': self.params.w_compactness,
                'min_cube_size': self.params.min_cube_size
            }
        }
        
        return analysis


def visualize_st_cubes_3d(segmentation: STCubeSegmentation, 
                         output_dir: str = "outputs/st_cube_analysis"):
    """Create 3D visualization of ST-cubes with RGB base image and single cube evolution"""
    print("Creating 3D ST-cube visualization with RGB base...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select the largest cube for visualization
    largest_cube = max(segmentation.cubes, key=lambda c: len(c.pixels))
    print(f"Visualizing cube {largest_cube.id} with {len(largest_cube.pixels)} pixels")
    
    # Create RGB base image from the first time step
    rgb_image = create_rgb_base_image(segmentation, time_step=0)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Add RGB base image as surface at z=0
    if rgb_image is not None:
        y_range = np.arange(rgb_image.shape[0])
        x_range = np.arange(rgb_image.shape[1])
        
        # Create surface plot for RGB image
        fig.add_trace(go.Surface(
            x=x_range,
            y=y_range,
            z=np.zeros_like(rgb_image[:, :, 0]),  # Base at z=0
            surfacecolor=rgb_image,
            opacity=0.8,
            showscale=False,
            name='RGB Base Image',
            hovertemplate='X: %{x}<br>Y: %{y}<br>RGB Base<extra></extra>'
        ))
    
    # Add the selected cube's evolution over time
    cube_color = 'rgb(255, 100, 100)'  # Red for the main cube
    
    # Group pixels by time step to show evolution
    time_steps = range(largest_cube.temporal_extent[0], largest_cube.temporal_extent[1] + 1)
    
    for t in time_steps:
        # Get NDVI values for this time step to vary marker size
        ndvi_values = []
        for y, x in largest_cube.pixels:
            try:
                if 'ndvi' in segmentation.data.variables:
                    value = float(segmentation.data['ndvi'].isel(time=t, y=y, x=x).values)
                    if not np.isnan(value):
                        ndvi_values.append(value)
                    else:
                        ndvi_values.append(0.3)  # Default for NaN
                else:
                    ndvi_values.append(0.3)  # Default
            except (IndexError, ValueError):
                ndvi_values.append(0.3)
        
        # Normalize NDVI values for marker size (range 3-8)
        if ndvi_values:
            min_ndvi = min(ndvi_values)
            max_ndvi = max(ndvi_values)
            if max_ndvi > min_ndvi:
                marker_sizes = [3 + 5 * (ndvi - min_ndvi) / (max_ndvi - min_ndvi) for ndvi in ndvi_values]
            else:
                marker_sizes = [5] * len(ndvi_values)
        else:
            marker_sizes = [5] * len(largest_cube.pixels)
        
        # Color intensity based on time (darker = earlier, brighter = later)
        time_ratio = (t - largest_cube.temporal_extent[0]) / max(1, largest_cube.temporal_extent[1] - largest_cube.temporal_extent[0])
        alpha = 0.3 + 0.7 * time_ratio  # Opacity from 0.3 to 1.0
        
        # Extract coordinates
        y_coords = [p[0] for p in largest_cube.pixels]
        x_coords = [p[1] for p in largest_cube.pixels]
        z_coords = [t] * len(largest_cube.pixels)
        
        # Add 3D scatter for this time step
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=cube_color,
                opacity=alpha,
                symbol='circle',
                line=dict(width=1, color='darkred')
            ),
            name=f'Time Step {t}',
            showlegend=True,
            hovertemplate=f'Time: {t}<br>X: %{{x}}<br>Y: %{{y}}<br>NDVI: {ndvi_values[0]:.3f}<extra></extra>'
        ))
    
    # Add connecting lines to show temporal evolution
    for i, (y, x) in enumerate(largest_cube.pixels[::5]):  # Sample every 5th pixel to avoid clutter
        z_line = list(range(largest_cube.temporal_extent[0], largest_cube.temporal_extent[1] + 1))
        x_line = [x] * len(z_line)
        y_line = [y] * len(z_line)
        
        fig.add_trace(go.Scatter3d(
            x=x_line,
            y=y_line,
            z=z_line,
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'ST-Cube 3D Evolution - Cube {largest_cube.id} (Sant Martí)',
        scene=dict(
            xaxis_title='X Coordinate (pixels)',
            yaxis_title='Y Coordinate (pixels)',
            zaxis_title='Time Step',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.5)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        height=900,
        width=1200,
        showlegend=True
    )
    
    # Save
    output_file = output_path / "st_cubes_3d_evolution.html"
    fig.write_html(output_file)
    print(f"3D evolution visualization saved: {output_file}")
    
    return fig


def create_rgb_base_image(segmentation: STCubeSegmentation, time_step: int = 0):
    """Create RGB base image from the dataset"""
    print(f"Creating RGB base image for time step {time_step}...")
    
    try:
        # Check if we have RGB bands
        if all(band in segmentation.data.variables for band in ['RED', 'GREEN', 'BLUE']):
            # Extract RGB bands
            red = segmentation.data['RED'].isel(time=time_step).values
            green = segmentation.data['GREEN'].isel(time=time_step).values
            blue = segmentation.data['BLUE'].isel(time=time_step).values
            
            # Handle NaN values
            red = np.nan_to_num(red, nan=0.0)
            green = np.nan_to_num(green, nan=0.0)
            blue = np.nan_to_num(blue, nan=0.0)
            
            # Normalize to 0-255 range (assuming values are in 0-1 range)
            if red.max() <= 1.0:
                red = (red * 255).astype(np.uint8)
                green = (green * 255).astype(np.uint8)
                blue = (blue * 255).astype(np.uint8)
            else:
                # Values might be in 0-10000 range (Landsat scaled)
                red = (red / red.max() * 255).astype(np.uint8)
                green = (green / green.max() * 255).astype(np.uint8)
                blue = (blue / blue.max() * 255).astype(np.uint8)
            
            # Stack into RGB image
            rgb_image = np.stack([red, green, blue], axis=-1)
            
            print(f"RGB image created with shape: {rgb_image.shape}")
            return rgb_image
            
        elif 'ndvi' in segmentation.data.variables:
            # Use NDVI as grayscale if no RGB bands
            ndvi = segmentation.data['ndvi'].isel(time=time_step).values
            ndvi = np.nan_to_num(ndvi, nan=0.0)
            
            # Normalize NDVI to 0-255 range
            ndvi_norm = ((ndvi + 1) / 2 * 255).astype(np.uint8)  # NDVI is -1 to 1
            
            # Create RGB from grayscale NDVI
            rgb_image = np.stack([ndvi_norm, ndvi_norm, ndvi_norm], axis=-1)
            
            print(f"NDVI-based RGB image created with shape: {rgb_image.shape}")
            return rgb_image
            
        else:
            print("No suitable bands found for RGB image creation")
            return None
            
    except Exception as e:
        print(f"Error creating RGB base image: {e}")
        return None


def visualize_st_cubes_time_series(segmentation: STCubeSegmentation, 
                                  output_dir: str = "outputs/st_cube_analysis"):
    """Create time series visualization of ST-cubes"""
    print("Creating ST-cube time series visualization...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select representative cubes
    large_cubes = [cube for cube in segmentation.cubes if len(cube.pixels) >= 10]
    sample_cubes = large_cubes[:12] if len(large_cubes) >= 12 else segmentation.cubes[:12]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f'Cube {cube.id} (n={len(cube.pixels)})' for cube in sample_cubes],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, cube in enumerate(sample_cubes):
        row = i // 4 + 1
        col = i % 4 + 1
        
        # Extract time series for this cube
        time_series = []
        time_indices = []
        
        for t in range(len(segmentation.time_coords)):
            values = []
            for y, x in cube.pixels:
                if 'ndvi' in segmentation.data.variables:
                    value = float(segmentation.data['ndvi'].isel(time=t, y=y, x=x).values)
                    if not np.isnan(value):
                        values.append(value)
            
            if values:
                time_series.append(np.mean(values))
                time_indices.append(t)
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=time_indices,
                y=time_series,
                mode='lines+markers',
                name=f'Cube {cube.id}',
                showlegend=False,
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Highlight cube's temporal extent
        if len(time_series) > 0:
            fig.add_vrect(
                x0=cube.temporal_extent[0],
                x1=cube.temporal_extent[1],
                fillcolor="rgba(255,0,0,0.2)",
                layer="below",
                line_width=0,
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title='ST-Cube Time Series Analysis - Sant Martí Municipality',
        height=900,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time Step")
    fig.update_yaxes(title_text="NDVI")
    
    # Save
    output_file = output_path / "st_cubes_time_series.html"
    fig.write_html(output_file)
    print(f"Time series visualization saved: {output_file}")
    
    return fig


def visualize_segmentation_map(segmentation: STCubeSegmentation, 
                              time_step: int = 0,
                              output_dir: str = "outputs/st_cube_analysis"):
    """Create spatial map of segmentation at specific time step"""
    print(f"Creating segmentation map for time step {time_step}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get segmentation map for specific time step
    seg_map = segmentation.segmentation_map[time_step, :, :]
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=seg_map,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Cube ID"),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Cube ID: %{z}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'ST-Cube Segmentation Map - Time Step {time_step}',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        height=600,
        width=800
    )
    
    # Save
    output_file = output_path / f"segmentation_map_t{time_step}.html"
    fig.write_html(output_file)
    print(f"Segmentation map saved: {output_file}")
    
    return fig


def visualize_multiple_cubes_3d(segmentation: STCubeSegmentation, 
                               output_dir: str = "outputs/st_cube_analysis",
                               n_cubes: int = 4):
    """Create 3D visualization showing multiple cube clusters"""
    print(f"Creating 3D visualization for {n_cubes} largest cubes...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select the largest cubes
    largest_cubes = sorted(segmentation.cubes, key=lambda c: len(c.pixels), reverse=True)[:n_cubes]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Cube {cube.id} ({len(cube.pixels)} pixels)' for cube in largest_cubes],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Color palette for different cubes
    colors = ['rgb(255, 100, 100)', 'rgb(100, 255, 100)', 'rgb(100, 100, 255)', 'rgb(255, 255, 100)']
    
    for i, cube in enumerate(largest_cubes):
        row = i // 2 + 1
        col = i % 2 + 1
        
        cube_color = colors[i % len(colors)]
        
        # Add cube evolution over time
        time_steps = range(cube.temporal_extent[0], cube.temporal_extent[1] + 1)
        
        for t in time_steps:
            # Time-based opacity
            time_ratio = (t - cube.temporal_extent[0]) / max(1, cube.temporal_extent[1] - cube.temporal_extent[0])
            alpha = 0.3 + 0.7 * time_ratio
            
            # Extract coordinates
            y_coords = [p[0] for p in cube.pixels]
            x_coords = [p[1] for p in cube.pixels]
            z_coords = [t] * len(cube.pixels)
            
            # Add 3D scatter for this time step
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=4,
                    color=cube_color,
                    opacity=alpha,
                    symbol='circle'
                ),
                name=f'Cube {cube.id} - T{t}',
                showlegend=(t == cube.temporal_extent[0]),  # Only show legend for first time step
                hovertemplate=f'Cube {cube.id}<br>Time: {t}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>'
            ), row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title='ST-Cube 3D Evolution - Multiple Clusters',
        height=900,
        width=1200,
        showlegend=True
    )
    
    # Update 3D scene properties for all subplots
    for i in range(1, 5):
        fig.update_scenes(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Time Step',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.5))
        )
    
    # Save
    output_file = output_path / "st_cubes_3d_multiple.html"
    fig.write_html(output_file)
    print(f"Multiple cubes 3D visualization saved: {output_file}")
    
    return fig


def create_analysis_report(analysis: Dict, output_dir: str = "outputs/st_cube_analysis"):
    """Create comprehensive analysis report"""
    print("Creating analysis report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save analysis as JSON
    report_file = output_path / "st_cube_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Create text summary
    summary_file = output_path / "st_cube_analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ST-CUBE SEGMENTATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"QUALITY METRICS:\n")
        f.write(f"- STGS (Spatiotemporal Global Score): {analysis['stgs']:.4f}\n")
        f.write(f"- Number of ST-cubes: {analysis['n_cubes']}\n\n")
        
        f.write(f"CUBE SIZE STATISTICS:\n")
        f.write(f"- Mean size: {analysis['cube_sizes']['mean']:.2f} pixels\n")
        f.write(f"- Std deviation: {analysis['cube_sizes']['std']:.2f} pixels\n")
        f.write(f"- Min size: {analysis['cube_sizes']['min']} pixels\n")
        f.write(f"- Max size: {analysis['cube_sizes']['max']} pixels\n\n")
        
        f.write(f"TEMPORAL EXTENT STATISTICS:\n")
        f.write(f"- Mean extent: {analysis['temporal_extents']['mean']:.2f} time steps\n")
        f.write(f"- Std deviation: {analysis['temporal_extents']['std']:.2f} time steps\n")
        f.write(f"- Min extent: {analysis['temporal_extents']['min']} time steps\n")
        f.write(f"- Max extent: {analysis['temporal_extents']['max']} time steps\n\n")
        
        f.write(f"HETEROGENEITY STATISTICS:\n")
        f.write(f"- Mean heterogeneity: {analysis['heterogeneities']['mean']:.6f}\n")
        f.write(f"- Std deviation: {analysis['heterogeneities']['std']:.6f}\n")
        f.write(f"- Min heterogeneity: {analysis['heterogeneities']['min']:.6f}\n")
        f.write(f"- Max heterogeneity: {analysis['heterogeneities']['max']:.6f}\n\n")
        
        f.write(f"ALGORITHM PARAMETERS:\n")
        f.write(f"- Spatial heterogeneity threshold: {analysis['parameters']['sh_threshold']}\n")
        f.write(f"- Temporal heterogeneity threshold: {analysis['parameters']['th_threshold']}\n")
        f.write(f"- Shape weight: {analysis['parameters']['w_shape']}\n")
        f.write(f"- Compactness weight: {analysis['parameters']['w_compactness']}\n")
        f.write(f"- Minimum cube size: {analysis['parameters']['min_cube_size']}\n")
    
    print(f"Analysis report saved: {report_file}")
    print(f"Summary saved: {summary_file}")


def main(netcdf_path: str = "data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc",
         municipality_name: str = "Sant Martí"):
    """Main function to run ST-Cube segmentation analysis"""
    print("=" * 70)
    print("ST-CUBE SEGMENTATION ANALYSIS")
    print("Using ST-MRS (Spatiotemporal Multiresolution Segmentation)")
    print("=" * 70)
    
    # Load data
    print(f"Loading data from: {netcdf_path}")
    ds = xr.open_dataset(netcdf_path)
    
    # Load municipality mapping
    municipality_mapping = None
    mapping_file = Path(netcdf_path).parent / "municipality_mapping.csv"
    if mapping_file.exists():
        municipality_mapping = pd.read_csv(mapping_file)
        print(f"Municipality mapping loaded: {len(municipality_mapping)} municipalities")
    
    # Find municipality ID
    municipality_id = None
    if municipality_mapping is not None:
        municipality_row = municipality_mapping[
            municipality_mapping['municipality_name'] == municipality_name
        ]
        if len(municipality_row) > 0:
            municipality_id = municipality_row.iloc[0]['municipality_id']
            print(f"Found {municipality_name} with ID: {municipality_id}")
        else:
            print(f"Municipality '{municipality_name}' not found in mapping")
            available_municipalities = municipality_mapping['municipality_name'].tolist()
            print(f"Available municipalities: {available_municipalities[:5]}...")
    
    # Set up parameters (optimized for large datasets)
    params = STCubeParameters(
        sh_threshold=0.1,     # Increased spatial heterogeneity threshold
        th_threshold=0.05,    # Increased temporal heterogeneity threshold
        w_shape=0.2,          # Reduced shape weight
        w_compactness=0.5,    # Balanced compactness weight
        min_cube_size=20,     # Increased minimum cube size
        max_iterations=20     # Reduced maximum iterations
    )
    
    # Initialize segmentation
    segmentation = STCubeSegmentation(params)
    
    # Load data
    segmentation.load_data(ds, municipality_id)
    
    # Run segmentation
    segmentation.segment_st_cubes()
    
    # Analyze results
    analysis = segmentation.analyze_cubes()
    
    # Create output directory
    output_dir = "outputs/st_cube_analysis"
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 3D visualization with RGB base and single cube evolution
    visualize_st_cubes_3d(segmentation, output_dir)
    
    # 3D visualization with multiple cubes
    visualize_multiple_cubes_3d(segmentation, output_dir, n_cubes=4)
    
    # Time series visualization
    visualize_st_cubes_time_series(segmentation, output_dir)
    
    # Segmentation maps for first few time steps
    for t in range(min(3, len(segmentation.time_coords))):
        visualize_segmentation_map(segmentation, t, output_dir)
    
    # Create analysis report
    create_analysis_report(analysis, output_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"Municipality: {municipality_name}")
    print(f"Number of ST-cubes: {analysis['n_cubes']}")
    print(f"STGS Quality Score: {analysis['stgs']:.4f}")
    print(f"Average cube size: {analysis['cube_sizes']['mean']:.2f} pixels")
    print(f"Average temporal extent: {analysis['temporal_extents']['mean']:.2f} time steps")
    print(f"Output directory: {output_dir}")
    
    return segmentation, analysis


if __name__ == "__main__":
    # Run the analysis
    segmentation, analysis = main()
