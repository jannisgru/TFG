#!/usr/bin/env python3
"""
Interactive Visualization Module for Vegetation ST-Cube Segmentation Results

This module provides interactive 3D and 2D visualizations using Plotly for the results of 
vegetation-focused spatiotemporal cube segmentation.
"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.offline as pyo
from pathlib import Path
from typing import List, Dict, Tuple, Union
from loguru import logger
import warnings
from ..config_loader import get_config
from ..core.cube import STCube
from tqdm import tqdm
import time
import random
import threading

warnings.filterwarnings('ignore')


class InteractiveVisualization:
    """Interactive HTML visualization generator for vegetation ST-cube segmentation results."""
    
    def __init__(self, output_directory: str = None):
        """Initialize the visualization generator."""
        if output_directory is None:
            output_directory = get_config().interactive_output_dir
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_safe_data(self, cube: Dict, key: str, default=None):
        """Safely extract data from cube, handling arrays and lists."""
        data = cube.get(key, default)
        if data is None:
            return default
        if isinstance(data, np.ndarray):
            return data.tolist() if data.size > 0 else (default or [])
        return data if data else (default or [])
    
    def _get_pixels_safely(self, cube: Dict) -> List[Tuple[int, int]]:
        """Extract pixel coordinates, checking multiple possible keys."""
        for key in ['pixels', 'coordinates']:
            pixels = self._extract_safe_data(cube, key, [])
            if pixels:
                if isinstance(pixels[0], (list, tuple)) and len(pixels[0]) == 2:
                    return [tuple(p) for p in pixels]
                elif len(pixels) == 2 and isinstance(pixels[0], (int, float)):
                    return [tuple(pixels)]
        return []
    
    def _get_ndvi_profile(self, cube: Dict) -> List[float]:
        """Extract NDVI temporal profile, checking multiple possible keys."""
        for key in ['ndvi_profile', 'mean_temporal_profile', 'ndvi_time_series']:
            profile = self._extract_safe_data(cube, key, [])
            if profile:
                return profile
        return []
    
    def _is_valid_cube(self, cube: Dict) -> bool:
        """Check if cube has both valid pixels and NDVI data."""
        return len(self._get_pixels_safely(cube)) > 0 and len(self._get_ndvi_profile(cube)) > 0
    
    def create_all_visualizations(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str], 
                                municipality_name: str = "Unknown") -> Dict[str, str]:
        """Create 3D spatiotemporal visualization for vegetation clusters."""        
        processed_cubes = self._process_cube_data(cubes, data)
        valid_cubes = [c for c in processed_cubes if self._is_valid_cube(c)]
                
        if not valid_cubes:
            logger.warning("No valid vegetation clusters found")
            return {}
        
        filename = f"3d_spatiotemporal_{municipality_name.replace(' ', '_')}.html"
        title = f"3D Spatiotemporal View - {municipality_name}"
        
        try:
            # Create a fake progress bar to simulate loading time, because its funny lol
            def fake_progress_bar(done_event, n_clusters):
                total_time = random.uniform(27, 35)
                breakpoints = sorted([random.uniform(0, total_time) for _ in range(n_clusters - 1)])
                intervals = [breakpoints[0]] + [breakpoints[i] - breakpoints[i-1] for i in range(1, n_clusters - 1)] + [total_time - breakpoints[-1]] if n_clusters > 1 else [total_time]
                with tqdm(total=n_clusters, desc="Generating 3D Visualization") as pbar:
                    for i in range(n_clusters):
                        if i == n_clusters - 1:
                            # Wait for done_event before finishing last step
                            while not done_event.is_set():
                                time.sleep(0.1)
                            pbar.update(1)
                            break
                        if done_event.is_set():
                            # Rapidly finish the bar
                            while pbar.n < n_clusters:
                                pbar.update(1)
                                time.sleep(0.01)
                            break
                        time.sleep(intervals[i])
                        pbar.update(1)
                    if not done_event.is_set():
                        pbar.n = n_clusters
                        pbar.refresh()

            n_clusters = len(valid_cubes)
            done_event = threading.Event()
            progress_thread = threading.Thread(target=fake_progress_bar, args=(done_event, n_clusters))
            progress_thread.start()

            result = self.create_3d_spatiotemporal_visualization(valid_cubes, data, filename, title)
            done_event.set()
            progress_thread.join()
            
            if result is not None:
                logger.success(f"✓ 3D visualization saved successfully")
                return {"3d_spatiotemporal": str(self.output_dir / filename)}
            else:
                logger.error(f"✗ 3D visualization generation failed")
                return {}
                
        except Exception as e:
            logger.error(f"✗ Error in 3D visualization: {str(e)}")
            return {}
    
    def _process_cube_data(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str]) -> List[Dict]:
        """Process cube data into a consistent format for visualization."""
        processed_cubes = []
        time_length = len(data.time) if hasattr(data, 'dims') and 'time' in data.dims else 1
        
        for i, cube in enumerate(cubes):
            if isinstance(cube, STCube):
                # Process STCube objects
                cube_dict = {k: getattr(cube, k, default) for k, default in [
                    ('id', i), ('pixels', []), ('area', 0), ('ndvi_profile', []), 
                    ('temporal_extent', (0, 0)), ('heterogeneity', 0.0)
                ]}
                cube_dict.update({
                    'mean_ndvi': np.mean(cube_dict['ndvi_profile']) if cube_dict['ndvi_profile'] else 0.5,
                })
            else:
                # Process dictionary cubes
                pixels = self._get_pixels_safely(cube)
                ndvi_profile = self._get_ndvi_profile(cube)
                cube_dict = {
                    'id': cube.get('id', i),
                    'pixels': pixels,
                    'area': cube.get('area', cube.get('size', len(pixels))),
                    'ndvi_profile': ndvi_profile,
                    'mean_ndvi': cube.get('mean_ndvi', np.mean(ndvi_profile) if ndvi_profile else 0.5),
                    'temporal_extent': cube.get('temporal_extent', (0, time_length)),
                    'heterogeneity': cube.get('heterogeneity', cube.get('temporal_variance', 0.0)),
                    'vegetation_type': cube.get('vegetation_type', 'Unknown'),
                    'trend_score': cube.get('trend_score', 0.0)
                }
            processed_cubes.append(cube_dict)
        
        return processed_cubes
    
    def create_3d_spatiotemporal_visualization(self, cubes: List[Dict], data: Union[xr.Dataset, str], filename: str, title: str = "3D Spatiotemporal View"):
        """Create a 3D visualization with X,Y spatial coordinates and Years (Z) axis, using a grayscale basemap."""
        config = get_config()
        if not cubes:
            logger.warning("No cubes provided for 3D visualization")
            return None

        # Determine temporal dimensions
        if hasattr(data, 'dims') and 'time' in data.dims:
            n_time_steps = len(data.time)
            actual_years = [1984 + i for i in range(n_time_steps)]
        else:
            first_valid = next((c for c in cubes if self._is_valid_cube(c)), None)
            if not first_valid:
                return None
            n_time_steps = len(self._get_ndvi_profile(first_valid))
            actual_years = [1984 + i for i in range(n_time_steps)]

        valid_cubes = [c for c in cubes if self._is_valid_cube(c)]
        fig = go.Figure()

        # Create basemap
        if hasattr(data, 'dims') and hasattr(data, 'x') and hasattr(data, 'y') and 'landsat' in data and 'band' in data.landsat.dims:
            x_coords_data = np.array(data.x)
            y_coords_data = np.array(data.y)
            X_raster, Y_raster = np.meshgrid(x_coords_data, y_coords_data)
            Z_raster = np.full_like(X_raster, 1983)
            self.create_basemap(data, fig, X_raster, Y_raster, Z_raster)

        # Transform pixel coordinates to geographic coordinates
        def transform_pixel_to_geo(px_y, px_x, data):
            if hasattr(data, 'x') and hasattr(data, 'y'):
                x_coords = np.array(data.x)
                y_coords = np.array(data.y)
                if 0 <= px_x < len(x_coords) and 0 <= px_y < len(y_coords):
                    return y_coords[px_y], x_coords[px_x]
                else:
                    x_interp = np.interp(px_x, range(len(x_coords)), x_coords)
                    y_interp = np.interp(px_y, range(len(y_coords)), y_coords)
                    return y_interp, x_interp
            return px_y, px_x

        # Add 3D spatiotemporal cubes using Mesh3D for proper cube visualization
        # Determine appropriate cube size based on coordinate system scale
        if hasattr(data, 'x') and hasattr(data, 'y'):
            x_coords_data = np.array(data.x)
            y_coords_data = np.array(data.y)
            if len(x_coords_data) > 1 and len(y_coords_data) > 1:
                x_resolution = abs(x_coords_data[1] - x_coords_data[0])
                y_resolution = abs(y_coords_data[1] - y_coords_data[0])
                cube_size = min(x_resolution, y_resolution) * 0.8  # 80% of pixel size
            else:
                cube_size = 0.0003  # Fallback small size
        else:
            cube_size = 0.5  # Fallback for pixel coordinates
        
        # Get config for visualization limits
        max_time_layers = config.max_time_layers 
        max_clusters = config.max_clusters_3d

        for cube_idx, cube in enumerate(valid_cubes[:max_clusters]):
            pixels = self._get_pixels_safely(cube)
            ndvi_profile = self._get_ndvi_profile(cube)
            legendgroup = f"cluster_{cube_idx+1}"
            for time_idx in range(min(max_time_layers, n_time_steps)):
                actual_year = actual_years[time_idx]
                if time_idx < len(ndvi_profile) and pixels:
                    geo_coords = [transform_pixel_to_geo(px_y, px_x, data) for px_y, px_x in pixels]
                    # Batch create cubes for better performance - combine multiple pixels into one mesh
                    if geo_coords:
                        all_x_coords = []
                        all_y_coords = []
                        all_z_coords = []
                        all_i_indices = []
                        all_j_indices = []
                        all_k_indices = []
                        half_size = cube_size / 2
                        time_half_size = 0.45  # Small temporal extent
                        for vertex_offset, (geo_y, geo_x) in enumerate(geo_coords):
                            # Define cube vertices (8 corners) for this pixel
                            base_idx = vertex_offset * 8
                            
                            # Add 8 vertices for this cube
                            cube_x = [geo_x - half_size, geo_x + half_size, geo_x + half_size, geo_x - half_size,
                                     geo_x - half_size, geo_x + half_size, geo_x + half_size, geo_x - half_size]
                            cube_y = [geo_y - half_size, geo_y - half_size, geo_y + half_size, geo_y + half_size,
                                     geo_y - half_size, geo_y - half_size, geo_y + half_size, geo_y + half_size]
                            cube_z = [actual_year - time_half_size, actual_year - time_half_size, actual_year - time_half_size, actual_year - time_half_size,
                                     actual_year + time_half_size, actual_year + time_half_size, actual_year + time_half_size, actual_year + time_half_size]
                            all_x_coords.extend(cube_x)
                            all_y_coords.extend(cube_y)
                            all_z_coords.extend(cube_z)
                            
                            # Define cube faces using vertex indices (12 triangles for 6 faces)
                            # Bottom face (z-min): vertices 0,1,2,3
                            # Top face (z-max): vertices 4,5,6,7
                            # Side faces connecting bottom to top
                            cube_i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
                            cube_j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
                            cube_k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
                            
                            # Offset indices for this cube
                            all_i_indices.extend([i + base_idx for i in cube_i])
                            all_j_indices.extend([j + base_idx for j in cube_j])
                            all_k_indices.extend([k + base_idx for k in cube_k])
                        ndvi_val = ndvi_profile[time_idx]
                        # Only show legend for the first year of each cluster
                        show_legend = (time_idx == 0)
                        fig.add_trace(go.Mesh3d(
                            x=all_x_coords, y=all_y_coords, z=all_z_coords,
                            i=all_i_indices, j=all_j_indices, k=all_k_indices,
                            intensity=np.full(len(all_x_coords), ndvi_val),
                            colorscale='RdYlGn',
                            cmin=0.0, cmax=1.0,
                            showscale=False,
                            name=f'Cluster {cube_idx+1}',
                            legendgroup=legendgroup,
                            showlegend=show_legend,
                            hovertemplate=f'Cluster {cube_idx+1}<br>Year: {actual_year}<br>NDVI: {ndvi_val:.3f}<extra></extra>',
                            visible='legendonly'
                        ))

        # Calculate fixed axis ranges to prevent rescaling when traces are hidden
        if hasattr(data, 'x') and hasattr(data, 'y'):
            x_coords_data = np.array(data.x)
            y_coords_data = np.array(data.y)
            
            # Handle case where coordinates might be constant
            x_span = x_coords_data.max() - x_coords_data.min()
            y_span = y_coords_data.max() - y_coords_data.min()
            
            if x_span > 0 and y_span > 0:
                # Add small padding to ensure all data is visible
                x_padding = x_span * 0.02
                y_padding = y_span * 0.02
                x_range = [float(x_coords_data.min() - x_padding), float(x_coords_data.max() + x_padding)]
                y_range = [float(y_coords_data.min() - y_padding), float(y_coords_data.max() + y_padding)]
            else:
                # Fallback for constant coordinates
                x_center = float(x_coords_data[0]) if len(x_coords_data) > 0 else 0
                y_center = float(y_coords_data[0]) if len(y_coords_data) > 0 else 0
                x_range = [x_center - 0.01, x_center + 0.01]
                y_range = [y_center - 0.01, y_center + 0.01]
        else:
            # Fallback for pixel coordinates
            x_range = [0, 100]
            y_range = [0, 100]
        
        # Fixed Z range based on actual years with padding
        z_range = [float(min(actual_years) - 1), float(max(actual_years) + 1)]
        
        # Define visibility states for show/hide all buttons
        n_traces = len(fig.data)
        # Check if first trace is basemap (has no name or name doesn't start with "Cluster")
        has_basemap = n_traces > 0 and (not hasattr(fig.data[0], 'name') or not fig.data[0].name or not fig.data[0].name.startswith('Cluster'))
        n_basemap = 1 if has_basemap else 0
        
        all_visible = []
        all_hidden = []
        for i, trace in enumerate(fig.data):
            if i < n_basemap:
                # Keep basemap always visible
                all_visible.append(True)
                all_hidden.append(True)
            else:
                # Clusters: show or hide
                all_visible.append(True)
                all_hidden.append('legendonly')
        
        # Update layout with fixed axis ranges and control buttons
        fig.update_layout(
            title=f'{title} - {len(valid_cubes)} Vegetation Clusters',
            scene=dict(
                xaxis=dict(
                    title='Longitude',
                    range=x_range
                ),
                yaxis=dict(
                    title='Latitude', 
                    range=y_range
                ),
                zaxis=dict(
                    title='Year',
                    range=z_range
                ),
                camera=dict(eye=dict(x=config.camera_x, y=config.camera_y, z=config.camera_z)),
                aspectmode='manual', 
                aspectratio=dict(x=config.aspect_x, y=config.aspect_y, z=config.aspect_z)
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.1,
                    y=1.02,
                    showactive=True,
                    buttons=[
                        dict(
                            label="Show All Clusters",
                            method="update",
                            args=[{"visible": all_visible}]
                        ),
                        dict(
                            label="Hide All Clusters",
                            method="update",
                            args=[{"visible": all_hidden}]
                        ),
                    ],
                )
            ],
            width=config.figure_width, 
            height=config.figure_height
        )

        # Save visualization
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        return str(self.output_dir / filename)

    def create_basemap(self, data, fig, X_raster, Y_raster, Z_raster):
        """Create smooth grayscale basemap with gentle contrast and lighter darks."""
        try:
            landsat_data = data.landsat.isel(time=-1)
            band_names = list(data.landsat.band.values)
            # Get any available band for grayscale
            if len(band_names) > 0:
                single_band = np.array(landsat_data.isel(band=0))
            else:
                return False
            # Check for valid data
            valid_mask = ~np.isnan(single_band)
            if np.any(valid_mask):
                # Gentle normalization using percentiles
                vmin, vmax = np.percentile(single_band[valid_mask], [5, 95])
                norm = np.clip((single_band - vmin) / (vmax - vmin), 0, 1)
                # Map to a lighter gray range: 0.0 -> #606060, 1.0 -> #c0c0c0
                def norm_to_gray_hex(val):
                    n_steps = 6  # Number of gray levels
                    val = np.clip(val, 0, 1)
                    val_q = round(val * (n_steps - 1)) / (n_steps - 1)
                    gray = int(70 + val_q * (120 - 70))  # 120=#c8 (light), 70=#50 (dark)
                    return f'#{gray:02x}{gray:02x}{gray:02x}'
                X_masked = np.where(valid_mask, X_raster, np.nan)
                Y_masked = np.where(valid_mask, Y_raster, np.nan)
                Z_masked = np.where(valid_mask, Z_raster, np.nan)
                x_coords_data = np.array(data.x)
                y_coords_data = np.array(data.y)
                if len(x_coords_data) > 1 and len(y_coords_data) > 1:
                    x_res = abs(x_coords_data[1] - x_coords_data[0]) / 2
                    y_res = abs(y_coords_data[1] - y_coords_data[0]) / 2
                else:
                    x_res = y_res = 0.0001
                all_x, all_y, all_z, all_i, all_j, all_k, all_colors = [], [], [], [], [], [], []
                for row in range(single_band.shape[0]):
                    for col in range(single_band.shape[1]):
                        if valid_mask[row, col]:
                            x_center = X_masked[row, col]
                            y_center = Y_masked[row, col]
                            z_center = Z_masked[row, col]
                            val = norm[row, col]
                            hex_color = norm_to_gray_hex(val)
                            base_idx = len(all_x)
                            corners_x = [x_center - x_res, x_center + x_res, x_center + x_res, x_center - x_res]
                            corners_y = [y_center - y_res, y_center - y_res, y_center + y_res, y_center + y_res]
                            corners_z = [z_center] * 4
                            all_x.extend(corners_x)
                            all_y.extend(corners_y)
                            all_z.extend(corners_z)
                            all_i.extend([base_idx, base_idx])
                            all_j.extend([base_idx + 1, base_idx + 2])
                            all_k.extend([base_idx + 2, base_idx + 3])
                            all_colors.extend([hex_color, hex_color])
                if all_x:
                    fig.add_trace(go.Mesh3d(
                        x=all_x, y=all_y, z=all_z,
                        i=all_i, j=all_j, k=all_k,
                        facecolor=all_colors,
                        opacity=0.9,
                        showscale=False,
                        showlegend=False,
                        hovertemplate='Basemap<extra></extra>'
                    ))
                return True
            return False
        except Exception as e:
            logger.error(f"Basemap creation failed: {e}")
            return False