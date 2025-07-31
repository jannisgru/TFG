#!/usr/bin/env python3
"""Interactive HTML Plotly Visualization for Vegetation ST-Cube Segmentation Results"""

# ==== CONFIGURABLE PARAMETERS ====
OUTPUT_DIRECTORY = "outputs/interactive_vegetation"
MAX_CLUSTERS_TO_DISPLAY = 50
DOWNSAMPLE_FACTOR = 2
COLOR_PALETTE = "Set3"
FIGURE_WIDTH = 1200
FIGURE_HEIGHT = 800
BASEMAP_TYPE = "rgb"  # Options: "rgb" or "ndvi"
# ================================

import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import warnings
from scipy.ndimage import zoom
from loguru import logger

warnings.filterwarnings('ignore')

import sys
import os
scripts_dir = Path(__file__).parent.parent.parent
sys.path.append(str(scripts_dir))

try:
    from .cube import STCube
except ImportError:
    from cube import STCube


class InteractiveVisualization:
    """Interactive HTML visualization generator for vegetation ST-cube segmentation results."""
    
    def __init__(self, output_directory: str = OUTPUT_DIRECTORY):
        """Initialize the visualization generator."""
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = getattr(px.colors.qualitative, COLOR_PALETTE)
        logger.info(f"Visualization generator initialized - Output: {self.output_dir}")
    
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
                                municipality_name: str = "Unknown", basemap_mode: str = BASEMAP_TYPE) -> Dict[str, str]:
        """Create spatial map and 3D spatiotemporal visualizations for vegetation clusters."""
        logger.info(f"Creating visualizations for '{municipality_name}' ({basemap_mode} basemap)")
        
        processed_cubes = self._process_cube_data(cubes, data)
        valid_cubes = [c for c in processed_cubes if self._is_valid_cube(c)]
        
        logger.info(f"Processing {len(valid_cubes)} valid clusters out of {len(processed_cubes)} total")
        
        if not valid_cubes:
            logger.warning("No valid vegetation clusters found")
            return {}
        
        visualizations = {}
        
        viz_configs = [
            ("spatial_map", self.create_interactive_spatial_map, f"Vegetation Clusters - {municipality_name}"),
            ("3d_spatiotemporal", self.create_3d_spatiotemporal_visualization, f"3D Spatiotemporal View - {municipality_name}")
        ]
        
        for viz_name, viz_func, title in viz_configs:
            filename = f"{viz_name}_{municipality_name.replace(' ', '_')}.html"
            try:
                logger.info(f"Generating {viz_name}...")
                
                args = [valid_cubes, filename, title]
                if viz_name == "3d_spatiotemporal":
                    args.insert(1, data)
                    args.append(basemap_mode)
                
                result = viz_func(*args)
                if result is not None:
                    visualizations[viz_name] = str(self.output_dir / filename)
                    logger.success(f"✓ {viz_name} saved successfully")
                else:
                    logger.error(f"✗ {viz_name} generation failed")
                    
            except Exception as e:
                logger.error(f"✗ Error in {viz_name}: {str(e)}")
        
        if visualizations:
            logger.success(f"Created {len(visualizations)} visualizations in: {self.output_dir}")
        
        return visualizations
    
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
                    'vegetation_type': 'Unknown', 'seasonality_score': 0.0, 'trend_score': 0.0
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
                    'seasonality_score': cube.get('seasonality_score', 0.0),
                    'trend_score': cube.get('trend_score', 0.0)
                }
            processed_cubes.append(cube_dict)
        
        return processed_cubes
    
    def create_interactive_spatial_map(self, cubes: List[Dict], filename: str, title: str = "Vegetation Clusters"):
        """Create an interactive spatial map showing vegetation cluster boundaries and NDVI patterns."""
        if not cubes:
            logger.warning("No cubes provided for spatial map")
            return None
        
        # Extract all pixels to determine spatial bounds
        all_pixels = [p for cube in cubes for p in self._get_pixels_safely(cube)]
        if not all_pixels:
            logger.warning("No valid pixels found for spatial map")
            return None
        
        y_coords, x_coords = zip(*all_pixels)
        y_min, y_max, x_min, x_max = min(y_coords), max(y_coords), min(x_coords), max(x_coords)
        
        # Create segmentation and NDVI maps
        seg_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        for i, cube in enumerate(cubes):
            for y, x in self._get_pixels_safely(cube):
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    seg_map[y - y_min, x - x_min] = i
                    ndvi_map[y - y_min, x - x_min] = cube['mean_ndvi']
        
        # Create subplot figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=['NDVI Distribution', 'Cluster Boundaries'],
                           specs=[[{"type": "heatmap"}, {"type": "scatter"}]])
        
        # Add NDVI heatmap
        fig.add_trace(go.Heatmap(z=ndvi_map, x=list(range(x_min, x_max + 1)), y=list(range(y_min, y_max + 1)),
                                colorscale='RdYlGn', name='NDVI', colorbar=dict(title="Mean NDVI", x=0.48),
                                hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'), row=1, col=1)
        
        # Add cluster scatter points
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if pixels:
                y_vals, x_vals = zip(*pixels)
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers',
                                       marker=dict(size=3, color=self.color_palette[i % len(self.color_palette)], 
                                                  line=dict(width=1, color='black')),
                                       name=f'Cluster {i} (NDVI: {cube["mean_ndvi"]:.3f})',
                                       hovertemplate=f'Cluster {i}<br>Area: {cube["area"]} pixels<br>Mean NDVI: {cube["mean_ndvi"]:.3f}<br>Type: {cube["vegetation_type"]}<extra></extra>'), row=1, col=2)
        
        # Update layout
        fig.update_layout(title=f'{title}<br>Total Clusters: {len(cubes)}', width=1400, height=600, hovermode='closest')
        for col in [1, 2]:
            fig.update_xaxes(title_text="X Coordinate", row=1, col=col)
            fig.update_yaxes(title_text="Y Coordinate", row=1, col=col)
        
        # Save visualization
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        return fig
    
    def create_3d_spatiotemporal_visualization(self, cubes: List[Dict], data: Union[xr.Dataset, str], filename: str, title: str = "3D Spatiotemporal View", basemap_mode: str = "rgb"):
        """Create a 3D visualization with X,Y spatial coordinates and Years (Z) axis."""
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
        basemap_created = False
        if hasattr(data, 'dims') and hasattr(data, 'x') and hasattr(data, 'y'):
            x_coords_data = np.array(data.x)
            y_coords_data = np.array(data.y)
            X_raster, Y_raster = np.meshgrid(x_coords_data, y_coords_data)
            Z_raster = np.full_like(X_raster, 1983)
            
            if basemap_mode == "rgb" and 'landsat' in data and 'band' in data.landsat.dims:
                basemap_created = self.create_rgb_basemap(data, fig, X_raster, Y_raster, Z_raster)
            elif basemap_mode == "ndvi" and 'ndvi' in data:
                basemap_created = self.create_ndvi_basemap(data, fig, X_raster, Y_raster, Z_raster)
            
            # Try fallback basemap
            if not basemap_created:
                logger.warning(f"Primary {basemap_mode} basemap failed, trying fallback...")
                if basemap_mode == "rgb" and 'ndvi' in data:
                    basemap_created = self.create_ndvi_basemap(data, fig, X_raster, Y_raster, Z_raster)
                elif basemap_mode == "ndvi" and 'landsat' in data and 'band' in data.landsat.dims:
                    basemap_created = self.create_rgb_basemap(data, fig, X_raster, Y_raster, Z_raster)

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
        
        for time_idx in range(min(15, n_time_steps)):  # Limit for performance
            actual_year = actual_years[time_idx]
            for cube_idx, cube in enumerate(valid_cubes[:8]):  # Limit clusters for performance
                pixels = self._get_pixels_safely(cube)
                ndvi_profile = self._get_ndvi_profile(cube)
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
                        
                        fig.add_trace(go.Mesh3d(
                            x=all_x_coords, y=all_y_coords, z=all_z_coords,
                            i=all_i_indices, j=all_j_indices, k=all_k_indices,
                            intensity=np.full(len(all_x_coords), ndvi_val),
                            colorscale='RdYlGn',
                            cmin=0.0, cmax=1.0,
                            showscale=False,
                            name=f'Cluster {cube_idx+1} - {actual_year}',
                            showlegend=(time_idx == 0),
                            hovertemplate=f'Cluster {cube_idx+1}<br>Year: {actual_year}<br>NDVI: {ndvi_val:.3f}<extra></extra>'
                        ))

        # Update layout
        fig.update_layout(
            title=f'{title} - {len(valid_cubes)} Vegetation Clusters ({basemap_mode.upper()} Basemap)',
            scene=dict(
                xaxis_title='Longitude', 
                yaxis_title='Latitude', 
                zaxis_title='Year',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode='manual', 
                aspectratio=dict(x=1, y=1, z=0.05)
            ),
            width=1400, 
            height=900
        )

        # Save visualization
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        return str(self.output_dir / filename)

    def create_rgb_basemap(self, data, fig, X_raster, Y_raster, Z_raster):
        """Create RGB natural color composite basemap."""
        try:
            landsat_data = data.landsat.isel(time=-1)
            band_names = list(data.landsat.band.values)
            
            # Find spectral bands
            bands = {}
            for color, targets in [('red', ['RED']), ('green', ['GREEN']), ('blue', ['BLUE']), ('nir', ['NIR'])]:
                for target in targets:
                    for i, band in enumerate(band_names):
                        if target in band.upper():
                            bands[color] = np.array(landsat_data.isel(band=i))
                            break
                    if color in bands:
                        break
            
            # Create composite based on available bands
            if 'nir' in bands and 'red' in bands and 'green' in bands:
                composite = 0.6 * bands['nir'] + 0.3 * bands['red'] + 0.1 * bands['green']
            elif 'red' in bands and 'green' in bands and 'blue' in bands:
                composite = 0.3 * bands['red'] + 0.6 * bands['green'] + 0.1 * bands['blue']
            elif 'nir' in bands and 'red' in bands:
                with np.errstate(divide='ignore', invalid='ignore'):
                    composite = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'])
            else:
                composite = list(bands.values())[0]
            
            valid_mask = ~np.isnan(composite)
            
            if np.any(valid_mask):
                valid_data = composite[valid_mask]
                p1, p99 = np.percentile(valid_data, [1, 99])
                
                # Normalize and stretch for better visualization
                composite_stretched = np.clip((composite - p1) / (p99 - p1), 0, 1)
                elevation_like = composite_stretched * 3000  # Scale for topographic colorscale
                
                X_masked = np.where(valid_mask, X_raster, np.nan)
                Y_masked = np.where(valid_mask, Y_raster, np.nan)
                Z_masked = np.where(valid_mask, Z_raster, np.nan)
                composite_masked = np.where(valid_mask, elevation_like, np.nan)
                
                # Add RGB basemap surface
                fig.add_trace(go.Surface(
                    x=X_masked, 
                    y=Y_masked, 
                    z=Z_masked,
                    surfacecolor=composite_masked,
                    colorscale='geyser',
                    opacity=0.9,
                    showscale=False
                ))
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"RGB basemap creation failed: {e}")
            return False

    def create_ndvi_basemap(self, data, fig, X_raster, Y_raster, Z_raster):
        """Create NDVI basemap with vegetation colors."""
        try:
            ndvi_2d = np.array(data.ndvi.isel(time=-1))
            
            valid_mask = ~np.isnan(ndvi_2d)
            ndvi_masked = np.where(valid_mask, ndvi_2d, np.nan)
            
                
            valid_ndvi = ndvi_2d[valid_mask]
            if len(valid_ndvi) == 0:
                logger.warning("No valid NDVI data found")
                return False
                
            X_masked = np.where(valid_mask, X_raster, np.nan)
            Y_masked = np.where(valid_mask, Y_raster, np.nan)
            Z_masked = np.where(valid_mask, Z_raster, np.nan)
            
            # Add NDVI basemap surface
            fig.add_trace(go.Surface(
                x=X_masked, 
                y=Y_masked, 
                z=Z_masked, 
                surfacecolor=ndvi_masked,
                colorscale='RdYlGn',
                cmin=-0.5,
                cmax=1.0,
                opacity=0.9, 
                showscale=False
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"NDVI basemap creation failed: {e}")
            return False


if __name__ == "__main__":
    print("Interactive Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides visualization tools for vegetation clustering results.")
    print("Use by importing InteractiveVisualization class and calling create_all_visualizations().")