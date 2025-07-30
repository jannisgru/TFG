#!/usr/bin/env python3
"""Interactive HTML Plotly Visualization for Vegetation ST-Cube Segmentation Results"""


# ==== CONFIGURABLE PARAMETERS ====
DEFAULT_OUTPUT_DIRECTORY = "outputs/interactive_vegetation"
DEFAULT_MAX_CLUSTERS_TO_DISPLAY = 50
DEFAULT_DOWNSAMPLE_FACTOR = 2
DEFAULT_COLOR_PALETTE = "Set3"
DEFAULT_FIGURE_WIDTH = 1200
DEFAULT_FIGURE_HEIGHT = 800
DEFAULT_BASEMAP_TYPE = "rgb"  # Options: "rgb" or "ndvi"
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
    
    def __init__(self, output_directory: str = DEFAULT_OUTPUT_DIRECTORY):
        """Initialize the visualization generator."""
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = getattr(px.colors.qualitative, DEFAULT_COLOR_PALETTE)
    
    def _extract_safe_data(self, cube: Dict, key: str, default=None):
        """Safely extract data from cube, handling arrays and lists."""
        data = cube.get(key, default)
        if data is None:
            return default
        if isinstance(data, np.ndarray):
            return data.tolist() if data.size > 0 else (default or [])
        return data if data else (default or [])
    
    def _get_pixels_safely(self, cube: Dict) -> List[Tuple[int, int]]:
        """Extract pixels, prioritizing different possible keys."""
        for key in ['pixels', 'coordinates']:
            pixels = self._extract_safe_data(cube, key, [])
            if pixels:
                if isinstance(pixels[0], (list, tuple)) and len(pixels[0]) == 2:
                    return [tuple(p) for p in pixels]
                elif len(pixels) == 2 and isinstance(pixels[0], (int, float)):
                    return [tuple(pixels)]
        return []
    
    def _get_ndvi_profile(self, cube: Dict) -> List[float]:
        """Extract NDVI profile, checking multiple possible keys."""
        for key in ['ndvi_profile', 'mean_temporal_profile', 'ndvi_time_series']:
            profile = self._extract_safe_data(cube, key, [])
            if profile:
                return profile
        return []
    
    def _is_valid_cube(self, cube: Dict) -> bool:
        """Check if cube has both valid pixels and NDVI data."""
        return len(self._get_pixels_safely(cube)) > 0 and len(self._get_ndvi_profile(cube)) > 0
    
    def create_all_visualizations(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str], 
                                municipality_name: str = "Unknown", basemap_mode: str = DEFAULT_BASEMAP_TYPE) -> Dict[str, str]:
        """Create spatial map and 3D spatiotemporal visualizations for vegetation clusters."""
        logger.info(f"Creating visualizations for {municipality_name} (basemap: {basemap_mode})...")
        
        processed_cubes = self._process_cube_data(cubes, data)
        visualizations = {}
        
        viz_configs = [
            ("spatial_map", self.create_interactive_spatial_map, f"Vegetation Clusters - {municipality_name}"),
            ("3d_spatiotemporal", self.create_3d_spatiotemporal_visualization, f"3D Spatiotemporal View - {municipality_name}")
        ]
        
        for viz_name, viz_func, title in viz_configs:
            filename = f"{viz_name}_{municipality_name.replace(' ', '_')}.html"
            try:
                args = [processed_cubes, filename, title]
                if viz_name == "3d_spatiotemporal":
                    args.insert(1, data)
                    args.append(basemap_mode)
                
                result = viz_func(*args)
                if result is not None:
                    visualizations[viz_name] = str(self.output_dir / filename)
                    logger.success(f"{viz_name} created successfully")
            except Exception as e:
                logger.error(f"Error in {viz_name}: {str(e)}")
        
        logger.success(f"Visualizations created successfully in: {self.output_dir}")
        return visualizations
    
    def _process_cube_data(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str]) -> List[Dict]:
        """Process cube data into a consistent format for visualization."""
        processed_cubes = []
        time_length = len(data.time) if hasattr(data, 'dims') and 'time' in data.dims else 1
        
        for i, cube in enumerate(cubes):
            if isinstance(cube, STCube):
                cube_dict = {k: getattr(cube, k, default) for k, default in [
                    ('id', i), ('pixels', []), ('area', 0), ('ndvi_profile', []), 
                    ('temporal_extent', (0, 0)), ('heterogeneity', 0.0)
                ]}
                cube_dict.update({
                    'mean_ndvi': np.mean(cube_dict['ndvi_profile']) if cube_dict['ndvi_profile'] else 0.5,
                    'vegetation_type': 'Unknown', 'seasonality_score': 0.0, 'trend_score': 0.0
                })
            else:
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
            return None
        
        all_pixels = [p for cube in cubes for p in self._get_pixels_safely(cube)]
        if not all_pixels:
            return None
        
        y_coords, x_coords = zip(*all_pixels)
        y_min, y_max, x_min, x_max = min(y_coords), max(y_coords), min(x_coords), max(x_coords)
        
        seg_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        for i, cube in enumerate(cubes):
            for y, x in self._get_pixels_safely(cube):
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    seg_map[y - y_min, x - x_min] = i
                    ndvi_map[y - y_min, x - x_min] = cube['mean_ndvi']
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['NDVI Distribution', 'Cluster Boundaries'],
                           specs=[[{"type": "heatmap"}, {"type": "scatter"}]])
        
        fig.add_trace(go.Heatmap(z=ndvi_map, x=list(range(x_min, x_max + 1)), y=list(range(y_min, y_max + 1)),
                                colorscale='RdYlGn', name='NDVI', colorbar=dict(title="Mean NDVI", x=0.48),
                                hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'), row=1, col=1)
        
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if pixels:
                y_vals, x_vals = zip(*pixels)
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers',
                                       marker=dict(size=3, color=self.color_palette[i % len(self.color_palette)], 
                                                  line=dict(width=1, color='black')),
                                       name=f'Cluster {i} (NDVI: {cube["mean_ndvi"]:.3f})',
                                       hovertemplate=f'Cluster {i}<br>Area: {cube["area"]} pixels<br>Mean NDVI: {cube["mean_ndvi"]:.3f}<br>Type: {cube["vegetation_type"]}<extra></extra>'), row=1, col=2)
        
        fig.update_layout(title=f'{title}<br>Total Clusters: {len(cubes)}', width=1400, height=600, hovermode='closest')
        for col in [1, 2]:
            fig.update_xaxes(title_text="X Coordinate", row=1, col=col)
            fig.update_yaxes(title_text="Y Coordinate", row=1, col=col)
        
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        return fig
    

    def create_3d_spatiotemporal_visualization(self, cubes: List[Dict], data: Union[xr.Dataset, str], filename: str, title: str = "3D Spatiotemporal View", basemap_mode: str = "rgb"):
        """Create a 3D visualization with X,Y spatial coordinates and Years (Z) axis."""
        if not cubes:
            return None

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

        # Create basemap - ONLY RGB or NDVI modes
        if hasattr(data, 'dims') and hasattr(data, 'x') and hasattr(data, 'y'):
            x_coords_data = np.array(data.x)
            y_coords_data = np.array(data.y)
            X_raster, Y_raster = np.meshgrid(x_coords_data, y_coords_data)
            Z_raster = np.full_like(X_raster, 1983)
            
            basemap_created = False
            
            if basemap_mode == "rgb" and 'landsat' in data and 'band' in data.landsat.dims:
                basemap_created = self.create_rgb_basemap(data, fig, X_raster, Y_raster, Z_raster)
            elif basemap_mode == "ndvi" and 'ndvi' in data:
                basemap_created = self.create_ndvi_basemap(data, fig, X_raster, Y_raster, Z_raster)
            
            if not basemap_created:
                print(f"Warning: Could not create {basemap_mode} basemap, trying fallback...")
                # Try the other mode as fallback
                if basemap_mode == "rgb" and 'ndvi' in data:
                    basemap_created = self.create_ndvi_basemap(data, fig, X_raster, Y_raster, Z_raster)
                elif basemap_mode == "ndvi" and 'landsat' in data and 'band' in data.landsat.dims:
                    basemap_created = self.create_rgb_basemap(data, fig, X_raster, Y_raster, Z_raster)

        # Transform coordinates (unchanged)
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

        # Add 3D points (unchanged)
        for time_idx in range(min(15, n_time_steps)):
            actual_year = actual_years[time_idx]
            for cube_idx, cube in enumerate(valid_cubes[:8]):
                pixels = self._get_pixels_safely(cube)
                ndvi_profile = self._get_ndvi_profile(cube)
                if time_idx < len(ndvi_profile) and pixels:
                    geo_coords = [transform_pixel_to_geo(px_y, px_x, data) for px_y, px_x in pixels]
                    y_vals, x_vals = zip(*geo_coords)
                    z_vals = np.full(len(pixels), actual_year)
                    ndvi_vals = np.full(len(pixels), ndvi_profile[time_idx])
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_vals, y=y_vals, z=z_vals, 
                        mode='markers',
                        marker=dict(
                            size=4, 
                            color=ndvi_vals, 
                            colorscale='RdYlGn', 
                            cmin=0.0, cmax=1.0, 
                            showscale=False, 
                            opacity=0.9
                        ),
                        name=f'Cluster {cube_idx+1} - {actual_year}',
                        showlegend=(time_idx == 0)
                    ))

        fig.update_layout(
            title=f'{title} - {len(valid_cubes)} Vegetation Clusters ({basemap_mode.upper()} Basemap)',
            scene=dict(
                xaxis_title='Longitude', 
                yaxis_title='Latitude', 
                zaxis_title='Year',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode='manual', 
                aspectratio=dict(x=1, y=1, z=0.6)
            ),
            width=1400, 
            height=900
        )

        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        return str(self.output_dir / filename)


    def create_rgb_basemap(self, data, fig, X_raster, Y_raster, Z_raster):
        """Create RGB basemap using satellite imagery colorscale"""
        try:
            landsat_data = data.landsat.isel(time=-1)
            band_names = list(data.landsat.band.values)
            
            # Find RGB bands
            rgb_indices = {}
            for color in ['red', 'green', 'blue']:
                for i, band in enumerate(band_names):
                    if color.upper() in band.upper():
                        rgb_indices[color] = i
                        break
            
            if len(rgb_indices) != 3:
                print("Could not find all RGB bands")
                return False
            
            # Extract all RGB bands and create a natural composite
            red_band = np.array(landsat_data.isel(band=rgb_indices['red']))
            green_band = np.array(landsat_data.isel(band=rgb_indices['green']))
            blue_band = np.array(landsat_data.isel(band=rgb_indices['blue']))
            
            # Clean NaN values
            red_clean = np.where(np.isnan(red_band), 0, red_band)
            green_clean = np.where(np.isnan(green_band), 0, green_band)
            blue_clean = np.where(np.isnan(blue_band), 0, blue_band)
            
            # Create a natural-looking composite using NIR-Red-Green (false color for vegetation)
            # This gives better contrast and more natural satellite imagery appearance
            composite = 0.5 * green_clean + 0.3 * red_clean + 0.2 * blue_clean
            
            # Create mask for valid data - same approach as visualize_multidim_raster
            valid_mask = ~np.isnan(red_band) & ~np.isnan(green_band) & ~np.isnan(blue_band)
            composite_masked = np.where(valid_mask, composite, np.nan)
            
            # Get valid data range for scaling
            valid_data = composite[valid_mask]
            if len(valid_data) == 0:
                print("No valid RGB data found")
                return False
            
            min_val = np.percentile(valid_data, 5)  # 5th percentile to avoid outliers
            max_val = np.percentile(valid_data, 95)  # 95th percentile to avoid outliers
            
            print(f"RGB composite range: {min_val:.0f} to {max_val:.0f}")
            
            # Apply same masking to coordinates as visualize_multidim_raster
            X_masked = np.where(valid_mask, X_raster, np.nan)
            Y_masked = np.where(valid_mask, Y_raster, np.nan)
            Z_masked = np.where(valid_mask, Z_raster, np.nan)
            
            fig.add_trace(go.Surface(
                x=X_masked, 
                y=Y_masked, 
                z=Z_masked,
                surfacecolor=composite_masked,
                #colorscale='balance',  # More natural earth/satellite colors
                cmin=min_val,
                cmax=max_val,
                opacity=0.9,
                name='RGB Basemap',
                showscale=False
            ))
            
            print("RGB basemap created successfully")
            return True
            
        except Exception as e:
            print(f"RGB basemap failed: {e}")
            return False


    def create_ndvi_basemap(self, data, fig, X_raster, Y_raster, Z_raster):
        """Create NDVI basemap with proper vegetation colors"""
        try:
            ndvi_2d = np.array(data.ndvi.isel(time=-1))
            
            # Create mask for valid data - same approach as visualize_multidim_raster
            valid_mask = ~np.isnan(ndvi_2d)
            ndvi_masked = np.where(valid_mask, ndvi_2d, np.nan)
            
            # Get valid NDVI data range (exclude zeros which are NaN replacements)
            valid_ndvi = ndvi_2d[valid_mask]
            if len(valid_ndvi) == 0:
                print("No valid NDVI data found")
                return False
                
            min_ndvi = np.min(valid_ndvi)
            max_ndvi = np.max(valid_ndvi)
            
            print(f"NDVI range: {min_ndvi:.3f} to {max_ndvi:.3f}")
            
            # Apply same masking to coordinates as visualize_multidim_raster
            X_masked = np.where(valid_mask, X_raster, np.nan)
            Y_masked = np.where(valid_mask, Y_raster, np.nan)
            Z_masked = np.where(valid_mask, Z_raster, np.nan)
            
            # Use NDVI data directly with appropriate range - just like the working visualization
            fig.add_trace(go.Surface(
                x=X_masked, 
                y=Y_masked, 
                z=Z_masked, 
                surfacecolor=ndvi_masked,
                colorscale='RdYlGn',  # Red-Yellow-Green (classic NDVI colors)
                cmin=-0.5,  # Fixed range like in working visualization
                cmax=1.0,   # Fixed range like in working visualization
                opacity=0.9, 
                name='NDVI Basemap',
                showscale=False
            ))
            
            print("NDVI basemap created successfully")
            return True
            
        except Exception as e:
            print(f"NDVI basemap failed: {e}")
            return False


if __name__ == "__main__":
    print("Interactive Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides comprehensive visualization tools for vegetation clustering results.")
    print("Use this module by importing InteractiveVisualization class and calling create_all_visualizations().")