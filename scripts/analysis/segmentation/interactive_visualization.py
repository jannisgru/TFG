#!/usr/bin/env python3
"""Interactive HTML Plotly Visualization for Vegetation ST-Cube Segmentation Results"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from typing import List, Dict, Tuple, Union
import warnings
from loguru import logger

warnings.filterwarnings('ignore')

import sys
scripts_dir = Path(__file__).parent.parent.parent
sys.path.append(str(scripts_dir))

try:
    from .cube import STCube
except ImportError:
    from cube import STCube


class InteractiveVisualization:
    """Interactive HTML visualization generator for vegetation ST-cube segmentation results."""
    
    def __init__(self, output_directory: str = "outputs/interactive_vegetation"):
        """Initialize the visualization generator."""
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = px.colors.qualitative.Set3
        logger.info(f"Initialized - Output: {self.output_dir}")
    
    def _get_pixels_safely(self, cube: Dict) -> List[Tuple[int, int]]:
        """Extract pixel coordinates from cube."""
        for key in ['pixels', 'coordinates']:
            pixels = cube.get(key, [])
            if pixels and isinstance(pixels[0], (list, tuple)) and len(pixels[0]) == 2:
                return [tuple(p) for p in pixels]
        return []
    
    def _get_ndvi_profile(self, cube: Dict) -> List[float]:
        """Extract NDVI temporal profile from cube."""
        for key in ['ndvi_profile', 'mean_temporal_profile', 'ndvi_time_series']:
            profile = cube.get(key, [])
            if profile:
                return profile if isinstance(profile, list) else profile.tolist()
        return []
    
    def _is_valid_cube(self, cube: Dict) -> bool:
        """Check if cube has valid pixels and NDVI data."""
        return len(self._get_pixels_safely(cube)) > 0 and len(self._get_ndvi_profile(cube)) > 0
    
    def create_all_visualizations(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str], 
                                municipality_name: str = "Unknown", basemap_mode: str = "rgb") -> Dict[str, str]:
        """Create spatial map and 3D spatiotemporal visualizations for vegetation clusters."""
        logger.info(f"Creating visualizations for '{municipality_name}' ({basemap_mode} basemap)")
        
        processed_cubes = self._process_cube_data(cubes, data)
        valid_cubes = [c for c in processed_cubes if self._is_valid_cube(c)]
        
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
                args = [valid_cubes, filename, title]
                if viz_name == "3d_spatiotemporal":
                    args.insert(1, data)
                    args.append(basemap_mode)
                
                result = viz_func(*args)
                if result is not None:
                    visualizations[viz_name] = str(self.output_dir / filename)
                    logger.success(f"Created {viz_name}")
                    
            except Exception as e:
                logger.error(f"Failed to create {viz_name}: {str(e)}")
        
        return visualizations
    
    def _process_cube_data(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str]) -> List[Dict]:
        """Process cube data into a consistent format for visualization."""
        processed_cubes = []
        time_length = len(data.time) if hasattr(data, 'dims') and 'time' in data.dims else 1
        
        for i, cube in enumerate(cubes):
            if isinstance(cube, STCube):
                # Convert STCube to dict
                cube_dict = {
                    'id': getattr(cube, 'id', i),
                    'pixels': getattr(cube, 'pixels', []),
                    'area': getattr(cube, 'area', 0),
                    'ndvi_profile': getattr(cube, 'ndvi_profile', []),
                    'temporal_extent': getattr(cube, 'temporal_extent', (0, time_length)),
                    'heterogeneity': getattr(cube, 'heterogeneity', 0.0),
                    'vegetation_type': 'Unknown',
                    'seasonality_score': 0.0,
                    'trend_score': 0.0
                }
                if cube_dict['ndvi_profile']:
                    cube_dict['mean_ndvi'] = np.mean(cube_dict['ndvi_profile'])
                else:
                    cube_dict['mean_ndvi'] = 0.5
            else:
                # Process dict cube
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
        
        # Get spatial bounds
        all_pixels = [p for cube in cubes for p in self._get_pixels_safely(cube)]
        if not all_pixels:
            return None
        
        y_coords, x_coords = zip(*all_pixels)
        y_min, y_max, x_min, x_max = min(y_coords), max(y_coords), min(x_coords), max(x_coords)
        
        # Create maps
        seg_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        for i, cube in enumerate(cubes):
            for y, x in self._get_pixels_safely(cube):
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    seg_map[y - y_min, x - x_min] = i
                    ndvi_map[y - y_min, x - x_min] = cube['mean_ndvi']
        
        # Create figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=['NDVI Distribution', 'Cluster Boundaries'],
                           specs=[[{"type": "heatmap"}, {"type": "scatter"}]])
        
        # Add NDVI heatmap
        fig.add_trace(go.Heatmap(z=ndvi_map, x=list(range(x_min, x_max + 1)), y=list(range(y_min, y_max + 1)),
                                colorscale='RdYlGn', name='NDVI', colorbar=dict(title="Mean NDVI", x=0.48),
                                hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'), row=1, col=1)
        
        # Add cluster points
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if pixels:
                y_vals, x_vals = zip(*pixels)
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers',
                                       marker=dict(size=3, color=self.color_palette[i % len(self.color_palette)], 
                                                  line=dict(width=1, color='black')),
                                       name=f'Cluster {i} (NDVI: {cube["mean_ndvi"]:.3f})',
                                       hovertemplate=f'Cluster {i}<br>Area: {cube["area"]} pixels<br>Mean NDVI: {cube["mean_ndvi"]:.3f}<extra></extra>'), row=1, col=2)
        
        # Update layout
        fig.update_layout(title=f'{title}<br>Total Clusters: {len(cubes)}', width=1400, height=600, hovermode='closest')
        for col in [1, 2]:
            fig.update_xaxes(title_text="X Coordinate", row=1, col=col)
            fig.update_yaxes(title_text="Y Coordinate", row=1, col=col)
        
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
            
            logger.debug(f"Attempting {basemap_mode} basemap creation...")
            
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

        # Add 3D spatiotemporal points
        for time_idx in range(min(15, n_time_steps)):  # Limit for performance
            actual_year = actual_years[time_idx]
            for cube_idx, cube in enumerate(valid_cubes[:8]):  # Limit clusters for performance
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

        # Update layout
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
            
            logger.debug(f"Found spectral bands: {list(bands.keys())}")
            
            # Create composite based on available bands
            if 'nir' in bands and 'red' in bands and 'green' in bands:
                composite = 0.6 * bands['nir'] + 0.3 * bands['red'] + 0.1 * bands['green']
                logger.debug("Using NIR-Red-Green false color composite")
            elif 'red' in bands and 'green' in bands and 'blue' in bands:
                composite = 0.3 * bands['red'] + 0.6 * bands['green'] + 0.1 * bands['blue']
                logger.debug("Using RGB natural color composite")
            elif 'nir' in bands and 'red' in bands:
                with np.errstate(divide='ignore', invalid='ignore'):
                    composite = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'])
                logger.debug("Using NDVI-like composite")
            else:
                composite = list(bands.values())[0]
                logger.debug("Using single band fallback")
            
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
                    name='RGB Basemap',
                    showscale=False
                ))
                
                logger.debug("RGB basemap created successfully")
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
                
            min_ndvi, max_ndvi = np.min(valid_ndvi), np.max(valid_ndvi)
            logger.debug(f"NDVI range: {min_ndvi:.3f} to {max_ndvi:.3f}")
            
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
                name='NDVI Basemap',
                showscale=False
            ))
            
            logger.debug("NDVI basemap created successfully")
            return True
            
        except Exception as e:
            logger.error(f"NDVI basemap creation failed: {e}")
            return False


if __name__ == "__main__":
    print("Interactive Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides visualization tools for vegetation clustering results.")
    print("Use by importing InteractiveVisualization class and calling create_all_visualizations().")