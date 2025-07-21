#!/usr/bin/env python3
"""
Interactive HTML Plotly Visualization for Vegetation ST-Cube Segmentation Results

This script creates interactive HTML visualizations using Plotly for
vegetation-focused ST-cube segmentation results, focusing on spatial patterns and NDVI evolution.
"""

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
warnings.filterwarnings('ignore')

# Add the scripts directory to Python path
import sys
import os
scripts_dir = Path(__file__).parent.parent.parent
sys.path.append(str(scripts_dir))

# Import only the cube module to avoid circular imports
try:
    from .cube import STCube
except ImportError:
    from cube import STCube


class InteractiveVisualization:
    """
    Interactive HTML visualization generator for vegetation ST-cube segmentation results.
    
    Creates interactive Plotly visualizations including spatial maps, time series,
    statistics dashboards, and 3D surfaces.
    """
    
    def __init__(self, output_directory: str = "outputs/interactive_vegetation"):
        """
        Initialize the visualization generator.
        
        Args:
            output_directory: Directory where HTML files will be saved
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = px.colors.qualitative.Set3
    
    def _get_pixels_safely(self, cube: Dict) -> List[Tuple[int, int]]:
        """Safely extract pixels from cube data, handling different formats."""
        pixels = cube.get('pixels', [])
        if pixels is None:
            return []
        
        if isinstance(pixels, np.ndarray):
            if pixels.size == 0:
                return []
            # Convert numpy array to list of tuples
            if pixels.ndim == 1 and len(pixels) == 2:
                return [tuple(pixels.tolist())]
            elif pixels.ndim == 2:
                return [tuple(row) for row in pixels.tolist()]
            else:
                return pixels.tolist()
        elif isinstance(pixels, list):
            return pixels
        else:
            return []
    
    def _has_valid_pixels(self, cube: Dict) -> bool:
        """Check if cube has valid pixel data."""
        return len(self._get_pixels_safely(cube)) > 0
    
    def create_all_visualizations(self, 
                                cubes: Union[List[STCube], List[Dict]], 
                                data: xr.Dataset, 
                                municipality_name: str = "Unknown") -> Dict[str, str]:
        """
        Create all visualizations for vegetation clusters.
        
        Args:
            cubes: List of STCube objects or dictionaries with cluster data
            data: Original xarray dataset
            municipality_name: Name of the municipality for titles
            
        Returns:
            Dictionary with visualization names and file paths
        """
        print(f"Creating comprehensive visualizations for {municipality_name}...")
        
        # Convert dict cubes to appropriate format if needed
        processed_cubes = self._process_cube_data(cubes, data)
        
        visualizations = {}
        
        try:
            # 1. Interactive spatial map
            spatial_file = f"spatial_map_{municipality_name.replace(' ', '_')}.html"
            self.create_interactive_spatial_map(processed_cubes, spatial_file, 
                                              f"Vegetation Clusters - {municipality_name}")
            visualizations["spatial_map"] = str(self.output_dir / spatial_file)
            
            # 2. NDVI time series for each cluster
            time_series_file = f"ndvi_time_series_{municipality_name.replace(' ', '_')}.html"
            self.create_interactive_time_series(processed_cubes, data, time_series_file,
                                              f"NDVI Evolution - {municipality_name}")
            visualizations["time_series"] = str(self.output_dir / time_series_file)
            
            # 3. 3D spatiotemporal visualization
            print("Creating 3D spatiotemporal visualization...")
            spatiotemporal_3d_file = f"3d_spatiotemporal_{municipality_name.replace(' ', '_')}.html"
            try:
                result = self.create_3d_spatiotemporal_visualization(processed_cubes, data, spatiotemporal_3d_file,
                                                          f"3D Spatiotemporal View - {municipality_name}")
                if result is not None:
                    visualizations["3d_spatiotemporal"] = str(self.output_dir / spatiotemporal_3d_file)
                    print(f"✅ 3D spatiotemporal visualization created successfully")
                else:
                    print(f"❌ 3D spatiotemporal visualization returned None")
            except Exception as e:
                print(f"❌ Error in 3D spatiotemporal visualization: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 4. Statistics dashboard
            stats_file = f"statistics_dashboard_{municipality_name.replace(' ', '_')}.html"
            self.create_interactive_statistics_dashboard(processed_cubes, stats_file,
                                                       f"Vegetation Statistics - {municipality_name}")
            visualizations["statistics"] = str(self.output_dir / stats_file)
            
            # 5. Individual cluster analysis
            cluster_analysis_file = f"cluster_analysis_{municipality_name.replace(' ', '_')}.html"
            self.create_individual_cluster_analysis(processed_cubes, data, cluster_analysis_file,
                                                  f"Individual Cluster Analysis - {municipality_name}")
            visualizations["cluster_analysis"] = str(self.output_dir / cluster_analysis_file)
            
            print(f"All visualizations created successfully in: {self.output_dir}")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return visualizations
    
    def _process_cube_data(self, cubes: Union[List[STCube], List[Dict]], data: xr.Dataset) -> List[Dict]:
        """Process cube data into a consistent format for visualization."""
        processed_cubes = []
        
        for i, cube in enumerate(cubes):
            if isinstance(cube, STCube):
                # Convert STCube to dict format
                cube_dict = {
                    'id': i,
                    'pixels': list(cube.pixels) if hasattr(cube, 'pixels') else [],
                    'area': cube.area if hasattr(cube, 'area') else 0,
                    'ndvi_profile': cube.ndvi_profile if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else [],
                    'mean_ndvi': np.mean(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5,
                    'temporal_extent': cube.temporal_extent if hasattr(cube, 'temporal_extent') else (0, 0),
                    'heterogeneity': cube.heterogeneity if hasattr(cube, 'heterogeneity') else 0.0,
                    'vegetation_type': 'Unknown',
                    'seasonality_score': 0.0,
                    'trend_score': 0.0
                }
            elif isinstance(cube, dict):
                # Already in dict format, ensure all required keys exist
                # Get pixels - prioritize coordinates if pixels is empty
                pixels = cube.get('pixels', [])
                if not pixels:
                    pixels = cube.get('coordinates', [])
                
                # Get NDVI profile - prioritize mean_temporal_profile for segmentation data
                ndvi_profile = cube.get('ndvi_profile', [])
                if ndvi_profile is None or (hasattr(ndvi_profile, '__len__') and len(ndvi_profile) == 0):
                    ndvi_profile = cube.get('mean_temporal_profile', [])
                if ndvi_profile is None or (hasattr(ndvi_profile, '__len__') and len(ndvi_profile) == 0):
                    ndvi_profile = cube.get('ndvi_time_series', [])
                
                cube_dict = {
                    'id': cube.get('id', i),
                    'pixels': pixels,
                    'area': cube.get('area', cube.get('size', 0)),
                    'ndvi_profile': ndvi_profile,
                    'mean_ndvi': cube.get('mean_ndvi', 0.5),
                    'temporal_extent': cube.get('temporal_extent', (0, len(data.time) if 'time' in data.dims else 1)),
                    'heterogeneity': cube.get('heterogeneity', cube.get('temporal_variance', 0.0)),
                    'vegetation_type': cube.get('vegetation_type', 'Unknown'),
                    'seasonality_score': cube.get('seasonality_score', 0.0),
                    'trend_score': cube.get('trend_score', 0.0)
                }
            else:
                continue
                
            processed_cubes.append(cube_dict)
        
        return processed_cubes
    
    def create_interactive_spatial_map(self, cubes: List[Dict], filename: str, title: str = "Vegetation Clusters"):
        """
        Create an interactive spatial map showing vegetation cluster boundaries and NDVI patterns.
        
        Args:
            cubes: List of processed cube dictionaries
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating interactive spatial map: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Get spatial extent
        all_pixels = []
        for cube in cubes:
            pixels = cube.get('pixels', [])
            if pixels is not None and len(pixels) > 0:
                # Handle both list of tuples and numpy arrays
                if isinstance(pixels, np.ndarray):
                    if pixels.size > 0:
                        all_pixels.extend(pixels.tolist() if pixels.ndim > 1 else [pixels.tolist()])
                else:
                    all_pixels.extend(pixels)
        
        if not all_pixels:
            print("Warning: No valid pixels found in cubes")
            return None
        
        # Create spatial grid
        y_coords = [p[0] for p in all_pixels]
        x_coords = [p[1] for p in all_pixels]
        
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Create segmentation map
        segmentation_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        # Fill maps with cube data
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if not pixels:
                continue
            
            for y, x in pixels:
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    segmentation_map[y - y_min, x - x_min] = i
                    ndvi_map[y - y_min, x - x_min] = cube['mean_ndvi']
        
        # Create the plotly figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['NDVI Distribution', 'Cluster Boundaries'],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Add NDVI heatmap
        fig.add_trace(
            go.Heatmap(
                z=ndvi_map,
                x=list(range(x_min, x_max + 1)),
                y=list(range(y_min, y_max + 1)),
                colorscale='RdYlGn',
                name='NDVI',
                colorbar=dict(title="Mean NDVI", x=0.48),
                hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add cluster boundaries as scatter plots
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if not pixels:
                continue
                
            color_idx = i % len(self.color_palette)
            y_coords = [p[0] for p in pixels]
            x_coords = [p[1] for p in pixels]
            
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=self.color_palette[color_idx],
                        line=dict(width=1, color='black')
                    ),
                    name=f'Cluster {i} (NDVI: {cube["mean_ndvi"]:.3f})',
                    hovertemplate=f'Cluster {i}<br>Area: {cube["area"]} pixels<br>Mean NDVI: {cube["mean_ndvi"]:.3f}<br>Type: {cube["vegetation_type"]}<extra></extra>',
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'{title}<br>Total Clusters: {len(cubes)}',
            width=1400,
            height=600,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="X Coordinate", row=1, col=1)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=1)
        fig.update_xaxes(title_text="X Coordinate", row=1, col=2)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=2)
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Interactive spatial map saved to: {output_file}")
        
        return fig
    
    def create_interactive_time_series(self, cubes: List[Dict], data: xr.Dataset, filename: str, title: str = "NDVI Time Series"):
        """
        Create an interactive time series plot showing NDVI evolution for each vegetation cube.
        
        Args:
            cubes: List of processed cube dictionaries
            data: The original xarray dataset with time information
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating interactive time series: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Get time coordinates
        if 'time' in data.dims:
            time_coords = pd.to_datetime(data.time.values)
        else:
            time_coords = list(range(len(cubes[0]['ndvi_profile']) if cubes[0]['ndvi_profile'] else 10))
        
        # Create subplots for different views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'All Clusters NDVI Evolution',
                'Individual Cluster Analysis',
                'NDVI Distribution Over Time',
                'Cluster Statistics'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Plot 1: All clusters time series
        valid_cubes = [cube for cube in cubes if cube['ndvi_profile'] and len(cube['ndvi_profile']) > 0]
        
        for i, cube in enumerate(valid_cubes[:10]):  # Limit to first 10 for readability
            color_idx = i % len(self.color_palette)
            
            fig.add_trace(
                go.Scatter(
                    x=time_coords,
                    y=cube['ndvi_profile'],
                    mode='lines+markers',
                    name=f'Cluster {i} (Area: {cube["area"]})',
                    line=dict(color=self.color_palette[color_idx], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'Cluster {i}<br>Time: %{{x}}<br>NDVI: %{{y:.3f}}<br>Area: {cube["area"]} pixels<br>Type: {cube["vegetation_type"]}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Select interesting clusters for detailed view
        interesting_clusters = sorted(valid_cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:3]
        
        for i, cube in enumerate(interesting_clusters):
            fig.add_trace(
                go.Scatter(
                    x=time_coords,
                    y=cube['ndvi_profile'],
                    mode='lines+markers',
                    name=f'High Seasonality Cluster {cube["id"]}',
                    line=dict(width=3),
                    marker=dict(size=6),
                    hovertemplate=f'Cluster {cube["id"]}<br>Time: %{{x}}<br>NDVI: %{{y:.3f}}<br>Seasonality: {cube.get("seasonality_score", 0):.3f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: NDVI heatmap over time
        if valid_cubes:
            ndvi_matrix = np.array([cube['ndvi_profile'][:len(time_coords)] for cube in valid_cubes[:20]])
            
            fig.add_trace(
                go.Heatmap(
                    z=ndvi_matrix,
                    x=time_coords,
                    y=[f'Cluster {i}' for i in range(len(ndvi_matrix))],
                    colorscale='RdYlGn',
                    name='NDVI Matrix',
                    hovertemplate='Time: %{x}<br>Cluster: %{y}<br>NDVI: %{z:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Cluster statistics
        mean_ndvis = [cube['mean_ndvi'] for cube in valid_cubes]
        cluster_names = [f'Cluster {cube["id"]}' for cube in valid_cubes]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names[:10],
                y=mean_ndvis[:10],
                name='Mean NDVI',
                marker_color='green',
                hovertemplate='%{x}<br>Mean NDVI: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=1600,
            height=900,
            hovermode='closest',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="NDVI Value", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="NDVI Value", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Cluster", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Mean NDVI", row=2, col=2)
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"NDVI time series plot saved to: {output_file}")
        
        return fig
    
    def create_3d_spatiotemporal_visualization(self, cubes: List[Dict], data: xr.Dataset, filename: str, title: str = "3D Spatiotemporal View"):
        """
        Create a 3D visualization with X,Y spatial coordinates and Time (Z) axis.
        
        This creates exactly what you requested:
        - X,Y axes = spatial coordinates
        - Z axis = time dimension  
        - 3D cubes = individual pixels evolving over time
        - Color = NDVI values (greener = higher NDVI)
        - Basemap = X,Y plane showing spatial context
        
        Args:
            cubes: List of processed cube dictionaries
            data: Original xarray dataset
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating 3D pixel evolution visualization: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Process cube data to ensure consistent format
        cubes = self._process_cube_data(cubes, data)
        
        # Get time coordinates
        if 'time' in data.dims:
            time_coords = np.arange(len(data.time))
            time_labels = [str(t)[:10] for t in pd.to_datetime(data.time.values)]
        else:
            time_coords = np.arange(10)
            time_labels = [f"Time {i}" for i in time_coords]
        
        # Prepare data for 3D visualization
        fig = go.Figure()
        
        # Filter valid cubes with both spatial and temporal data
        print(f"🔍 Filtering {len(cubes)} cubes for 3D visualization...")
        valid_cubes = []
        for i, cube in enumerate(cubes):
            has_pixels = self._has_valid_pixels(cube)
            ndvi_profile = cube.get('ndvi_profile', [])
            has_ndvi = ndvi_profile is not None and len(ndvi_profile) > 0
            print(f"   Cube {i}: pixels={has_pixels}, ndvi_profile={has_ndvi} (len={len(ndvi_profile) if ndvi_profile is not None else 0})")
            if has_pixels and has_ndvi:
                valid_cubes.append(cube)
        
        print(f"🎯 Found {len(valid_cubes)} valid cubes for 3D visualization")
        
        if not valid_cubes:
            print("Warning: No valid cubes with spatial and temporal data")
            return None
        
        # Get all pixels from all cubes for spatial extent
        all_pixels = []
        pixel_to_cube = {}  # Map pixel coordinates to cube info
        
        for cube_idx, cube in enumerate(valid_cubes):
            pixels = self._get_pixels_safely(cube)
            all_pixels.extend(pixels)
            
            # Map each pixel to its cube information
            for pixel in pixels:
                pixel_to_cube[pixel] = {
                    'cube_idx': cube_idx,
                    'ndvi_profile': cube.get('ndvi_profile', []),
                    'area': cube.get('area', 1),
                    'mean_ndvi': cube.get('mean_ndvi', 0.5)
                }
        
        if not all_pixels:
            print("Warning: No pixels found in cubes")
            return None
        
        y_coords = [p[0] for p in all_pixels]
        x_coords = [p[1] for p in all_pixels]
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        print(f"Creating 3D visualization for {len(all_pixels)} pixels over {len(time_coords)} time steps")
        
        # 1. CREATE BASEMAP ON X,Y PLANE (Z=0)
        # Create a grid for the basemap
        x_grid_size = min(50, x_max - x_min + 1)
        y_grid_size = min(50, y_max - y_min + 1)
        
        x_basemap = np.linspace(x_min, x_max, x_grid_size)
        y_basemap = np.linspace(y_min, y_max, y_grid_size)
        X_base, Y_base = np.meshgrid(x_basemap, y_basemap)
        Z_base = np.zeros_like(X_base)
        
        # Create NDVI basemap by interpolating from pixel data
        basemap_ndvi = np.full_like(X_base, 0.3)  # Default background NDVI
        
        # Fill basemap with actual NDVI values where we have data
        for pixel, cube_info in pixel_to_cube.items():
            px_y, px_x = pixel
            # Find closest grid point
            x_idx = np.argmin(np.abs(x_basemap - px_x))
            y_idx = np.argmin(np.abs(y_basemap - px_y))
            basemap_ndvi[y_idx, x_idx] = cube_info['mean_ndvi']
        
        # Add basemap surface
        fig.add_trace(go.Surface(
            x=X_base,
            y=Y_base,
            z=Z_base,
            surfacecolor=basemap_ndvi,
            colorscale='RdYlGn',
            cmin=0.0,
            cmax=1.0,
            opacity=0.7,
            name='NDVI Basemap',
            showscale=True,
            colorbar=dict(
                title="Basemap NDVI",
                x=1.1,
                len=0.3,
                y=0.8
            ),
            hovertemplate='Basemap<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>NDVI: %{surfacecolor:.3f}<extra></extra>'
        ))
        
        # 2. CREATE 3D CUBES FOR EACH PIXEL AT EACH TIME STEP
        # Sample pixels for performance (limit to ~200 pixels max)
        max_pixels = 200
        if len(all_pixels) > max_pixels:
            step = len(all_pixels) // max_pixels
            sampled_pixels = all_pixels[::step]
            print(f"Sampling {len(sampled_pixels)} pixels for visualization performance")
        else:
            sampled_pixels = all_pixels
        
        # Limit time steps for performance
        max_time_steps = min(12, len(time_coords))
        sampled_time_coords = time_coords[:max_time_steps]
        
        # Create 3D cubes for each pixel at each time step
        for t_idx, t in enumerate(sampled_time_coords):
            z_position = t_idx + 1  # Start at z=1 (above basemap)
            
            for pixel in sampled_pixels:
                cube_info = pixel_to_cube.get(pixel)
                if not cube_info or t_idx >= len(cube_info['ndvi_profile']):
                    continue
                
                px_y, px_x = pixel
                ndvi_value = cube_info['ndvi_profile'][t_idx]
                
                # Skip very low NDVI values for cleaner visualization
                if ndvi_value < 0.2:
                    continue
                
                # Color based on NDVI (greener = higher NDVI)
                if ndvi_value >= 0.7:
                    color = 'darkgreen'
                    opacity = 0.9
                elif ndvi_value >= 0.5:
                    color = 'green'
                    opacity = 0.8
                elif ndvi_value >= 0.4:
                    color = 'lightgreen'
                    opacity = 0.7
                else:
                    color = 'yellow'
                    opacity = 0.6
                
                # Size based on NDVI (higher NDVI = bigger cube)
                cube_size = max(3, min(15, ndvi_value * 20))
                
                # Add 3D cube as a marker
                fig.add_trace(go.Scatter3d(
                    x=[px_x],
                    y=[px_y],
                    z=[z_position],
                    mode='markers',
                    marker=dict(
                        size=cube_size,
                        color=color,
                        opacity=opacity,
                        symbol='square',  # Use square instead of cube (cube not supported)
                        line=dict(width=1, color='black')
                    ),
                    name=f'T{t_idx}',
                    legendgroup=f'time_{t_idx}',
                    showlegend=(pixel == sampled_pixels[0]),  # Only show legend for first pixel of each time
                    hovertemplate=f'Pixel ({px_x}, {px_y})<br>' +
                                f'Time: {time_labels[t_idx] if t_idx < len(time_labels) else f"T{t}"}<br>' +
                                f'NDVI: {ndvi_value:.3f}<br>' +
                                f'Cluster: {cube_info["cube_idx"]}<extra></extra>'
                ))
        
        # 3. ADD TIME AXIS LABELS
        # Create time axis markers
        for t_idx, t in enumerate(sampled_time_coords):
            z_position = t_idx + 1
            time_label = time_labels[t_idx] if t_idx < len(time_labels) else f"T{t}"
            
            # Add time axis label
            fig.add_trace(go.Scatter3d(
                x=[x_max + 2],
                y=[y_max],
                z=[z_position],
                mode='markers+text',
                marker=dict(size=8, color='black', symbol='diamond'),
                text=[time_label],
                textposition='middle right',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 4. UPDATE LAYOUT FOR 3D VISUALIZATION
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>X,Y=Coordinates | Z=Time | Color=NDVI | Size=Vegetation Density</sub>",
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title='X Coordinate (Longitude)',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='lightgray',
                    showbackground=True,
                    zerolinecolor='gray'
                ),
                yaxis=dict(
                    title='Y Coordinate (Latitude)', 
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='lightgray',
                    showbackground=True,
                    zerolinecolor='gray'
                ),
                zaxis=dict(
                    title='Time Progression →',
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='lightgray',
                    showbackground=True,
                    zerolinecolor='gray',
                    range=[0, max_time_steps + 1]
                ),
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5),
                    center=dict(x=0, y=0, z=0.2)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.8)
            ),
            width=1200,
            height=900,
            hovermode='closest',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        # Add annotation explaining the visualization
        fig.add_annotation(
            text="🌱 Each cube = 1 pixel for 1 year | Greener = Higher NDVI | Basemap shows spatial context",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12, color="darkgreen"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="green",
            borderwidth=1
        )
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"🎯 3D pixel evolution visualization saved to: {output_file}")
        print(f"   📊 Shows {len(sampled_pixels)} pixels over {max_time_steps} time steps")
        print(f"   🌍 Basemap shows NDVI spatial context on X,Y plane")
        print(f"   📈 Z-axis shows temporal evolution")
        
        return fig
    
    def create_interactive_statistics_dashboard(self, cubes: List[Dict], filename: str, title: str = "Vegetation Statistics"):
        """
        Create an interactive dashboard with vegetation cube statistics.
        
        Args:
            cubes: List of processed cube dictionaries
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating statistics dashboard: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Prepare data
        cube_data = []
        for cube in cubes:
            ndvi_profile = cube.get('ndvi_profile', [])
            if ndvi_profile:
                max_ndvi = np.max(ndvi_profile)
                min_ndvi = np.min(ndvi_profile)
                ndvi_range = max_ndvi - min_ndvi
            else:
                max_ndvi = min_ndvi = ndvi_range = 0
            
            cube_data.append({
                'cube_id': cube['id'],
                'area': cube['area'],
                'mean_ndvi': cube['mean_ndvi'],
                'max_ndvi': max_ndvi,
                'min_ndvi': min_ndvi,
                'ndvi_range': ndvi_range,
                'temporal_variance': cube.get('heterogeneity', 0),
                'seasonality_score': cube.get('seasonality_score', 0),
                'vegetation_type': cube.get('vegetation_type', 'Unknown')
            })
        
        df = pd.DataFrame(cube_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cube Area Distribution',
                'NDVI vs Area (colored by vegetation type)',
                'Seasonality vs Mean NDVI',
                'Vegetation Type Distribution'
            ],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Area distribution
        fig.add_trace(
            go.Histogram(
                x=df['area'],
                nbinsx=20,
                name='Area Distribution',
                marker_color='lightblue',
                hovertemplate='Area: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # NDVI vs Area scatter (colored by vegetation type)
        veg_types = df['vegetation_type'].unique()
        colors = px.colors.qualitative.Set1
        
        for i, veg_type in enumerate(veg_types):
            subset = df[df['vegetation_type'] == veg_type]
            fig.add_trace(
                go.Scatter(
                    x=subset['area'],
                    y=subset['mean_ndvi'],
                    mode='markers',
                    name=veg_type,
                    marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7),
                    text=[f'Cluster {i}' for i in subset['cube_id']],
                    hovertemplate=f'{veg_type}<br>Cluster: %{{text}}<br>Area: %{{x}}<br>NDVI: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Seasonality vs Mean NDVI
        fig.add_trace(
            go.Scatter(
                x=df['mean_ndvi'],
                y=df['seasonality_score'],
                mode='markers',
                name='Seasonality Analysis',
                marker=dict(
                    size=df['area']/10,  # Size represents area
                    color=df['seasonality_score'],
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="Seasonality")
                ),
                text=[f'Cluster {i}' for i in df['cube_id']],
                hovertemplate='Cluster: %{text}<br>Mean NDVI: %{x:.3f}<br>Seasonality: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Vegetation type pie chart
        type_counts = df['vegetation_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                name='Vegetation Types',
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=1400,
            height=900,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Area (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Area (pixels)", row=1, col=2)
        fig.update_yaxes(title_text="Mean NDVI", row=1, col=2)
        fig.update_xaxes(title_text="Mean NDVI", row=2, col=1)
        fig.update_yaxes(title_text="Seasonality Score", row=2, col=1)
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Statistics dashboard saved to: {output_file}")
        
        return fig
    
    def create_individual_cluster_analysis(self, cubes: List[Dict], data: xr.Dataset, filename: str, title: str = "Individual Cluster Analysis"):
        """
        Create detailed analysis for individual clusters.
        
        Args:
            cubes: List of processed cube dictionaries
            data: Original xarray dataset
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating individual cluster analysis: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Select most interesting clusters for detailed analysis
        interesting_cubes = sorted(cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:6]
        
        # Create subplots for each interesting cluster
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Cluster {cube["id"]} - {cube.get("vegetation_type", "Unknown")}' for cube in interesting_cubes],
            specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)]
        )
        
        # Get time coordinates
        if 'time' in data.dims:
            time_coords = pd.to_datetime(data.time.values)
        else:
            time_coords = list(range(len(interesting_cubes[0]['ndvi_profile']) if interesting_cubes[0]['ndvi_profile'] else 10))
        
        # Plot each cluster
        for idx, cube in enumerate(interesting_cubes):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            if cube['ndvi_profile']:
                # NDVI time series
                fig.add_trace(
                    go.Scatter(
                        x=time_coords,
                        y=cube['ndvi_profile'],
                        mode='lines+markers',
                        name=f'Cluster {cube["id"]}',
                        line=dict(color=self.color_palette[idx % len(self.color_palette)], width=3),
                        marker=dict(size=6),
                        hovertemplate=f'Cluster {cube["id"]}<br>' +
                                    f'Time: %{{x}}<br>' +
                                    f'NDVI: %{{y:.3f}}<br>' +
                                    f'Type: {cube.get("vegetation_type", "Unknown")}<br>' +
                                    f'Area: {cube["area"]} pixels<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add trend line
                if len(cube['ndvi_profile']) > 2:
                    x_numeric = np.arange(len(cube['ndvi_profile']))
                    z = np.polyfit(x_numeric, cube['ndvi_profile'], 1)
                    trend_line = np.poly1d(z)(x_numeric)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_coords,
                            y=trend_line,
                            mode='lines',
                            name=f'Trend {cube["id"]}',
                            line=dict(color='red', width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=1500,
            height=800,
            hovermode='closest'
        )
        
        # Update all axes
        for i in range(1, 3):  # rows
            for j in range(1, 4):  # cols
                fig.update_xaxes(title_text="Time", row=i, col=j)
                fig.update_yaxes(title_text="NDVI", row=i, col=j)
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Individual cluster analysis saved to: {output_file}")
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("Interactive Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides comprehensive visualization tools for vegetation clustering results.")
    print("Use this module by importing InteractiveVisualization class and calling create_all_visualizations().")
