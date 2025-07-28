#!/usr/bin/env python3
"""
Interactive HTML Plotly Visualization for Vegetation ST-Cube Segmentation Results

This script creates interactive HTML visualizations using Plotly for
vegetation-focused ST-cube segmentation results, focusing on spatial patterns and NDVI evolution.
"""

# ==== CONFIGURABLE PARAMETERS ====
DEFAULT_OUTPUT_DIRECTORY = "outputs/interactive_vegetation"    # Default output directory
DEFAULT_MAX_CLUSTERS_TO_DISPLAY = 50                          # Maximum clusters to display
DEFAULT_DOWNSAMPLE_FACTOR = 2                                 # Downsampling factor for large datasets
DEFAULT_COLOR_PALETTE = "Set3"                                # Default color palette
DEFAULT_FIGURE_WIDTH = 1200                                   # Default figure width
DEFAULT_FIGURE_HEIGHT = 800                                   # Default figure height
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

# Add the scripts directory to Python path
import sys
import os
scripts_dir = Path(__file__).parent.parent.parent
sys.path.append(str(scripts_dir))

# Import only the cube module to avoid circular imports
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
                # Handle different formats
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
    
    def create_all_visualizations(self, 
                                cubes: Union[List[STCube], List[Dict]], 
                                data: Union[xr.Dataset, str], 
                                municipality_name: str = "Unknown") -> Dict[str, str]:
        """Create all visualizations for vegetation clusters."""
        logger.info(f"Creating comprehensive visualizations for {municipality_name}...")
        
        # Convert and process cube data
        processed_cubes = self._process_cube_data(cubes, data)
        visualizations = {}
        
        # Create all visualizations
        viz_configs = [
            ("spatial_map", self.create_interactive_spatial_map, f"Vegetation Clusters - {municipality_name}"),
            ("time_series", self.create_interactive_time_series, f"NDVI Evolution - {municipality_name}"),
            ("3d_spatiotemporal", self.create_3d_spatiotemporal_visualization, f"3D Spatiotemporal View - {municipality_name}"),
            ("statistics", self.create_interactive_statistics_dashboard, f"Vegetation Statistics - {municipality_name}"),
            ("cluster_analysis", self.create_individual_cluster_analysis, f"Individual Cluster Analysis - {municipality_name}")
        ]
        
        for viz_name, viz_func, title in viz_configs:
            filename = f"{viz_name}_{municipality_name.replace(' ', '_')}.html"
            try:
                if viz_name == "time_series" or viz_name == "cluster_analysis" or viz_name == "3d_spatiotemporal":
                    result = viz_func(processed_cubes, data, filename, title)
                else:
                    result = viz_func(processed_cubes, filename, title)
                
                if result is not None:
                    visualizations[viz_name] = str(self.output_dir / filename)
                    logger.success(f"{viz_name} created successfully")
                else:
                    logger.warning(f"{viz_name} returned None")
            except Exception as e:
                logger.error(f"Error in {viz_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        logger.success(f"All visualizations created successfully in: {self.output_dir}")
        return visualizations
    
    def _process_cube_data(self, cubes: Union[List[STCube], List[Dict]], data: Union[xr.Dataset, str]) -> List[Dict]:
        """Process cube data into a consistent format for visualization."""
        processed_cubes = []
        
        # Determine time dimension safely
        time_length = 1
        if hasattr(data, 'dims') and 'time' in data.dims:
            time_length = len(data.time)
        
        for i, cube in enumerate(cubes):
            if isinstance(cube, STCube):
                # Convert STCube to dict format
                cube_dict = {
                    'id': i,
                    'pixels': getattr(cube, 'pixels', []),
                    'area': getattr(cube, 'area', 0),
                    'ndvi_profile': getattr(cube, 'ndvi_profile', []),
                    'mean_ndvi': np.mean(getattr(cube, 'ndvi_profile', [0.5])),
                    'temporal_extent': getattr(cube, 'temporal_extent', (0, 0)),
                    'heterogeneity': getattr(cube, 'heterogeneity', 0.0),
                    'vegetation_type': 'Unknown',
                    'seasonality_score': 0.0,
                    'trend_score': 0.0
                }
            else:
                # Process dict format
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
        print(f"Creating interactive spatial map: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Get all pixels and spatial extent
        all_pixels = []
        for cube in cubes:
            all_pixels.extend(self._get_pixels_safely(cube))
        
        if not all_pixels:
            print("Warning: No valid pixels found in cubes")
            return None
        
        y_coords, x_coords = zip(*all_pixels)
        y_min, y_max, x_min, x_max = min(y_coords), max(y_coords), min(x_coords), max(x_coords)
        
        # Create maps
        seg_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        # Fill maps
        for i, cube in enumerate(cubes):
            for y, x in self._get_pixels_safely(cube):
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    seg_map[y - y_min, x - x_min] = i
                    ndvi_map[y - y_min, x - x_min] = cube['mean_ndvi']
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['NDVI Distribution', 'Cluster Boundaries'],
            specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # NDVI heatmap
        fig.add_trace(go.Heatmap(
            z=ndvi_map, x=list(range(x_min, x_max + 1)), y=list(range(y_min, y_max + 1)),
            colorscale='RdYlGn', name='NDVI', colorbar=dict(title="Mean NDVI", x=0.48),
            hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Cluster boundaries
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if pixels:
                y_vals, x_vals = zip(*pixels)
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='markers',
                    marker=dict(size=3, color=self.color_palette[i % len(self.color_palette)], 
                               line=dict(width=1, color='black')),
                    name=f'Cluster {i} (NDVI: {cube["mean_ndvi"]:.3f})',
                    hovertemplate=f'Cluster {i}<br>Area: {cube["area"]} pixels<br>Mean NDVI: {cube["mean_ndvi"]:.3f}<br>Type: {cube["vegetation_type"]}<extra></extra>',
                ), row=1, col=2)
        
        # Update layout
        fig.update_layout(title=f'{title}<br>Total Clusters: {len(cubes)}', width=1400, height=600, hovermode='closest')
        for col in [1, 2]:
            fig.update_xaxes(title_text="X Coordinate", row=1, col=col)
            fig.update_yaxes(title_text="Y Coordinate", row=1, col=col)
        
        # Save
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        print(f"Interactive spatial map saved to: {self.output_dir / filename}")
        return fig
    
    def create_interactive_time_series(self, cubes: List[Dict], data: Union[xr.Dataset, str], filename: str, title: str = "NDVI Time Series"):
        """Create an interactive time series plot showing NDVI evolution for each vegetation cube."""
        print(f"Creating interactive time series: {filename}")
        
        if not cubes:
            return None
        
        # Get time coordinates - handle both Dataset and non-Dataset inputs
        if hasattr(data, 'dims') and 'time' in data.dims:
            time_coords = pd.to_datetime(data.time.values)
        else:
            # Fallback: use first valid cube to determine time dimension
            first_valid = next((c for c in cubes if self._get_ndvi_profile(c)), None)
            if first_valid:
                time_coords = list(range(len(self._get_ndvi_profile(first_valid))))
            else:
                time_coords = list(range(10))  # Default fallback
        
        # Filter valid cubes
        valid_cubes = [c for c in cubes if self._is_valid_cube(c)]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['All Clusters NDVI Evolution', 'Individual Cluster Analysis', 
                           'NDVI Distribution Over Time', 'Cluster Statistics'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Plot 1: All clusters (limit to 10 for readability)
        for i, cube in enumerate(valid_cubes[:10]):
            fig.add_trace(go.Scatter(
                x=time_coords, y=cube['ndvi_profile'], mode='lines+markers',
                name=f'Cluster {i} (Area: {cube["area"]})',
                line=dict(color=self.color_palette[i % len(self.color_palette)], width=2),
                hovertemplate=f'Cluster {i}<br>Time: %{{x}}<br>NDVI: %{{y:.3f}}<br>Area: {cube["area"]} pixels<extra></extra>'
            ), row=1, col=1)
        
        # Plot 2: High seasonality clusters
        interesting_clusters = sorted(valid_cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:3]
        for cube in interesting_clusters:
            fig.add_trace(go.Scatter(
                x=time_coords, y=cube['ndvi_profile'], mode='lines+markers',
                name=f'High Seasonality Cluster {cube["id"]}', line=dict(width=3),
                hovertemplate=f'Cluster {cube["id"]}<br>Seasonality: {cube.get("seasonality_score", 0):.3f}<extra></extra>'
            ), row=1, col=2)
        
        # Plot 3: NDVI heatmap
        if valid_cubes:
            ndvi_matrix = np.array([cube['ndvi_profile'][:len(time_coords)] for cube in valid_cubes[:20]])
            fig.add_trace(go.Heatmap(
                z=ndvi_matrix, x=time_coords, y=[f'Cluster {i}' for i in range(len(ndvi_matrix))],
                colorscale='RdYlGn', hovertemplate='Time: %{x}<br>Cluster: %{y}<br>NDVI: %{z:.3f}<extra></extra>'
            ), row=2, col=1)
        
        # Plot 4: Statistics bar chart
        fig.add_trace(go.Bar(
            x=[f'Cluster {c["id"]}' for c in valid_cubes[:10]],
            y=[c['mean_ndvi'] for c in valid_cubes[:10]],
            marker_color='green', hovertemplate='%{x}<br>Mean NDVI: %{y:.3f}<extra></extra>'
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(title=title, width=1600, height=900, hovermode='closest')
        axes_titles = [("Time", "NDVI Value"), ("Time", "NDVI Value"), ("Time", "Cluster"), ("Cluster", "Mean NDVI")]
        for i, (x_title, y_title) in enumerate(axes_titles):
            row, col = (i // 2) + 1, (i % 2) + 1
            fig.update_xaxes(title_text=x_title, row=row, col=col)
            fig.update_yaxes(title_text=y_title, row=row, col=col)
        
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        print(f"NDVI time series plot saved to: {self.output_dir / filename}")
        return fig
    
    def create_3d_spatiotemporal_visualization(self, cubes: List[Dict], data: Union[xr.Dataset, str], filename: str, title: str = "3D Spatiotemporal View"):
        """Create a 3D visualization with X,Y spatial coordinates and Years (Z) axis."""
        print(f"Creating 3D spatiotemporal visualization: {filename}")
        
        if not cubes:
            return None
        
        # Setup time coordinates - handle both Dataset and non-Dataset inputs
        if hasattr(data, 'dims') and 'time' in data.dims:
            n_time_steps = len(data.time)
            actual_years = [1984 + i for i in range(n_time_steps)]
            time_coords = np.arange(n_time_steps)
        else:
            # Fallback: use first valid cube to determine time dimension
            first_valid = next((c for c in cubes if self._is_valid_cube(c)), None)
            if not first_valid:
                return None
            n_time_steps = len(self._get_ndvi_profile(first_valid))
            actual_years = [1984 + i for i in range(n_time_steps)]
            time_coords = np.arange(n_time_steps)
        
        # Filter valid cubes
        valid_cubes = [c for c in cubes if self._is_valid_cube(c)]
        if not valid_cubes:
            return None
        
        # Get spatial extent
        all_pixels = []
        for cube in valid_cubes:
            all_pixels.extend(self._get_pixels_safely(cube))
        
        y_coords, x_coords = zip(*all_pixels)
        y_min, y_max, x_min, x_max = min(y_coords), max(y_coords), min(x_coords), max(x_coords)
        
        fig = go.Figure()
        
        # Create NDVI basemap
        x_base = np.linspace(x_min, x_max, min(50, x_max - x_min + 1))
        y_base = np.linspace(y_min, y_max, min(50, y_max - y_min + 1))
        X_base, Y_base = np.meshgrid(x_base, y_base)
        Z_base = np.full_like(X_base, 1984)
        
        # Create basemap from vegetation data
        basemap_ndvi = np.full((len(y_base), len(x_base)), 0.2)
        for cube in valid_cubes:
            pixels = self._get_pixels_safely(cube)
            mean_ndvi = np.clip(cube.get('mean_ndvi', 0.5), 0.0, 1.0)
            
            for px_y, px_x in pixels:
                x_idx = np.argmin(np.abs(x_base - px_x))
                y_idx = np.argmin(np.abs(y_base - px_y))
                if 0 <= y_idx < len(y_base) and 0 <= x_idx < len(x_base):
                    basemap_ndvi[y_idx, x_idx] = max(basemap_ndvi[y_idx, x_idx], mean_ndvi)
        
        # Add basemap
        fig.add_trace(go.Surface(
            x=X_base, y=Y_base, z=Z_base, surfacecolor=basemap_ndvi,
            colorscale='RdYlGn', cmin=0.0, cmax=1.0, opacity=0.7,
            showscale=True, hoverinfo='none',
            colorbar=dict(title="NDVI", x=1.1, len=0.5, y=0.75)
        ))
        
        # Add 3D cubes for clusters
        max_time_steps = min(15, len(time_coords))
        for time_idx in range(max_time_steps):
            actual_year = actual_years[time_idx]
            
            for cube_idx, cube in enumerate(valid_cubes[:8]):  # Limit for performance
                pixels = self._get_pixels_safely(cube)
                ndvi_profile = self._get_ndvi_profile(cube)
                
                if time_idx < len(ndvi_profile) and pixels:
                    px_coords = np.array(pixels)
                    y_vals, x_vals = px_coords[:, 0], px_coords[:, 1]
                    z_vals = np.full(len(pixels), actual_year)
                    ndvi_vals = np.full(len(pixels), ndvi_profile[time_idx])
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_vals, y=y_vals, z=z_vals, mode='markers',
                        marker=dict(size=3, color=ndvi_vals, colorscale='RdYlGn', 
                                  cmin=0.0, cmax=1.0, showscale=False, opacity=0.8),
                        name=f'Cluster {cube_idx+1} - {actual_year}',
                        text=[f'Cluster {cube_idx+1}<br>Year: {actual_year}<br>NDVI: {ndvi_profile[time_idx]:.3f}' 
                              for _ in range(len(pixels))],
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=(time_idx == 0)
                    ))
        
        # Update layout
        fig.update_layout(
            title=f'{title} - {len(valid_cubes)} Vegetation Clusters (1984-2025)',
            scene=dict(
                xaxis_title='X Coordinate', yaxis_title='Y Coordinate', zaxis_title='Year',
                zaxis=dict(tickmode='array', tickvals=[1984, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025],
                          ticktext=['1984', '1990', '1995', '2000', '2005', '2010', '2015', '2020', '2025']),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6)
            ),
            width=1200, height=800, margin=dict(l=0, r=0, t=50, b=0)
        )
        
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        print(f"3D spatiotemporal visualization saved to: {self.output_dir / filename}")
        return str(self.output_dir / filename)
    
    def create_interactive_statistics_dashboard(self, cubes: List[Dict], filename: str, title: str = "Vegetation Statistics"):
        """Create an interactive dashboard with vegetation cube statistics."""
        print(f"Creating statistics dashboard: {filename}")
        
        if not cubes:
            return None
        
        # Prepare data efficiently
        cube_data = []
        for cube in cubes:
            ndvi_profile = self._get_ndvi_profile(cube)
            if ndvi_profile:
                max_ndvi, min_ndvi = np.max(ndvi_profile), np.min(ndvi_profile)
                ndvi_range = max_ndvi - min_ndvi
            else:
                max_ndvi = min_ndvi = ndvi_range = 0
            
            cube_data.append({
                'cube_id': cube['id'], 'area': cube['area'], 'mean_ndvi': cube['mean_ndvi'],
                'max_ndvi': max_ndvi, 'min_ndvi': min_ndvi, 'ndvi_range': ndvi_range,
                'temporal_variance': cube.get('heterogeneity', 0),
                'seasonality_score': cube.get('seasonality_score', 0),
                'vegetation_type': cube.get('vegetation_type', 'Unknown')
            })
        
        df = pd.DataFrame(cube_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Cube Area Distribution', 'NDVI vs Area (colored by vegetation type)',
                           'Seasonality vs Mean NDVI', 'Vegetation Type Distribution'],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Area histogram
        fig.add_trace(go.Histogram(
            x=df['area'], nbinsx=20, marker_color='lightblue',
            hovertemplate='Area: %{x}<br>Count: %{y}<extra></extra>'
        ), row=1, col=1)
        
        # NDVI vs Area scatter by vegetation type
        colors = px.colors.qualitative.Set1
        for i, veg_type in enumerate(df['vegetation_type'].unique()):
            subset = df[df['vegetation_type'] == veg_type]
            fig.add_trace(go.Scatter(
                x=subset['area'], y=subset['mean_ndvi'], mode='markers',
                name=veg_type, marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7),
                text=[f'Cluster {i}' for i in subset['cube_id']],
                hovertemplate=f'{veg_type}<br>Cluster: %{{text}}<br>Area: %{{x}}<br>NDVI: %{{y:.3f}}<extra></extra>'
            ), row=1, col=2)
        
        # Seasonality vs NDVI
        fig.add_trace(go.Scatter(
            x=df['mean_ndvi'], y=df['seasonality_score'], mode='markers',
            marker=dict(size=df['area']/10, color=df['seasonality_score'], 
                       colorscale='Viridis', opacity=0.7, colorbar=dict(title="Seasonality")),
            text=[f'Cluster {i}' for i in df['cube_id']],
            hovertemplate='Cluster: %{text}<br>Mean NDVI: %{x:.3f}<br>Seasonality: %{y:.3f}<extra></extra>'
        ), row=2, col=1)
        
        # Vegetation type pie
        type_counts = df['vegetation_type'].value_counts()
        fig.add_trace(go.Pie(
            labels=type_counts.index, values=type_counts.values,
            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(title=title, width=1400, height=900, hovermode='closest')
        
        # Update axes
        axes_config = [("Area (pixels)", "Frequency"), ("Area (pixels)", "Mean NDVI"),
                      ("Mean NDVI", "Seasonality Score"), (None, None)]
        for i, (x_title, y_title) in enumerate(axes_config):
            if x_title:  # Skip pie chart
                row, col = (i // 2) + 1, (i % 2) + 1
                fig.update_xaxes(title_text=x_title, row=row, col=col)
                fig.update_yaxes(title_text=y_title, row=row, col=col)
        
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        print(f"Statistics dashboard saved to: {self.output_dir / filename}")
        return fig
    
    def create_individual_cluster_analysis(self, cubes: List[Dict], data: Union[xr.Dataset, str], filename: str, title: str = "Individual Cluster Analysis"):
        """Create detailed analysis for individual clusters."""
        print(f"Creating individual cluster analysis: {filename}")
        
        if not cubes:
            return None
        
        # Select most interesting clusters
        interesting_cubes = sorted(cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:6]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Cluster {c["id"]} - {c.get("vegetation_type", "Unknown")}' for c in interesting_cubes],
            specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)]
        )
        
        # Get time coordinates - handle both Dataset and non-Dataset inputs
        if hasattr(data, 'dims') and 'time' in data.dims:
            time_coords = pd.to_datetime(data.time.values)
        else:
            # Fallback: use first interesting cube to determine time dimension
            first_valid = next((c for c in interesting_cubes if self._get_ndvi_profile(c)), None)
            if first_valid:
                time_coords = list(range(len(self._get_ndvi_profile(first_valid))))
            else:
                time_coords = list(range(10))  # Default fallback
        
        # Plot each cluster
        for idx, cube in enumerate(interesting_cubes):
            row, col = (idx // 3) + 1, (idx % 3) + 1
            ndvi_profile = self._get_ndvi_profile(cube)
            
            if ndvi_profile:
                # NDVI time series
                fig.add_trace(go.Scatter(
                    x=time_coords, y=ndvi_profile, mode='lines+markers',
                    line=dict(color=self.color_palette[idx % len(self.color_palette)], width=3),
                    hovertemplate=f'Cluster {cube["id"]}<br>Time: %{{x}}<br>NDVI: %{{y:.3f}}<br>Type: {cube.get("vegetation_type", "Unknown")}<br>Area: {cube["area"]} pixels<extra></extra>',
                    showlegend=False
                ), row=row, col=col)
                
                # Add trend line
                if len(ndvi_profile) > 2:
                    x_numeric = np.arange(len(ndvi_profile))
                    trend_line = np.poly1d(np.polyfit(x_numeric, ndvi_profile, 1))(x_numeric)
                    fig.add_trace(go.Scatter(
                        x=time_coords, y=trend_line, mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False, hoverinfo='skip'
                    ), row=row, col=col)
        
        # Update layout
        fig.update_layout(title=title, width=1500, height=800, hovermode='closest')
        
        # Update all axes
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(title_text="Time", row=i, col=j)
                fig.update_yaxes(title_text="NDVI", row=i, col=j)
        
        pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        print(f"Individual cluster analysis saved to: {self.output_dir / filename}")
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("Interactive Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides comprehensive visualization tools for vegetation clustering results.")
    print("Use this module by importing InteractiveVisualization class and calling create_all_visualizations().")