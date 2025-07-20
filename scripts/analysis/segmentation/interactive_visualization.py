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
from typing import List, Dict, Any
import pandas as pd

# Add the scripts directory to Python path
import sys
import os
scripts_dir = Path(__file__).parent.parent.parent
sys.path.append(str(scripts_dir))

# Import only the cube module to avoid circular imports
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
    
    def create_interactive_spatial_map(self, cubes: List[STCube], filename: str, title: str = "Vegetation Clusters"):
        """
        Create an interactive spatial map showing vegetation cluster boundaries and NDVI patterns.
        
        Args:
            cubes: List of STCube objects representing vegetation clusters
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
            all_pixels.extend(cube.pixels)
        
        if not all_pixels:
            print("Warning: No pixels in cubes")
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
        colors = px.colors.qualitative.Set3
        for i, cube in enumerate(cubes):
            color_idx = i % len(colors)
            mean_ndvi = np.mean(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5
            
            for y, x in cube.pixels:
                map_y = y - y_min
                map_x = x - x_min
                segmentation_map[map_y, map_x] = i
                ndvi_map[map_y, map_x] = mean_ndvi
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Add heatmap for NDVI values
        fig.add_trace(go.Heatmap(
            z=ndvi_map,
            x=list(range(x_min, x_max + 1)),
            y=list(range(y_min, y_max + 1)),
            colorscale='RdYlGn',
            name='NDVI',
            colorbar=dict(title="Mean NDVI"),
            hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'
        ))
        
        # Add cube boundaries as scatter plots
        for i, cube in enumerate(cubes):
            color_idx = i % len(colors)
            y_coords = [p[0] for p in cube.pixels]
            x_coords = [p[1] for p in cube.pixels]
            
            mean_ndvi = np.mean(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors[color_idx],
                    line=dict(width=1, color='black')
                ),
                name=f'Cube {i} (NDVI: {mean_ndvi:.3f})',
                hovertemplate=f'Cube {i}<br>Area: {cube.area} pixels<br>Mean NDVI: {mean_ndvi:.3f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{title}<br>Total Cubes: {len(cubes)}',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            width=800,
            height=600,
            hovermode='closest'
        )
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Interactive spatial map saved to: {output_file}")
        
        return fig
    
    def create_interactive_time_series(self, cubes: List[STCube], data: xr.Dataset, filename: str, title: str = "NDVI Time Series"):
        """
        Create an interactive time series plot showing NDVI evolution for each vegetation cube.
        
        Args:
            cubes: List of STCube objects
            data: The original xarray dataset with time information
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating interactive time series: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        time_coords = data.time.values if 'time' in data.dims else list(range(len(cubes[0].ndvi_profile) if hasattr(cubes[0], 'ndvi_profile') and cubes[0].ndvi_profile is not None else []))
        
        for i, cube in enumerate(cubes):
            if not hasattr(cube, 'ndvi_profile') or cube.ndvi_profile is None:
                continue
                
            color_idx = i % len(colors)
            
            fig.add_trace(go.Scatter(
                x=time_coords,
                y=cube.ndvi_profile,
                mode='lines+markers',
                name=f'Cube {i} (Area: {cube.area})',
                line=dict(color=colors[color_idx], width=2),
                marker=dict(size=4),
                hovertemplate=f'Cube {i}<br>Time: %{{x}}<br>NDVI: %{{y:.3f}}<br>Area: {cube.area} pixels<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='NDVI Value',
            width=1000,
            height=600,
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"NDVI time series plot saved to: {output_file}")
        
        return fig
    
    def create_interactive_statistics_dashboard(self, cubes: List[STCube], filename: str, title: str = "Vegetation Statistics"):
        """
        Create an interactive dashboard with vegetation cube statistics.
        
        Args:
            cubes: List of STCube objects
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating statistics dashboard: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Prepare data
        cube_data = []
        for i, cube in enumerate(cubes):
            mean_ndvi = np.mean(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5
            max_ndvi = np.max(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5
            min_ndvi = np.min(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5
            
            cube_data.append({
                'cube_id': i,
                'area': cube.area,
                'mean_ndvi': mean_ndvi,
                'max_ndvi': max_ndvi,
                'min_ndvi': min_ndvi,
                'ndvi_range': max_ndvi - min_ndvi,
                'temporal_variance': cube.temporal_variance if hasattr(cube, 'temporal_variance') else 0,
                'compactness': cube.compactness if hasattr(cube, 'compactness') else 0
            })
        
        df = pd.DataFrame(cube_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cube Area Distribution',
                'NDVI vs Area',
                'Temporal Variance vs Area',
                'NDVI Range Distribution'
            ],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
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
        
        # NDVI vs Area scatter
        fig.add_trace(
            go.Scatter(
                x=df['area'],
                y=df['mean_ndvi'],
                mode='markers',
                name='NDVI vs Area',
                marker=dict(size=8, color='green', opacity=0.7),
                text=[f'Cube {i}' for i in df['cube_id']],
                hovertemplate='Cube: %{text}<br>Area: %{x}<br>NDVI: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Temporal Variance vs Area
        fig.add_trace(
            go.Scatter(
                x=df['area'],
                y=df['temporal_variance'],
                mode='markers',
                name='Temporal Variance vs Area',
                marker=dict(size=8, color='red', opacity=0.7),
                text=[f'Cube {i}' for i in df['cube_id']],
                hovertemplate='Cube: %{text}<br>Area: %{x}<br>Temporal Var: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # NDVI Range distribution
        fig.add_trace(
            go.Histogram(
                x=df['ndvi_range'],
                nbinsx=20,
                name='NDVI Range Distribution',
                marker_color='orange',
                hovertemplate='NDVI Range: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            width=1000,
            showlegend=False
        )
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Statistics dashboard saved to: {output_file}")
        
        return fig
    
    def create_3d_surface_plot(self, cubes: List[STCube], data: xr.Dataset, filename: str, title: str = "3D NDVI Surface"):
        """
        Create a 3D surface plot of NDVI patterns.
        
        Args:
            cubes: List of STCube objects
            data: The original xarray dataset
            filename: Output HTML filename
            title: Plot title
        """
        print(f"Creating 3D surface plot: {filename}")
        
        if not cubes:
            print("Warning: No cubes to visualize")
            return None
        
        # Get spatial extent
        all_pixels = []
        for cube in cubes:
            all_pixels.extend(cube.pixels)
        
        if not all_pixels:
            print("Warning: No pixels in cubes")
            return None
        
        # Create spatial grid
        y_coords = [p[0] for p in all_pixels]
        x_coords = [p[1] for p in all_pixels]
        
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Create NDVI surface
        ndvi_surface = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        # Fill surface with mean NDVI values
        for cube in cubes:
            mean_ndvi = np.mean(cube.ndvi_profile) if hasattr(cube, 'ndvi_profile') and cube.ndvi_profile is not None else 0.5
            
            for y, x in cube.pixels:
                map_y = y - y_min
                map_x = x - x_min
                ndvi_surface[map_y, map_x] = mean_ndvi
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=ndvi_surface,
            x=list(range(x_min, x_max + 1)),
            y=list(range(y_min, y_max + 1)),
            colorscale='RdYlGn',
            colorbar=dict(title="Mean NDVI"),
            hovertemplate='X: %{x}<br>Y: %{y}<br>NDVI: %{z:.3f}<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='NDVI Value',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        # Save as HTML
        output_file = self.output_dir / filename
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"3D surface plot saved to: {output_file}")
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("Interactive Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides visualization tools for vegetation clustering results.")
    print("Use this module by importing InteractiveVisualization class.")
