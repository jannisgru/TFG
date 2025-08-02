#!/usr/bin/env python3
"""
Static Visualization Module for Vegetation ST-Cube Segmentation Results

Provides visualizations using Matplotlib for the results of vegetation-focused spatiotemporal cube segmentation. Includes spatial maps and temporal NDVI analyses for clusters/cubes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import warnings
from loguru import logger
from ..config_loader import get_config

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')


class StaticVisualization:
    """
    Static visualization generator for vegetation ST-cube segmentation results.
    """
    
    def __init__(self, output_directory: str = None):
        """Initialize the static visualization generator."""
        if output_directory is None:
            output_directory = get_config().static_output_dir
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = get_config()
        self.colors = plt.cm.get_cmap(config.color_map)(np.linspace(0, 1, 12))
    
    def _get_pixels_safely(self, cube: Dict) -> List[Tuple[int, int]]:
        """Safely extract pixels from cube data, handling different formats."""
        # Check multiple possible field names
        for field_name in ['pixels', 'coordinates']:
            pixels = cube.get(field_name, [])
            if pixels is not None and len(pixels) > 0:
                break
        else:
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
    
    def _get_ndvi_profile(self, cube: Dict) -> List[float]:
        """Extract NDVI temporal profile, checking multiple possible keys."""
        for key in ['ndvi_profile', 'mean_temporal_profile', 'ndvi_time_series']:
            profile = cube.get(key, None)
            if profile is not None:
                if hasattr(profile, 'tolist'):
                    return profile.tolist()
                elif hasattr(profile, '__len__') and len(profile) > 0:
                    return list(profile)
        return []

    def _has_valid_ndvi_profile(self, cube: Dict) -> bool:
        """Check if cube has a valid NDVI profile."""
        return len(self._get_ndvi_profile(cube)) > 0

    def _has_valid_pixels(self, cube: Dict) -> bool:
        """Check if cube has valid pixel data."""
        return len(self._get_pixels_safely(cube)) > 0

    def _calculate_summary_statistics(self, cubes: List[Dict], municipality_name: str) -> Dict[str, Any]:
        """Calculate summary statistics for the clustering results."""
        total_area = sum(cube.get('area', 0) for cube in cubes)
        valid_ndvis = [cube.get('mean_ndvi', 0) for cube in cubes if cube.get('mean_ndvi') is not None and not np.isnan(cube.get('mean_ndvi', 0))]
        mean_ndvi = np.mean(valid_ndvis) if valid_ndvis else 0
        largest_cube = max(cubes, key=lambda x: x.get('area', 0)) if cubes else {}
        highest_ndvi_cube = max(cubes, key=lambda x: x.get('mean_ndvi', 0) if x.get('mean_ndvi') is not None and not np.isnan(x.get('mean_ndvi', 0)) else 0) if cubes else {}
        return {
            'municipality_name': municipality_name,
            'total_clusters': len(cubes),
            'total_area_pixels': total_area,
            'mean_cluster_size': total_area/len(cubes) if cubes else 0,
            'overall_mean_ndvi': mean_ndvi,
            'largest_cluster_size': largest_cube.get('area', 0),
            'highest_ndvi_value': highest_ndvi_cube.get('mean_ndvi', 0),
            'clusters_with_valid_ndvi': len(valid_ndvis)
        }
        
    def create_all_static_visualizations(self, 
                                       cubes: List[Dict], 
                                       data: Any, 
                                       municipality_name: str = "Unknown") -> Dict[str, str]:
        """Create all static visualizations for vegetation clusters, including analysis report and interactive NDVI plot."""
        try:
            import plotly.graph_objs as go
            import plotly.offline as pyo
        except ImportError:
            logger.error("Plotly not available. Please install plotly: pip install plotly")
            return {}
            
        from ..config_loader import get_config
        visualizations = {}
        
        try:
            # 1. Export analysis report with config parameters and summary statistics
            config = get_config()
            report_file = self.output_dir / f"vegetation_analysis_report_{municipality_name.replace(' ', '_')}.txt"
            
            # Calculate summary statistics
            stats = self._calculate_summary_statistics(cubes, municipality_name)
            # Only include selected parameter groups
            param_sections = [
                ("Segmentation Parameters", ['min_cube_size', 'max_spatial_distance', 'min_vegetation_ndvi', 'ndvi_variance_threshold', 'n_clusters', 'temporal_weight']),
                ("Clustering Parameters", ['spatial_weight', 'min_samples_ratio', 'eps_search_attempts']),
                ("Bridging Parameters", ['enable_spatial_bridging', 'bridge_similarity_tolerance', 'max_bridge_gap', 'min_bridge_density', 'connectivity_radius', 'max_bridge_length', 'min_cluster_size_for_bridging']),
                ("Data Parameters", ['default_netcdf_path', 'municipalities_data', 'default_municipality', 'default_output_dir']),
                ("Analysis Parameters", ['chunk_size', 'max_pixels_for_sampling', 'spatial_margin', 'temporal_margin', 'max_neighbors', 'search_margin', 'adjacency_search_neighbors'])
            ]
            config_dict = vars(config)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Vegetation Clustering Analysis Report - {municipality_name}\n")
                f.write("=" * 60 + "\n\n")
                # Summary Statistics Section
                f.write("ANALYSIS RESULTS SUMMARY\n")
                f.write("=" * 30 + "\n")
                f.write(f"Municipality: {stats['municipality_name']}\n")
                f.write(f"Total Clusters Identified: {stats['total_clusters']}\n")
                f.write(f"Total Area Analyzed: {stats['total_area_pixels']:,} pixels\n")
                f.write(f"Mean Cluster Size: {stats['mean_cluster_size']:.1f} pixels\n")
                f.write(f"Overall Mean NDVI: {stats['overall_mean_ndvi']:.3f}\n")
                f.write(f"Largest Cluster Size: {stats['largest_cluster_size']:,} pixels\n")
                f.write(f"Highest NDVI Value: {stats['highest_ndvi_value']:.3f}\n")
                f.write(f"Clusters with Valid NDVI: {stats['clusters_with_valid_ndvi']}\n")
                f.write("\n")
                # Configuration Parameters Section
                f.write("CONFIGURATION PARAMETERS\n")
                f.write("=" * 30 + "\n")
                for section_title, keys in param_sections:
                    f.write(f"\n{section_title}:\n")
                    f.write("-" * len(section_title) + ":\n")
                    for k in keys:
                        if k in config_dict:
                            f.write(f"  {k}: {config_dict[k]}\n")
                    f.write("\n")
            visualizations["analysis_report"] = str(report_file)

            # 2. Spatial distribution map
            spatial_file = f"spatial_distribution_{municipality_name.replace(' ', '_')}.png"
            self.create_spatial_distribution_map(cubes, data, spatial_file, municipality_name)
            visualizations["spatial_distribution"] = str(self.output_dir / spatial_file)

            # 3. Interactive NDVI evolution plot (HTML)
            ndvi_html_file = self.create_interactive_ndvi_evolution(cubes, data, municipality_name)
            if ndvi_html_file:
                visualizations["ndvi_evolution_html"] = ndvi_html_file

        except Exception as e:
            logger.error(f"Error creating static visualizations: {str(e)}")
            import traceback
            traceback.print_exc()

        return visualizations
    
    def create_interactive_ndvi_evolution(self, cubes: List[Dict], data: Any, municipality_name: str) -> Optional[str]:
        """Create an interactive HTML plot of NDVI evolution over time with toggleable cluster lines."""
        try:
            import plotly.graph_objs as go
            import plotly.offline as pyo
            import numpy as np
        except ImportError:
            logger.error("Plotly not available for interactive NDVI plot")
            return None
            
        try:
            # Extract time axis from data
            time_axis = None
            time_labels = None
            
            if hasattr(data, 'coords') and 'time' in data.coords:
                time_axis = data.coords['time'].values
                # Convert to years - the data contains years directly
                try:
                    # Handle different time formats - assume they are years
                    if hasattr(time_axis[0], 'item'):
                        # Handle numpy scalar types
                        time_labels = [int(year.item()) for year in time_axis]
                    else:
                        # Handle regular integers/years
                        time_labels = [int(year) for year in time_axis]
                except:
                    # Fallback to string representation
                    time_labels = [str(t) for t in time_axis]
            elif hasattr(data, 'time'):
                time_axis = data.time.values
                try:
                    # Handle different time formats - assume they are years
                    if hasattr(time_axis[0], 'item'):
                        # Handle numpy scalar types
                        time_labels = [int(year.item()) for year in time_axis]
                    else:
                        # Handle regular integers/years
                        time_labels = [int(year) for year in time_axis]
                except:
                    # Fallback to string representation
                    time_labels = [str(t) for t in time_axis]
            
            # Fallback: create year-based time axis like interactive visualization
            if time_axis is None and cubes:
                # Find the longest NDVI profile to determine time axis length
                max_length = 0
                for cube in cubes:
                    ndvi_profile = self._get_ndvi_profile(cube)
                    if ndvi_profile:
                        max_length = max(max_length, len(ndvi_profile))
                
                if max_length > 0:
                    # Create years starting from 1984 like the interactive visualization
                    time_axis = list(range(max_length))
                    time_labels = [1984 + i for i in range(max_length)]
                else:
                    logger.warning("No valid NDVI profiles found in cubes")
                    return None
            
            if time_axis is None:
                logger.warning("Could not determine time axis for NDVI evolution plot")
                return None
                        
            # Create the interactive plot
            fig = go.Figure()
            
            # Add traces for each cluster
            valid_clusters = 0
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                     '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10  # Repeat colors if needed
            
            for idx, cube in enumerate(cubes):
                ndvi_profile = self._get_ndvi_profile(cube)
                
                if ndvi_profile and len(ndvi_profile) > 0:
                    # Ensure time axis and NDVI values have same length
                    min_length = min(len(time_labels), len(ndvi_profile))
                    x_vals = time_labels[:min_length]
                    y_vals = ndvi_profile[:min_length]
                    
                    # Create cluster label
                    cluster_id = cube.get('id', idx) + 1
                    cluster_size = cube.get('area', cube.get('size', len(self._get_pixels_safely(cube))))
                    label = f"Cluster {cluster_id} (size: {cluster_size})"
                    
                    # Add trace with toggle capability
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=4),
                        visible=True,  # All visible by default
                        hovertemplate=f'<b>{label}</b><br>' +
                                    'Time: %{x}<br>' +
                                    'NDVI: %{y:.3f}<br>' +
                                    '<extra></extra>'
                    ))
                    valid_clusters += 1
            
            if valid_clusters == 0:
                logger.warning("No valid clusters with NDVI profiles found")
                return None
            
            # Update layout with better styling and interactive features
            fig.update_layout(
                title=dict(
                    text=f"Interactive NDVI Evolution - {municipality_name}<br><sub>Click legend items to toggle cluster visibility</sub>",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Time",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title="NDVI",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    range=[0, 1]  # NDVI typically ranges from 0 to 1
                ),
                legend=dict(
                    title="Clusters (click to toggle)",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode="x unified",
                plot_bgcolor='white',
                width=1200,
                height=700,
                margin=dict(r=200)  # Extra margin for legend
            )
            
            # Save the HTML file
            html_file = self.output_dir / f"ndvi_evolution_{municipality_name.replace(' ', '_')}.html"
            fig.write_html(
                str(html_file),
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                }
            )
            return str(html_file)
            
        except Exception as e:
            logger.error(f"Error creating interactive NDVI evolution plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_spatial_distribution_map(self, cubes: List[Dict], data: Any, filename: str, municipality_name: str):
        """Create a spatial distribution map showing all cluster pixel locations in lat/lon coordinates."""
        
        # Check if cubes have spatial data using the same method as other functions
        valid_cubes = []
        for cube in cubes:
            pixels = self._get_pixels_safely(cube)
            if len(pixels) > 0:
                valid_cubes.append(cube)
        
        if not valid_cubes:
            logger.warning("No spatial data available for mapping")
            return
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Cluster Spatial Distribution - {municipality_name}', fontsize=16, fontweight='bold')
        
        # Define colors for different clusters
        config = get_config()
        cmap = plt.cm.get_cmap(config.color_map)
        colors = cmap(np.linspace(0, 1, len(valid_cubes)))
        
        # Calculate pixel size in data coordinates
        x_res = abs(float(data.x.values[1] - data.x.values[0]))
        y_res = abs(float(data.y.values[1] - data.y.values[0]))
        
        # Process each cluster
        for cluster_idx, cube in enumerate(valid_cubes):
            pixels = self._get_pixels_safely(cube)
            if not pixels:
                continue
                        
            # Convert all pixels to lat/lon coordinates and create rectangles
            for y_coord, x_coord in pixels:
                lat, lon = self._convert_pixel_to_latlon(data, y_coord, x_coord)
                if lat is not None and lon is not None:
                    # Create a rectangle representing the 30x30m pixel
                    # Rectangle centered at (lon, lat) with width=x_res, height=y_res
                    rect = plt.Rectangle(
                        (lon - x_res/2, lat - y_res/2),  # Bottom-left corner
                        x_res,  # Width
                        y_res,  # Height
                        facecolor=colors[cluster_idx % len(colors)],
                        edgecolor='black',
                        linewidth=0.1,
                        alpha=0.7
                    )
                    ax.add_patch(rect)
        
        # Add legend by creating dummy scatter points
        for cluster_idx, cube in enumerate(valid_cubes):
            cluster_id = cube.get('id', cluster_idx) + 1
            ax.scatter([], [], 
                    color=colors[cluster_idx % len(colors)], 
                    s=100, alpha=0.7, 
                    label=f'Cluster {cluster_id}',
                    marker='s')
        
        # Set labels and formatting
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Vegetation Cluster Pixel Locations (30x30m pixels)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set equal aspect ratio to preserve geographic proportions
        ax.set_aspect('equal', adjustable='box')
        
        # Auto-adjust the view to fit all data
        ax.autoscale(tight=True)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
            
    def _convert_pixel_to_latlon(self, data: Any, y_coord: int, x_coord: int) -> Tuple[float, float]:
        """Convert pixel coordinates to latitude/longitude using the same method as json_exporter."""
        try:
            # Get latitude and longitude from the dataset coordinates
            lat = float(data.y.isel(y=y_coord).values)
            lon = float(data.x.isel(x=x_coord).values)
        except (IndexError, KeyError, AttributeError):
            # Fallback: try alternative coordinate names
            try:
                lat = float(data.latitude.isel(latitude=y_coord).values) if 'latitude' in data.coords else float(data.lat.isel(lat=y_coord).values)
                lon = float(data.longitude.isel(longitude=x_coord).values) if 'longitude' in data.coords else float(data.lon.isel(lon=x_coord).values)
            except:
                # If we can't convert, skip this pixel
                logger.warning(f"Could not convert pixel coordinates ({y_coord}, {x_coord}) to lat/lon")
                return None, None
        
        return lat, lon