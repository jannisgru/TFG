#!/usr/bin/env python3
"""
Static Visualization Module for Vegetation ST-Cube Segmentation Results

Provides publication-ready static visualizations using Matplotlib for the results of vegetation-focused spatiotemporal cube segmentation. Includes summary plots, spatial maps, and temporal NDVI analyses for clusters/cubes.
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
        
    def create_all_static_visualizations(self, 
                                       cubes: List[Dict], 
                                       data: Any, 
                                       municipality_name: str = "Unknown") -> Dict[str, str]:
        """Create all static visualizations for vegetation clusters, including config export and interactive NDVI plot."""
        try:
            import plotly.graph_objs as go
            import plotly.offline as pyo
        except ImportError:
            logger.error("Plotly not available. Please install plotly: pip install plotly")
            return {}
            
        from ..config_loader import get_config
        visualizations = {}
        
        try:
            # 0. Export config parameters to txt
            config = get_config()
            config_txt_file = self.output_dir / f"config_parameters_{municipality_name.replace(' ', '_')}.txt"
            # Only include selected parameter groups
            param_sections = [
                ("Segmentation Parameters", ['min_cube_size', 'max_spatial_distance', 'min_vegetation_ndvi', 'ndvi_variance_threshold', 'n_clusters', 'temporal_weight']),
                ("Clustering Parameters", ['spatial_weight', 'min_samples_ratio', 'eps_search_attempts']),
                ("Bridging Parameters", ['enable_spatial_bridging', 'bridge_similarity_tolerance', 'max_bridge_gap', 'min_bridge_density', 'connectivity_radius', 'max_bridge_length', 'min_cluster_size_for_bridging']),
                ("Data Parameters", ['default_netcdf_path', 'municipalities_data', 'default_municipality', 'default_output_dir']),
                ("Analysis Parameters", ['chunk_size', 'max_pixels_for_sampling', 'spatial_margin', 'temporal_margin', 'max_neighbors', 'search_margin', 'adjacency_search_neighbors'])
            ]
            config_dict = vars(config)
            with open(config_txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Configuration Parameters for {municipality_name}\n")
                f.write("=" * 50 + "\n\n")
                for section_title, keys in param_sections:
                    f.write(f"{'='*20} {section_title} {'='*20}\n")
                    for k in keys:
                        if k in config_dict:
                            f.write(f"{k}: {config_dict[k]}\n")
                    f.write("\n")
            visualizations["config_txt"] = str(config_txt_file)

            # 1. Comprehensive summary plot
            summary_file = f"comprehensive_summary_{municipality_name.replace(' ', '_')}.png"
            self.create_comprehensive_summary(cubes, summary_file, municipality_name)
            visualizations["comprehensive_summary"] = str(self.output_dir / summary_file)

            # 2. Spatial distribution map
            spatial_file = f"spatial_distribution_{municipality_name.replace(' ', '_')}.png"
            self.create_spatial_distribution_map(cubes, spatial_file, municipality_name)
            visualizations["spatial_distribution"] = str(self.output_dir / spatial_file)

            # 3. NDVI temporal analysis (static)
            temporal_file = f"temporal_analysis_{municipality_name.replace(' ', '_')}.png"
            self.create_temporal_analysis(cubes, data, temporal_file, municipality_name)
            visualizations["temporal_analysis"] = str(self.output_dir / temporal_file)

            # 4. Interactive NDVI evolution plot (HTML)
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
    
    def create_comprehensive_summary(self, cubes: List[Dict], filename: str, municipality_name: str):
        """Create a comprehensive summary plot with multiple panels."""

        config = get_config()
        fig, axes = plt.subplots(2, 3, figsize=config.figure_size)
        fig.suptitle(f'Vegetation Clustering Analysis - {municipality_name}', fontsize=16, fontweight='bold')
        
        # Panel 1: Cluster size distribution
        sizes = []
        for cube in cubes:
            area = cube.get('area', 0)
            if area > 0:
                sizes.append(area)
        
        if sizes:
            axes[0, 0].hist(sizes, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        else:
            axes[0, 0].text(0.5, 0.5, 'No size data available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_xlabel('Cluster Size (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].grid(True, alpha=config.grid_alpha)
        
        # Panel 2: NDVI distribution
        ndvis = []
        for cube in cubes:
            mean_ndvi = cube.get('mean_ndvi', None)
            if mean_ndvi is not None and not np.isnan(mean_ndvi):
                ndvis.append(mean_ndvi)
        
        if ndvis:
            axes[0, 1].hist(ndvis, bins=20, alpha=0.7, color='green', edgecolor='black')
        else:
            axes[0, 1].text(0.5, 0.5, 'No NDVI data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_xlabel('Mean NDVI')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('NDVI Distribution')
        axes[0, 1].grid(True, alpha=config.grid_alpha)
        
        # Panel 3: Vegetation type pie chart
        veg_types = [cube.get('vegetation_type', 'Unknown') for cube in cubes]
        type_counts = pd.Series(veg_types).value_counts()
        axes[0, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Vegetation Types')
        
        # Panel 4: NDVI vs Area scatter
        areas = []
        ndvis = []
        seasonality_colors = []
        
        for cube in cubes:
            area = cube.get('area', 0)
            mean_ndvi = cube.get('mean_ndvi', None)
            seasonality = cube.get('seasonality_score', 0)
            
            if area > 0 and mean_ndvi is not None and not np.isnan(mean_ndvi):
                areas.append(area)
                ndvis.append(mean_ndvi)
                seasonality_colors.append(seasonality)
        
        if areas and ndvis:
            scatter = axes[1, 0].scatter(areas, ndvis, c=seasonality_colors, cmap='viridis', 
                                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=axes[1, 0], shrink=0.8)
            cbar.set_label('Seasonality Score')
        else:
            axes[1, 0].text(0.5, 0.5, 'No scatter plot data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            
        axes[1, 0].set_xlabel('Area (pixels)')
        axes[1, 0].set_ylabel('Mean NDVI')
        axes[1, 0].set_title('NDVI vs Area (colored by seasonality)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 5: Time series sample
        valid_cubes = [cube for cube in cubes if self._has_valid_ndvi_profile(cube)]
        
        if valid_cubes:
            top_cubes = sorted(valid_cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:5]
            
            for i, cube in enumerate(top_cubes):
                ndvi_profile = self._get_ndvi_profile(cube)
                if ndvi_profile:
                    time_coords = list(range(len(ndvi_profile)))
                    axes[1, 1].plot(time_coords, ndvi_profile, 
                                  linewidth=2, alpha=0.8, label=f"Cluster {cube['id']}")
            
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('NDVI')
            axes[1, 1].set_title('NDVI Evolution - Most Seasonal Clusters')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Panel 6: Summary statistics
        axes[1, 2].axis('off')
        
        total_area = sum(cube.get('area', 0) for cube in cubes)
        valid_ndvis = [cube.get('mean_ndvi', 0) for cube in cubes if cube.get('mean_ndvi') is not None and not np.isnan(cube.get('mean_ndvi', 0))]
        mean_ndvi = np.mean(valid_ndvis) if valid_ndvis else 0
        valid_seasonality = [cube.get('seasonality_score', 0) for cube in cubes if cube.get('seasonality_score') is not None]
        mean_seasonality = np.mean(valid_seasonality) if valid_seasonality else 0
        
        largest_cube = max(cubes, key=lambda x: x.get('area', 0)) if cubes else {}
        highest_ndvi_cube = max(cubes, key=lambda x: x.get('mean_ndvi', 0) if x.get('mean_ndvi') is not None and not np.isnan(x.get('mean_ndvi', 0)) else 0) if cubes else {}
        
        stats_text = f"""Summary Statistics:

Total Clusters: {len(cubes)}
Total Area: {total_area:,} pixels
Mean Cluster Size: {total_area/len(cubes):.1f} pixels

Overall Mean NDVI: {mean_ndvi:.3f}
Mean Seasonality Score: {mean_seasonality:.3f}

Largest Cluster: {largest_cube.get('area', 0):,} pixels
Highest NDVI: {highest_ndvi_cube.get('mean_ndvi', 0):.3f}"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=config.dpi, bbox_inches=config.bbox_inches, facecolor='white')
        plt.close()
        
    
    def create_spatial_distribution_map(self, cubes: List[Dict], filename: str, municipality_name: str):
        """Create a detailed spatial distribution map."""
        
        # Check if cubes have spatial data
        valid_cubes = []
        for cube in cubes:
            pixels = cube.get('pixels', [])
            if pixels is not None and len(pixels) > 0:
                valid_cubes.append(cube)
        
        if not valid_cubes:
            print("Warning: No spatial data available for mapping")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Spatial Distribution Analysis - {municipality_name}', fontsize=14, fontweight='bold')
        
        # Get spatial extent
        all_pixels = []
        for cube in cubes:
            pixels = self._get_pixels_safely(cube)
            if pixels:
                all_pixels.extend(pixels)
        
        if not all_pixels:
            print("Warning: No valid pixels found")
            return
        
        y_coords = [p[0] for p in all_pixels]
        x_coords = [p[1] for p in all_pixels]
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Create maps
        cluster_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        area_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        seasonality_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        # Fill maps
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if not pixels:
                continue
            
            for y, x in pixels:
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    map_y, map_x = y - y_min, x - x_min
                    cluster_map[map_y, map_x] = i
                    ndvi_map[map_y, map_x] = cube['mean_ndvi']
                    area_map[map_y, map_x] = cube['area']
                    seasonality_map[map_y, map_x] = cube.get('seasonality_score', 0)
        
        # Plot 1: Cluster boundaries
        im1 = axes[0, 0].imshow(cluster_map, cmap='tab20', aspect='equal', origin='lower')
        axes[0, 0].set_title('Cluster Boundaries')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        
        # Plot 2: NDVI distribution
        im2 = axes[0, 1].imshow(ndvi_map, cmap='RdYlGn', aspect='equal', origin='lower', vmin=0, vmax=1)
        axes[0, 1].set_title('NDVI Distribution')
        axes[0, 1].set_xlabel('X Coordinate')
        axes[0, 1].set_ylabel('Y Coordinate')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Mean NDVI')
        
        # Plot 3: Cluster size
        im3 = axes[1, 0].imshow(np.log1p(area_map), cmap='plasma', aspect='equal', origin='lower')
        axes[1, 0].set_title('Cluster Size (log scale)')
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        cbar3.set_label('Log(Area + 1)')
        
        # Plot 4: Seasonality
        im4 = axes[1, 1].imshow(seasonality_map, cmap='viridis', aspect='equal', origin='lower')
        axes[1, 1].set_title('Seasonality Score')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Y Coordinate')
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        cbar4.set_label('Seasonality Score')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Spatial distribution map saved to: {output_file}")
    
    def create_temporal_analysis(self, cubes: List[Dict], data: Any, filename: str, municipality_name: str):
        """Create temporal analysis plots."""
        
        valid_cubes = [cube for cube in cubes if self._has_valid_ndvi_profile(cube)]
        
        if not valid_cubes:
            print("Warning: No temporal data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Temporal Analysis - {municipality_name}', fontsize=14, fontweight='bold')
        
        # Get time coordinates
        if valid_cubes:
            first_profile = self._get_ndvi_profile(valid_cubes[0])
            n_times = len(first_profile) if first_profile else 0
        else:
            n_times = 0
        
        if n_times == 0:
            logger.warning("No valid NDVI profiles found for temporal analysis")
            return
            
        time_coords = list(range(n_times))
        
        # Plot 1: Time series of top clusters
        interesting_cubes = sorted(valid_cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:8]
        
        for i, cube in enumerate(interesting_cubes):
            ndvi_profile = self._get_ndvi_profile(cube)
            if ndvi_profile:
                color = self.colors[i % len(self.colors)]
                axes[0, 0].plot(time_coords, ndvi_profile, 
                              color=color, linewidth=2, alpha=0.8,
                              label=f"Cluster {cube['id']} (Area: {cube['area']})")
        
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('NDVI')
        axes[0, 0].set_title('NDVI Evolution - Most Seasonal Clusters')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average NDVI evolution by vegetation type
        veg_types = set(cube.get('vegetation_type', 'Unknown') for cube in valid_cubes)
        
        for veg_type in veg_types:
            type_cubes = [cube for cube in valid_cubes if cube.get('vegetation_type') == veg_type]
            if type_cubes:
                profiles = []
                for cube in type_cubes:
                    ndvi_profile = self._get_ndvi_profile(cube)
                    if ndvi_profile:
                        profiles.append(ndvi_profile)
                
                if profiles:
                    profiles = np.array(profiles)
                    mean_profile = np.mean(profiles, axis=0)
                    std_profile = np.std(profiles, axis=0)
                
                axes[0, 1].plot(time_coords, mean_profile, linewidth=3, label=veg_type)
                axes[0, 1].fill_between(time_coords, 
                                      mean_profile - std_profile, 
                                      mean_profile + std_profile, 
                                      alpha=0.2)
        
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Mean NDVI')
        axes[0, 1].set_title('Average NDVI by Vegetation Type')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: NDVI heatmap
        sample_cubes = valid_cubes[::max(1, len(valid_cubes)//20)]
        ndvi_matrix_list = []
        for cube in sample_cubes:
            ndvi_profile = self._get_ndvi_profile(cube)
            if ndvi_profile:
                ndvi_matrix_list.append(ndvi_profile)
        
        if ndvi_matrix_list:
            ndvi_matrix = np.array(ndvi_matrix_list)
            im3 = axes[1, 0].imshow(ndvi_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Cluster Index')
            axes[1, 0].set_title('NDVI Evolution Heatmap')
            cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            cbar3.set_label('NDVI')
        else:
            axes[1, 0].text(0.5, 0.5, 'No NDVI data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('NDVI Evolution Heatmap - No Data')
        
        # Plot 4: Temporal statistics distribution
        temporal_means = []
        temporal_stds = []
        for cube in valid_cubes:
            ndvi_profile = self._get_ndvi_profile(cube)
            if ndvi_profile:
                temporal_means.append(np.mean(ndvi_profile))
                temporal_stds.append(np.std(ndvi_profile))
        
        if temporal_means and temporal_stds:
            axes[1, 1].hist(temporal_means, bins=20, alpha=0.7, label='Mean NDVI', color='green')
            ax_twin = axes[1, 1].twinx()
            ax_twin.hist(temporal_stds, bins=20, alpha=0.7, label='NDVI Std Dev', color='orange')
            
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Frequency (Mean NDVI)', color='green')
            ax_twin.set_ylabel('Frequency (Std Dev)', color='orange')
            axes[1, 1].set_title('Temporal Statistics Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No temporal statistics available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Temporal Statistics - No Data')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        

# Example usage
if __name__ == "__main__":
    print("Static Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides publication-ready static visualizations.")
    print("Use this module by importing StaticVisualization class.")
