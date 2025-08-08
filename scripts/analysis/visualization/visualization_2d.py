#!/usr/bin/env python3
"""
2D Visualization Module for Vegetation ST-Cube Segmentation Results

Provides static 2D visualizations using Matplotlib for the results of vegetation-focused spatiotemporal trace segmentation. Includes spatial maps and temporal NDVI analyses for clusters/traces.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import plotly.graph_objs as go
import warnings
from loguru import logger
import traceback
import datetime
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
    
    def _get_pixels_safely(self, trace: Dict) -> List[Tuple[int, int]]:
        """Safely extract pixels from trace data, handling different formats."""
        # Check multiple possible field names
        for field_name in ['pixels', 'coordinates']:
            pixels = trace.get(field_name, [])
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
    
    def _get_ndvi_profile(self, trace: Dict) -> List[float]:
        """Extract NDVI temporal profile, checking multiple possible keys."""
        for key in ['ndvi_profile', 'mean_temporal_profile', 'ndvi_time_series']:
            profile = trace.get(key, None)
            if profile is not None:
                if hasattr(profile, 'tolist'):
                    return profile.tolist()
                elif hasattr(profile, '__len__') and len(profile) > 0:
                    return list(profile)
        return []

    def _has_valid_ndvi_profile(self, trace: Dict) -> bool:
        """Check if trace has a valid NDVI profile."""
        return len(self._get_ndvi_profile(trace)) > 0

    def _has_valid_pixels(self, trace: Dict) -> bool:
        """Check if trace has valid pixel data."""
        return len(self._get_pixels_safely(trace)) > 0

    def _calculate_summary_statistics(self, traces: List[Dict], municipality_name: str) -> Dict[str, Any]:
        """Calculate summary statistics for the clustering results."""
        total_area = sum(trace.get('area', 0) for trace in traces)
        valid_ndvis = [trace.get('mean_ndvi', 0) for trace in traces if trace.get('mean_ndvi') is not None and not np.isnan(trace.get('mean_ndvi', 0))]
        mean_ndvi = np.mean(valid_ndvis) if valid_ndvis else 0
        largest_trace = max(traces, key=lambda x: x.get('area', 0)) if traces else {}
        highest_ndvi_trace = max(traces, key=lambda x: x.get('mean_ndvi', 0) if x.get('mean_ndvi') is not None and not np.isnan(x.get('mean_ndvi', 0)) else 0) if traces else {}
        return {
            'municipality_name': municipality_name,
            'total_clusters': len(traces),
            'total_area_pixels': total_area,
            'mean_cluster_size': total_area/len(traces) if traces else 0,
            'overall_mean_ndvi': mean_ndvi,
            'largest_cluster_size': largest_trace.get('area', 0),
            'highest_ndvi_value': highest_ndvi_trace.get('mean_ndvi', 0),
            'clusters_with_valid_ndvi': len(valid_ndvis)
        }
        
    def create_all_static_visualizations(self, 
                                       traces: List[Dict], 
                                       data: Any, 
                                       municipality_name: str = "Unknown") -> Dict[str, str]:
        """Create all static visualizations for vegetation clusters."""
            
        visualizations = {}
        
        try:
            # 1. Interactive NDVI evolution plot (HTML)
            ndvi_html_file = self.create_interactive_ndvi_evolution(traces, data, municipality_name)
            if ndvi_html_file:
                visualizations["ndvi_evolution_html"] = ndvi_html_file

        except Exception as e:
            logger.error(f"Error creating static visualizations: {str(e)}")
            traceback.print_exc()

        return visualizations
    
    def create_combined_analysis_report(self, 
                                      results: Dict[str, List[Dict]], 
                                      municipality_name: str = "Unknown") -> str:
        """Create a combined analysis report for both greening and browning trends."""

        config = get_config()
        report_file = self.output_dir / f"vegetation_analysis_report_{municipality_name.replace(' ', '_')}.txt"
        
        # Calculate summary statistics for each trend
        trend_stats = {}
        for trend, traces in results.items():
            trend_stats[trend] = self._calculate_summary_statistics(traces, municipality_name)
        
        # Configuration parameters section
        param_sections = [
            ("Segmentation Parameters", ['min_cluster_size', 'max_spatial_distance', 'min_vegetation_ndvi', 'ndvi_variance_threshold']),
            ("Clustering Parameters", ['eps', 'min_samples', 'temporal_weight', 'spatial_weight']),
            ("Data Parameters", ['netcdf_path', 'municipalities_data', 'municipality', 'output_dir']),
            ("Analysis Parameters", ['chunk_size', 'max_pixels_for_sampling', 'spatial_margin', 'temporal_margin', 'max_neighbors', 'search_margin', 'adjacency_search_neighbors'])
        ]
        
        config_dict = vars(config)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Vegetation Clustering Analysis Report - {municipality_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Combined Analysis Results Summary
            f.write("ANALYSIS RESULTS SUMMARY\n")
            f.write("=" * 30 + "\n")
            f.write(f"Municipality: {municipality_name}\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trends Analyzed: {', '.join(results.keys())}\n\n")
            
            # Summary for each trend
            for trend, stats in trend_stats.items():
                f.write(f"{trend.upper()} TRENDS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"  Total Clusters: {stats['total_clusters']}\n")
                f.write(f"  Total Area: {stats['total_area_pixels']:,} pixels\n")
                f.write(f"  Mean Cluster Size: {stats['mean_cluster_size']:.1f} pixels\n")
                f.write(f"  Overall Mean NDVI: {stats['overall_mean_ndvi']:.3f}\n")
                f.write(f"  Largest Cluster: {stats['largest_cluster_size']:,} pixels\n")
                f.write(f"  Highest NDVI: {stats['highest_ndvi_value']:.3f}\n")
                f.write(f"  Clusters with Valid NDVI: {stats['clusters_with_valid_ndvi']}\n")
                f.write("\n")
            
            # Comparison summary
            if len(trend_stats) == 2:
                inc_stats = trend_stats.get('greening', {})
                dec_stats = trend_stats.get('browning', {})
                f.write("TREND COMPARISON:\n")
                f.write("-" * 17 + "\n")
                f.write(f"  Ratio (Greening/Browning clusters): {inc_stats.get('total_clusters', 0)}/{dec_stats.get('total_clusters', 0)}\n")
                f.write(f"  Area Ratio (Greening/Browning): {inc_stats.get('total_area_pixels', 0):,}/{dec_stats.get('total_area_pixels', 0):,} pixels\n")
                f.write("\n")
            
            # Configuration Parameters Section
            f.write("CONFIGURATION PARAMETERS\n")
            f.write("=" * 30 + "\n")
            for section_title, keys in param_sections:
                f.write(f"\n{section_title}:\n")
                f.write("-" * len(section_title) + "\n")
                for k in keys:
                    if k in config_dict:
                        f.write(f"  {k}: {config_dict[k]}\n")
                f.write("\n")
        
        return str(report_file)
    
    def create_interactive_ndvi_evolution(self, traces: List[Dict], data: Any, municipality_name: str) -> Optional[str]:
        """Create an interactive HTML plot of NDVI evolution over time with toggleable cluster lines."""

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
            if time_axis is None and traces:
                # Find the longest NDVI profile to determine time axis length
                max_length = 0
                for trace in traces:
                    ndvi_profile = self._get_ndvi_profile(trace)
                    if ndvi_profile:
                        max_length = max(max_length, len(ndvi_profile))
                
                if max_length > 0:
                    # Create years starting from 1984 like the interactive visualization
                    time_axis = list(range(max_length))
                    time_labels = [1984 + i for i in range(max_length)]
                else:
                    logger.warning("No valid NDVI profiles found in traces")
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
            
            for idx, trace in enumerate(traces):
                ndvi_profile = self._get_ndvi_profile(trace)
                
                if ndvi_profile and len(ndvi_profile) > 0:
                    # Ensure time axis and NDVI values have same length
                    min_length = min(len(time_labels), len(ndvi_profile))
                    x_vals = time_labels[:min_length]
                    y_vals = ndvi_profile[:min_length]
                    
                    # Create cluster label
                    cluster_id = trace.get('id', idx) + 1
                    cluster_size = trace.get('area', trace.get('size', len(self._get_pixels_safely(trace))))
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
                    text=f"NDVI Evolution - {municipality_name}<br><sub>Click legend items to toggle cluster visibility</sub>",
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
            traceback.print_exc()
            return None