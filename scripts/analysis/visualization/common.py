#!/usr/bin/env python3
"""
Common Visualization Functions for Vegetation ST-Cube Segmentation

Shared visualization utilities for dual trend analysis and comparative visualizations.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any, Tuple
import warnings
from loguru import logger
from ..config_loader import get_config
import requests
from io import BytesIO
from PIL import Image

warnings.filterwarnings('ignore')


class CommonVisualization:
    """
    Common visualization utilities for vegetation ST-cube segmentation results.
    """
    
    def __init__(self, output_directory: str = None):
        """Initialize the common visualization utilities."""
        if output_directory is None:
            output_directory = get_config().output_dir
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_icgc_basemap_url(self, data):
        """Get ICGC basemap WMS URL for the given data bounds."""
        # Get geographic bounds from the data
        bounds = self._get_data_bounds(data)
        bbox_str = f"{bounds[1]},{bounds[0]},{bounds[3]},{bounds[2]}"
        basemap_layer = get_config().basemap_layer

        # Calculate pixel size based on bounds for consistent resolution across images
        avg_lat = (bounds[1] + bounds[3]) / 2
        cos_lat = math.cos(math.radians(avg_lat))
        width_m = abs(bounds[2] - bounds[0]) * 111320 * cos_lat
        height_m = abs(bounds[3] - bounds[1]) * 110574
        width_px = max(1, int(round(width_m / 3)))
        height_px = max(1, int(round(height_m / 3)))

        # Construct the WMS URL for ICGC RGB basemap
        icgc_wms_url = (
            "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms?"
            "REQUEST=GetMap&"
            "VERSION=1.3.0&"
            "SERVICE=WMS&"
            "CRS=EPSG:4326&"
            f"BBOX={bbox_str}&"
            f"WIDTH={width_px}&HEIGHT={height_px}&"
            f"LAYERS={basemap_layer}&"
            "STYLES=&"
            "FORMAT=JPEG&"
        )
        return icgc_wms_url, bounds

    def _add_icgc_basemap(self, ax, data):
        """Add ICGC RGB basemap using WMS service for matplotlib plots."""
        logger.info("Adding ICGC rgb basemap")
        icgc_wms_url, bounds = self._get_icgc_basemap_url(data)
        response = requests.get(icgc_wms_url)
        img = Image.open(BytesIO(response.content))
        ax.imshow(
            img,
            extent=[bounds[0], bounds[2], bounds[3], bounds[1]],
            origin='lower',
            alpha=1.0
        )

    def _add_icgc_basemap_to_plotly(self, fig, data):
        """Add ICGC basemap as background image to plotly figure."""
        icgc_wms_url, bounds = self._get_icgc_basemap_url(data)
        response = requests.get(icgc_wms_url)
        img = Image.open(BytesIO(response.content))
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=bounds[0],
                y=bounds[3],
                sizex=(bounds[2] - bounds[0]),
                sizey=(bounds[3] - bounds[1]),
                sizing="stretch",
                layer="below"
            )
        )
        fig.update_layout(
            xaxis=dict(
                range=[bounds[0], bounds[2]],
                title="Longitude",
                gridcolor='rgba(0,0,0,0.08)',
                gridwidth=1,
            ),
            yaxis=dict(
                range=[bounds[1], bounds[3]],
                title="Latitude",
                scaleanchor="x",
                scaleratio=1,
                gridcolor='rgba(0,0,0,0.08)',
                gridwidth=1,
            )
        )
        return True

    def _get_data_bounds(self, data):
        """Get the geographic bounds from the xarray dataset."""
        x_coords = data.x.values
        y_coords = data.y.values
        lon_min, lon_max = float(x_coords.min()), float(x_coords.max())
        lat_min, lat_max = float(y_coords.min()), float(y_coords.max())
        bounds = [lon_min, lat_min, lon_max, lat_max]
        return bounds

    def _get_pixels_safely(self, trace: Dict) -> List[Tuple[int, int]]:
        """Safely extract pixels from trace data, handling different formats."""
        pixels = trace['coordinates']
        return [tuple(p) for p in pixels]

    def _convert_pixel_to_latlon(self, data: Any, y_coord: int, x_coord: int) -> Tuple[float, float]:
        """Convert pixel coordinates to latitude/longitude using the same method as other modules."""
        lat = float(data.y.isel(y=y_coord).values)
        lon = float(data.x.isel(x=x_coord).values)
        return lat, lon


    def create_interactive_temporal_trend_map(self, results: Dict[str, List[Dict]], 
                                             data: Any, 
                                             municipality_name: str) -> str:
        """
        Create an interactive 2D map showing yearly NDVI changes for each cluster with a time slider.
        
        Args:
            results: Dictionary with 'greening' and 'browning' trend results
            data: xarray Dataset with coordinate and temporal information
            municipality_name: Name of the municipality
        
        Returns:
            Path to the saved interactive HTML file
        """        
        # Setup output
        filename = f"interactive_temporal_trends_{municipality_name.replace(' ', '_')}.html"
        output_file = self.output_dir / filename
        
        # Extract time coordinates (years)
        time_coords = data.time.values.tolist()
            
        # Prepare data for all clusters and all years
        cluster_data = []
        
        # Process both greening and browning clusters
        for trend_type in ['greening', 'browning']:
            if trend_type not in results:
                continue
                
            traces = results[trend_type]
            
            for trace_idx, trace in enumerate(traces):
                pixels = self._get_pixels_safely(trace)
                if not pixels:
                    continue
                    
                # Get cluster ID from the trace data
                actual_cluster_id = trace.get('id', trace_idx)
                    
                # Get NDVI profile for this cluster - use mean_temporal_profile from segmentation
                ndvi_profile = None
                for key in ['mean_temporal_profile', 'ndvi_profile', 'ndvi_time_series']:
                    profile = trace.get(key, None)
                    if profile is not None:
                        if hasattr(profile, 'tolist'):
                            ndvi_profile = profile.tolist()
                        elif hasattr(profile, '__len__') and len(profile) > 0:
                            ndvi_profile = list(profile)
                        break
                
                if not ndvi_profile or len(ndvi_profile) != len(time_coords):
                    logger.debug(f"Cluster {trace_idx}: Skipping cluster - NDVI profile length {len(ndvi_profile) if ndvi_profile else 0} != time coords length {len(time_coords)}, available keys: {list(trace.keys())}")
                    continue
                
                # Calculate year-over-year NDVI changes
                ndvi_changes = []
                for i in range(1, len(ndvi_profile)):
                    change = ndvi_profile[i] - ndvi_profile[i-1]
                    ndvi_changes.append(change)
                
                # Convert pixels to lat/lon
                for y_coord, x_coord in pixels:
                    try:
                        lat, lon = self._convert_pixel_to_latlon(data, y_coord, x_coord)
                        if lat is None or lon is None:
                            logger.debug(f"Could not convert coordinates ({y_coord}, {x_coord}) to lat/lon")
                            continue
                    except Exception as e:
                        logger.debug(f"Error converting coordinates ({y_coord}, {x_coord}): {e}")
                        continue
                    
                    # Create data point for each year (except first year)
                    for year_idx, change in enumerate(ndvi_changes):
                        year = time_coords[year_idx + 1]  # +1 because change is relative to previous year
                        
                        cluster_data.append({
                            'cluster_id': actual_cluster_id + 1,
                            'trend_type': trend_type,
                            'lat': lat,
                            'lon': lon,
                            'year': year,
                            'ndvi_change': change,
                            'ndvi_value': ndvi_profile[year_idx + 1],
                            'pixel_x': x_coord,
                            'pixel_y': y_coord
                        })
            
        # Create DataFrame
        df = pd.DataFrame(cluster_data) 
        
        # Check if we have any data to visualize
        if df.empty:
            logger.warning("No cluster data found for interactive temporal trend map")
            return

        fig = go.Figure()
        
        # Define color scale: red (-1) to white (0) to green (+1)
        colorscale = [
            [0.0, '#6B0000'],
            [0.37, '#8B0000'],
            [0.47, '#D3D3D3'],
            [0.5, '#FFFFFF'],
            [0.53, '#D3D3D3'],
            [0.63, '#006400'],
            [1.0, '#005800']
        ]
        
        # Create traces for each year
        years = sorted(df['year'].unique())
        
        for i, year in enumerate(years):
            year_data = df[df['year'] == year]
            
            ndvi_changes_clamped = np.clip(year_data['ndvi_change'], -1, 1)
            
            fig.add_trace(
                go.Scatter(
                    x=year_data['lon'],
                    y=year_data['lat'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='square',
                        color=ndvi_changes_clamped,
                        colorscale=colorscale,
                        cmin=-1,
                        cmax=1,
                        colorbar=dict(
                            title="NDVI Change",
                            tickmode="linear",
                            tick0=-1,
                            dtick=0.5,
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=['-1.0 (Browning)', '-0.5', '0.0 (No Change)', '+0.5', '+1.0 (Greening)']
                        ),
                        opacity=0.8,
                        line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                    ),
                    text=[
                        f"Cluster {row['cluster_id']}<br>"
                        f"Type: {row['trend_type']}<br>"
                        f"Year: {row['year']}<br>"
                        f"NDVI Change: {row['ndvi_change']:.3f}<br>"
                        f"NDVI Value: {row['ndvi_value']:.3f}<br>"
                        f"Lat: {row['lat']:.4f}<br>"
                        f"Lon: {row['lon']:.4f}<br>"
                        for _, row in year_data.iterrows()
                    ],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Year {year}',
                    visible=(i == 0)
                )
            )
        
        # Create slider steps
        steps = []
        for i, year in enumerate(years):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(years)}],
                label=str(year)
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        
        # Add slider
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Year: "},
            pad={"t": 50},
            steps=steps
        )]
        
        # Add ICGC basemap
        self._add_icgc_basemap_to_plotly(fig, data)
        
        # Update layout with title and slider
        fig.update_layout(
            title=dict(
                text=f'Interactive Temporal NDVI Trends - {municipality_name}<br>'
                     f'<sub>Use slider to navigate through years. Square pixels show NDVI change. Color indicates yearly change intensity.</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            sliders=sliders,
            height=800,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Save
        fig.write_html(
            output_file,
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'scrollZoom': True
            }
        )
        
        logger.success(f"Interactive temporal trend map saved successfully")
        return str(output_file)




    def create_spatial_distribution_map(self, results: Dict[str, List[Dict]], data: Any, municipality_name: str) -> str:
        """
        Create a spatial distribution map showing individual clusters with numbered labels.
        Greening clusters in green, browning clusters in red, with cluster numbers displayed on map.
        
        Args:
            results: Dictionary with 'greening' and 'browning' trend results
            data: xarray Dataset with coordinate information  
            municipality_name: Name of the municipality
            
        Returns:
            Path to the saved visualization file
        """
        # Setup output
        filename = f"spatial_distribution_{municipality_name.replace(' ', '_')}.png"
        output_file = self.output_dir / filename       
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Set axis limits based on data extent
        bounds = self._get_data_bounds(data)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # Add ICGC basemap
        self._add_icgc_basemap(ax, data)
        
        greening_color = '#2E8B57'
        browning_color = '#CD5C5C'
        
        # Calculate pixel size in data coordinates for rectangles
        x_res = abs(float(data.x.values[1] - data.x.values[0]))
        y_res = abs(float(data.y.values[1] - data.y.values[0]))
        
        total_clusters_plotted = 0
        
        # Process each trend type
        for trend_type, color in [('greening', greening_color), ('browning', browning_color)]:
            if trend_type not in results:
                continue
                
            traces = results[trend_type]
            
            for trace in traces:
                pixels = self._get_pixels_safely(trace)
                if not pixels:
                    continue
                
                cluster_id = trace.get('id', 0) + 1
                
                # Calculate cluster centroid for label placement
                lats, lons = [], []
                
                # Convert all pixels to lat/lon coordinates and create rectangles
                for y_coord, x_coord in pixels:
                    lat, lon = self._convert_pixel_to_latlon(data, y_coord, x_coord)
                    if lat is not None and lon is not None:
                        lats.append(lat)
                        lons.append(lon)
                        
                        # Create a rectangle representing the 30x30m pixel
                        rect = plt.Rectangle(
                            (lon - x_res/2, lat - y_res/2),  # Bottom-left corner
                            x_res,  # Width
                            y_res,  # Height
                            facecolor=color,
                            edgecolor='none',
                            alpha=0.8
                        )
                        ax.add_patch(rect)
                
                # Add cluster number label at centroid
                if lats and lons:
                    centroid_lat = np.mean(lats)
                    centroid_lon = np.mean(lons)                    
                    ax.text(centroid_lon, centroid_lat, str(cluster_id),
                           fontsize=10, fontweight='bold', color='white',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                total_clusters_plotted += 1
        
        # Set labels and title
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Vegetation Cluster Distribution - {municipality_name}\n'
                    f'Green: Greening Trends, Red: Browning Trends ({total_clusters_plotted} clusters)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)       
        ax.set_aspect('equal', adjustable='box')   
        ax.autoscale(tight=True)
        plt.tight_layout()  
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.success(f"Spatial distribution map saved successfully")
        return str(output_file)