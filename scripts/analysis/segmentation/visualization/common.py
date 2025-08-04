#!/usr/bin/env python3
"""
Common Visualization Functions for Vegetation ST-Cube Segmentation

Shared visualization utilities for dual trend analysis and comparative visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings
from loguru import logger
from ..config_loader import get_config
import requests
from io import BytesIO
from PIL import Image

warnings.filterwarnings('ignore')

def _add_icgc_basemap(ax, data):
    """Add ICGC RGB basemap using WMS service."""
    logger.info("Adding ICGC rgb basemap")

    try:
        data_crs = 'EPSG:4326'
        if hasattr(data, 'rio') and hasattr(data.rio, 'crs') and data.rio.crs:
            data_crs = str(data.rio.crs)

        # Get geographic bounds from the data
        bounds = _get_data_bounds(data)
        if bounds is None:
            logger.warning("Could not determine geographic bounds from data")
            return

        bbox_str = f"{bounds[1]},{bounds[0]},{bounds[3]},{bounds[2]}"

        # WMS URL for ICGC RGB basemap
        icgc_wms_url = (
            "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms?"
            "REQUEST=GetMap&"
            "VERSION=1.3.0&"
            "SERVICE=WMS&"
            "CRS=EPSG:4326&"
            f"BBOX={bbox_str}&"
            "WIDTH=3000&HEIGHT=3000&"
            "LAYERS=ortofoto_color_vigent&"
            "STYLES=&"
            "FORMAT=JPEG&"
        )

        response = requests.get(icgc_wms_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Plot image on graph
        ax.imshow(
            img,
            extent=[bounds[0], bounds[2], bounds[3], bounds[1]],
            origin='lower',
            alpha=1.0
        )
    except Exception as e:
        logger.warning(f"ICGC basemap failed: {e}")

def _get_data_bounds(data):
    """Get the geographic bounds from the xarray dataset."""
    try:
        # Get coordinate arrays from the data
        if hasattr(data, 'x') and hasattr(data, 'y'):
            x_coords = data.x.values
            y_coords = data.y.values
        elif hasattr(data, 'longitude') and hasattr(data, 'latitude'):
            x_coords = data.longitude.values
            y_coords = data.latitude.values
        elif hasattr(data, 'lon') and hasattr(data, 'lat'):
            x_coords = data.lon.values
            y_coords = data.lat.values
        else:
            logger.warning("Could not find coordinate arrays in data")
            return None
            
        # Calculate bounds
        lon_min, lon_max = float(x_coords.min()), float(x_coords.max())
        lat_min, lat_max = float(y_coords.min()), float(y_coords.max())
        bounds = [lon_min, lat_min, lon_max, lat_max]

        return bounds
        
    except Exception as e:
        logger.error(f"Error calculating data bounds: {e}")
        return None


def create_dual_trend_spatial_map(results: Dict[str, List[Dict]], 
                                 data: Any, 
                                 municipality_name: str,
                                 output_directory: str) -> str:
    """
    Create a spatial map showing greening and browning vegetation trends grouped by type.
    
    Args:
        results: Dictionary with 'greening' and 'browning' trend results
        data: xarray Dataset with coordinate information
        municipality_name: Name of the municipality
        output_directory: Directory to save the visualization
    
    Returns:
        Path to the saved visualization file
    """
    logger.info(f"Creating greening/browning visualization for {municipality_name}")
    
    # Setup output
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ndvi_evo_map_{municipality_name.replace(' ', '_')}.png"
    output_file = output_dir / filename
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Set axis limits based on data extent
    bounds = _get_data_bounds(data)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # Add ICGC basemap
    _add_icgc_basemap(ax, data)
    
    greening_color = '#2E8B57'
    browning_color = '#CD5C5C'

    # Calculate pixel size in data coordinates for rectangles
    x_res = abs(float(data.x.values[1] - data.x.values[0]))
    y_res = abs(float(data.y.values[1] - data.y.values[0]))
    
    total_pixels_plotted = 0
    trend_counts = {}
    
    # Process each trend type
    for trend_type, color in [('greening', greening_color), ('browning', browning_color)]:
        if trend_type not in results:
            continue
            
        cubes = results[trend_type]
        trend_pixels = 0
                
        # Process all clusters of this trend type
        for cube in cubes:
            pixels = _get_pixels_safely(cube)
            if not pixels:
                continue
                
            # Convert all pixels to lat/lon coordinates and create rectangles
            for y_coord, x_coord in pixels:
                lat, lon = _convert_pixel_to_latlon(data, y_coord, x_coord)
                if lat is not None and lon is not None:
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
                    trend_pixels += 1
                    total_pixels_plotted += 1
        
        trend_counts[trend_type] = trend_pixels
    
    # Add legend and formatting
    legend_elements = []
    if 'greening' in trend_counts:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=greening_color, alpha=0.9, 
                                           label=f'Greening NDVI ({trend_counts["greening"]:,} pixels)'))
    if 'browning' in trend_counts:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=browning_color, alpha=0.9,
                                           label=f'Browning NDVI ({trend_counts["browning"]:,} pixels)'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Vegetation Trend Analysis - {municipality_name}\n'
                f'Grouped by NDVI Change Direction ({total_pixels_plotted:,} total pixels)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio to preserve geographic proportions
    ax.set_aspect('equal', adjustable='box')   
    ax.autoscale(tight=True)
    
    # Text box with statistics
    stats_text = []
    if 'greening' in trend_counts and 'browning' in trend_counts:
        inc_count = trend_counts['greening']
        dec_count = trend_counts['browning']
        total = inc_count + dec_count
        if total > 0:
            inc_pct = (inc_count / total) * 100
            dec_pct = (dec_count / total) * 100
            stats_text.extend([
                f"Greening: {inc_pct:.1f}% ({inc_count:,} pixels)",
                f"Browning: {dec_pct:.1f}% ({dec_count:,} pixels)",
                f"Ratio (Greening/Browning): {inc_count/dec_count:.2f}" if dec_count > 0 else "Ratio: ∞"
            ])
    
    if stats_text:
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()  
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.success(f"Trend spatial map saved successfully")
    return str(output_file)


def _get_pixels_safely(cube: Dict) -> List[Tuple[int, int]]:
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


def _convert_pixel_to_latlon(data: Any, y_coord: int, x_coord: int) -> Tuple[float, float]:
    """Convert pixel coordinates to latitude/longitude using the same method as other modules."""
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