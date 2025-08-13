"""
Cluster Visualizer with ICGC WMS Basemap
Combines cluster visualization from JSON files with ICGC orthophoto basemap.
"""

# ==== CONFIGURABLE PARAMETERS ====
JSON_PATH = "outputs/Sant_Adria_del_Besos/20250812_185543/vegetation_clusters_combined_Sant_Adria_del_Besos.json"
CLUSTER_IDS = [1, 2, 3, 5, 6, 7, 8, 9]  # List of cluster IDs to visualize (can be single ID or multiple)
# =================================

import json
import matplotlib.pyplot as plt
import geopandas as gpd
import requests
from PIL import Image
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
from io import BytesIO
import math
from matplotlib.patches import Rectangle
import unicodedata

warnings.filterwarnings('ignore')

# Fixed parameters (no need to configure)
BASEMAP_LAYER = "ortofoto_color_vigent"
BOUNDARIES_PATH = "data/boundaries/AMB_Municipalities.shp"

# Configure logger
logger.add(
    "logs/cluster_visualizer_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def normalize_text(text):
    """Remove accents and normalize text for comparison."""
    # Normalize unicode characters and remove accents
    normalized = unicodedata.normalize('NFD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text.lower().strip()


def load_municipality_boundaries(boundaries_path, municipality_name):
    """Load municipality boundaries from shapefile."""
    logger.info(f"Loading municipality boundaries from {boundaries_path}")
    
    gdf = gpd.read_file(boundaries_path)
    
    # Find municipality name column
    name_columns = ['NOMMUNI', 'NOM', 'NAME', 'MUNICIPALITY', 'nom', 'name']
    municipality_column = None
    
    for col in name_columns:
        if col in gdf.columns:
            municipality_column = col
            break
    
    if municipality_column is None:
        municipality_column = gdf.columns[0]
    
    logger.info(f"Using column '{municipality_column}' for municipality names")
    logger.info(f"Looking for municipality: '{municipality_name}'")
    
    # Normalize the search term
    normalized_search = normalize_text(municipality_name)
    
    # Filter for specific municipality using accent-insensitive matching
    mask = gdf[municipality_column].apply(lambda x: normalize_text(str(x)) == normalized_search)
    gdf_filtered = gdf[mask]
    
    if gdf_filtered.empty:
        # Try partial matching if exact match fails
        mask = gdf[municipality_column].apply(lambda x: normalized_search in normalize_text(str(x)))
        gdf_filtered = gdf[mask]
        
        if not gdf_filtered.empty:
            logger.info(f"Found municipality using partial matching")
        else:
            alt_names = [
                municipality_name.replace('_', ' '),
                municipality_name.replace(' ', '_'),
                municipality_name.replace("'", "'"),
                municipality_name.replace("'", "'")
            ]
            
            for alt_name in alt_names:
                normalized_alt = normalize_text(alt_name)
                mask = gdf[municipality_column].apply(lambda x: normalize_text(str(x)) == normalized_alt)
                gdf_filtered = gdf[mask]
                if not gdf_filtered.empty:
                    logger.info(f"Found municipality using alternative name: '{alt_name}'")
                    break
            
            if gdf_filtered.empty:
                available_names = gdf[municipality_column].tolist()
                logger.error(f"Municipality '{municipality_name}' not found in shapefile")
                logger.error(f"Available municipalities: {available_names[:10]}...")
                raise ValueError(f"Municipality '{municipality_name}' not found in shapefile")
    
    bbox = gdf_filtered.total_bounds  # [minx, miny, maxx, maxy]
    logger.info(f"Municipality bbox: {bbox}")
    return gdf_filtered, bbox


def load_json_data(json_path):
    """Load and parse JSON cluster data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_file_path = Path(json_path)
    output_dir = json_file_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    return data, str(output_dir)


def find_cluster_in_data(data, cluster_id):
    """Find cluster by ID in any trend category."""
    for trend_name, trend_data in data['trends'].items():
        for cluster in trend_data.get('clusters', []):
            if cluster['cluster_id'] == cluster_id:
                return cluster, trend_name
    
    raise ValueError(f"Cluster {cluster_id} not found in any trend category")


def get_cluster_coordinates(cluster):
    """Extract cluster coordinates."""
    lats, lons = [], []
    
    for trace in cluster['traces']:
        coords = trace['coordinates']
        if 'latitude' in coords and 'longitude' in coords:
            lats.append(coords['latitude'])
            lons.append(coords['longitude'])
    
    if not lats:
        raise ValueError("No valid coordinates found in cluster")
    
    return lats, lons


def get_wms_image(bbox, layer):
    """Fetch WMS image from ICGC service with consistent resolution based on bbox."""
    logger.info("Fetching WMS image...")
    
    # Calculate pixel size based on bounds for consistent resolution
    avg_lat = (bbox[1] + bbox[3]) / 2
    cos_lat = math.cos(math.radians(avg_lat))
    width_m = abs(bbox[2] - bbox[0]) * 111320 * cos_lat
    height_m = abs(bbox[3] - bbox[1]) * 110574
    width_px = max(1, int(round(width_m / 3)))
    height_px = max(1, int(round(height_m / 3)))
    
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    
    icgc_wms_url = (
        "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms?"
        "REQUEST=GetMap&"
        "VERSION=1.3.0&"
        "SERVICE=WMS&"
        "CRS=EPSG:4326&"
        f"BBOX={bbox_str}&"
        f"WIDTH={width_px}&HEIGHT={height_px}&"
        f"LAYERS={layer}&"
        "STYLES=&"
        "FORMAT=JPEG"
    )
    
    logger.info(f"WMS request: {icgc_wms_url}")
    logger.info(f"Calculated dimensions: {width_px}x{height_px} pixels")
    
    response = requests.get(icgc_wms_url, timeout=30)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '')
    
    if 'image' not in content_type:
        raise ValueError("WMS service did not return an image")
    
    img = Image.open(BytesIO(response.content))
    return img


def create_spatial_map(clusters_data, data, municipality_gdf, municipality_bbox, output_dir):
    """Create spatial map with municipality basemap and all cluster points overlay."""
    logger.info("Creating spatial map with municipality basemap for all clusters...")
    
    municipality = data['metadata']['municipality']
    
    # Get WMS basemap for the entire municipality
    wms_image = get_wms_image(municipality_bbox, BASEMAP_LAYER)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display WMS image as background
    ax.imshow(np.array(wms_image), extent=[municipality_bbox[0], municipality_bbox[2], municipality_bbox[1], municipality_bbox[3]], 
              origin='upper', alpha=0.9)
    
    # Add municipality boundaries
    municipality_gdf.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1, alpha=0.0)
    
    # Define trend-based colors
    def get_trend_color(trend, cluster_id):
        if 'greening' in trend.lower():
            green_colors = ['forestgreen', 'limegreen', 'darkgreen', 'seagreen', 'mediumseagreen', 'lightgreen', 'palegreen', 'yellowgreen']
            return green_colors[cluster_id % len(green_colors)]
        elif 'browning' in trend.lower():
            red_colors = ['darkred', 'red', 'crimson', 'firebrick', 'brown']
            return red_colors[cluster_id % len(red_colors)]
        else:
            other_colors = ['blue', 'orange', 'purple', 'pink', 'gray']
            return other_colors[cluster_id % len(other_colors)]
    
    # Overlay cluster points
    lon_range = municipality_bbox[2] - municipality_bbox[0]
    lat_range = municipality_bbox[3] - municipality_bbox[1]
    
    for cluster_id, cluster, trend in clusters_data:
        lats, lons = get_cluster_coordinates(cluster)
        color = get_trend_color(trend, cluster_id)
        
        for lon, lat in zip(lons, lats):
            # Calculate dynamic pixel size based on WMS image resolution
            # Get WMS image dimensions to calculate pixel density
            avg_lat = (municipality_bbox[1] + municipality_bbox[3]) / 2
            cos_lat = math.cos(math.radians(avg_lat))
            width_m = abs(municipality_bbox[2] - municipality_bbox[0]) * 111320 * cos_lat
            height_m = abs(municipality_bbox[3] - municipality_bbox[1]) * 110574
            width_px = max(1, int(round(width_m / 3)))
            height_px = max(1, int(round(height_m / 3)))
            
            # Calculate pixel size in geographic coordinates
            pixel_size_lon = lon_range / width_px
            pixel_size_lat = lat_range / height_px
            
            # Make rectangles represent approximately 30x30m pixels (10 pixels)
            rect_width = pixel_size_lon * 10 * 0.75
            rect_height = pixel_size_lat * 13  * 0.8 # Always taller
            
            rect = Rectangle((lon - rect_width/2, lat - rect_height/2), 
                           rect_width, rect_height,
                           facecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        # Add legend entry
        ax.scatter([], [], c=color, s=80, alpha=0.8, marker='s',
                   label=f'Cluster {cluster_id} ({len(lats)} traces)')
    
    # Set extent and labels
    ax.set_xlim(municipality_bbox[0], municipality_bbox[2])
    ax.set_ylim(municipality_bbox[1], municipality_bbox[3])
    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    #ax.set_title(f'{municipality} - {len(clusters_data)} Clusters\nICGC Orthophoto', fontsize=30, fontweight='bold')
    
    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Place legend outside the plot area with bigger font
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cluster_ids_str = '_'.join([str(cid) for cid, _, _ in clusters_data])
    filename = f"clusters_{cluster_ids_str}_spatial_map.png"
    filepath = output_path / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def create_temporal_plot(clusters_data, data, output_dir):
    """Create NDVI temporal evolution plot for all clusters."""
    logger.info("Creating temporal evolution plot for all clusters...")
    
    municipality = data['metadata']['municipality']
    years = data['metadata']['years']
    
    # Define trend-based colors for temporal plot
    def get_trend_color_temporal(trend, cluster_id):
        if 'greening' in trend.lower():
            green_colors = ['forestgreen', 'limegreen', 'darkgreen', 'seagreen', 'mediumseagreen']
            return green_colors[cluster_id % len(green_colors)]
        elif 'browning' in trend.lower():
            red_colors = ['darkred', 'red', 'crimson', 'firebrick', 'brown']
            return red_colors[cluster_id % len(red_colors)]
        else:
            other_colors = ['blue', 'orange', 'purple', 'pink', 'gray']
            return other_colors[cluster_id % len(other_colors)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each cluster and store final positions for labels
    all_stats = []
    cluster_final_positions = []  # Store (x, y, label, color) for end labels
    for cluster_id, cluster, trend in clusters_data:
        temporal_profile = cluster['temporal_profile']
        mean_ndvi_dict = temporal_profile['mean_ndvi_per_year']
        std_ndvi_dict = temporal_profile.get('std_ndvi_per_year', {})
        
        mean_ndvi = [mean_ndvi_dict[str(year)] for year in years if str(year) in mean_ndvi_dict]
        std_ndvi = [std_ndvi_dict.get(str(year), 0) for year in years if str(year) in mean_ndvi_dict]
        valid_years = [year for year in years if str(year) in mean_ndvi_dict]
        
        if not mean_ndvi:
            continue
        
        years_array = np.array(valid_years)
        mean_ndvi_array = np.array(mean_ndvi)
        std_ndvi_array = np.array(std_ndvi)
        
        color = get_trend_color_temporal(trend, cluster_id)
        
        # Plot mean NDVI line
        ax.plot(years_array, mean_ndvi_array, 
               color=color, linewidth=2, marker='o', markersize=4)
        
        # Store final position for end label
        if len(years_array) > 0 and len(mean_ndvi_array) > 0:
            final_x = years_array[-1]
            final_y = mean_ndvi_array[-1]
            cluster_final_positions.append((final_x, final_y, f'C{cluster_id}', color))

        # Add standard deviation
        if np.any(std_ndvi_array > 0):
            ax.fill_between(years_array, 
                           mean_ndvi_array - std_ndvi_array, 
                           mean_ndvi_array + std_ndvi_array,
                           alpha=0.2, color=color)
        
        # Collect stats
        overall_std = cluster['summary']['overall_cluster_std']
        n_traces = len(cluster['traces'])
        all_stats.append(f'C{cluster_id}: {n_traces} traces, std: {overall_std:.3f}')
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=20)
    ax.set_ylabel('NDVI', fontsize=20)
    # ax.set_title(f'{municipality} - {len(clusters_data)} Clusters NDVI Evolution', fontsize=30, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax.set_xticks(np.arange(min(years)+1, max(years)+1, 5))
    ax.set_xticklabels(np.arange(min(years)+1, max(years)+1, 5))
    ax.set_xlim(min(years), max(years))
    
    # Create secondary y-axis for cluster labels
    ax2 = ax.twinx()
    ax2.set_ylabel('Clusters', fontsize=20, labelpad=40)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelsize=0, length=0)
    
    # Sort cluster positions by y-coordinate for overlap prevention
    cluster_final_positions.sort(key=lambda x: x[1]) 
    min_spacing = 0.03
    adjusted_positions = []
    
    for i, (final_x, final_y, label, color) in enumerate(cluster_final_positions):
        adjusted_y = final_y
        
        # Check for overlaps with previously placed labels
        for prev_y in adjusted_positions:
            if abs(adjusted_y - prev_y) < min_spacing:
                if adjusted_y < prev_y:
                    adjusted_y = prev_y - min_spacing
                else:
                    adjusted_y = prev_y + min_spacing        
        adjusted_y = max(0.02, min(0.90, adjusted_y))
        adjusted_positions.append(adjusted_y)
        
        # Use the secondary axis for positioning cluster labels
        ax2.text(max(years) + 0.5, adjusted_y, label, 
               color=color, fontsize=14,
               verticalalignment='center', horizontalalignment='left',
               transform=ax2.transData)

    # stats_text = f'Total Clusters: {len(clusters_data)}\n' + '\n'.join(all_stats)
    # 
    # ax.text(0.02, 0.98, stats_text, 
    #        transform=ax.transAxes, 
    #        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    #        verticalalignment='top', fontsize=10)
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"clusters_{'_'.join([str(cid) for cid, _, _ in clusters_data])}_temporal_evolution.png"
    filepath = output_path / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved temporal plot: {filepath}")
    return str(filepath)


def main(json_path=JSON_PATH, cluster_ids=CLUSTER_IDS):
    """Main function to create combined cluster visualizations."""
    try:
        # Load data
        data, output_dir = load_json_data(json_path)
        municipality = data['metadata']['municipality']
        municipality_gdf, municipality_bbox = load_municipality_boundaries(BOUNDARIES_PATH, municipality)
        
        # Collect cluster data
        clusters_data = []
        for cluster_id in cluster_ids:
            cluster, trend = find_cluster_in_data(data, cluster_id)
            clusters_data.append((cluster_id, cluster, trend))
        
        # Create visualizations
        spatial_path = create_spatial_map(clusters_data, data, municipality_gdf, municipality_bbox, output_dir)
        temporal_path = create_temporal_plot(clusters_data, data, output_dir)
        
        # Print summary
        cluster_ids_str = ', '.join([str(cid) for cid, _, _ in clusters_data])
        print(f"Created combined visualizations for clusters {cluster_ids_str}:")
        print(f"  Spatial map: {spatial_path}")
        print(f"  Temporal plot: {temporal_path}")
        
        print(f"\nCluster details:")
        for cluster_id, cluster, trend in clusters_data:
            n_traces = len(cluster['traces'])
            print(f"  Cluster {cluster_id}: {n_traces} traces ({trend} trend)")
        
        return {
            'spatial_map': spatial_path,
            'temporal_plot': temporal_path,
            'clusters': clusters_data
        }
        
    except Exception as e:
        logger.error(f"Error in cluster visualization: {e}")
        raise


if __name__ == "__main__":
    main()