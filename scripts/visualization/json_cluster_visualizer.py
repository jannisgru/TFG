"""
Cluster Visualizer with ICGC WMS Basemap
Combines cluster visualization from JSON files with ICGC orthophoto basemap.
"""

# ==== CONFIGURABLE PARAMETERS ====
JSON_PATH = "outputs/PEIN/CLR/20250814_220347/vegetation_clusters_combined_None.json"
CLUSTER_IDS = [1, 4, 7, 9 ,10, 16, 17, 18]  # List of cluster IDs to visualize (can be single ID or multiple)
MUNICIPALITY_STATS_JSON = "outputs/municipality_statistics/municipality_ndvi_statistics.json"  # Path to municipality statistics JSON
CREATE_COMPARISON_CSV = True  # Set to True to generate municipality-cluster comparison CSV
COLOR_CODE_NDVI = True  # Set to True to color-code cells based on NDVI values (higher = greener)
# =================================

import json
import matplotlib.pyplot as plt
import geopandas as gpd
import requests
from PIL import Image
import numpy as np
import pandas as pd
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

    # WMS servers often limit image dimensions (common max is 4096). If calculated
    # dimensions exceed that, scale the larger side down to the maximum and
    # reduce the other side proportionally to keep aspect ratio.
    MAX_PIX = 4096
    if width_px > MAX_PIX or height_px > MAX_PIX:
        if width_px >= height_px:
            scale = MAX_PIX / float(width_px)
        else:
            scale = MAX_PIX / float(height_px)

        new_width = max(1, int(round(width_px * scale)))
        new_height = max(1, int(round(height_px * scale)))
        logger.warning(f"Calculated dimensions {width_px}x{height_px} exceed WMS limits; scaling to {new_width}x{new_height}")
        width_px, height_px = new_width, new_height
    
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
    
    try:
        response = requests.get(icgc_wms_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"WMS request failed: {e}")
        raise

    content_type = response.headers.get('content-type', '')

    if 'image' not in content_type:
        # If the server returned an XML ServiceException or non-image content,
        # include the body (shortened) in the error to help debugging.
        body = response.content.decode('utf-8', errors='ignore')[:2000]
        logger.error(f"WMS server response not an image. Content starts: {body}")
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
    ax.set_title(f'{municipality} - {len(clusters_data)} Clusters', fontsize=20, fontweight='bold')
    
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
    ax.set_title(f'{municipality} - {len(clusters_data)} Clusters NDVI Evolution', fontsize=20, fontweight='bold')

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
               fontsize=16,
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


def load_municipality_stats(json_path):
    """Load municipality NDVI statistics from JSON file."""
    logger.info(f"Loading municipality statistics from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def find_and_aggregate_park_stats(municipality_stats, park_name):
    """Search all municipalities for a park (pein/xpn) and aggregate stats across them.

    Returns an aggregated data dict that mimics a municipality stats structure
    with keys 'years' and 'general'->'yearly', or (None, None) if not found.
    Also returns the list of municipality names that contributed.
    """
    normalized_search = normalize_text(park_name)
    contributors = []

    # Collect per-year accumulators
    yearly_acc = {}  # year_str -> { 'weighted_sum': float, 'weight': float, 'vegetation_pixels': int, 'category_counts': {cat:count} }
    overall_veg_pixels = 0

    # Iterate municipalities
    for muni_name, muni_data in municipality_stats.get('municipalities', {}).items():
        for park_type in ('pein', 'xpn'):
            parks = muni_data.get(park_type, {}) or {}
            for candidate_name, park_data in parks.items():
                if not candidate_name:
                    continue
                if normalize_text(candidate_name) == normalized_search:
                    # Found park in this municipality
                    # Ensure park_data has yearly info
                    yearly = park_data.get('yearly', {}) if isinstance(park_data, dict) else {}
                    found = False
                    for year_str, ydata in yearly.items():
                        found = True
                        # Try multiple keys for vegetation pixel count
                        veg_pixels = ydata.get('vegetation_pixels') or ydata.get('vegetation_data_points') or ydata.get('spatial_pixels') or ydata.get('vegetation') or 0

                        mean_ndvi = ydata.get('vegetation_mean_ndvi') or ydata.get('mean_ndvi') or ydata.get('mean')

                        if year_str not in yearly_acc:
                            yearly_acc[year_str] = {
                                'weighted_sum': 0.0,
                                'weight': 0.0,
                                'vegetation_pixels': 0,
                                'category_counts': {}
                            }

                        # Weight by vegetation pixels when available
                        weight = veg_pixels if veg_pixels else 1
                        if mean_ndvi is not None:
                            try:
                                val = float(mean_ndvi)
                                yearly_acc[year_str]['weighted_sum'] += val * weight
                                yearly_acc[year_str]['weight'] += weight
                            except Exception:
                                pass

                        yearly_acc[year_str]['vegetation_pixels'] += int(veg_pixels or 0)

                        # Aggregate category counts if present
                        cat_counts = ydata.get('category_counts', {})
                        if isinstance(cat_counts, dict):
                            for cat, cnt in cat_counts.items():
                                try:
                                    yearly_acc[year_str]['category_counts'][cat] = yearly_acc[year_str]['category_counts'].get(cat, 0) + int(cnt or 0)
                                except Exception:
                                    pass

                    # Aggregate overall vegetation pixels if present in park_data
                    overall = park_data.get('overall', {}) if isinstance(park_data, dict) else {}
                    overall_veg_pixels += int(overall.get('vegetation_data_points') or overall.get('vegetation_pixels') or 0)

                    if found:
                        contributors.append(muni_name)

    if not contributors:
        return None, None

    # Build aggregated structure
    years = sorted([int(y) for y in yearly_acc.keys()])
    agg_yearly = {}
    for y in years:
        ys = yearly_acc[str(y)] if str(y) in yearly_acc else yearly_acc.get(y, yearly_acc.get(str(y), None))
        if ys is None:
            continue
        weight = ys['weight']
        if weight and weight > 0:
            avg = ys['weighted_sum'] / weight
        else:
            avg = None

        agg_yearly[str(y)] = {
            'mean_ndvi': avg if avg is not None else None,
            'vegetation_mean_ndvi': avg if avg is not None else None,
            'vegetation_pixels': ys['vegetation_pixels'],
            'category_counts': ys['category_counts']
        }

    aggregated = {
        'municipality_name': park_name,
        'years': years,
        'analysis_date': None,
        'general': {
            'overall': {
                'vegetation_data_points': overall_veg_pixels
            },
            'yearly': agg_yearly
        }
    }

    return aggregated, contributors


def extract_municipality_name_from_cluster_data(cluster_data):
    """Extract municipality name from cluster JSON metadata."""
    if 'metadata' in cluster_data and 'municipality' in cluster_data['metadata']:
        return cluster_data['metadata']['municipality']
    elif 'metadata' in cluster_data and 'config_parameters' in cluster_data['metadata']:
        return cluster_data['metadata']['config_parameters'].get('municipality_name', 'Unknown')
    else:
        return 'Unknown'


def find_cluster_by_id(cluster_data, cluster_id):
    """Find a specific cluster by ID in the cluster data."""
    # Check both greening and browning trends
    for trend_type in ['greening', 'browning']:
        if trend_type in cluster_data.get('trends', {}):
            clusters = cluster_data['trends'][trend_type].get('clusters', [])
            for cluster in clusters:
                if cluster.get('cluster_id') == cluster_id:
                    return cluster, trend_type
    return None, None


def extract_cluster_ndvi_profile(cluster):
    """Extract NDVI temporal profile from cluster data."""
    if 'temporal_profile' in cluster:
        temporal_profile = cluster['temporal_profile']
        if 'mean_ndvi_per_year' in temporal_profile:
            return temporal_profile['mean_ndvi_per_year']
        elif 'yearly_mean_ndvi' in temporal_profile:
            return temporal_profile['yearly_mean_ndvi']
        elif 'mean_ndvi_by_year' in temporal_profile:
            return temporal_profile['mean_ndvi_by_year']
    return {}


def create_comparison_table(municipality_stats, cluster_data, cluster_ids, municipality_name, output_dir):
    """Create the comparison table combining municipality and cluster data."""
    logger.info(f"Creating comparison table for {municipality_name}")
    
    # Find municipality data with better matching
    municipality_data = None
    if 'municipalities' in municipality_stats:
        # First try exact match
        if municipality_name in municipality_stats['municipalities']:
            municipality_data = municipality_stats['municipalities'][municipality_name]
        else:
            # Try normalized matching
            normalized_search = normalize_text(municipality_name)            
            for muni_name, muni_data in municipality_stats['municipalities'].items():
                normalized_muni = normalize_text(muni_name)                
                if normalized_muni == normalized_search:
                    municipality_data = muni_data
                    break
            
            # If still not found, try partial matching
            if not municipality_data:
                for muni_name, muni_data in municipality_stats['municipalities'].items():
                    normalized_muni = normalize_text(muni_name)
                    if normalized_search in normalized_muni or normalized_muni in normalized_search:
                        municipality_data = muni_data
                        break
    
    if not municipality_data:
        # List available municipalities for debugging
        available_munis = list(municipality_stats.get('municipalities', {}).keys())[:10]
        logger.warning(f"Municipality data not found for: '{municipality_name}'. Trying to find as a park name across municipalities...")

        # Try aggregating park stats across municipalities (pein/xpn)
        aggregated, contributors = find_and_aggregate_park_stats(municipality_stats, municipality_name)
        if aggregated is not None:
            logger.info(f"Aggregated park stats found for '{municipality_name}' from {len(contributors)} municipalities: {contributors}")
            municipality_data = aggregated
        else:
            logger.error(f"Municipality or park data not found for: '{municipality_name}'")
            logger.error(f"Available municipalities (first 10): {available_munis}")
            return None, None, None
    
    # Get years from municipality data
    years = municipality_data.get('years', [])
    if not years:
        logger.error("No years found in municipality data")
        return None, None, None
    
    # Initialize the comparison table
    comparison_data = []
    
    for year in years:
        year_str = str(year)
        row = {
            'Year': year,
            'Municipality': municipality_name
        }
        
        # Add municipality general statistics
        general_stats = municipality_data.get('general', {}).get('yearly', {})
        if year_str in general_stats:
            year_data = general_stats[year_str]
            row['Municipality_Avg_NDVI'] = year_data.get('mean_ndvi', np.nan)
            row['Municipality_Vegetation_Avg_NDVI'] = year_data.get('vegetation_mean_ndvi', np.nan)
        else:
            row['Municipality_Avg_NDVI'] = np.nan
            row['Municipality_Vegetation_Avg_NDVI'] = np.nan
        
        # Add cluster statistics
        for cluster_id in cluster_ids:
            cluster, trend_type = find_cluster_by_id(cluster_data, cluster_id)
            
            if cluster:
                # Extract NDVI profile for this cluster
                ndvi_profile = extract_cluster_ndvi_profile(cluster)
                
                # Get NDVI value for this year
                cluster_ndvi = np.nan
                if year_str in ndvi_profile:
                    cluster_ndvi = ndvi_profile[year_str]
                
                # Add to row (only NDVI value)
                row[f'Cluster_{cluster_id}_NDVI'] = cluster_ndvi
            else:
                # Cluster not found
                row[f'Cluster_{cluster_id}_NDVI'] = np.nan
        
        comparison_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    fixed_columns = ['Year', 'Municipality_Avg_NDVI', 'Municipality_Vegetation_Avg_NDVI']
    
    cluster_columns = []
    for cluster_id in cluster_ids:
        cluster_columns.append(f'Cluster_{cluster_id}_NDVI')
    
    df = df[fixed_columns + cluster_columns]
    
    column_mapping = {
        'Municipality_Avg_NDVI': 'Average',
        'Municipality_Vegetation_Avg_NDVI': 'Avg > 0.2'
    }
    
    # Add cluster column mappings
    for cluster_id in cluster_ids:
        column_mapping[f'Cluster_{cluster_id}_NDVI'] = f'Cluster {cluster_id}'
    
    df = df.rename(columns=column_mapping)
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cluster_ids_str = '_'.join([str(cid) for cid in cluster_ids])
    csv_filename = f"municipality_cluster_comparison_{cluster_ids_str}.csv"
    csv_filepath = output_path / csv_filename
    
    df.to_csv(csv_filepath, index=False, float_format='%.4f')
    
    # Save table as PNG image
    png_filename = f"municipality_cluster_comparison_{cluster_ids_str}.png"
    png_filepath = output_path / png_filename
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.4)))
    ax.axis('tight')
    ax.axis('off')  
    df_display = df.copy()

    # Format table
    df_display['Year'] = df_display['Year'].astype(int).astype(str)  
    numeric_columns = df_display.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "NaN")
    
    # Create the table
    table = ax.table(cellText=df_display.values, 
                    colLabels=df_display.columns,
                    cellLoc='center', 
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor("#f0f0f0")
        table[(0, i)].set_text_props(weight='bold', color='black')
    if COLOR_CODE_NDVI:
        df_numeric = df.copy()
        colormap = plt.cm.Greens
        ndvi_columns = [col for col in df_numeric.columns if col != 'Year']
        all_ndvi_values = []
        for col in ndvi_columns:
            values = df_numeric[col].dropna()
            all_ndvi_values.extend(values.tolist())
        
        if all_ndvi_values:
            min_ndvi = min(all_ndvi_values)
            max_ndvi = max(all_ndvi_values)
            
            # Apply colors to data cells
            for i in range(1, len(df_display) + 1):
                for j in range(len(df_display.columns)):
                    col_name = df_display.columns[j]
                    
                    if col_name == 'Year':
                        table[(i, j)].set_facecolor('#f8f8f8')
                    else:
                        try:
                            value = df_numeric.iloc[i-1, j]
                            if not pd.isna(value) and min_ndvi != max_ndvi:
                                normalized_value = (value - min_ndvi) / (max_ndvi - min_ndvi)
                                color = colormap(normalized_value)
                                table[(i, j)].set_facecolor(color)
                                if value > 0.5:
                                    table[(i, j)].set_text_props(color='white')
                                else:
                                    table[(i, j)].set_text_props(color='black')
                            else:
                                table[(i, j)].set_facecolor('#f0f0f0')
                                table[(i, j)].set_text_props(color='black')
                        except:
                            table[(i, j)].set_facecolor('#f0f0f0')
                            table[(i, j)].set_text_props(color='black')
    else:
        for i in range(1, len(df_display) + 1):
            for j in range(len(df_display.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
    
    plt.title(f'{municipality_name} - Municipality vs Cluster NDVI Comparison', fontsize=20, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(png_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(csv_filepath), str(png_filepath), df


def main(json_path=JSON_PATH, cluster_ids=CLUSTER_IDS):
    """Main function to create combined cluster visualizations."""
    try:
        # Load data
        data, output_dir = load_json_data(json_path)

        # Collect cluster data first (we may need coordinates to derive a bbox if municipality lookup fails)
        clusters_data = []
        for cluster_id in cluster_ids:
            cluster, trend = find_cluster_in_data(data, cluster_id)
            clusters_data.append((cluster_id, cluster, trend))

        # Derive municipality name from metadata, but fall back to other metadata fields when it's missing or 'None'
        municipality = data.get('metadata', {}).get('municipality')
        if not municipality or str(municipality).strip().lower() in ('none', 'null', ''):
            # Try other metadata fields
            municipality = extract_municipality_name_from_cluster_data(data)
            logger.info(f"Derived municipality name from cluster data: '{municipality}'")

        # Try to load municipality boundaries; if that fails, fall back to a bbox built from cluster coordinates
        try:
            municipality_gdf, municipality_bbox = load_municipality_boundaries(BOUNDARIES_PATH, municipality)
        except Exception as e:
            logger.warning(f"Could not load municipality boundaries for '{municipality}': {e}. Falling back to cluster-derived bbox.")
            # Aggregate coordinates from selected clusters
            all_lats, all_lons = [], []
            for _, cluster, _ in clusters_data:
                try:
                    lats, lons = get_cluster_coordinates(cluster)
                    all_lats.extend(lats)
                    all_lons.extend(lons)
                except Exception:
                    continue

            if all_lats and all_lons:
                min_lon, max_lon = min(all_lons), max(all_lons)
                min_lat, max_lat = min(all_lats), max(all_lats)
                # Add a small padding
                lon_pad = (max_lon - min_lon) * 0.05 if max_lon != min_lon else 0.01
                lat_pad = (max_lat - min_lat) * 0.05 if max_lat != min_lat else 0.01
                municipality_bbox = [min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad]
            else:
                # Last-resort: use metadata bbox if present or a small default bbox
                meta_bbox = data.get('metadata', {}).get('bbox')
                if meta_bbox and isinstance(meta_bbox, (list, tuple)) and len(meta_bbox) == 4:
                    municipality_bbox = meta_bbox
                else:
                    municipality_bbox = [0.0, 0.0, 0.1, 0.1]

            # Create a GeoDataFrame from the bbox polygon so plotting code has an active
            # geometry column and a defined CRS (EPSG:4326).
            try:
                from shapely.geometry import box
                poly = box(municipality_bbox[0], municipality_bbox[1], municipality_bbox[2], municipality_bbox[3])
                municipality_gdf = gpd.GeoDataFrame({'geometry': [poly]}, crs='EPSG:4326')
            except Exception:
                # As a last resort, create an empty GeoDataFrame with a geometry column and CRS
                municipality_gdf = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
        
        # Create visualizations
        spatial_path = create_spatial_map(clusters_data, data, municipality_gdf, municipality_bbox, output_dir)
        temporal_path = create_temporal_plot(clusters_data, data, output_dir)
        
        # Create comparison CSV if requested
        csv_path = None
        png_path = None
        comparison_df = None
        if CREATE_COMPARISON_CSV:
            try:
                municipality_stats = load_municipality_stats(MUNICIPALITY_STATS_JSON)
                # If municipality is missing or 'None', try to auto-detect a park name
                comparison_target = municipality
                if not comparison_target or str(comparison_target).strip().lower() in ('none', 'null', ''):
                    logger.info("Municipality missing; attempting to auto-detect park name from cluster JSON...")
                    try:
                        json_text = json.dumps(data)
                        # Build candidate park names from municipality stats
                        candidates = set()
                        for muni_name, muni_data in municipality_stats.get('municipalities', {}).items():
                            for park_type in ('pein', 'xpn'):
                                parks = muni_data.get(park_type, {}) or {}
                                for park_name in parks.keys():
                                    if park_name:
                                        candidates.add(park_name)

                        detected = None
                        low_json = normalize_text(json_text)
                        for cand in candidates:
                            if normalize_text(cand) in low_json:
                                detected = cand
                                break

                        if detected:
                            comparison_target = detected
                            logger.info(f"Auto-detected park for comparison: '{comparison_target}'")
                        else:
                            logger.info("No park name detected automatically from cluster JSON; will fall back to existing lookup logic.")
                    except Exception as e:
                        logger.warning(f"Auto-detect park failed: {e}")

                result = create_comparison_table(
                    municipality_stats, data, cluster_ids, comparison_target, output_dir
                )
                if result and isinstance(result, (list, tuple)) and len(result) == 3:
                    csv_path, png_path, comparison_df = result
                else:
                    logger.warning("Comparison data not created (no matching municipality or park). Skipping comparison output.")
                
            except Exception as e:
                logger.error(f"Error creating comparison CSV: {e}")
                
        return {
            'spatial_map': spatial_path,
            'temporal_plot': temporal_path,
            'comparison_csv': csv_path,
            'comparison_png': png_path,
            'comparison_df': comparison_df,
            'clusters': clusters_data
        }
        
    except Exception as e:
        logger.error(f"Error in cluster visualization: {e}")
        raise


if __name__ == "__main__":
    main()