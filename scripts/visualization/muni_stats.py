"""
Single File NDVI Statistics Analyzer

Analyzes NDVI statistics for a single NetCDF file using multidimensional datasets.
Generates comprehensive JSON statistics including:
- Average NDVI per year
- NDVI statistics for traces with mean NDVI > 0.2
- Trace counts by NDVI categories
- Natural parks (PEIN/XPN) specific statistics if overlapping
- Standard deviations and pixel counts
"""

# ==== CONFIGURABLE PARAMETERS ====
INPUT_FILE = "data/processed/mdim_Torrelles_de_Llobregat_sen.nc"  # Specify the single file to analyze
OUTPUT_DIR = "outputs/single_file_statistics"
PEIN_SHAPEFILE = "data/boundaries/PEIN_clipped.shp"
XPN_SHAPEFILE = "data/boundaries/XPN_clipped.shp"

# NDVI Categories for classification
NDVI_CATEGORIES = [
    (-1.0, 0.0, 'Water'),
    (0.0, 0.1, '0.0-0.1'),
    (0.1, 0.2, '0.1-0.2'),
    (0.2, 0.3, '0.2-0.3'),
    (0.3, 0.4, '0.3-0.4'),
    (0.4, 0.5, '0.4-0.5'),
    (0.5, 0.6, '0.5-0.6'),
    (0.6, 0.7, '0.6-0.7'),
    (0.7, 0.8, '0.7-0.8'),
    (0.8, 0.9, '0.8-0.9'),
    (0.9, 1.0, '0.9-1.0')
]

VEGETATION_THRESHOLD = 0.2  # Minimum NDVI for vegetation analysis
# =================================

import json
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from loguru import logger
import warnings
import unicodedata
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

warnings.filterwarnings('ignore')

# Configure logger
logger.add(
    f"logs/single_file_ndvi_stats_{datetime.now().strftime('%Y-%m-%d')}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def load_natural_parks():
    """Load natural parks data (PEIN and XPN)."""
    natural_parks = {}
    
    # Load PEIN data
    pein_path = Path(PEIN_SHAPEFILE)
    if pein_path.exists():
        try:
            natural_parks['PEIN'] = gpd.read_file(pein_path)
            logger.info(f"Loaded PEIN data: {len(natural_parks['PEIN'])} features")
        except Exception as e:
            logger.warning(f"Could not load PEIN data: {e}")
            natural_parks['PEIN'] = None
    else:
        logger.warning(f"PEIN shapefile not found: {pein_path}")
        natural_parks['PEIN'] = None
    
    # Load XPN data
    xpn_path = Path(XPN_SHAPEFILE)
    if xpn_path.exists():
        try:
            natural_parks['XPN'] = gpd.read_file(xpn_path)
            logger.info(f"Loaded XPN data: {len(natural_parks['XPN'])} features")
        except Exception as e:
            logger.warning(f"Could not load XPN data: {e}")
            natural_parks['XPN'] = None
    else:
        logger.warning(f"XPN shapefile not found: {xpn_path}")
        natural_parks['XPN'] = None
    
    return natural_parks


def create_individual_park_statistics(dataset, natural_park_gdf, park_type, ndvi_data):
    """Create individual statistics for each natural park that overlaps with the dataset."""
    if natural_park_gdf is None or len(natural_park_gdf) == 0:
        return {}
    
    park_statistics = {}
    
    try:
        # Get dataset bounds - handle different coordinate naming conventions
        if 'longitude' in dataset.dims and 'latitude' in dataset.dims:
            lon_min, lon_max = float(dataset.longitude.min()), float(dataset.longitude.max())
            lat_min, lat_max = float(dataset.latitude.min()), float(dataset.latitude.max())
            height, width = len(dataset.latitude), len(dataset.longitude)
        elif 'x' in dataset.dims and 'y' in dataset.dims:
            lon_min, lon_max = float(dataset.x.min()), float(dataset.x.max())
            lat_min, lat_max = float(dataset.y.min()), float(dataset.y.max())
            height, width = len(dataset.y), len(dataset.x)
        elif 'lon' in dataset.dims and 'lat' in dataset.dims:
            lon_min, lon_max = float(dataset.lon.min()), float(dataset.lon.max())
            lat_min, lat_max = float(dataset.lat.min()), float(dataset.lat.max())
            height, width = len(dataset.lat), len(dataset.lon)
        else:
            logger.warning(f"Cannot determine spatial dimensions for {park_type} statistics")
            return {}
        
        # Create transform
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        
        # Find name column in the natural park GDF
        if park_type.upper() == 'PEIN':
            name_columns = ['NOM_PEIN']
        if park_type.upper() == 'XPN':
            name_columns = ['NOM_OFIC']
            
        name_col = next((col for col in name_columns if col in natural_park_gdf.columns), None)
        
        if not name_col:
            logger.warning(f"No name column found in {park_type} shapefile")
            return {}
        
        # Process each park individually
        for idx, row in natural_park_gdf.iterrows():
            park_name = str(row[name_col])
            
            try:
                # Create individual mask for this park
                individual_mask = geometry_mask(
                    [row.geometry],
                    transform=transform,
                    invert=True,
                    out_shape=(height, width)
                )
                
                pixel_count = np.sum(individual_mask)
                if pixel_count > 0:
                    # Calculate statistics for this specific park
                    park_stats = calculate_ndvi_statistics(ndvi_data, individual_mask, [park_name])
                    
                    # Only include park if it has meaningful statistics
                    if park_stats and ('overall' in park_stats or 
                                     ('yearly' in park_stats and len(park_stats['yearly']) > 0)):
                        park_statistics[park_name] = park_stats
                    
            except Exception as e:
                logger.warning(f"Error processing {park_type} park {park_name}: {e}")
        
        return park_statistics
        
    except Exception as e:
        logger.warning(f"Could not create {park_type} individual statistics: {e}")
        return {}


def classify_ndvi_values(ndvi_array, categories):
    """Classify NDVI values into categories and return counts per category."""
    ndvi_flat = ndvi_array.flatten()
    valid_ndvi = ndvi_flat[~np.isnan(ndvi_flat)]
    
    category_counts = {}
    for min_val, max_val, name in categories:
        mask = (valid_ndvi >= min_val) & (valid_ndvi < max_val)
        category_counts[name] = int(np.sum(mask))
    
    return category_counts


def calculate_ndvi_statistics(ndvi_data, mask=None, park_names=None):
    """Calculate comprehensive NDVI statistics for given data and optional mask."""
    if mask is not None:
        # Apply mask to NDVI data
        masked_data = np.where(mask[np.newaxis, :, :], ndvi_data, np.nan)
    else:
        masked_data = ndvi_data
    
    stats = {}
    time_length = masked_data.shape[0]
    
    # Add park names if provided
    if park_names:
        stats['park_names'] = park_names
    
    # Calculate spatial area statistics
    first_year_data = masked_data[0, :, :]
    spatial_valid_pixels = first_year_data[~np.isnan(first_year_data)]
    total_spatial_pixels = int(len(spatial_valid_pixels))
    
    # Calculate per-year statistics
    yearly_stats = {}
    for t in range(time_length):
        year_data = masked_data[t, :, :]
        valid_pixels = year_data[~np.isnan(year_data)]
        
        if len(valid_pixels) > 0:
            yearly_stats[str(1984 + t)] = {
                'mean_ndvi': float(np.mean(valid_pixels)),
                'std_ndvi': float(np.std(valid_pixels)),
                'spatial_pixels': total_spatial_pixels,
                'category_counts': classify_ndvi_values(year_data, NDVI_CATEGORIES)
            }
            
            # Calculate vegetation-only statistics (NDVI > 0.2)
            vegetation_pixels = valid_pixels[valid_pixels >= VEGETATION_THRESHOLD]
            if len(vegetation_pixels) > 0:
                yearly_stats[str(1984 + t)]['vegetation_mean_ndvi'] = float(np.mean(vegetation_pixels))
                yearly_stats[str(1984 + t)]['vegetation_std_ndvi'] = float(np.std(vegetation_pixels))
                yearly_stats[str(1984 + t)]['vegetation_pixels'] = int(len(vegetation_pixels))
            else:
                yearly_stats[str(1984 + t)]['vegetation_mean_ndvi'] = None
                yearly_stats[str(1984 + t)]['vegetation_std_ndvi'] = None
                yearly_stats[str(1984 + t)]['vegetation_pixels'] = 0
    
    # Calculate overall statistics
    all_valid_data = masked_data[~np.isnan(masked_data)]
    if len(all_valid_data) > 0:
        stats['overall'] = {
            'mean_ndvi': float(np.mean(all_valid_data)),
            'std_ndvi': float(np.std(all_valid_data)),
            'spatial_pixels': total_spatial_pixels,
            'total_data_points': int(len(all_valid_data))
        }
        
        # Overall vegetation statistics
        vegetation_data = all_valid_data[all_valid_data >= VEGETATION_THRESHOLD]
        if len(vegetation_data) > 0:
            stats['overall']['vegetation_mean_ndvi'] = float(np.mean(vegetation_data))
            stats['overall']['vegetation_std_ndvi'] = float(np.std(vegetation_data))
            stats['overall']['vegetation_data_points'] = int(len(vegetation_data))
        else:
            stats['overall']['vegetation_mean_ndvi'] = None
            stats['overall']['vegetation_std_ndvi'] = None
            stats['overall']['vegetation_data_points'] = 0
    
    stats['yearly'] = yearly_stats
    return stats


def analyze_single_file_ndvi(file_path, natural_parks):
    """Analyze NDVI statistics for a single NetCDF file."""
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        logger.info(f"Analyzing NDVI data from: {file_path}")
        
        # Load dataset
        dataset = xr.open_dataset(file_path)
        
        # Extract NDVI data
        if 'ndvi' in dataset.data_vars:
            ndvi_data = dataset.ndvi.values
        else:
            logger.error(f"No NDVI data found in {file_path}")
            return None
        
        # Get time coordinates
        if 'time' in dataset.dims:
            time_coords = dataset.time.values
            years = [1984 + i for i in range(len(time_coords))]
        else:
            logger.warning(f"No time dimension found in {file_path}")
            years = [1984]
        
        # Extract file name without extension for identification
        file_name = file_path.stem
        area_name = file_name.replace('mdim_', '') if file_name.startswith('mdim_') else file_name
        
        file_stats = {
            'area_name': area_name,
            'file_path': str(file_path),
            'file_name': file_name,
            'years': years,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Calculate general file statistics
        logger.info("Calculating general NDVI statistics...")
        file_stats['general'] = calculate_ndvi_statistics(ndvi_data)
        
        # Calculate individual natural parks statistics
        logger.info("Calculating natural parks statistics...")
        for park_type, park_gdf in natural_parks.items():
            if park_gdf is not None:
                logger.info(f"Processing {park_type} parks...")
                individual_park_stats = create_individual_park_statistics(dataset, park_gdf, park_type, ndvi_data)
                if individual_park_stats:
                    file_stats[park_type.lower()] = individual_park_stats
                    logger.info(f"Found {len(individual_park_stats)} {park_type} parks overlapping with the dataset")
                else:
                    logger.info(f"No {park_type} parks found overlapping with the dataset")
        
        dataset.close()
        logger.success(f"Successfully analyzed {file_path}")
        return file_stats
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return None


def create_single_file_ndvi_statistics(input_file=None):
    """Main function to create comprehensive NDVI statistics for a single file."""
    
    # Use provided file or default from configuration
    target_file = input_file if input_file else INPUT_FILE
    
    logger.info(f"Starting single file NDVI statistics analysis for: {target_file}")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load natural parks data
    natural_parks = load_natural_parks()
    
    # Analyze the single file
    file_stats = analyze_single_file_ndvi(target_file, natural_parks)
    
    if not file_stats:
        logger.error("Failed to analyze the file")
        return None
    
    # Initialize results structure
    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'input_file': str(target_file),
            'vegetation_threshold': VEGETATION_THRESHOLD,
            'ndvi_categories': [{'range': f"{cat[0]} to {cat[1]}", 'name': cat[2]} for cat in NDVI_CATEGORIES],
            'analysis_type': 'single_file'
        },
        'file_analysis': file_stats
    }
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = Path(target_file).stem
    output_file = output_dir / f"single_file_ndvi_statistics_{file_name}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Analysis complete. Results saved to: {output_file}")
    
    # Print summary
    general_stats = file_stats.get('general', {})
    overall_stats = general_stats.get('overall', {})
    
    if overall_stats:
        logger.info(f"Overall NDVI Statistics:")
        logger.info(f"  - Mean NDVI: {overall_stats.get('mean_ndvi', 'N/A'):.4f}")
        logger.info(f"  - Std NDVI: {overall_stats.get('std_ndvi', 'N/A'):.4f}")
        logger.info(f"  - Spatial pixels: {overall_stats.get('spatial_pixels', 'N/A')}")
        logger.info(f"  - Vegetation mean NDVI: {overall_stats.get('vegetation_mean_ndvi', 'N/A')}")
        logger.info(f"  - Vegetation data points: {overall_stats.get('vegetation_data_points', 'N/A')}")
    
    # Report natural parks found
    for park_type in ['pein', 'xpn']:
        if park_type in file_stats:
            park_count = len(file_stats[park_type])
            logger.info(f"Found {park_count} {park_type.upper()} parks in the analysis area")
    
    return str(output_file)


if __name__ == "__main__":
    import sys
    
    # Allow command line argument for input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        logger.info(f"Using command line input file: {input_file}")
        create_single_file_ndvi_statistics(input_file)
    else:
        create_single_file_ndvi_statistics()