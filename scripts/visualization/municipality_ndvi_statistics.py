"""
Municipality NDVI Statistics Analyzer

Analyzes NDVI statistics for all AMB municipalities using multidimensional NetCDF datasets.
Generates comprehensive JSON statistics including:
- Average NDVI per municipality per year
- NDVI statistics for traces with mean NDVI > 0.2
- Trace counts by NDVI categories
- Natural parks (PEIN/XPN) specific statistics
- Standard deviations and pixel counts
"""

# ==== CONFIGURABLE PARAMETERS ====
MDIM_DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/municipality_statistics"
USE_INDIVIDUAL_FILES = True  # True: use individual mdim_Municipality.nc files, False: use mdim_AMB.nc
BOUNDARIES_PATH = "data/boundaries/AMB_Municipalities.shp"
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
from tqdm import tqdm
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

warnings.filterwarnings('ignore')

# Configure logger
logger.add(
    f"logs/municipality_ndvi_stats_{datetime.now().strftime('%Y-%m-%d')}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def normalize_text(text):
    """Remove accents and normalize text for comparison."""
    normalized = unicodedata.normalize('NFD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text.lower().strip()


def load_municipalities(boundaries_path):
    """Load municipality boundaries from shapefile."""
    logger.info(f"Loading municipalities from: {boundaries_path}")
    
    for encoding in ['utf-8', 'cp1252']:
        try:
            gdf = gpd.read_file(boundaries_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        gdf = gpd.read_file(boundaries_path)
    
    # Find municipality name column
    name_columns = ['NOMMUNI', 'NOM', 'NAME', 'MUNICIPALITY', 'nom', 'name']
    name_col = next((col for col in name_columns if col in gdf.columns), None)
    
    if not name_col:
        name_col = gdf.columns[0]
        logger.warning(f"No standard name column found, using: {name_col}")
    
    gdf['municipality_name'] = gdf[name_col]
    logger.info(f"Loaded {len(gdf)} municipalities")
    return gdf


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


def find_municipality_file(municipality_name, data_dir):
    """Find the NetCDF file for a specific municipality."""
    data_path = Path(data_dir)
    
    # Normalize municipality name for file matching
    normalized_name = normalize_text(municipality_name).replace(' ', '_').replace('-', '_')
    
    # Try different file name patterns
    patterns = [
        f"mdim_{normalized_name}.nc",
        f"mdim_{municipality_name.replace(' ', '_')}.nc",
        f"mdim_{municipality_name.replace(' ', '')}.nc",
        f"mdim_{municipality_name.replace(' ', '_').replace('-', '_')}.nc",
        f"mdim_{municipality_name.replace(' ', '_').replace(chr(39), '').replace('-', '_')}.nc",
        f"mdim_{municipality_name.replace(' ', '_').replace(' - ', '_')}.nc"
    ]
    
    for pattern in patterns:
        file_path = data_path / pattern
        if file_path.exists():
            return str(file_path)
    
    # If not found, try fuzzy matching with all files
    for file_path in data_path.glob("mdim_*.nc"):
        file_name = file_path.stem.replace('mdim_', '')
        file_normalized = normalize_text(file_name.replace('_', ' '))
        municipality_normalized = normalize_text(municipality_name)
        
        # Try exact match
        if file_normalized == municipality_normalized:
            return str(file_path)
        
        # Try partial match (both directions)
        if municipality_normalized in file_normalized or file_normalized in municipality_normalized:
            return str(file_path)
        
        # Special cases for common name variations
        if 'horta' in municipality_normalized and 'guinardo' in municipality_normalized:
            if 'horta' in file_normalized and 'guinardo' in file_normalized:
                return str(file_path)
        
        if 'sarria' in municipality_normalized and 'sant gervasi' in municipality_normalized:
            if 'sarria' in file_normalized and 'sant gervasi' in file_normalized:
                return str(file_path)
    
    return None


def create_natural_park_mask(dataset, natural_park_gdf, park_type):
    """Create a mask for natural park areas within the dataset bounds and return park names."""
    if natural_park_gdf is None or len(natural_park_gdf) == 0:
        return None, []
    
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
            # Try to find coordinate variables in data_vars
            coord_vars = list(dataset.coords.keys())
            logger.warning(f"Cannot determine spatial dimensions for {park_type} mask. Available coords: {coord_vars}")
            return None, []
        
        # Ensure natural parks are in same CRS (assuming WGS84)
        if natural_park_gdf.crs != 'EPSG:4326':
            natural_park_gdf = natural_park_gdf.to_crs('EPSG:4326')
        
        # Create transform
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        
        # Create mask
        park_mask = geometry_mask(
            natural_park_gdf.geometry,
            transform=transform,
            invert=True,
            out_shape=(height, width)
        )
        
        # Extract park names that intersect with the dataset bounds
        park_names = []
        pixel_count = np.sum(park_mask)
        if pixel_count > 0:
            # Find name column in the natural park GDF
            name_columns = ['NOM', 'TOPONIM', 'NAME', 'nom', 'name', 'Name']
            name_col = next((col for col in name_columns if col in natural_park_gdf.columns), None)
            
            if name_col:
                # Get unique park names that have overlapping pixels
                for idx, row in natural_park_gdf.iterrows():
                    # Create individual mask for this park
                    individual_mask = geometry_mask(
                        [row.geometry],
                        transform=transform,
                        invert=True,
                        out_shape=(height, width)
                    )
                    if np.sum(individual_mask) > 0:
                        park_names.append(str(row[name_col]))
                    
        return park_mask, park_names
        
    except Exception as e:
        logger.warning(f"Could not create {park_type} mask: {e}")
        return None, []


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


def analyze_municipality_ndvi(municipality_name, municipalities_gdf, natural_parks, data_dir):
    """Analyze NDVI statistics for a single municipality."""
    
    # Find municipality file
    if USE_INDIVIDUAL_FILES:
        file_path = find_municipality_file(municipality_name, data_dir)
        if not file_path:
            logger.warning(f"No NetCDF file found for {municipality_name}")
            return None
    else:
        file_path = Path(data_dir) / "mdim_AMB.nc"
        if not file_path.exists():
            logger.error(f"AMB file not found: {file_path}")
            return None
    
    try:
        # Load dataset
        dataset = xr.open_dataset(file_path)
        
        # Extract NDVI data
        if 'ndvi' in dataset.data_vars:
            ndvi_data = dataset.ndvi.values
        else:
            logger.warning(f"No NDVI data found in {file_path}")
            return None
        
        # Get time coordinates
        if 'time' in dataset.dims:
            time_coords = dataset.time.values
            years = [1984 + i for i in range(len(time_coords))]
        else:
            logger.warning(f"No time dimension found in {file_path}")
            years = [1984]
        
        municipality_stats = {
            'municipality_name': municipality_name,
            'file_path': str(file_path),
            'years': years,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Calculate general municipality statistics
        municipality_stats['general'] = calculate_ndvi_statistics(ndvi_data)
        
        # Calculate individual natural parks statistics
        for park_type, park_gdf in natural_parks.items():
            individual_park_stats = create_individual_park_statistics(dataset, park_gdf, park_type, ndvi_data)
            if individual_park_stats:
                municipality_stats[park_type.lower()] = individual_park_stats
        
        dataset.close()
        return municipality_stats
        
    except Exception as e:
        logger.error(f"Error analyzing {municipality_name}: {e}")
        return None


def create_municipality_ndvi_statistics():
    """Main function to create comprehensive NDVI statistics for all municipalities."""
    logger.info("Starting municipality NDVI statistics analysis")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load spatial data
    municipalities_gdf = load_municipalities(BOUNDARIES_PATH)
    natural_parks = load_natural_parks()
    
    # Initialize results structure
    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'data_source': MDIM_DATA_DIR,
            'use_individual_files': USE_INDIVIDUAL_FILES,
            'vegetation_threshold': VEGETATION_THRESHOLD,
            'ndvi_categories': [{'range': f"{cat[0]} to {cat[1]}", 'name': cat[2]} for cat in NDVI_CATEGORIES],
            'total_municipalities': len(municipalities_gdf)
        },
        'municipalities': {}
    }
    
    # Process each municipality
    for idx, municipality_row in tqdm(municipalities_gdf.iterrows(), 
                                     total=len(municipalities_gdf), 
                                     desc="Processing municipalities"):
        
        municipality_name = municipality_row['municipality_name']
        municipality_stats = analyze_municipality_ndvi(
            municipality_name, municipalities_gdf, natural_parks, MDIM_DATA_DIR
        )
        
        if municipality_stats:
            results['municipalities'][municipality_name] = municipality_stats
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"municipality_ndvi_statistics_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Analysis complete. Results saved to: {output_file}")
    
    # Print summary
    successful_analyses = len([m for m in results['municipalities'].values() if m is not None])
    logger.info(f"Successfully analyzed {successful_analyses} out of {len(municipalities_gdf)} municipalities")
    
    return str(output_file)


if __name__ == "__main__":
    create_municipality_ndvi_statistics()
