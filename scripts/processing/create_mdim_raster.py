"""
Create a multidimensional raster from Landsat time series for ALL AMB municipalities with NDVI values.
This script creates individual municipality datasets while also allowing overall analysis.
"""

# ==== CONFIGURABLE PARAMETERS ====
CONFIG_PATH = "config/config.yaml"
LOG_PATH = "logs/landsat_processing_{time:YYYY-MM-DD}.log"

NDVI_THRESHOLDS = [(-1.0, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
NDVI_CLASS_NAMES = ['Water', 'Bare', 'Sparse vegetation', 'Moderate vegetation', 'Dense vegetation', 'Very dense vegetation']
BAND_NAMES = ['BLUE', 'GREEN', 'RED', 'NIR']
MUNICIPALITY_NAME_COLS = ['name']
OUTPUT_DTYPE = 'float32'  # More memory-efficient dtype (changable if needed to 'float64')
OUTPUT_FILE_NAME = "mdim_Sant_Feliu_de_Llobregat.nc"
START_YEAR = None   # Set to None to use config file value
END_YEAR = None     # Set to None to use config file value
YEAR_STEP = None       # Set to None to use config file value
# Optionally filter to a single municipality (set to None for all, or e.g. "L'Eixample")
FILTER_MUNICIPALITY = "Sant Feliu de Llobregat"  # e.g. "L'Eixample" or None
# ================================

import warnings
import numpy as np
import xarray as xr
import rasterio
import geopandas as gpd
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from rasterio.mask import mask
from loguru import logger
from scipy.ndimage import zoom

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# Configure loguru
logger.add(
    LOG_PATH,
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def load_config(config_path=CONFIG_PATH):
    """Load configuration from YAML file with proper encoding."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_municipalities(boundaries_path):
    """Load all municipalities from shapefile."""
    logger.info(f"Loading municipalities from: {boundaries_path}")
    for encoding in ['utf-8', 'cp1252']:
        try:
            gdf = gpd.read_file(boundaries_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not read shapefile with any encoding")
    name_col = next((col for col in MUNICIPALITY_NAME_COLS if col in gdf.columns), None)
    if not name_col:
        name_col = next((col for col in gdf.columns if gdf[col].dtype == 'object' and col != 'geometry'), None)
    if not name_col:
        raise ValueError(f"Could not find name column in {list(gdf.columns)}")
    gdf['municipality_name'] = gdf[name_col]
    return gdf


def create_municipality_masks(src, boundaries_gdf, out_transform, height, width):
    """Create municipality ID raster with individual masks."""
    municipality_ids = np.zeros((height, width), dtype=np.int16)
    municipality_names = []
    
    for idx, (_, row) in enumerate(boundaries_gdf.iterrows(), 1):
        municipality_names.append(row['municipality_name'])
        try:
            muni_image, _ = mask(src, [row['geometry']], crop=False, 
                               all_touched=True, filled=False)
            
            # Extract clipped region
            muni_clipped = muni_image[0][
                int((out_transform[5] - src.transform[5]) / src.transform[4]):
                int((out_transform[5] - src.transform[5]) / src.transform[4]) + height,
                int((out_transform[2] - src.transform[2]) / src.transform[0]):
                int((out_transform[2] - src.transform[2]) / src.transform[0]) + width
            ]
            
            # Resize if needed
            scale_y = municipality_ids.shape[0] / muni_clipped.shape[0]
            scale_x = municipality_ids.shape[1] / muni_clipped.shape[1]
            muni_clipped = zoom(muni_clipped, (scale_y, scale_x), order=0)    
            municipality_ids[~muni_clipped.mask] = idx
            
        except Exception as e:
            logger.warning(f"Could not process municipality {row['municipality_name']}: {e}")
            continue
    
    return municipality_ids, municipality_names


def load_and_clip_landsat_file(file_path, year, boundaries_gdf):
    """Load a Landsat file and clip it to boundaries."""
    with rasterio.open(file_path) as src:
        # Clip to boundaries
        out_image, out_transform = mask(src, boundaries_gdf.geometry, crop=True)
        
        # Create coordinates - use float32 for consistency
        height, width = out_image.shape[1], out_image.shape[2]
        x_coords = np.linspace(out_transform[2], out_transform[2] + width*out_transform[0], width, dtype=np.float32)
        y_coords = np.linspace(out_transform[5], out_transform[5] + height*out_transform[4], height, dtype=np.float32)
        
        # Handle NoData values properly - convert -9999 to NaN
        landsat_data = out_image.astype(np.float32)
        landsat_data[landsat_data == -9999] = np.nan
        
        # Create dataset with landsat structure
        ds = xr.Dataset(
            {
                'landsat': (['band', 'y', 'x'], landsat_data)
            },
            coords={
                'x': x_coords,
                'y': y_coords,
                'time': year,  # Use simple integer year instead of datetime
                'band': BAND_NAMES  # ['BLUE', 'GREEN', 'RED', 'NIR']
            }
        )
        
        # Add municipality masks (only if processing multiple municipalities)
        if FILTER_MUNICIPALITY is None:
            municipality_ids, municipality_names = create_municipality_masks(
                src, boundaries_gdf, out_transform, height, width
            )
            
            ds['municipality_id'] = (['y', 'x'], municipality_ids)
            ds.attrs.update({
                'municipality_names': municipality_names,
                'n_municipalities': len(municipality_names)
            })
        
        return ds


def calculate_ndvi(combined_ds):
    """Calculate NDVI with proper NoData handling."""
    logger.info("Calculating NDVI...")
    
    # Extract RED and NIR from the landsat variable using band selection
    red = combined_ds['landsat'].sel(band='RED')
    nir = combined_ds['landsat'].sel(band='NIR')
    
    # Calculate NDVI only where both RED and NIR are valid
    denominator = (nir + red)
    # Avoid division by zero and handle NaN properly
    ndvi = xr.where(
        np.abs(denominator) > 0.001,
        (nir - red) / denominator,
        np.nan
    )
    
    # Clip to valid NDVI range
    ndvi = ndvi.clip(-1.0, 1.0)
    
    # Log statistics
    valid_ndvi = ndvi.values[~np.isnan(ndvi.values)]
    if len(valid_ndvi) > 0:
        logger.info(f"NDVI: Min={valid_ndvi.min():.3f}, Max={valid_ndvi.max():.3f}, Mean={valid_ndvi.mean():.3f}")
        logger.info(f"Valid NDVI pixels: {len(valid_ndvi):,}/{ndvi.size:,} ({100*len(valid_ndvi)/ndvi.size:.1f}%)")
    else:
        logger.warning("No valid NDVI values calculated!")
    
    return ndvi.astype('float32')


def classify_ndvi(ndvi):
    """Classify NDVI into 6 categories."""
    logger.info("Classifying NDVI values into 6 categories...")

    ndvi_class = xr.zeros_like(ndvi, dtype='int8')

    for i, (min_val, max_val) in enumerate(NDVI_THRESHOLDS):
        if i == len(NDVI_THRESHOLDS) - 1:  # Last class includes upper bound
            mask = (ndvi >= min_val) & (ndvi <= max_val)
        else:
            mask = (ndvi >= min_val) & (ndvi < max_val)
        ndvi_class = xr.where(mask, i, ndvi_class)

    # Handle NoData
    ndvi_class = ndvi_class.where(~ndvi.isnull(), -1)
    
    # Log classification results
    logger.info("NDVI Classification Results:")
    for i, name in enumerate(NDVI_CLASS_NAMES):
        count = int((ndvi_class == i).sum())
        logger.info(f"  Class {i}: {count} pixels - {name}")
    logger.info(f"  NoData (-1): {int((ndvi_class == -1).sum())} pixels")
    return ndvi_class.astype('int8')


def create_multidimensional_raster_all_municipalities(config_path=CONFIG_PATH):
    """Create a multidimensional raster for ALL AMB municipalities."""
    config = load_config(config_path)

    # Setup paths
    raw_data_path = Path(config['paths']['raw_data'])
    processed_data_path = Path(config['paths']['processed_data'])
    boundaries_path = Path(config['paths']['boundaries'])

    # Use script values if set, otherwise config values
    start_year = START_YEAR if START_YEAR is not None else config.get('analysis', {}).get('start_year')
    end_year = END_YEAR if END_YEAR is not None else config.get('analysis', {}).get('end_year')
    year_step = YEAR_STEP if YEAR_STEP is not None else config.get('analysis', {}).get('year_step')
    file_pattern = config['data']['file_pattern']

    logger.info(f"Creating multidimensional raster for ALL AMB municipalities ({start_year}-{end_year})")

    # Load municipalities
    boundaries_gdf = load_municipalities(boundaries_path)
    if FILTER_MUNICIPALITY is not None:
        boundaries_gdf = boundaries_gdf[boundaries_gdf['municipality_name'] == FILTER_MUNICIPALITY]
        logger.info(f"Filtered to municipality: {FILTER_MUNICIPALITY} ({len(boundaries_gdf)} found)")
    else:
        logger.info(f"Processing {len(boundaries_gdf)} municipalities")
    
    # Find available files
    available_files = []
    available_years = []
    
    for year in range(start_year, end_year + 1, year_step):
        file_path = raw_data_path / file_pattern.format(year=year)
        if file_path.exists():
            available_files.append(file_path)
            available_years.append(year)
    
    if not available_files:
        raise FileNotFoundError(f"No files found in {raw_data_path} with pattern {file_pattern}")

    logger.info(f"Found {len(available_files)} files for years: {min(available_years)}-{max(available_years)}")
    
    # Process files
    datasets = []
    logger.info("Loading and clipping files...")
    for file_path, year in tqdm(zip(available_files, available_years), total=len(available_files), desc="Processing files"):
        try:
            ds = load_and_clip_landsat_file(file_path, year, boundaries_gdf)
            # Convert landsat data to desired output type
            ds['landsat'] = ds['landsat'].astype(OUTPUT_DTYPE)
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    if not datasets:
        raise RuntimeError("No datasets were successfully processed")
    
    # Combine datasets
    logger.info("Combining datasets...")
    combined_ds = xr.concat(datasets, dim='time').sortby('time')
    
    # Ensure time coordinate is int64 (simple years) to match target structure
    combined_ds = combined_ds.assign_coords(time=np.array(available_years, dtype=np.int64))
    
    # Calculate NDVI 
    ndvi = calculate_ndvi(combined_ds)
    combined_ds['ndvi'] = (['time', 'y', 'x'], ndvi.data)
    
    # Only add classification if processing multiple municipalities
    if FILTER_MUNICIPALITY is None:
        ndvi_class = classify_ndvi(combined_ds['ndvi'])
        combined_ds['ndvi_class'] = (['time', 'y', 'x'], ndvi_class.data)
    
    # Add attributes
    combined_ds['landsat'].attrs = {
        'long_name': 'Landsat Collection 2 Level 2 Surface Reflectance',
        'units': 'dimensionless',
        'description': 'Surface reflectance values for BLUE, GREEN, RED, NIR bands',
        'bands': ', '.join(BAND_NAMES),
        'source': 'Google Earth Engine Landsat Collection 2 Level 2'
    }
    combined_ds['ndvi'].attrs = {
        'long_name': 'Normalized Difference Vegetation Index',
        'units': 'dimensionless',
        'valid_range': np.array([-1.0, 1.0], dtype=np.float32),
        'description': 'NDVI calculated from NIR and RED bands: (NIR - RED) / (NIR + RED)'
    }
    
    if 'ndvi_class' in combined_ds:
        combined_ds['ndvi_class'].attrs = {
            'long_name': 'NDVI Classification Categories',
            'units': 'class',
            'valid_range': [0, 5],
            'description': 'NDVI classified into 6 vegetation categories',
            'classification_scheme': 'Class 0: -1 to 0 (Water/Bare/Built-up), Class 1: 0 to 0.1 (Very sparse vegetation), Class 2: 0.1 to 0.2 (Sparse vegetation), Class 3: 0.2 to 0.4 (Moderate vegetation), Class 4: 0.4 to 0.6 (Dense vegetation), Class 5: 0.6 to 1 (Very dense vegetation)',
            'nodata_value': -1
        }
    combined_ds.attrs.update({
        'title': f'AMB Landsat Time Series - {FILTER_MUNICIPALITY or "All Municipalities"}',
        'description': f'Landsat Collection 2 Level 2 data for {FILTER_MUNICIPALITY or f"all {len(boundaries_gdf)} AMB municipalities"} ({min(available_years)}-{max(available_years)})',
        'source': 'Google Earth Engine',
        'processing_level': 'Collection 2 Level 2',
        'spatial_resolution': '30m',
        'projection': 'EPSG:4326',
        'total_municipalities': len(boundaries_gdf),
        'municipality_names': FILTER_MUNICIPALITY or ', '.join(boundaries_gdf['municipality_name'].tolist()),
        'n_years': len(available_years),
        'bands': ', '.join(BAND_NAMES),
        'derived_variables': 'NDVI' + (', NDVI_CLASS' if 'ndvi_class' in combined_ds else ''),
        'created_date': datetime.now().isoformat(),
        'individual_analysis_supported': 'true',
        'nodata_handling': 'NaN for invalid values'
    })
    
    # Save dataset
    output_file_name = OUTPUT_FILE_NAME
    processed_data_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_data_path / output_file_name

    # Set encoding to ensure proper data types matching target structure
    encoding = {
        'landsat': {'dtype': 'float32', 'zlib': True, 'complevel': 6},
        'ndvi': {'dtype': 'float32', 'zlib': True, 'complevel': 6},
        'time': {'dtype': 'int64'},
        'x': {'dtype': 'float32'},
        'y': {'dtype': 'float32'}
    }
    
    if 'ndvi_class' in combined_ds:
        encoding['ndvi_class'] = {'dtype': 'int8', 'zlib': True, 'complevel': 6}
    if 'municipality_id' in combined_ds:
        encoding['municipality_id'] = {'dtype': 'int16', 'zlib': True, 'complevel': 6}
    
    combined_ds.to_netcdf(output_file, engine='netcdf4', encoding=encoding)
    
    # Save municipality mapping (only if processing multiple municipalities)
    if FILTER_MUNICIPALITY is None and 'municipality_id' in combined_ds:
        municipality_info = boundaries_gdf['municipality_name'].copy()
        municipality_info['municipality_id'] = range(1, len(municipality_info) + 1)
        municipality_info_file = processed_data_path / "municipality_mapping.csv"
        municipality_info.to_csv(municipality_info_file, index=False, encoding='utf-8')
        logger.info(f"Municipality mapping saved to: {municipality_info_file}")

    logger.success("Dataset creation complete!")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Dataset dimensions: {combined_ds.dims}")
    logger.info(f"Variables: {list(combined_ds.data_vars.keys())}")
    
    # Log data coverage statistics
    landsat_valid = np.sum(~np.isnan(combined_ds['landsat'].values))
    landsat_total = combined_ds['landsat'].size
    ndvi_valid = np.sum(~np.isnan(combined_ds['ndvi'].values))
    ndvi_total = combined_ds['ndvi'].size
    
    logger.info(f"Landsat valid data: {landsat_valid:,}/{landsat_total:,} ({100*landsat_valid/landsat_total:.1f}%)")
    logger.info(f"NDVI valid data: {ndvi_valid:,}/{ndvi_total:,} ({100*ndvi_valid/ndvi_total:.1f}%)")
    
    return output_file


if __name__ == "__main__":
    create_multidimensional_raster_all_municipalities()