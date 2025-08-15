"""
Create a multidimensional raster from Landsat time series for different spatial entities.
This script supports creating MDIM rasters for:
- AMB municipalities (default)
- PEIN natural parks 
- XPN natural parks

USAGE EXAMPLES:
1. All municipalities: Set PROCESSING_MODE = "municipalities", FILTER_ENTITY = None
2. Single municipality: Set PROCESSING_MODE = "municipalities", FILTER_ENTITY = "Barcelona"
3. All PEIN parks: Set PROCESSING_MODE = "pein", FILTER_ENTITY = None
4. Single PEIN park: Set PROCESSING_MODE = "pein", FILTER_ENTITY = "Serra de Collserola"
5. All XPN parks: Set PROCESSING_MODE = "xpn", FILTER_ENTITY = None
6. Single XPN park: Set PROCESSING_MODE = "xpn", FILTER_ENTITY = "Parc del Garraf"
"""

# ==== CONFIGURABLE PARAMETERS ====
# Data paths
RAW_DATA_PATH = "data/raw/GEE_raw"
PROCESSED_DATA_PATH = "data/processed"
BOUNDARIES_SHAPEFILE = "data/boundaries/AMB_Municipalities.shp"  # Main boundaries shapefile
LOG_PATH = "logs/landsat_processing_{time:YYYY-MM-DD}.log"

# Time range configuration
START_YEAR = 1984
END_YEAR = 2025
YEAR_STEP = 1
FILE_PATTERN = "{year}.tif"  # Pattern for Landsat files

# Processing options
PROCESSING_MODE = "pein"  # Options: "municipalities", "pein", "xpn"
FILTER_ENTITY = "CLR"  # Filter to specific entity (municipality/park name), set to None for all
OUTPUT_FILE_NAME = "mdim_clr.nc"  # Will be auto-modified based on processing mode
OUTPUT_DTYPE = 'float32'  # Data type for output arrays

# NDVI classification
NDVI_THRESHOLDS = [(-1.0, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
NDVI_CLASS_NAMES = ['Water', 'Bare', 'Sparse vegetation', 'Moderate vegetation', 'Dense vegetation', 'Very dense vegetation']

# Landsat band configuration
BAND_NAMES = ['BLUE', 'GREEN', 'RED', 'NIR']
NODATA_VALUE = -9999

# Shapefile-specific configurations
SHAPEFILE_CONFIGS = {
    "municipalities": {
        "shapefile": "data/boundaries/AMB_Municipalities.shp",
        "name_columns": ['name', 'NAME', 'nom', 'NOM'],
        "id_column": None,  # Will use auto-generated IDs
        "output_prefix": "mdim_",
        "description": "AMB Municipality"
    },
    "pein": {
        "shapefile": "data/boundaries/PEIN_clipped.shp",
        "name_columns": ['NOM', 'nom', 'name', 'NAME', 'CODI_PEIN'],
        "id_column": 'CODI_PEIN',
        "output_prefix": "mdim_pein_",
        "description": "PEIN Natural Park"
    },
    "xpn": {
        "shapefile": "data/boundaries/XPN_clipped.shp", 
        "name_columns": ['NOM', 'nom', 'name', 'NAME', 'ACRONIM'],
        "id_column": 'ACRONIM',
        "output_prefix": "mdim_xpn_",
        "description": "XPN Natural Park"
    }
}

# Additional features to include
INCLUDE_NATURAL_PARKS = True  # Whether to include natural park overlays (only for municipality mode)
PEIN_SHAPEFILE = "data/boundaries/PEIN_clipped.shp"  # PEIN natural parks overlay
XPN_SHAPEFILE = "data/boundaries/XPN_clipped.shp"    # XPN natural parks overlay
# ================================

import warnings
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
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


def load_config(config_path=None):
    """Load configuration from parameters (no longer uses YAML file)."""
    return {
        'paths': {
            'raw_data': RAW_DATA_PATH,
            'processed_data': PROCESSED_DATA_PATH,
            'boundaries': BOUNDARIES_SHAPEFILE
        },
        'analysis': {
            'start_year': START_YEAR,
            'end_year': END_YEAR,
            'year_step': YEAR_STEP
        },
        'data': {
            'file_pattern': FILE_PATTERN,
            'bands': BAND_NAMES,
            'nodata_value': NODATA_VALUE
        }
    }


def load_boundaries(shapefile_path, processing_mode):
    """Load boundaries from shapefile based on processing mode."""
    logger.info(f"Loading {processing_mode} boundaries from: {shapefile_path}")
    
    # Get configuration for the processing mode
    config = SHAPEFILE_CONFIGS.get(processing_mode)
    if not config:
        raise ValueError(f"Unknown processing mode: {processing_mode}. Available: {list(SHAPEFILE_CONFIGS.keys())}")
    
    # Try different encodings
    for encoding in ['utf-8', 'cp1252', 'latin1']:
        try:
            gdf = gpd.read_file(shapefile_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not read shapefile with any encoding")
    
    # Find the name column
    name_col = None
    for col in config['name_columns']:
        if col in gdf.columns:
            name_col = col
            break
    
    if not name_col:
        # Fallback: find any object column that's not geometry
        name_col = next((col for col in gdf.columns if gdf[col].dtype == 'object' and col != 'geometry'), None)
    
    if not name_col:
        raise ValueError(f"Could not find name column in {list(gdf.columns)}. Expected one of: {config['name_columns']}")
    
    # Set standardized name column
    gdf['entity_name'] = gdf[name_col]
    
    # Add ID column if specified
    if config['id_column'] and config['id_column'] in gdf.columns:
        gdf['entity_id'] = gdf[config['id_column']]
    else:
        gdf['entity_id'] = range(1, len(gdf) + 1)
    
    logger.info(f"Loaded {len(gdf)} {processing_mode} entities using name column: {name_col}")
    return gdf


def load_natural_parks():
    """Load natural parks data (PEIN and XPN) - only used for municipality mode overlays."""
    natural_parks = {}
    
    if not INCLUDE_NATURAL_PARKS or PROCESSING_MODE != "municipalities":
        return natural_parks
    
    # Load PEIN data
    pein_path = Path(PEIN_SHAPEFILE)
    if pein_path.exists():
        try:
            pein_gdf = gpd.read_file(pein_path)
            natural_parks['pein'] = pein_gdf
            logger.info(f"Loaded PEIN overlay data: {len(pein_gdf)} polygons")
        except Exception as e:
            logger.warning(f"Could not load PEIN data from {pein_path}: {e}")
    else:
        logger.warning(f"PEIN shapefile not found at {pein_path}")
    
    # Load XPN data
    xpn_path = Path(XPN_SHAPEFILE)
    if xpn_path.exists():
        try:
            xpn_gdf = gpd.read_file(xpn_path)
            natural_parks['xpn'] = xpn_gdf
            logger.info(f"Loaded XPN overlay data: {len(xpn_gdf)} polygons")
        except Exception as e:
            logger.warning(f"Could not load XPN data from {xpn_path}: {e}")
    else:
        logger.warning(f"XPN shapefile not found at {xpn_path}")
    
    return natural_parks


def create_natural_park_masks(src, natural_parks, out_transform, height, width):
    """Create natural park rasters (PEIN and XPN) with integer codes."""
    masks = {}
    
    if not INCLUDE_NATURAL_PARKS or not natural_parks:
        return masks
    
    # Process PEIN
    if 'pein' in natural_parks:
        logger.info("Creating PEIN mask...")
        pein_mask = np.zeros((height, width), dtype=np.int16)  # Integer array for PEIN codes
        
        for idx, (_, row) in enumerate(natural_parks['pein'].iterrows(), 1):
            try:
                park_image, _ = mask(src, [row['geometry']], crop=False, 
                                   all_touched=True, filled=False)
                
                # Extract clipped region
                park_clipped = park_image[0][
                    int((out_transform[5] - src.transform[5]) / src.transform[4]):
                    int((out_transform[5] - src.transform[5]) / src.transform[4]) + height,
                    int((out_transform[2] - src.transform[2]) / src.transform[0]):
                    int((out_transform[2] - src.transform[2]) / src.transform[0]) + width
                ]
                
                # Resize if needed
                if park_clipped.shape != (height, width):
                    scale_y = height / park_clipped.shape[0]
                    scale_x = width / park_clipped.shape[1]
                    park_clipped = zoom(park_clipped, (scale_y, scale_x), order=0)
                
                # Set integer code where park intersects
                pein_mask[~park_clipped.mask] = idx
                    
            except Exception as e:
                logger.warning(f"Could not process PEIN polygon {row.get('CODI_PEIN', 'Unknown')}: {e}")
                continue
        
        masks['pein'] = pein_mask
    
    # Process XPN
    if 'xpn' in natural_parks:
        logger.info("Creating XPN mask...")
        xpn_mask = np.zeros((height, width), dtype=np.int16)  # Integer array for XPN codes
        
        for idx, (_, row) in enumerate(natural_parks['xpn'].iterrows(), 1):
            try:
                park_image, _ = mask(src, [row['geometry']], crop=False, 
                                   all_touched=True, filled=False)
                
                # Extract clipped region
                park_clipped = park_image[0][
                    int((out_transform[5] - src.transform[5]) / src.transform[4]):
                    int((out_transform[5] - src.transform[5]) / src.transform[4]) + height,
                    int((out_transform[2] - src.transform[2]) / src.transform[0]):
                    int((out_transform[2] - src.transform[2]) / src.transform[0]) + width
                ]
                
                # Resize if needed
                if park_clipped.shape != (height, width):
                    scale_y = height / park_clipped.shape[0]
                    scale_x = width / park_clipped.shape[1]
                    park_clipped = zoom(park_clipped, (scale_y, scale_x), order=0)
                
                # Set integer code where park intersects
                xpn_mask[~park_clipped.mask] = idx
                    
            except Exception as e:
                logger.warning(f"Could not process XPN polygon {row.get('ACRONIM', 'Unknown')}: {e}")
                continue
        
        masks['xpn'] = xpn_mask
    
    return masks


def create_entity_masks(src, boundaries_gdf, out_transform, height, width, processing_mode):
    """Create entity ID raster with individual masks (works for municipalities, PEIN, or XPN)."""
    entity_ids = np.zeros((height, width), dtype=np.int16)
    entity_names = []
    entity_original_ids = []
    
    for idx, (_, row) in enumerate(boundaries_gdf.iterrows(), 1):
        entity_names.append(row['entity_name'])
        entity_original_ids.append(row['entity_id'])
        try:
            entity_image, _ = mask(src, [row['geometry']], crop=False, 
                               all_touched=True, filled=False)
            
            # Extract clipped region
            entity_clipped = entity_image[0][
                int((out_transform[5] - src.transform[5]) / src.transform[4]):
                int((out_transform[5] - src.transform[5]) / src.transform[4]) + height,
                int((out_transform[2] - src.transform[2]) / src.transform[0]):
                int((out_transform[2] - src.transform[2]) / src.transform[0]) + width
            ]
            
            # Resize if needed
            scale_y = entity_ids.shape[0] / entity_clipped.shape[0]
            scale_x = entity_ids.shape[1] / entity_clipped.shape[1]
            entity_clipped = zoom(entity_clipped, (scale_y, scale_x), order=0)    
            entity_ids[~entity_clipped.mask] = idx
            
        except Exception as e:
            logger.warning(f"Could not process {processing_mode} entity {row['entity_name']}: {e}")
            continue
    
    return entity_ids, entity_names, entity_original_ids


def load_and_clip_landsat_file(file_path, year, boundaries_gdf, natural_park_masks=None, entity_masks=None, processing_mode="municipalities"):
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
        landsat_data[landsat_data == NODATA_VALUE] = np.nan
        
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
        
        # Add pre-computed entity masks (for all processing modes)
        if entity_masks is not None:
            entity_ids, entity_names, entity_original_ids = entity_masks
            ds['entity_id'] = (['y', 'x'], entity_ids)
            ds.attrs.update({
                'entity_names': entity_names,
                'entity_original_ids': entity_original_ids,
                'n_entities': len(entity_names),
                'processing_mode': processing_mode
            })
        
        # Add pre-computed natural park masks as grouped variable (only for municipality mode)
        if natural_park_masks and processing_mode == "municipalities":
            # Create a list to store natural park data and park names
            natural_data = []
            park_names = []
            
            if 'pein' in natural_park_masks:
                natural_data.append(natural_park_masks['pein'])
                park_names.append('pein')
            
            if 'xpn' in natural_park_masks:
                natural_data.append(natural_park_masks['xpn'])
                park_names.append('xpn')
            
            if natural_data:
                # Stack the natural park data into a multi-dimensional array
                natural_array = np.stack(natural_data, axis=0)
                ds['natural'] = (['park', 'y', 'x'], natural_array)
                
                # Add coordinate for park dimension
                ds = ds.assign_coords(park=park_names)
                
                # Add attributes
                ds['natural'].attrs = {
                    'long_name': 'Natural Parks Data',
                    'description': 'Natural park identifiers using integer codes for PEIN and XPN',
                    'source': 'PEIN_clipped.shp, XPN_clipped.shp',
                    'parks': ', '.join(park_names),
                    'nodata_value': 0,
                    'units': 'code'
                }
        
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


def create_multidimensional_raster(processing_mode=None, filter_entity=None):
    """Create a multidimensional raster for specified processing mode (municipalities, pein, or xpn)."""
    # Use global parameters if not specified
    mode = processing_mode or PROCESSING_MODE
    entity_filter = filter_entity or FILTER_ENTITY
    
    # Validate processing mode
    if mode not in SHAPEFILE_CONFIGS:
        raise ValueError(f"Invalid processing mode: {mode}. Available: {list(SHAPEFILE_CONFIGS.keys())}")
    
    config = load_config()
    shape_config = SHAPEFILE_CONFIGS[mode]
    
    # Setup paths
    raw_data_path = Path(config['paths']['raw_data'])
    processed_data_path = Path(config['paths']['processed_data'])
    
    # Use the shapefile for the selected processing mode
    boundaries_path = Path(shape_config['shapefile'])
    
    logger.info(f"Creating multidimensional raster for {mode.upper()} ({START_YEAR}-{END_YEAR})")

    # Load boundaries
    boundaries_gdf = load_boundaries(boundaries_path, mode)
    
    # Filter to specific entity if requested
    if entity_filter is not None:
        # capture possible values from the unfiltered boundaries
        all_possible_values = sorted(set(str(x) for x in boundaries_gdf['entity_name'].values))
        original_count = len(boundaries_gdf)
        filtered = boundaries_gdf[boundaries_gdf['entity_name'] == entity_filter]
        logger.info(f"Filtered to {mode}: {entity_filter} ({len(filtered)} found out of {original_count})")
        if len(filtered) == 0:
            # log and print possible values from the original list, then exit cleanly
            values_str = "\n".join([f"  - {val}" for val in all_possible_values])
            logger.error(f"No {mode} found with name: {entity_filter}. Possible values:\n{values_str}")
            print(f"\nPossible values for {mode}:\n{values_str}")
            logger.info("Exiting without error because filter did not match any entity.")
            return None
        # use filtered dataframe going forward
        boundaries_gdf = filtered
    else:
        logger.info(f"Processing all {len(boundaries_gdf)} {mode} entities")
    
    # Load natural parks data (only for municipality mode as overlays)
    natural_parks = load_natural_parks()
    if natural_parks and mode == "municipalities":
        park_info = []
        if 'pein' in natural_parks:
            park_info.append(f"PEIN: {len(natural_parks['pein'])} polygons")
        if 'xpn' in natural_parks:
            park_info.append(f"XPN: {len(natural_parks['xpn'])} polygons")
        logger.info(f"Natural parks overlays loaded: {', '.join(park_info)}")
    else:
        logger.info(f"No natural parks overlays (mode: {mode})")
    
    # Find available files
    available_files = []
    available_years = []
    
    for year in range(START_YEAR, END_YEAR + 1, YEAR_STEP):
        file_path = raw_data_path / FILE_PATTERN.format(year=year)
        if file_path.exists():
            available_files.append(file_path)
            available_years.append(year)
    
    if not available_files:
        raise FileNotFoundError(f"No files found in {raw_data_path} with pattern {FILE_PATTERN}")

    logger.info(f"Found {len(available_files)} files for years: {min(available_years)}-{max(available_years)}")
    
    # Create masks once before processing all files (optimization)
    natural_park_masks = None
    entity_masks = None
    
    # Use the first file to get the spatial reference for creating masks
    with rasterio.open(available_files[0]) as first_src:
        # Get the clipped transform and dimensions
        _, out_transform = mask(first_src, boundaries_gdf.geometry, crop=True)
        clipped_bounds = mask(first_src, boundaries_gdf.geometry, crop=True)[0]
        height, width = clipped_bounds.shape[1], clipped_bounds.shape[2]
        
        # Create entity masks (works for municipalities, PEIN, or XPN)
        if entity_filter is None:  # Only create masks when processing multiple entities
            logger.info(f"Creating {mode} entity masks (one-time operation)...")
            entity_masks = create_entity_masks(
                first_src, boundaries_gdf, out_transform, height, width, mode
            )
        
        # Create natural park masks (only for municipality mode)
        if natural_parks and mode == "municipalities":
            logger.info("Creating natural park overlay masks (one-time operation)...")
            natural_park_masks = create_natural_park_masks(
                first_src, natural_parks, out_transform, height, width
            )
    
    # Process files
    datasets = []
    logger.info("Loading and clipping files...")
    for file_path, year in tqdm(zip(available_files, available_years), total=len(available_files), desc="Processing files"):
        try:
            ds = load_and_clip_landsat_file(file_path, year, boundaries_gdf, natural_park_masks, entity_masks, mode)
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
    
    # Add classification if processing multiple entities
    if entity_filter is None:
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
    
    # Calculate derived variables string including natural parks
    derived_vars = ['NDVI']
    if 'ndvi_class' in combined_ds:
        derived_vars.append('NDVI_CLASS')
    if 'natural' in combined_ds:
        derived_vars.append('NATURAL_PARKS')
    
    # Generate output filename
    if entity_filter:
        # Single entity
        safe_name = entity_filter.replace(' ', '_').replace('/', '_').replace("'", "").replace('-', '_')
        output_filename = f"{shape_config['output_prefix']}{safe_name}.nc"
    else:
        # All entities
        output_filename = f"{shape_config['output_prefix']}all.nc"
    
    entity_description = entity_filter or f"all {len(boundaries_gdf)} {mode}"
    
    combined_ds.attrs.update({
        'title': f'{shape_config["description"]} Landsat Time Series - {entity_filter or f"All {mode.title()}"}',
        'description': f'Landsat Collection 2 Level 2 data for {entity_description} ({min(available_years)}-{max(available_years)})',
        'source': 'Google Earth Engine',
        'processing_level': 'Collection 2 Level 2',
        'spatial_resolution': '30m',
        'projection': 'EPSG:4326',
        'processing_mode': mode,
        'total_entities': len(boundaries_gdf),
        'entity_names': entity_filter or ', '.join(boundaries_gdf['entity_name'].tolist()),
        'n_years': len(available_years),
        'bands': ', '.join(BAND_NAMES),
        'derived_variables': ', '.join(derived_vars),
        'natural_parks_included': 'true' if (INCLUDE_NATURAL_PARKS and mode == "municipalities") else 'false',
        'created_date': datetime.now().isoformat(),
        'individual_analysis_supported': 'true',
        'nodata_handling': 'NaN for invalid values'
    })
    
    # Save dataset
    processed_data_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_data_path / output_filename

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
    if 'entity_id' in combined_ds:
        encoding['entity_id'] = {'dtype': 'int16', 'zlib': True, 'complevel': 6}
    if 'natural' in combined_ds:
        encoding['natural'] = {'dtype': 'int16', 'zlib': True, 'complevel': 6}  # Integer encoding for natural parks
    
    combined_ds.to_netcdf(output_file, engine='netcdf4', encoding=encoding)
    
    # Save natural parks mapping if included (only for municipality mode)
    if INCLUDE_NATURAL_PARKS and natural_parks and mode == "municipalities":
        park_mappings = []
        
        if 'pein' in natural_parks:
            for idx, (_, row) in enumerate(natural_parks['pein'].iterrows(), 1):
                park_mappings.append({
                    'park_type': 'pein',
                    'park_code': idx,
                    'original_code': row.get('CODI_PEIN', ''),
                    'name': row.get('NOM', ''),
                    'description': f"PEIN park: {row.get('CODI_PEIN', '')}"
                })
        
        if 'xpn' in natural_parks:
            for idx, (_, row) in enumerate(natural_parks['xpn'].iterrows(), 1):
                park_mappings.append({
                    'park_type': 'xpn', 
                    'park_code': idx,
                    'original_code': row.get('ACRONIM', ''),
                    'name': row.get('NOM', ''),
                    'description': f"XPN park: {row.get('ACRONIM', '')}"
                })
        
        if park_mappings:
            park_mapping_df = pd.DataFrame(park_mappings)
            park_mapping_file = processed_data_path / "natural_parks_mapping.csv"
            park_mapping_df.to_csv(park_mapping_file, index=False, encoding='utf-8')
            logger.info(f"Natural parks mapping saved to: {park_mapping_file}")
    
    # Save entity mapping (only if processing multiple entities)
    if entity_filter is None and entity_masks is not None:
        entity_ids, entity_names, entity_original_ids = entity_masks
        entity_info = pd.DataFrame({
            'entity_name': entity_names,
            'entity_id': range(1, len(entity_names) + 1),
            'original_id': entity_original_ids
        })
        entity_info_file = processed_data_path / f"{mode}_mapping.csv"
        entity_info.to_csv(entity_info_file, index=False, encoding='utf-8')
        logger.info(f"{mode.title()} mapping saved to: {entity_info_file}")

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
    # Example usage:
    # For municipalities (default): python create_mdim_raster.py
    # For PEIN parks: Set PROCESSING_MODE = "pein" above
    # For XPN parks: Set PROCESSING_MODE = "xpn" above
    # For single entity: Set FILTER_ENTITY = "entity_name" above
    
    create_multidimensional_raster()