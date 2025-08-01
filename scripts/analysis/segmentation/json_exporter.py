"""
JSON Export Module for Vegetation Cluster Data

This module handles the export of vegetation cluster analysis results to structured JSON files
for future analysis, visualization, and research purposes.
"""

import json
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger
from config_loader import get_config


class VegetationClusterJSONExporter:
    """
    Handles export of vegetation cluster data to structured JSON files.
    
    This class provides functionality to export cluster analysis results including:
    - Cluster metadata and summary statistics
    - Temporal NDVI profiles for each cluster
    - Individual pixel coordinates (lat/lon) and NDVI time series
    - Configuration parameters used for analysis
    """
    
    def __init__(self):
        """Initialize the JSON exporter."""
        self.config = get_config()
        self.logger = logger
    
    def export_clusters_to_json(self, 
                               vegetation_cubes: List[Dict], 
                               data: xr.Dataset, 
                               output_dir: str, 
                               municipality_name: str,
                               parameters: Dict) -> str:
        """
        Export cluster information to a structured JSON file for future analysis and visualization.
        
        Args:
            vegetation_cubes: List of cluster dictionaries
            data: Original xarray dataset
            output_dir: Output directory path
            municipality_name: Name of the municipality
            parameters: Configuration parameters used for segmentation
            
        Returns:
            Path to the exported JSON file
        """
        self.logger.info("Exporting cluster data to JSON...")
        
        # Get configuration settings
        include_pixels = self.config.include_pixel_level_data
        json_indent = self.config.json_indent
        
        # Extract time coordinates (years)
        time_coords = self._extract_time_coordinates(data)
        
        # Prepare the main data structure
        cluster_data = self._create_metadata_structure(
            municipality_name, vegetation_cubes, time_coords, include_pixels, parameters
        )
        
        # Process each cluster
        for i, cube in enumerate(vegetation_cubes):            
            cluster_info = self._process_cluster(cube, data, time_coords, include_pixels, i)
            cluster_data["clusters"].append(cluster_info)
        
        # Save to JSON file
        json_filepath = self._save_json_file(cluster_data, output_dir, municipality_name, json_indent)
        
        return str(json_filepath)
    
    def _extract_time_coordinates(self, data: xr.Dataset) -> List[int]:
        """Extract and convert time coordinates to a list of years."""
        time_coords = data.time.values.tolist() if 'time' in data.dims else []
        
        # Convert numpy types to native Python types for JSON serialization
        if time_coords:
            time_coords = [int(year) if hasattr(year, 'item') else int(year) for year in time_coords]
        
        return time_coords
    
    def _create_metadata_structure(self, 
                                 municipality_name: str, 
                                 vegetation_cubes: List[Dict], 
                                 time_coords: List[int], 
                                 include_pixels: bool, 
                                 parameters: Dict) -> Dict[str, Any]:
        """Create the main JSON data structure with metadata."""
        return {
            "metadata": {
                "municipality": municipality_name,
                "export_date": datetime.now().isoformat(),
                "total_clusters": len(vegetation_cubes),
                "years": time_coords,
                "include_pixel_level_data": include_pixels,
                "data_structure_version": "1.0",
                "description": "Vegetation cluster analysis results with NDVI time series data",
                "config_parameters": parameters
            },
            "clusters": []
        }
    
    def _process_cluster(self, 
                        cube: Dict, 
                        data: xr.Dataset, 
                        time_coords: List[int], 
                        include_pixels: bool, 
                        cluster_index: int) -> Dict[str, Any]:
        """Process a single cluster and return its JSON representation."""
        
        # Get cluster pixel coordinates and NDVI profiles
        coordinates, pixel_ndvi_profiles = self._extract_cluster_data(cube)
        
        # Initialize cluster info structure
        cluster_info = self._create_cluster_info_structure(cube, coordinates, cluster_index)
        
        # Add temporal profile data
        self._add_temporal_profile_data(cluster_info, cube, time_coords)
        
        # Process individual pixels if enabled
        if include_pixels and coordinates and pixel_ndvi_profiles:
            self._process_cluster_pixels(
                cluster_info, coordinates, pixel_ndvi_profiles, data, time_coords
            )
        
        return cluster_info
    
    def _extract_cluster_data(self, cube: Dict) -> tuple:
        """Extract coordinates and NDVI profiles from cluster data."""
        
        # Get cluster pixel coordinates
        coordinates = cube.get('coordinates', [])
        if hasattr(coordinates, 'tolist'):
            coordinates = coordinates.tolist()
        elif isinstance(coordinates, np.ndarray):
            coordinates = coordinates.tolist()
        elif isinstance(coordinates, set):
            coordinates = list(coordinates)
        
        # Get individual pixel NDVI profiles
        pixel_ndvi_profiles = cube.get('ndvi_profiles', [])
        if hasattr(pixel_ndvi_profiles, 'tolist'):
            pixel_ndvi_profiles = pixel_ndvi_profiles.tolist()
        elif isinstance(pixel_ndvi_profiles, np.ndarray):
            pixel_ndvi_profiles = pixel_ndvi_profiles.tolist()
        
        return coordinates, pixel_ndvi_profiles
    
    def _create_cluster_info_structure(self, cube: Dict, coordinates: List, cluster_index: int) -> Dict[str, Any]:
        """Create the basic cluster information structure."""
        return {
            "cluster_id": int(cube.get('id', cluster_index)),
            "summary": {
                "area_pixels": int(cube.get('area', len(coordinates))),
                "mean_ndvi": self._safe_float_conversion(cube.get('mean_ndvi')),
                "vegetation_type": str(cube.get('vegetation_type', 'Unknown')),
                "seasonality_score": self._safe_float_conversion(cube.get('seasonality_score')),
                "trend_score": self._safe_float_conversion(cube.get('trend_score')),
                "temporal_variance": self._safe_float_conversion(cube.get('temporal_variance'))
            },
            "temporal_profile": {
                "mean_ndvi_per_year": {},
                "cluster_ndvi_profile": []
            },
            "pixels": []
        }
    
    def _safe_float_conversion(self, value) -> float:
        """Safely convert a value to float, handling NaN and None values."""
        if value is not None and not np.isnan(value):
            return float(value)
        return None
    
    def _add_temporal_profile_data(self, cluster_info: Dict, cube: Dict, time_coords: List[int]):
        """Add temporal profile data to cluster info."""
        
        # Get cluster-level NDVI temporal profile (mean across all pixels)
        ndvi_profile = cube.get('mean_temporal_profile', [])
        if hasattr(ndvi_profile, 'tolist'):
            ndvi_profile = ndvi_profile.tolist()
        elif isinstance(ndvi_profile, np.ndarray):
            ndvi_profile = [float(x) if not np.isnan(x) else None for x in ndvi_profile]
        elif ndvi_profile is not None:
            # Ensure it's a list of proper Python types
            ndvi_profile = [float(x) if x is not None and not np.isnan(x) else None for x in ndvi_profile]
        else:
            ndvi_profile = []
        
        cluster_info["temporal_profile"]["cluster_ndvi_profile"] = ndvi_profile
        
        # Create mean NDVI per year mapping
        if len(ndvi_profile) == len(time_coords):
            for year, ndvi_val in zip(time_coords, ndvi_profile):
                cluster_info["temporal_profile"]["mean_ndvi_per_year"][str(year)] = ndvi_val
    
    def _process_cluster_pixels(self, 
                               cluster_info: Dict, 
                               coordinates: List, 
                               pixel_ndvi_profiles: List, 
                               data: xr.Dataset, 
                               time_coords: List[int]):
        """Process individual pixels within a cluster."""
        
        for pixel_idx, (pixel_coord, pixel_ndvi_series) in enumerate(zip(coordinates, pixel_ndvi_profiles)):
            if not isinstance(pixel_coord, (list, tuple)) or len(pixel_coord) < 2:
                self.logger.warning(f"Invalid pixel coordinate format: {pixel_coord}")
                continue
            
            y_coord, x_coord = int(pixel_coord[0]), int(pixel_coord[1])
            
            # Process NDVI time series for this pixel
            pixel_ndvi_dict, pixel_ndvi_values = self._process_pixel_ndvi_series(
                pixel_ndvi_series, time_coords
            )
            
            # Convert pixel coordinates to latitude/longitude
            lat, lon = self._convert_pixel_to_latlon(data, y_coord, x_coord)
            
            # Create pixel information structure
            pixel_info = self._create_pixel_info_structure(
                pixel_idx, lat, lon, pixel_ndvi_dict, pixel_ndvi_values
            )
            
            cluster_info["pixels"].append(pixel_info)
    
    def _process_pixel_ndvi_series(self, pixel_ndvi_series, time_coords: List[int]) -> tuple:
        """Process NDVI time series for a single pixel."""
        
        pixel_ndvi_dict = {}
        pixel_ndvi_values = []
        
        # Convert pixel NDVI series to proper format
        if hasattr(pixel_ndvi_series, 'tolist'):
            pixel_ndvi_series = pixel_ndvi_series.tolist()
        elif isinstance(pixel_ndvi_series, np.ndarray):
            pixel_ndvi_series = pixel_ndvi_series.tolist()
        
        # Create year-to-NDVI mapping
        for year, ndvi_val in zip(time_coords, pixel_ndvi_series[:len(time_coords)]):
            # Convert numpy types to Python native types
            if hasattr(ndvi_val, 'item'):
                ndvi_val = ndvi_val.item()
            ndvi_val = float(ndvi_val) if not np.isnan(ndvi_val) else None
            
            pixel_ndvi_dict[str(int(year))] = ndvi_val
            pixel_ndvi_values.append(ndvi_val)
        
        return pixel_ndvi_dict, pixel_ndvi_values
    
    def _convert_pixel_to_latlon(self, data: xr.Dataset, y_coord: int, x_coord: int) -> tuple:
        """Convert pixel coordinates to latitude/longitude."""
        
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
                # Final fallback: use pixel coordinates as placeholders
                self.logger.warning(f"Could not convert pixel coordinates ({y_coord}, {x_coord}) to lat/lon. Using pixel coordinates as fallback.")
                lat = float(y_coord)
                lon = float(x_coord)
        
        return lat, lon
    
    def _create_pixel_info_structure(self, 
                                   pixel_idx: int, 
                                   lat: float, 
                                   lon: float, 
                                   pixel_ndvi_dict: Dict, 
                                   pixel_ndvi_values: List) -> Dict[str, Any]:
        """Create the pixel information structure."""
        
        pixel_info = {
            "pixel_id": pixel_idx,
            "coordinates": {
                "latitude": lat,
                "longitude": lon
            },
            "ndvi_time_series": pixel_ndvi_dict,
            "ndvi_profile": pixel_ndvi_values,
            "pixel_statistics": self._calculate_pixel_statistics(pixel_ndvi_values)
        }
        
        return pixel_info
    
    def _calculate_pixel_statistics(self, pixel_ndvi_values: List) -> Dict[str, float]:
        """Calculate statistics for a pixel's NDVI time series."""
        
        valid_values = [v for v in pixel_ndvi_values if v is not None]
        
        if not valid_values:
            return {
                "mean_ndvi": None,
                "min_ndvi": None,
                "max_ndvi": None,
                "std_ndvi": None
            }
        
        stats = {
            "mean_ndvi": np.nanmean(valid_values),
            "min_ndvi": np.nanmin(valid_values),
            "max_ndvi": np.nanmax(valid_values),
            "std_ndvi": np.nanstd(valid_values)
        }
        
        # Convert numpy values to Python native types for JSON serialization
        for stat_key, stat_val in stats.items():
            if stat_val is not None and hasattr(stat_val, 'item'):
                stats[stat_key] = float(stat_val.item())
            elif stat_val is not None and not np.isnan(stat_val):
                stats[stat_key] = float(stat_val)
            else:
                stats[stat_key] = None
        
        return stats
    
    def _save_json_file(self, 
                       cluster_data: Dict, 
                       output_dir: str, 
                       municipality_name: str, 
                       json_indent: int) -> Path:
        """Save cluster data to JSON file."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_filename = f"vegetation_clusters_{municipality_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_filepath = output_path / json_filename
        
        try:
            # Convert all numpy types to native Python types before JSON serialization
            cluster_data_converted = self._convert_numpy_types(cluster_data)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(cluster_data_converted, f, indent=json_indent, ensure_ascii=False)
            
            return json_filepath
            
        except Exception as e:
            self.logger.error(f"Failed to export cluster data to JSON: {e}")
            raise
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        else:
            return obj