"""
Configuration loader for vegetation segmentation.

This module provides a centralized way to load and access configuration
parameters from the YAML configuration file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import warnings

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "segment_config.yaml"


@dataclass
class SegmentationConfig:
    """Configuration data class for vegetation segmentation."""
    
    # Core segmentation parameters
    min_cube_size: int = 20
    max_spatial_distance: int = 10
    min_vegetation_ndvi: float = 0.4
    ndvi_variance_threshold: float = 0.05
    n_clusters: Optional[int] = 10
    temporal_weight: float = 0.5
    chunk_size: int = 1000
    max_pixels_for_sampling: int = 50000
    
    # Clustering parameters
    eps_search_attempts: int = 5
    min_samples_ratio: float = 0.01
    spatial_weight: float = 0.2
    
    # Spatial bridging parameters
    enable_spatial_bridging: bool = True
    bridge_similarity_tolerance: float = 0.1
    max_bridge_gap: int = 2
    min_bridge_density: float = 0.7
    connectivity_radius: int = 3
    max_bridge_length: int = 20
    min_cluster_size_for_bridging: int = 5
    
    # Spatial parameters
    spatial_margin: int = 1
    temporal_margin: int = 0
    max_neighbors: int = 10
    search_margin: int = 3
    adjacency_search_neighbors: int = 50
    
    # Data paths
    default_netcdf_path: str = "data/processed/landsat_mdim_all_muni.nc"
    municipalities_data: str = "data/processed/municipality_mapping.csv"
    default_municipality: str = "Sant Martí"
    default_output_dir: str = "outputs/vegetation_clustering"
    
    # Visualization parameters
    interactive_output_dir: str = "outputs/interactive_vegetation"
    figure_width: int = 1400
    figure_height: int = 900
    static_output_dir: str = "outputs/static_vegetation"
    figure_size: list = field(default_factory=lambda: [18, 12])
    dpi: int = 300
    color_map: str = "Set3"
    grid_alpha: float = 0.3
    
    # 3D visualization
    cube_size_multiplier: float = 0.01
    max_time_layers: int = 42
    max_clusters_3d: int = 50
    camera_x: float = 1.5
    camera_y: float = 1.5
    camera_z: float = 1.2
    aspect_x: float = 1.0
    aspect_y: float = 1.0
    aspect_z: float = 0.05
    
    # Logging
    segmentation_log: str = "logs/segmentation_{time:YYYY-MM-DD}.log"
    visualization_log: str = "logs/visualization_{time:YYYY-MM-DD}.log"
    log_level: str = "INFO"
    log_rotation: str = "1 day"
    
    # Legacy visualization
    simple_netcdf_path: str = "data/processed/landsat_mdim_all_muni.nc"
    simple_output_dir: str = "outputs/interactive"
    time_series_name: str = "simple_time_series.html"
    spatial_map_name: str = "simple_spatial_map.html"
    raster_output_dir: str = "outputs/3d_raster"
    raster_output_name: str = "3d_raster_barcelona_test.html"
    downsample_factor: int = 2
    target_years: list = field(default_factory=lambda: [2012, 2014, 2016, 2018, 2020, 2022])
    
    # Analysis parameters
    dense_vegetation_threshold: float = 0.7
    moderate_vegetation_threshold: float = 0.5
    significant_greening_threshold: float = 0.005
    significant_browning_threshold: float = -0.005
    min_seasonal_variance: float = 0.01
    
    # Processing
    max_dataset_pixels: int = 50000
    n_jobs: int = -1
    max_retries: int = 3
    
    # Export
    csv_encoding: str = "utf-8"
    plotly_js_mode: str = "cdn"
    auto_open: bool = False
    bbox_inches: str = "tight"
    enable_json_export: bool = True
    include_pixel_level_data: bool = True
    json_indent: int = 2


class ConfigLoader:
    """Configuration loader for vegetation segmentation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration loader."""
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config_data = None
        self._config_obj = None
        
    def load_config(self) -> SegmentationConfig:
        """Load configuration from YAML file."""
        if self._config_obj is not None:
            return self._config_obj
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
        except FileNotFoundError:
            warnings.warn(f"Config file not found at {self.config_path}, using defaults")
            self._config_data = {}
        except yaml.YAMLError as e:
            warnings.warn(f"Error parsing config file: {e}, using defaults")
            self._config_data = {}
            
        # Create config object with flattened parameters
        config_dict = self._flatten_config(self._config_data)
        self._config_obj = SegmentationConfig(**config_dict)
        
        return self._config_obj
    
    def _flatten_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested configuration dictionary."""
        flattened = {}
        
        # Core segmentation parameters
        seg_params = config_data.get('segmentation', {})
        flattened.update({
            'min_cube_size': seg_params.get('min_cube_size', 20),
            'max_spatial_distance': seg_params.get('max_spatial_distance', 10),
            'min_vegetation_ndvi': seg_params.get('min_vegetation_ndvi', 0.4),
            'ndvi_variance_threshold': seg_params.get('ndvi_variance_threshold', 0.05),
            'n_clusters': seg_params.get('n_clusters', 10),
            'temporal_weight': seg_params.get('temporal_weight', 0.5),
        })
        
        # Clustering parameters
        cluster_params = config_data.get('clustering', {})
        flattened.update({
            'eps_search_attempts': cluster_params.get('eps_search_attempts', 5),
            'min_samples_ratio': cluster_params.get('min_samples_ratio', 0.01),
            'spatial_weight': cluster_params.get('spatial_weight', 0.2),
        })
        
        # Processing parameters  
        process_params = config_data.get('processing', {})
        flattened.update({
            'chunk_size': process_params.get('chunk_size', 1000),
            'max_pixels_for_sampling': process_params.get('max_pixels_for_sampling', 50000),
            'max_dataset_pixels': process_params.get('max_dataset_pixels', 50000),
            'n_jobs': process_params.get('n_jobs', -1),
            'max_retries': process_params.get('max_retries', 3),
        })
        
        # Spatial parameters (with defaults for rarely used parameters)
        spatial_params = config_data.get('spatial', {})
        flattened.update({
            'spatial_margin': spatial_params.get('spatial_margin', 1),
            'temporal_margin': spatial_params.get('temporal_margin', 0),
            'max_neighbors': spatial_params.get('max_neighbors', 10),
            'search_margin': spatial_params.get('search_margin', 3),
            'adjacency_search_neighbors': spatial_params.get('adjacency_search_neighbors', 50),
        })
        
        # Data paths
        data_params = config_data.get('data', {})
        flattened.update({
            'default_netcdf_path': data_params.get('default_netcdf_path', "data/processed/landsat_mdim_all_muni.nc"),
            'municipalities_data': data_params.get('municipalities_data', "data/processed/municipality_mapping.csv"),
            'default_municipality': data_params.get('default_municipality', "Sant Martí"),
            'default_output_dir': data_params.get('default_output_dir', "outputs/vegetation_clustering"),
        })
        
        # Visualization parameters
        viz_params = config_data.get('visualization', {})
        flattened.update({
            'interactive_output_dir': viz_params.get('interactive_output_dir', "outputs/interactive_vegetation"),
            'figure_width': viz_params.get('figure_width', 1400),
            'figure_height': viz_params.get('figure_height', 900),
            'static_output_dir': viz_params.get('static_output_dir', "outputs/static_vegetation"),
            'figure_size': viz_params.get('figure_size', [18, 12]),
            'dpi': viz_params.get('dpi', 300),
            'color_map': viz_params.get('color_map', "Set3"),
            'grid_alpha': viz_params.get('grid_alpha', 0.3),
            'cube_size_multiplier': viz_params.get('cube_size_multiplier', 0.01),
            'max_time_layers': viz_params.get('max_time_layers', 42),
            'max_clusters_3d': viz_params.get('max_clusters_3d', 50),
            'camera_x': viz_params.get('camera_x', 1.5),
            'camera_y': viz_params.get('camera_y', 1.5),
            'camera_z': viz_params.get('camera_z', 1.2),
            'aspect_x': viz_params.get('aspect_x', 1.0),
            'aspect_y': viz_params.get('aspect_y', 1.0),
            'aspect_z': viz_params.get('aspect_z', 0.2),
        })
        
        # Analysis parameters
        analysis_params = config_data.get('analysis', {})
        veg_class_params = analysis_params.get('vegetation_classification', {})
        trend_params = analysis_params.get('trend_analysis', {})
        season_params = analysis_params.get('seasonality_analysis', {})
        flattened.update({
            'dense_vegetation_threshold': veg_class_params.get('dense_vegetation_threshold', 0.7),
            'moderate_vegetation_threshold': veg_class_params.get('moderate_vegetation_threshold', 0.5),
            'significant_greening_threshold': trend_params.get('significant_greening_threshold', 0.005),
            'significant_browning_threshold': trend_params.get('significant_browning_threshold', -0.005),
            'min_seasonal_variance': season_params.get('min_seasonal_variance', 0.01),
        })
        
        # Logging parameters
        log_params = config_data.get('logging', {})
        flattened.update({
            'segmentation_log': log_params.get('segmentation_log', "logs/segmentation_{time:YYYY-MM-DD}.log"),
            'visualization_log': log_params.get('visualization_log', "logs/visualization_{time:YYYY-MM-DD}.log"),
            'log_level': log_params.get('log_level', "INFO"),
            'log_rotation': log_params.get('log_rotation', "1 day"),
        })
        
        # Export parameters
        export_params = config_data.get('export', {})
        flattened.update({
            'csv_encoding': export_params.get('csv_encoding', "utf-8"),
            'plotly_js_mode': export_params.get('plotly_js_mode', "cdn"),
            'auto_open': export_params.get('auto_open', False),
            'bbox_inches': export_params.get('bbox_inches', "tight"),
        })
        
        # Legacy visualization parameters (with defaults)
        flattened.update({
            'simple_netcdf_path': "data/processed/landsat_mdim_all_muni.nc",
            'simple_output_dir': "outputs/interactive",
            'time_series_name': "simple_time_series.html",
            'spatial_map_name': "simple_spatial_map.html", 
            'raster_output_dir': "outputs/3d_raster",
            'raster_output_name': "3d_raster_barcelona_test.html",
            'downsample_factor': 2,
            'target_years': [2012, 2014, 2016, 2018, 2020, 2022],
        })
        
        return flattened
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a specific parameter value."""
        config = self.load_config()
        return getattr(config, key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all parameters from a specific section."""
        if self._config_data is None:
            self.load_config()
        return self._config_data.get(section, {})


# Global config loader instance
_config_loader = ConfigLoader()

def get_config() -> SegmentationConfig:
    """Get the global configuration object."""
    return _config_loader.load_config()

def get_parameter(key: str, default: Any = None) -> Any:
    """Get a specific parameter value."""
    return _config_loader.get_parameter(key, default)

def get_section(section: str) -> Dict[str, Any]:
    """Get all parameters from a specific section."""
    return _config_loader.get_section(section)

def reload_config(config_path: Optional[Path] = None) -> SegmentationConfig:
    """Reload configuration from file."""
    global _config_loader
    _config_loader = ConfigLoader(config_path)
    return _config_loader.load_config()
