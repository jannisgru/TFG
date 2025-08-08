"""
Configuration Loader for Vegetation Segmentation

Centralized module for loading, accessing, and managing all configuration parameters for the segmentation pipeline. Supports YAML config, default values, and section-based access for reproducible experiments.
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
    
    # Core segmentation parameters (defined in YAML)
    min_cube_size: int
    max_spatial_distance: int
    min_vegetation_ndvi: float
    ndvi_variance_threshold: float
    n_clusters: Optional[int]
    temporal_weight: float
    ndvi_trend_filter: Optional[str]
    
    # Clustering parameters (defined in YAML)
    eps_search_attempts: int
    min_samples_ratio: float
    spatial_weight: float
    
    # Processing parameters (defined in YAML)
    chunk_size: int
    max_pixels_for_sampling: int
    
    # Data paths (defined in YAML)
    netcdf_path: str
    municipality: str
    output_dir: str
    # Data paths (defined in YAML)
    netcdf_path: str
    municipality: str
    output_dir: str
    

    # Export settings (defined in YAML)
    enable_json_export: bool

    # Basemap settings (defined in YAML)
    basemap_layer: str
    
    # Parameters with defaults (not in YAML - rarely changed)
    spatial_margin: int = 1
    temporal_margin: int = 0
    max_neighbors: int = 10
    search_margin: int = 3
    adjacency_search_neighbors: int = 50
    municipalities_data: str = "data/processed/municipality_mapping.csv"
    
    # Visualization parameters (defaults for parameters not in YAML)
    interactive_output_dir: str = "outputs/interactive_vegetation"
    figure_width: int = 1400
    figure_height: int = 900
    static_output_dir: str = "outputs/static_vegetation"
    figure_size: list = field(default_factory=lambda: [18, 12])
    dpi: int = 300
    color_map: str = "Set3"
    grid_alpha: float = 0.3
    bbox_inches: str = "tight"
    
    # 3D visualization
    cube_size_multiplier: float = 0.01
    max_time_layers: int = 42
    max_clusters_3d: int = 50
    camera_x: float = 1.5
    camera_y: float = 1.5
    camera_z: float = 1.2
    aspect_x: float = 1.0
    aspect_y: float = 1.0
    aspect_z: float = 0.15
    
    # Logging
    segmentation_log: str = "logs/segmentation_{time:YYYY-MM-DD}.log"
    visualization_log: str = "logs/visualization_{time:YYYY-MM-DD}.log"
    log_level: str = "INFO"
    log_rotation: str = "1 day"
    
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
        
        # Core segmentation parameters (required from YAML)
        seg_params = config_data.get('segmentation', {})
        flattened.update({
            'min_cube_size': seg_params['min_cube_size'],
            'max_spatial_distance': seg_params['max_spatial_distance'],
            'min_vegetation_ndvi': seg_params['min_vegetation_ndvi'],
            'ndvi_variance_threshold': seg_params['ndvi_variance_threshold'],
            'n_clusters': seg_params['n_clusters'],
            'temporal_weight': seg_params['temporal_weight'],
            'ndvi_trend_filter': seg_params['ndvi_trend_filter'],
        })
        
        # Clustering parameters (required from YAML)
        cluster_params = config_data.get('clustering', {})
        flattened.update({
            'eps_search_attempts': cluster_params['eps_search_attempts'],
            'min_samples_ratio': cluster_params['min_samples_ratio'],
            'spatial_weight': cluster_params['spatial_weight'],
        })
        
        # Processing parameters (required from YAML)
        process_params = config_data.get('processing', {})
        flattened.update({
            'chunk_size': process_params['chunk_size'],
            'max_pixels_for_sampling': process_params['max_pixels_for_sampling'],
        })
        
        # Data paths (required from YAML)
        data_params = config_data.get('data', {})
        flattened.update({
            'netcdf_path': data_params['netcdf_path'],
            'municipality': data_params['municipality'],
            'output_dir': data_params['output_dir'],
        })
        
        # Export settings (required from YAML)
        export_params = config_data.get('export', {})
        flattened.update({
            'enable_json_export': export_params['enable_json_export'],
        })
        
        # Basemap settings (required from YAML)
        basemap_params = config_data.get('basemap', {})
        flattened.update({
            'basemap_layer': basemap_params.get('basemap_layer', "ortofoto_color_vigent"),
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
