"""
Main Pipeline Orchestrator for Vegetation-Focused ST-Cube Segmentation

Main entry point and orchestration for NDVI-based spatiotemporal trace segmentation.
Coordinates the segmentation workflow, manages dual trend processing, and handles
export/visualization generation. Contains the main API functions.
"""

import xarray as xr
from pathlib import Path
import warnings
from typing import Optional, Dict, List
from dataclasses import dataclass
from loguru import logger
import datetime
from .config_loader import get_config
from .json_exporter import VegetationClusterJSONExporter
from .visualization.visualization_2d import StaticVisualization
from .segmentation_engine import VegetationSegmenter

warnings.filterwarnings('ignore')

@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation segmentation."""
    max_spatial_distance: int = None
    min_vegetation_ndvi: float = None
    min_cluster_size: int = None
    ndvi_variance_threshold: float = None
    chunk_size: int = None
    eps: float = None
    min_pts: int = None
    temporal_weight: float = None
    spatial_weight: float = None
    ndvi_trend_filter: Optional[str] = None
    
    def __post_init__(self):
        # Load config and set defaults if not provided
        config = get_config()
        
        if self.max_spatial_distance is None:
            self.max_spatial_distance = config.max_spatial_distance
        if self.min_vegetation_ndvi is None:
            self.min_vegetation_ndvi = config.min_vegetation_ndvi
        if self.min_cluster_size is None:
            self.min_cluster_size = config.min_cluster_size
        if self.ndvi_variance_threshold is None:
            self.ndvi_variance_threshold = config.ndvi_variance_threshold
        if self.chunk_size is None:
            self.chunk_size = config.chunk_size
        if self.eps is None:
            self.eps = config.eps
        if self.min_pts is None:
            self.min_pts = config.min_pts
        if self.temporal_weight is None:
            self.temporal_weight = config.temporal_weight
        if self.spatial_weight is None:
            self.spatial_weight = config.spatial_weight
        if self.ndvi_trend_filter is None:
            self.ndvi_trend_filter = config.ndvi_trend_filter


def segment_vegetation(netcdf_path: str = None, 
                               parameters: Optional[VegetationSegmentationParameters] = None,
                               municipality_name: str = None,
                               create_visualizations: bool = True,
                               output_dir: str = None) -> Dict[str, List[Dict]]:
    """
    Vegetation segmentation function with improved performance and memory usage.

    Args:
        netcdf_path (str): Path to the input NetCDF file.
        parameters (VegetationSegmentationParameters): Segmentation parameters.
        municipality_name (str): Name of the municipality to analyze.
        create_visualizations (bool): Whether to create visualizations.
        output_dir (str): Base output directory.

    Returns:
        Dict[str, List[Dict]]: Segmentation results categorized by trend.
    """
    config = get_config()
    
    try:
        netcdf_path = config.netcdf_path
        municipality_name = config.municipality
        output_dir = config.output_dir
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
    
    if parameters is None:
        parameters = VegetationSegmentationParameters()
    
    # Load data for dual trend visualization (if needed)
    data = None
    if netcdf_path:
        try:
            data = xr.open_dataset(netcdf_path, chunks={'time': 10, 'x': 500, 'y': 500})
            if municipality_name and 'municipality' in data.dims:
                data = data.sel(municipality=municipality_name)
        except Exception as e:
            logger.warning(f"Could not load data for dual trend visualization: {e}")
            data = None
    
    # Always add timestamp at the top level
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = str(Path(output_dir) / timestamp)
    
    results = {}
    
    # Initialize global cluster counter
    global_cluster_counter = 0
    
    # Determine which trends to process
    if parameters.ndvi_trend_filter is None:
        # Process both trends
        trends_to_process = ['greening', 'browning']
        logger.info("Running segmentation for both greening and browning trends...")
    else:
        # Process single trend
        trends_to_process = [parameters.ndvi_trend_filter]
        logger.info(f"Running segmentation for {parameters.ndvi_trend_filter} trends...")
    
    # Process each trend
    for trend in trends_to_process:
        logger.info(f"PROCESSING {trend.upper()} TRENDS")
        
        # Create parameters for this trend
        trend_params = VegetationSegmentationParameters(
            min_cluster_size=parameters.min_cluster_size,
            max_spatial_distance=parameters.max_spatial_distance,
            min_vegetation_ndvi=parameters.min_vegetation_ndvi,
            eps=parameters.eps,
            min_pts=parameters.min_pts,
            ndvi_variance_threshold=parameters.ndvi_variance_threshold,
            temporal_weight=parameters.temporal_weight,
            spatial_weight=parameters.spatial_weight,
            ndvi_trend_filter=trend
        )
        
        # Create trend-specific output directory
        trend_output_dir = str(Path(timestamped_output_dir) / trend)
        
        # Run segmentation for this trend using the segmentation engine
        segmenter = VegetationSegmenter(trend_params)
        trend_results = segmenter.segment_vegetation(
            netcdf_path=netcdf_path,
            municipality_name=municipality_name,
            create_visualizations=create_visualizations,
            output_dir=trend_output_dir,
            global_cluster_counter=global_cluster_counter
        )
        
        # Update global counter for next trend
        if trend_results:
            global_cluster_counter += len(trend_results)
        
        results[trend] = trend_results
    
    # Create JSON export
    if config.enable_json_export and data is not None:
        logger.info("Creating JSON export...")
        # Get configuration parameters for export
        config_params = {
            "min_cluster_size": parameters.min_cluster_size,
            "max_spatial_distance": parameters.max_spatial_distance,
            "min_vegetation_ndvi": parameters.min_vegetation_ndvi,
            "eps": parameters.eps,
            "min_pts": parameters.min_pts,
            "ndvi_variance_threshold": parameters.ndvi_variance_threshold,
            "temporal_weight": parameters.temporal_weight,
            "netcdf_path": netcdf_path,
            "municipality_name": municipality_name
        }
        
        # Export results
        json_exporter = VegetationClusterJSONExporter()
        json_exporter.export_combined_clusters_to_json(
            results, data, timestamped_output_dir, municipality_name, config_params
        )
    
    # Create additional visualizations and reports
    if len(trends_to_process) > 1:
        static_viz = StaticVisualization(output_directory=timestamped_output_dir)
        static_viz.create_combined_analysis_report(results, municipality_name)

        # Create common visualizations using CommonVisualization class
        from .visualization.common import CommonVisualization
        common_viz = CommonVisualization(output_directory=timestamped_output_dir)
        
        # Create spatial distribution map with cluster numbering
        common_viz.create_spatial_distribution_map(
            results=results,
            data=data,
            municipality_name=municipality_name
        )
        
        # Create interactive temporal trend map
        common_viz.create_interactive_temporal_trend_map(
            results=results,
            data=data,
            municipality_name=municipality_name
        )
    
    return results


if __name__ == "__main__":    
    config = get_config()
    
    # Get parameters from config and defaults
    params = VegetationSegmentationParameters()
    results = segment_vegetation(
        parameters=params,
        create_visualizations=True
    )
    
    # Final summary log
    total_clusters = sum(len(clusters) for clusters in results.values())
    if total_clusters == 0:
        logger.warning("No vegetation clusters found. Check your data and parameters.")
    else:
        trend_summary = " and ".join([f"{len(clusters)} {trend}" for trend, clusters in results.items()])
        logger.success(f"Found {trend_summary} clusters (total: {total_clusters}).")