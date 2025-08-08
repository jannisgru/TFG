"""
Main Pipeline for Vegetation-Focused ST-Cube Segmentation

Implements the main workflow for NDVI-based spatiotemporal trace segmentation, including data loading, preprocessing, clustering, 
and result export/visualization. Entry point for running the segmentation pipeline with configurable parameters.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import warnings
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import gc
from loguru import logger
import datetime
import traceback
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from .config_loader import get_config, get_section
from .json_exporter import VegetationClusterJSONExporter
from .visualization.visualization_3d import InteractiveVisualization
from .visualization.visualization_2d import StaticVisualization

warnings.filterwarnings('ignore')

@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation segmentation."""
    max_spatial_distance: int = None
    min_vegetation_ndvi: float = None
    min_cluster_size: int = None
    ndvi_variance_threshold: float = None
    chunk_size: int = None
    eps: float = None  # DBSCAN eps parameter (maximum distance between samples)
    min_samples: int = None  # DBSCAN min_samples parameter
    temporal_weight: float = None
    spatial_weight: float = None
    ndvi_trend_filter: Optional[str] = None  # 'greening', 'browning', or None
    
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
        if self.min_samples is None:
            self.min_samples = config.min_samples
        if self.temporal_weight is None:
            self.temporal_weight = config.temporal_weight
        if self.spatial_weight is None:
            self.spatial_weight = config.spatial_weight
        if self.ndvi_trend_filter is None:
            self.ndvi_trend_filter = config.ndvi_trend_filter


class VegetationSegmenter:
    """Vegetation segmentation with memory-efficient processing."""
    
    def __init__(self, parameters: VegetationSegmentationParameters):
        self.params = parameters
        self.config = get_config()
    
    def segment_vegetation(self, netcdf_path: str, 
                          municipality_name,
                          create_visualizations: bool = True,
                          output_dir: str = "outputs/vegetation_clustering",
                          is_dual_trend_processing: bool = False,
                          global_cluster_counter: int = 0
                          ) -> List[Dict[str, Any]]:
        """
        Run vegetation-focused NDVI clustering segmentation.
        
        Args:
            netcdf_path: Path to the input NetCDF file.
            municipality_name: Name of the municipality for analysis.
            create_visualizations: If True, generates visualizations.
            output_dir: Directory for saving output files.
            is_dual_trend_processing: If True, indicates this is part of dual trend processing.
                          Set to True when called from dual trend processing.
            global_cluster_counter: Starting cluster ID for consistent numbering across trends.

        Returns:
            List of vegetation cluster dictionaries with spatial and temporal info
        """

        logger.info(f"Data: {netcdf_path}")
        logger.info(f"Municipality: {municipality_name}")
        
        try:
            # Step 1: Load and validate data with lazy loading
            logger.info("1. Loading and validating data...")
            data, valid_mask, spatial_coords = self.load_and_prepare_data(netcdf_path, municipality_name)

            if data is None:
                logger.error("Failed to load data")
                return []
            
            # Step 2: Extract vegetation pixels efficiently
            logger.info("2. Extracting vegetation pixels...")
            vegetation_pixels, vegetation_coords = self.extract_vegetation_pixels(data, valid_mask)

            if len(vegetation_pixels) < self.params.min_cluster_size:
                logger.warning(f"Insufficient vegetation pixels: {len(vegetation_pixels)}")
                return []
            
            # Step 3: Perform clustering
            logger.info("3. Performing spatially-constrained clustering...")
            clusters = self.perform_spatially_constrained_clustering(vegetation_pixels, vegetation_coords, global_cluster_counter)
            
            # Step 4: Create vegetation traces
            logger.info("4. Creating vegetation ST-cubes...")
            vegetation_traces = self.create_vegetation_traces(clusters)
            
            # Step 5: Export cluster data to JSON (only for single trend processing)
            if vegetation_traces and not is_dual_trend_processing:
                config = get_config()
                if config.enable_json_export:
                    logger.info("5. Exporting cluster data to JSON...")
                    # Get configuration parameters for export
                    config_params = {
                        "min_cluster_size": self.params.min_cluster_size,
                        "max_spatial_distance": self.params.max_spatial_distance,
                        "min_vegetation_ndvi": self.params.min_vegetation_ndvi,
                        "eps": self.params.eps,
                        "min_samples": self.params.min_samples,
                        "ndvi_variance_threshold": self.params.ndvi_variance_threshold,
                        "temporal_weight": self.params.temporal_weight,
                        "netcdf_path": netcdf_path,
                        "municipality_name": municipality_name
                    }
                    
                    # Use the dedicated JSON exporter
                    json_exporter = VegetationClusterJSONExporter()
                    json_exporter.export_clusters_to_json(
                        vegetation_traces, data, output_dir, municipality_name, config_params
                    )
                else:
                    logger.info("5. JSON export disabled in configuration")
            elif is_dual_trend_processing:
                logger.info("5. Individual JSON export skipped (will be done in combined export)")
            else:
                logger.info("5. No vegetation clusters found for JSON export")
            
            # Step 6: Generate visualizations if requested
            if create_visualizations and vegetation_traces:
                logger.info("6. Creating visualizations...")
                self.create_visualizations(
                    vegetation_traces, data, output_dir, municipality_name
                )
            
            return vegetation_traces
            
        except Exception as e:
            logger.error(f"Error during vegetation segmentation: {str(e)}")
            traceback.print_exc()
            return []
        finally:
            # Clean up memory
            gc.collect()

    def load_and_prepare_data(self, netcdf_path: str,
                               municipality_name: str) -> Tuple[Optional[xr.Dataset],
                                                                  Optional[np.ndarray],
                                                                  Optional[Dict]]:
        """Data loading with chunking and validation."""

        try:
            # Load with chunking for memory efficiency
            data = xr.open_dataset(netcdf_path, chunks={'time': 10, 'x': 500, 'y': 500})
            #logger.info(f"Loaded dataset with shape: {dict(data.dims)}")
            
            # Validate required variables
            required_vars = ['ndvi']
            missing_vars = [var for var in required_vars if var not in data.variables]
            if missing_vars:
                logger.error(f"Missing required variables: {missing_vars}")
                return None, None, None
            
            # Filter by municipality with better error handling
            if 'municipality' in data.dims:
                available_munis = list(data.municipality.values)
                if municipality_name not in available_munis:
                    logger.warning(f"Municipality '{municipality_name}' not found.")
                    logger.info(f"Available: {available_munis[:5]}...")  # Show first 5
                    municipality_name = available_munis[0]
                    logger.info(f"Using: {municipality_name}")
                
                data = data.sel(municipality=municipality_name)
            
            # Create valid mask with memory-efficient operations
            ndvi_data = data['ndvi']
            
            # Process in chunks to avoid memory issues
            valid_mask = self.create_valid_mask_chunked(ndvi_data)
            
            n_valid = int(valid_mask.sum())
            n_total = valid_mask.size
            
            #logger.info(f"Valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            
            # Extract spatial coordinates
            spatial_coords = self.extract_spatial_coordinates(data)
            
            # Validate sufficient data
            if n_valid < self.params.min_cluster_size:
                logger.error(f"Insufficient valid pixels: {n_valid}")
                return None, None, None
            
            return data, valid_mask, spatial_coords
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None, None
    
    def create_valid_mask_chunked(self, ndvi_data: xr.DataArray) -> np.ndarray:
        """Create valid mask using chunked processing to handle large datasets."""
        
        # Use dask for efficient computation if available
        if hasattr(ndvi_data, 'chunks'):
            valid_mask = (~ndvi_data.isnull()).all(dim='time').compute()
        else:
            # Fallback to numpy for smaller datasets
            valid_mask = ~np.isnan(ndvi_data.values).any(axis=0)
        
        return valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
    
    def extract_spatial_coordinates(self, data: xr.Dataset) -> Dict[str, Any]:
        """Extract spatial coordinate information."""
        
        coords = {}
        
        # Get spatial dimensions (flexible naming)
        spatial_dims = []
        for dim in ['x', 'y', 'longitude', 'latitude', 'lon', 'lat']:
            if dim in data.dims:
                spatial_dims.append(dim)
                coords[dim] = data.coords[dim].values
        
        coords['spatial_dims'] = spatial_dims[:2]  # Use first two found
        
        return coords

    def extract_vegetation_pixels(self, data: xr.Dataset, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract vegetation pixels based on NDVI thresholds and trend filtering."""
        
        ndvi_data = data['ndvi'].values
        
        # Calculate mean NDVI across time for each pixel
        mean_ndvi = np.nanmean(ndvi_data, axis=0)
        
        # Calculate temporal variance for filtering static areas
        ndvi_variance = np.nanvar(ndvi_data, axis=0)
        
        # Create vegetation mask
        vegetation_mask = (
            valid_mask & 
            (mean_ndvi >= self.params.min_vegetation_ndvi) &
            (ndvi_variance >= self.params.ndvi_variance_threshold)
        )
        
        # Apply NDVI trend filtering if specified
        if hasattr(self.params, 'ndvi_trend_filter') and self.params.ndvi_trend_filter is not None:
            logger.info(f"Applying NDVI trend filter: {self.params.ndvi_trend_filter}")
            trend_mask = self._calculate_trend_mask(ndvi_data, self.params.ndvi_trend_filter)
            vegetation_mask = vegetation_mask & trend_mask
            logger.info(f"Pixels after trend filtering: {np.sum(vegetation_mask)}")
        
        # Get coordinates of vegetation pixels
        y_indices, x_indices = np.where(vegetation_mask)
        vegetation_coords = np.column_stack([y_indices, x_indices])
        
        # Extract NDVI time series for vegetation pixels
        vegetation_pixels = ndvi_data[:, vegetation_mask.astype(bool)]
        
        logger.info(f"Found {len(vegetation_coords)} vegetation pixels")
        #logger.info(f"Mean NDVI range: {mean_ndvi[vegetation_mask].min():.3f} - {mean_ndvi[vegetation_mask].max():.3f}")
        
        return vegetation_pixels.T, vegetation_coords  # Transpose for sklearn compatibility
    
    def perform_spatially_constrained_clustering(self, vegetation_pixels: np.ndarray, vegetation_coords: np.ndarray, global_cluster_counter: int = 0) -> List[Dict]:
        """Perform spatially-constrained clustering."""
        
        n_pixels = len(vegetation_pixels)
        # logger.info(f"Clustering {n_pixels} vegetation pixels...")
        
        # Standardize temporal features
        scaler = StandardScaler()
        temporal_features = scaler.fit_transform(vegetation_pixels)
        
        # Normalize spatial coordinates
        spatial_features = vegetation_coords.astype(float)
        spatial_features = (spatial_features - spatial_features.mean(axis=0)) / spatial_features.std(axis=0)
        
        # Combine temporal and spatial features with weighting
        combined_features = np.column_stack([
            temporal_features * self.params.temporal_weight,
            spatial_features * self.params.spatial_weight
        ])
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.params.eps,
            min_samples=self.params.min_samples
        )
        cluster_labels = dbscan.fit_predict(combined_features)
        
        # Filter out noise points (labeled as -1 by DBSCAN)
        noise_mask = cluster_labels != -1
        if np.sum(noise_mask) == 0:
            logger.warning("DBSCAN found no clusters, all points classified as noise")
            return []
        
        # Remove noise points from further processing
        cluster_labels = cluster_labels[noise_mask]
        vegetation_coords = vegetation_coords[noise_mask]
        vegetation_pixels = vegetation_pixels[noise_mask]
        
        logger.info(f"DBSCAN found {len(np.unique(cluster_labels))} clusters, filtered out {np.sum(~noise_mask)} noise points")
        
        # Apply spatial constraints - filter clusters based on spatial distance
        clusters = self.apply_spatial_constraints(
            cluster_labels, vegetation_coords, vegetation_pixels, global_cluster_counter
        )
        
        logger.info(f"Created {len(clusters)} spatially-constrained clusters")
        
        return clusters
    
    def apply_spatial_constraints(self, cluster_labels: np.ndarray, coords: np.ndarray, pixels: np.ndarray, global_cluster_counter: int = 0) -> List[Dict]:
        """Apply spatial distance constraints to clusters."""
                
        clusters = []
        
        for cluster_id in np.unique(cluster_labels):
            # Get pixels in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_pixels = pixels[cluster_mask]
            
            if len(cluster_coords) < self.params.min_cluster_size:
                continue
            
            # Apply spatial constraints to maintain cluster coherence
            if len(cluster_coords) > 1:
                # Calculate pairwise distances
                distances = cdist(cluster_coords, cluster_coords)
                mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                
                # Keep only pixels within reasonable distance from cluster centroid
                centroid = np.mean(cluster_coords, axis=0)
                distances_to_centroid = np.linalg.norm(cluster_coords - centroid, axis=1)
                
                spatial_threshold = min(self.params.max_spatial_distance, mean_distance * 1.5)
                valid_pixels = distances_to_centroid <= spatial_threshold
                
                if valid_pixels.sum() >= self.params.min_cluster_size:
                    clusters.append({
                        'id': global_cluster_counter + len(clusters),
                        'coordinates': cluster_coords[valid_pixels],
                        'ndvi_profiles': cluster_pixels[valid_pixels],
                        'size': valid_pixels.sum(),
                        'centroid': centroid,
                        'mean_ndvi': np.mean(cluster_pixels[valid_pixels]),
                        'temporal_variance': np.var(cluster_pixels[valid_pixels])
                    })
            else:
                # Single pixel cluster - keep it if it meets minimum size
                clusters.append({
                    'id': global_cluster_counter + len(clusters),
                    'coordinates': cluster_coords,
                    'ndvi_profiles': cluster_pixels,
                    'size': len(cluster_coords),
                    'centroid': cluster_coords[0] if len(cluster_coords) > 0 else [0, 0],
                    'mean_ndvi': np.mean(cluster_pixels),
                    'temporal_variance': np.var(cluster_pixels)
                })
        
        return clusters
    
    def create_vegetation_traces(self, clusters: List[Dict]) -> List[Dict]:
        """Create vegetation ST-cubes from clusters."""
        
        vegetation_traces = []
        
        for i, cluster in enumerate(clusters, 1):
            try:
                # Calculate additional statistics
                ndvi_profiles = cluster['ndvi_profiles']
                
                trace = {
                    **cluster,  # Include all cluster info
                    'area': cluster['size'],
                    'mean_temporal_profile': np.mean(ndvi_profiles, axis=0),
                    'std_temporal_profile': np.std(ndvi_profiles, axis=0),
                    'trend_score': self.calculate_trend_score(ndvi_profiles),
                    'vegetation_type': self.classify_vegetation_type(cluster)
                }
                
                vegetation_traces.append(trace)
                
            except Exception as e:
                logger.warning(f"Error creating trace for cluster {cluster['id']}: {e}")
                continue
        
        return vegetation_traces
    
    def calculate_trend_score(self, ndvi_profiles: np.ndarray) -> float:
        """Calculate trend score (positive for greening NDVI, negative for browning)."""
        
        mean_profile = np.mean(ndvi_profiles, axis=0)
        if len(mean_profile) < 3:
            return 0.0
        
        # Simple linear trend using least squares
        x = np.arange(len(mean_profile))
        trend = np.polyfit(x, mean_profile, 1)[0]  # Slope of linear fit
        
        return trend
    
    def classify_vegetation_type(self, cluster: Dict) -> str:
        """Simple vegetation type classification based on NDVI characteristics."""
        
        mean_ndvi = cluster['mean_ndvi']
        variance = cluster['temporal_variance']
        
        if mean_ndvi > 0.7:
            return "Dense Vegetation"
        elif mean_ndvi > 0.5:
            if variance > 0.02:
                return "Seasonal Vegetation"
            else:
                return "Moderate Vegetation"
        else:
            return "Sparse Vegetation"
    
    def create_visualizations(self, vegetation_traces: List[Dict], 
                                      data: xr.Dataset, 
                                      output_dir: str,
                                      municipality_name: str):
        """Create visualizations for the segmentation results."""
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)    
        visualizations_created = {}
            
        # 1. Interactive HTML visualizations
        interactive_viz = InteractiveVisualization(output_directory=str(output_path))
        interactive_files = interactive_viz.create_all_visualizations(
            traces=vegetation_traces,
            data=data,
            municipality_name=municipality_name
        )
        visualizations_created.update(interactive_files)

        # 2. Static publication-ready visualizations
        logger.info("Generating 2D Visualizations...")
        static_viz = StaticVisualization(output_directory=str(output_path))
        static_files = static_viz.create_all_static_visualizations(
            traces=vegetation_traces,
            data=data,
            municipality_name=municipality_name
        )
        visualizations_created.update(static_files)
                        
        return visualizations_created

    def _calculate_trend_mask(self, ndvi_data: np.ndarray, trend_filter: str) -> np.ndarray:
        """Calculate trend mask for filtering pixels by NDVI trend direction."""
        
        logger.info(f"Calculating NDVI trends for filtering...")
        
        trend_mask = np.zeros((ndvi_data.shape[1], ndvi_data.shape[2]), dtype=bool)
        
        # Create time array for regression
        time_array = np.arange(ndvi_data.shape[0])
        
        for i in range(ndvi_data.shape[1]):
            for j in range(ndvi_data.shape[2]):
                pixel_values = ndvi_data[:, i, j]
                
                # Skip pixels with too many NaN values
                valid_mask = ~np.isnan(pixel_values)
                if np.sum(valid_mask) < ndvi_data.shape[0] * 0.5:  # Need at least 50% valid data
                    continue
                
                # Perform linear regression
                valid_time = time_array[valid_mask]
                valid_ndvi = pixel_values[valid_mask]
                
                try:
                    slope, _, _, _, _ = linregress(valid_time, valid_ndvi)
                    
                    # Apply filter based on trend direction
                    if trend_filter == 'greening' and slope > 0:
                        trend_mask[i, j] = True
                    elif trend_filter == 'browning' and slope < 0:
                        trend_mask[i, j] = True
                        
                except Exception as e:
                    # Skip pixels where regression fails
                    continue
        
        greening_count = np.sum(trend_mask)
        total_pixels = ndvi_data.shape[1] * ndvi_data.shape[2]
        logger.info(f"Trend analysis: {greening_count}/{total_pixels} pixels match '{trend_filter}' trend")
        
        return trend_mask


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
            import xarray as xr
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
            min_samples=parameters.min_samples,
            ndvi_variance_threshold=parameters.ndvi_variance_threshold,
            temporal_weight=parameters.temporal_weight,
            spatial_weight=parameters.spatial_weight,
            ndvi_trend_filter=trend
        )
        
        # Create trend-specific output directory
        trend_output_dir = str(Path(timestamped_output_dir) / trend)
        
        # Run segmentation for this trend
        segmenter = VegetationSegmenter(trend_params)
        trend_results = segmenter.segment_vegetation(
            netcdf_path=netcdf_path,
            municipality_name=municipality_name,
            create_visualizations=create_visualizations,
            output_dir=trend_output_dir,
            is_dual_trend_processing=(len(trends_to_process) > 1),
            global_cluster_counter=global_cluster_counter
        )
        
        # Update global counter for next trend
        if trend_results:
            global_cluster_counter += len(trend_results)
        
        results[trend] = trend_results
    
    # Create combined analysis report and visualizations if processing multiple trends
    if len(trends_to_process) > 1:
        # Create combined JSON export first
        if config.enable_json_export and data is not None:
            logger.info("Creating combined JSON export...")
            # Get configuration parameters for export
            config_params = {
                "min_cluster_size": parameters.min_cluster_size,
                "max_spatial_distance": parameters.max_spatial_distance,
                "min_vegetation_ndvi": parameters.min_vegetation_ndvi,
                "eps": parameters.eps,
                "min_samples": parameters.min_samples,
                "ndvi_variance_threshold": parameters.ndvi_variance_threshold,
                "temporal_weight": parameters.temporal_weight,
                "netcdf_path": netcdf_path,
                "municipality_name": municipality_name
            }
            
            # Export combined results to main output directory
            json_exporter = VegetationClusterJSONExporter()
            json_exporter.export_combined_clusters_to_json(
                results, data, timestamped_output_dir, municipality_name, config_params
            )
        
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
    else:
        # For single trend processing, still create combined JSON export for consistency
        if config.enable_json_export and data is not None:
            logger.info("Creating combined JSON export for single trend...")
            # Get configuration parameters for export
            config_params = {
                "min_cluster_size": parameters.min_cluster_size,
                "max_spatial_distance": parameters.max_spatial_distance,
                "min_vegetation_ndvi": parameters.min_vegetation_ndvi,
                "eps": parameters.eps,
                "min_samples": parameters.min_samples,
                "ndvi_variance_threshold": parameters.ndvi_variance_threshold,
                "temporal_weight": parameters.temporal_weight,
                "netcdf_path": netcdf_path,
                "municipality_name": municipality_name
            }
            
            # Export combined results to main output directory
            json_exporter = VegetationClusterJSONExporter()
            json_exporter.export_combined_clusters_to_json(
                results, data, timestamped_output_dir, municipality_name, config_params
            )
    
    return results


# Example usage
if __name__ == "__main__":    
    config = get_config()
    
    # Use config defaults (no parameter overrides) to respect YAML configuration
    params = VegetationSegmentationParameters()  # Uses all config defaults
    
    # Run segmentation using config defaults for paths and municipality
    results = segment_vegetation(
        parameters=params,
        create_visualizations=True
    )
    
    # Handle the simplified return format - always contains trend keys
    total_clusters = sum(len(clusters) for clusters in results.values())
    if total_clusters == 0:
        logger.warning("No vegetation clusters found. Check your data and parameters.")
    else:
        trend_summary = " and ".join([f"{len(clusters)} {trend}" for trend, clusters in results.items()])
        logger.success(f"Found {trend_summary} clusters (total: {total_clusters}).")