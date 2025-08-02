"""
Main Pipeline for Vegetation-Focused ST-Cube Segmentation

Implements the main workflow for NDVI-based spatiotemporal cube segmentation, including data loading, preprocessing, clustering, 
spatial bridging, and result export/visualization. Entry point for running the segmentation pipeline with configurable parameters.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import warnings
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import gc
from loguru import logger
from .config_loader import get_config, get_section
from .json_exporter import VegetationClusterJSONExporter
import datetime
from .spatial_bridging import apply_spatial_bridging_to_clusters, BridgingParameters
from .visualization.interactive import InteractiveVisualization
from .visualization.static import StaticVisualization

warnings.filterwarnings('ignore')

@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation segmentation."""
    max_spatial_distance: int = None
    min_vegetation_ndvi: float = None
    min_cube_size: int = None
    ndvi_variance_threshold: float = None
    chunk_size: int = None
    n_clusters: int = None
    temporal_weight: float = None
    
    def __post_init__(self):
        # Load config and set defaults if not provided
        config = get_config()
        
        if self.max_spatial_distance is None:
            self.max_spatial_distance = config.max_spatial_distance
        if self.min_vegetation_ndvi is None:
            self.min_vegetation_ndvi = config.min_vegetation_ndvi
        if self.min_cube_size is None:
            self.min_cube_size = config.min_cube_size
        if self.ndvi_variance_threshold is None:
            self.ndvi_variance_threshold = config.ndvi_variance_threshold
        if self.chunk_size is None:
            self.chunk_size = config.chunk_size
        if self.n_clusters is None:
            self.n_clusters = config.n_clusters
        if self.temporal_weight is None:
            self.temporal_weight = config.temporal_weight


class VegetationSegmenter:
    """Vegetation segmentation with memory-efficient processing."""
    
    def __init__(self, parameters: VegetationSegmentationParameters):
        self.params = parameters
    
    def segment_vegetation(self, netcdf_path: str, 
                          municipality_name,
                          create_visualizations: bool = True,
                          output_dir: str = "outputs/vegetation_clustering") -> List[Dict[str, Any]]:
        """
        Run vegetation-focused NDVI clustering segmentation.
        
        Returns:
            List of vegetation cluster dictionaries with spatial and temporal info
        """
        # Add datetime subfolder inside the given output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(output_dir) / timestamp)

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

            if len(vegetation_pixels) < self.params.min_cube_size:
                logger.warning(f"Insufficient vegetation pixels: {len(vegetation_pixels)}")
                return []
            
            # Step 3: Perform clustering
            logger.info("3. Performing spatially-constrained clustering...")
            clusters = self.perform_spatially_constrained_clustering(vegetation_pixels, vegetation_coords)
            
            # Step 4: Create vegetation cubes
            logger.info("4. Creating vegetation ST-cubes...")
            vegetation_cubes = self.create_vegetation_cubes(clusters)
            
            # Step 5: Export cluster data to JSON
            if vegetation_cubes:
                config = get_config()
                if config.enable_json_export:
                    logger.info("5. Exporting cluster data to JSON...")
                    # Get configuration parameters for export
                    config_params = {
                        "min_cube_size": self.params.min_cube_size,
                        "max_spatial_distance": self.params.max_spatial_distance,
                        "min_vegetation_ndvi": self.params.min_vegetation_ndvi,
                        "n_clusters": self.params.n_clusters,
                        "ndvi_variance_threshold": self.params.ndvi_variance_threshold,
                        "temporal_weight": self.params.temporal_weight,
                        "netcdf_path": netcdf_path,
                        "municipality_name": municipality_name
                    }
                    
                    # Use the dedicated JSON exporter
                    json_exporter = VegetationClusterJSONExporter()
                    json_exporter.export_clusters_to_json(
                        vegetation_cubes, data, output_dir, municipality_name, config_params
                    )
                else:
                    logger.info("5. JSON export disabled in configuration")
            
            # Step 6: Generate visualizations if requested
            if create_visualizations and vegetation_cubes:
                logger.info("6. Creating visualizations...")
                self.create_visualizations(
                    vegetation_cubes, data, output_dir, municipality_name
                )
            
            return vegetation_cubes
            
        except Exception as e:
            logger.error(f"Error during vegetation segmentation: {str(e)}")
            import traceback
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
            if n_valid < self.params.min_cube_size:
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
        """Extract vegetation pixels based on NDVI thresholds."""
        
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
        
        # Get coordinates of vegetation pixels
        y_indices, x_indices = np.where(vegetation_mask)
        vegetation_coords = np.column_stack([y_indices, x_indices])
        
        # Extract NDVI time series for vegetation pixels
        vegetation_pixels = ndvi_data[:, vegetation_mask.astype(bool)]
        
        logger.info(f"Found {len(vegetation_coords)} vegetation pixels")
        #logger.info(f"Mean NDVI range: {mean_ndvi[vegetation_mask].min():.3f} - {mean_ndvi[vegetation_mask].max():.3f}")
        
        return vegetation_pixels.T, vegetation_coords  # Transpose for sklearn compatibility
    
    def perform_spatially_constrained_clustering(self, vegetation_pixels: np.ndarray, vegetation_coords: np.ndarray) -> List[Dict]:
        """Perform spatially-constrained clustering."""
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from scipy.spatial.distance import pdist, squareform
        except ImportError:
            logger.error("Required packages not available. Install scikit-learn and scipy.")
            return []
        
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
            spatial_features * (1 - self.params.temporal_weight)
        ])
        
        # Perform k-means clustering
        n_clusters = min(self.params.n_clusters, n_pixels // self.params.min_cube_size)
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_features)
        
        # Apply spatial bridging if enabled
        try:
            bridging_cfg = get_section('bridging')
            
            enable_bridging = bridging_cfg.get('enable_spatial_bridging', False) if bridging_cfg else False
            
            if enable_bridging:
                logger.info("Applying spatial bridging...")
                
                # Convert coordinates to list of tuples format expected by bridging
                pixel_coords_list = [(int(coord[0]), int(coord[1])) for coord in vegetation_coords]
                
                # Create bridging parameters from config
                bridging_params = BridgingParameters(
                    bridge_similarity_tolerance=bridging_cfg.get('bridge_similarity_tolerance', 0.1),
                    max_bridge_gap=bridging_cfg.get('max_bridge_gap', 2),
                    min_bridge_density=bridging_cfg.get('min_bridge_density', 0.7),
                    connectivity_radius=bridging_cfg.get('connectivity_radius', 3),
                    max_bridge_length=bridging_cfg.get('max_bridge_length', 20),
                    min_cluster_size_for_bridging=bridging_cfg.get('min_cluster_size_for_bridging', 5)
                )
                
                # Log initial cluster count
                initial_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
                logger.info(f"Initial clusters before bridging: {initial_clusters}")
                
                # Apply spatial bridging
                cluster_labels = apply_spatial_bridging_to_clusters(
                    cluster_labels, pixel_coords_list, vegetation_pixels, bridging_params
                )
                
                # Log final cluster count
                final_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
                logger.info(f"Final clusters after bridging: {final_clusters} (merged {initial_clusters - final_clusters} clusters)")
                
        except Exception as e:
            logger.warning(f"Spatial bridging failed: {e}, continuing without bridging")
            import traceback
            traceback.print_exc()
        
        # Apply spatial constraints - filter clusters based on spatial distance
        clusters = self.apply_spatial_constraints(
            cluster_labels, vegetation_coords, vegetation_pixels
        )
        
        logger.info(f"Created {len(clusters)} spatially-constrained clusters")
        
        return clusters
    
    def apply_spatial_constraints(self, cluster_labels: np.ndarray, coords: np.ndarray, pixels: np.ndarray) -> List[Dict]:
        """Apply spatial distance constraints to clusters."""
        
        from scipy.spatial.distance import cdist
        
        clusters = []
        
        for cluster_id in np.unique(cluster_labels):
            # Get pixels in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_pixels = pixels[cluster_mask]
            
            if len(cluster_coords) < self.params.min_cube_size:
                continue
            
            # For bridged clusters, be more lenient with spatial constraints
            # since spatial bridging already ensures connectivity through similar pixels
            try:
                from .config_loader import get_section
                bridging_cfg = get_section('bridging')
                bridging_enabled = bridging_cfg.get('enable_spatial_bridging', False) if bridging_cfg else False
            except:
                bridging_enabled = False
            
            if bridging_enabled:
                # Less aggressive spatial filtering for bridged clusters
                # Only remove extreme outliers, not moderate distances
                if len(cluster_coords) > 1:
                    centroid = np.mean(cluster_coords, axis=0)
                    distances_to_centroid = np.linalg.norm(cluster_coords - centroid, axis=1)
                    
                    # Use a more lenient threshold for bridged clusters
                    spatial_threshold = self.params.max_spatial_distance * 3  # 3x more lenient
                    valid_pixels = distances_to_centroid <= spatial_threshold
                    
                    if valid_pixels.sum() >= self.params.min_cube_size:
                        clusters.append({
                            'id': len(clusters),
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
                        'id': len(clusters),
                        'coordinates': cluster_coords,
                        'ndvi_profiles': cluster_pixels,
                        'size': len(cluster_coords),
                        'centroid': cluster_coords[0] if len(cluster_coords) > 0 else [0, 0],
                        'mean_ndvi': np.mean(cluster_pixels),
                        'temporal_variance': np.var(cluster_pixels)
                    })
            else:
                # Original strict spatial filtering for non-bridged clusters
                if len(cluster_coords) > 1:
                    # Calculate pairwise distances
                    distances = cdist(cluster_coords, cluster_coords)
                    mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                    
                    # Keep only pixels within reasonable distance from cluster centroid
                    centroid = np.mean(cluster_coords, axis=0)
                    distances_to_centroid = np.linalg.norm(cluster_coords - centroid, axis=1)
                    
                    spatial_threshold = min(self.params.max_spatial_distance, mean_distance * 1.5)
                    valid_pixels = distances_to_centroid <= spatial_threshold
                    
                    if valid_pixels.sum() >= self.params.min_cube_size:
                        clusters.append({
                            'id': len(clusters),
                            'coordinates': cluster_coords[valid_pixels],
                            'ndvi_profiles': cluster_pixels[valid_pixels],
                            'size': valid_pixels.sum(),
                            'centroid': centroid,
                            'mean_ndvi': np.mean(cluster_pixels[valid_pixels]),
                            'temporal_variance': np.var(cluster_pixels[valid_pixels])
                        })
        
        return clusters
    
    def create_vegetation_cubes(self, clusters: List[Dict]) -> List[Dict]:
        """Create vegetation ST-cubes from clusters."""
        
        vegetation_cubes = []
        
        for i, cluster in enumerate(clusters, 1):
            try:
                # Calculate additional statistics
                ndvi_profiles = cluster['ndvi_profiles']
                
                cube = {
                    **cluster,  # Include all cluster info
                    'area': cluster['size'],
                    'mean_temporal_profile': np.mean(ndvi_profiles, axis=0),
                    'std_temporal_profile': np.std(ndvi_profiles, axis=0),
                    'trend_score': self.calculate_trend_score(ndvi_profiles),
                    'vegetation_type': self.classify_vegetation_type(cluster)
                }
                
                vegetation_cubes.append(cube)
                
            except Exception as e:
                logger.warning(f"Error creating cube for cluster {cluster['id']}: {e}")
                continue
        
        return vegetation_cubes
    
    def calculate_trend_score(self, ndvi_profiles: np.ndarray) -> float:
        """Calculate trend score (positive for increasing NDVI, negative for decreasing)."""
        
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
    
    def create_visualizations(self, vegetation_cubes: List[Dict], 
                                      data: xr.Dataset, 
                                      output_dir: str,
                                      municipality_name: str):
        """Create comprehensive visualizations using both interactive and static modules."""
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
            
        # Create all visualizations
        visualizations_created = {}
            
        # 1. Interactive HTML visualizations
        interactive_viz = InteractiveVisualization(output_directory=str(output_path / "interactive"))
        interactive_files = interactive_viz.create_all_visualizations(
            cubes=vegetation_cubes,
            data=data,
            municipality_name=municipality_name
        )
        visualizations_created.update(interactive_files)
        
        # 2. Static publication-ready visualizations
        logger.info("Generating 2D Visualizations...")
        static_viz = StaticVisualization(output_directory=str(output_path / "static"))
        static_files = static_viz.create_all_static_visualizations(
            cubes=vegetation_cubes,
            data=data,
            municipality_name=municipality_name
        )
        visualizations_created.update(static_files)
            
        logger.info(f"Output saved to: {output_path}")
            
        return visualizations_created


def segment_vegetation(netcdf_path: str = None, 
                               parameters: Optional[VegetationSegmentationParameters] = None,
                               municipality_name: str = None,
                               create_visualizations: bool = True,
                               output_dir: str = None) -> List[Dict]:
    """
    Vegetation segmentation function with improved performance and memory usage.
    """
    config = get_config()
    
    # Use config defaults if not provided
    if netcdf_path is None:
        netcdf_path = config.default_netcdf_path
    if municipality_name is None:
        municipality_name = config.default_municipality
    if output_dir is None:
        output_dir = config.default_output_dir
    
    if parameters is None:
        parameters = VegetationSegmentationParameters()
    
    segmenter = VegetationSegmenter(parameters)
    
    return segmenter.segment_vegetation(
        netcdf_path=netcdf_path,
        municipality_name=municipality_name,
        create_visualizations=create_visualizations,
        output_dir=output_dir
    )


# Example usage
if __name__ == "__main__":    
    config = get_config()
    
    # Use config defaults (no parameter overrides) to respect YAML configuration
    params = VegetationSegmentationParameters()  # Uses all config defaults
    
    # Run segmentation using config defaults for paths and municipality
    vegetation_cubes = segment_vegetation(
        parameters=params,
        create_visualizations=True
    )
    if len(vegetation_cubes) == 0:
        logger.warning("No vegetation clusters found. Check your data and parameters.")
    else:
        logger.success(f"Completed! Found {len(vegetation_cubes)} vegetation clusters.")