"""
Vegetation Segmentation Engine

Core implementation of NDVI-based spatiotemporal trace segmentation with clustering calculations, data processing, and spatial constraint algorithms.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import warnings
from typing import List, Optional, Tuple, Dict, Any
import gc
from loguru import logger
import traceback
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from .config_loader import get_config
from .visualization.visualization_3d import InteractiveVisualization
from .visualization.visualization_2d import StaticVisualization

warnings.filterwarnings('ignore')


class VegetationSegmenter:
    """Vegetation segmentation with memory-efficient processing."""
    
    def __init__(self, parameters):
        self.params = parameters
        self.config = get_config()
    
    def segment_vegetation(self, netcdf_path: str, 
                          municipality_name,
                          create_visualizations: bool = True,
                          output_dir: str = "outputs/vegetation_clustering",
                          global_cluster_counter: int = 0
                          ) -> List[Dict[str, Any]]:
        """
        Run vegetation-focused NDVI clustering segmentation.
        
        Args:
            netcdf_path: Path to the input NetCDF file.
            municipality_name: Name of the municipality for analysis.
            create_visualizations: If True, generates visualizations.
            output_dir: Directory for saving output files.
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
            
            # Step 2: Extract vegetation traces efficiently
            logger.info("2. Extracting vegetation traces...")
            vegetation_traces, vegetation_coords = self.extract_vegetation_traces(data, valid_mask)

            if len(vegetation_traces) < self.params.min_cluster_size:
                logger.warning(f"Insufficient vegetation traces: {len(vegetation_traces)}")
                return []
            
            # Step 3: Perform clustering
            logger.info("3. Performing spatially-constrained clustering...")
            clusters = self.perform_spatially_constrained_clustering(vegetation_traces, vegetation_coords, global_cluster_counter)
            
            # Step 4: Create vegetation traces
            logger.info("4. Creating vegetation ST-cubes...")
            vegetation_traces = self.create_vegetation_traces(clusters)
            
            # Step 5: Generate visualizations
            if create_visualizations and vegetation_traces:
                logger.info("5. Creating visualizations...")
                self.create_visualizations(
                    vegetation_traces, data, output_dir, municipality_name
                )
            return vegetation_traces
            
        except Exception as e:
            logger.error(f"Error during vegetation segmentation: {str(e)}")
            traceback.print_exc()
            return []
        finally:
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
                    logger.info(f"Available: {available_munis[:5]}...")
                    municipality_name = available_munis[0]
                    logger.info(f"Using: {municipality_name}")
                
                data = data.sel(municipality=municipality_name)
            
            ndvi_data = data['ndvi']
            valid_mask = self.create_valid_mask_chunked(ndvi_data)
            n_valid = int(valid_mask.sum())            
            #logger.info(f"Valid traces: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            
            # Extract spatial coordinates
            spatial_coords = self.extract_spatial_coordinates(data)
            
            # Validate sufficient data
            if n_valid < self.params.min_cluster_size:
                logger.error(f"Insufficient valid traces: {n_valid}")
                return None, None, None
            
            return data, valid_mask, spatial_coords
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None, None
    
    def create_valid_mask_chunked(self, ndvi_data: xr.DataArray) -> np.ndarray:
        """Create valid mask using chunked processing to handle large datasets."""
        
        # Use dask for efficient computation
        if hasattr(ndvi_data, 'chunks'):
            valid_mask = (~ndvi_data.isnull()).all(dim='time').compute()
        else:
            valid_mask = ~np.isnan(ndvi_data.values).any(axis=0)
        
        return valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
    
    def extract_spatial_coordinates(self, data: xr.Dataset) -> Dict[str, Any]:
        """Extract spatial coordinate information."""
        
        coords = {}
        
        # Get spatial dimensions
        spatial_dims = []
        for dim in ['x', 'y', 'longitude', 'latitude', 'lon', 'lat']:
            if dim in data.dims:
                spatial_dims.append(dim)
                coords[dim] = data.coords[dim].values
        
        coords['spatial_dims'] = spatial_dims[:2]
        
        return coords

    def extract_vegetation_traces(self, data: xr.Dataset, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract vegetation traces based on NDVI thresholds and trend filtering."""
        
        ndvi_data = data['ndvi'].values
        
        # Calculate mean NDVI across time for each trace (spatial location)
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
            logger.info(f"Traces after trend filtering: {np.sum(vegetation_mask)}")
        
        # Get coordinates of vegetation traces (spatial locations)
        y_indices, x_indices = np.where(vegetation_mask)
        vegetation_coords = np.column_stack([y_indices, x_indices])
        
        # Extract NDVI time series for vegetation traces
        vegetation_traces = ndvi_data[:, vegetation_mask.astype(bool)]
        
        logger.info(f"Found {len(vegetation_coords)} vegetation traces")
        #logger.info(f"Mean NDVI range: {mean_ndvi[vegetation_mask].min():.3f} - {mean_ndvi[vegetation_mask].max():.3f}")
        
        return vegetation_traces.T, vegetation_coords  # Transpose for sklearn compatibility
    
    def perform_spatially_constrained_clustering(self, vegetation_traces: np.ndarray, vegetation_coords: np.ndarray, global_cluster_counter: int = 0) -> List[Dict]:
        """Perform spatially-constrained clustering."""
        
        # n_traces = len(vegetation_traces)
        # logger.info(f"Clustering {n_traces} vegetation traces...")
        
        # Standardize temporal features
        scaler = StandardScaler()
        temporal_features = scaler.fit_transform(vegetation_traces)
        
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
            min_samples=self.params.min_pts
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
        vegetation_traces = vegetation_traces[noise_mask]
        
        logger.info(f"DBSCAN found {len(np.unique(cluster_labels))} clusters, filtered out {np.sum(~noise_mask)} noise points")
        
        # Apply spatial constraints - filter clusters based on spatial distance
        clusters = self.apply_spatial_constraints(
            cluster_labels, vegetation_coords, vegetation_traces, global_cluster_counter
        )
        
        logger.info(f"Created {len(clusters)} spatially-constrained clusters")
        
        return clusters
    
    def apply_spatial_constraints(self, cluster_labels: np.ndarray, coords: np.ndarray, traces: np.ndarray, global_cluster_counter: int = 0) -> List[Dict]:
        """Apply spatial distance constraints to clusters."""
                
        clusters = []
        
        for cluster_id in np.unique(cluster_labels):
            # Get traces in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_traces = traces[cluster_mask]
            
            if len(cluster_coords) < self.params.min_cluster_size:
                continue
            
            # Apply spatial constraints to maintain cluster coherence
            if len(cluster_coords) > 1:
                # Calculate pairwise distances
                distances = cdist(cluster_coords, cluster_coords)
                mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
                
                # Keep only traces within reasonable distance from cluster centroid
                centroid = np.mean(cluster_coords, axis=0)
                distances_to_centroid = np.linalg.norm(cluster_coords - centroid, axis=1)
                
                spatial_threshold = min(self.params.max_spatial_distance, mean_distance * 1.5)
                valid_traces = distances_to_centroid <= spatial_threshold
                
                if valid_traces.sum() >= self.params.min_cluster_size:
                    clusters.append({
                        'id': global_cluster_counter + len(clusters),
                        'coordinates': cluster_coords[valid_traces],
                        'ndvi_profiles': cluster_traces[valid_traces],
                        'size': valid_traces.sum(),
                        'centroid': centroid,
                        'mean_ndvi': np.mean(cluster_traces[valid_traces]),
                        'temporal_variance': np.var(cluster_traces[valid_traces])
                    })
            else:
                # Single trace cluster - keep it if it meets minimum size
                clusters.append({
                    'id': global_cluster_counter + len(clusters),
                    'coordinates': cluster_coords,
                    'ndvi_profiles': cluster_traces,
                    'size': len(cluster_coords),
                    'centroid': cluster_coords[0] if len(cluster_coords) > 0 else [0, 0],
                    'mean_ndvi': np.mean(cluster_traces),
                    'temporal_variance': np.var(cluster_traces)
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
                    'trend_score': self.calculate_trend_score(ndvi_profiles)
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
        """Calculate trend mask for filtering traces by NDVI trend direction."""
        
        logger.info(f"Calculating NDVI trends for filtering...")
        
        trend_mask = np.zeros((ndvi_data.shape[1], ndvi_data.shape[2]), dtype=bool)
        
        # Create time array for regression
        time_array = np.arange(ndvi_data.shape[0])
        
        for i in range(ndvi_data.shape[1]):
            for j in range(ndvi_data.shape[2]):
                trace_values = ndvi_data[:, i, j]
                
                # Skip traces with too many NaN values
                valid_mask = ~np.isnan(trace_values)
                if np.sum(valid_mask) < ndvi_data.shape[0] * 0.5:  # Need at least 50% valid data
                    continue
                
                # Perform linear regression
                valid_time = time_array[valid_mask]
                valid_ndvi = trace_values[valid_mask]
                
                try:
                    slope, _, _, _, _ = linregress(valid_time, valid_ndvi)
                    
                    # Apply filter based on trend direction
                    if trend_filter == 'greening' and slope > 0:
                        trend_mask[i, j] = True
                    elif trend_filter == 'browning' and slope < 0:
                        trend_mask[i, j] = True
                        
                except Exception as e:
                    # Skip traces where regression fails
                    continue
        
        greening_count = np.sum(trend_mask)
        total_traces = ndvi_data.shape[1] * ndvi_data.shape[2]
        logger.info(f"Trend analysis: {greening_count}/{total_traces} traces match '{trend_filter}' trend")
        
        return trend_mask