"""
Vegetation-focused ST-Cube Segmentation Main Script

This script provides vegetation-specific NDVI clustering segmentation
with local spatial constraints for analyzing vegetation patterns.
"""

# ==== CONFIGURABLE PARAMETERS ====
DEFAULT_MAX_SPATIAL_DISTANCE = 10          # Maximum spatial distance for clustering
DEFAULT_MIN_VEGETATION_NDVI = 0.4           # Minimum NDVI for vegetation
DEFAULT_MIN_CUBE_SIZE = 20                  # Minimum pixels per cube
DEFAULT_NDVI_VARIANCE_THRESHOLD = 0.01      # Filter out static areas
DEFAULT_CHUNK_SIZE = 1000                   # For memory-efficient processing
DEFAULT_N_CLUSTERS = 10                     # Number of temporal clusters
DEFAULT_TEMPORAL_WEIGHT = 0.7               # Weight for temporal vs spatial similarity
# ================================

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import warnings
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import gc
import logging
from loguru import logger

warnings.filterwarnings('ignore')

@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation segmentation."""
    max_spatial_distance: int = DEFAULT_MAX_SPATIAL_DISTANCE
    min_vegetation_ndvi: float = DEFAULT_MIN_VEGETATION_NDVI
    min_cube_size: int = DEFAULT_MIN_CUBE_SIZE
    ndvi_variance_threshold: float = DEFAULT_NDVI_VARIANCE_THRESHOLD  # Filter out static areas
    chunk_size: int = DEFAULT_CHUNK_SIZE  # For memory-efficient processing
    n_clusters: int = DEFAULT_N_CLUSTERS  # Number of temporal clusters
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT  # Weight for temporal vs spatial similarity


class VegetationSegmenter:
    """Vegetation segmentation with memory-efficient processing."""
    
    def __init__(self, parameters: VegetationSegmentationParameters):
        self.params = parameters
        self.logger = logging.getLogger(__name__)
    
    def segment_vegetation(self, netcdf_path: str, 
                          municipality_name: str = "Sant Martí",
                          create_visualizations: bool = True,
                          output_dir: str = "outputs/vegetation_clustering") -> List[Dict[str, Any]]:
        """
        Run vegetation-focused NDVI clustering segmentation.
        
        Returns:
            List of vegetation cluster dictionaries with spatial and temporal info
        """
        
        self.logger.info(f"=== Starting Vegetation NDVI Clustering Segmentation ===")
        self.logger.info(f"Data: {netcdf_path}")
        self.logger.info(f"Municipality: {municipality_name}")
        
        try:
            # Step 1: Load and validate data with lazy loading
            self.logger.info("1. Loading and validating data...")
            data, valid_mask, spatial_coords = self.load_and_prepare_data(
                netcdf_path, municipality_name
            )
            
            if data is None:
                self.logger.error("Failed to load data")
                return []
            
            # Step 2: Extract vegetation pixels efficiently
            self.logger.info("2. Extracting vegetation pixels...")
            vegetation_pixels, vegetation_coords = self.extract_vegetation_pixels(
                data, valid_mask, spatial_coords
            )
            
            if len(vegetation_pixels) < self.params.min_cube_size:
                self.logger.warning(f"Insufficient vegetation pixels: {len(vegetation_pixels)}")
                return []
            
            # Step 3: Perform clustering
            self.logger.info("3. Performing spatially-constrained clustering...")
            clusters = self.perform_spatially_constrained_clustering(
                vegetation_pixels, vegetation_coords
            )
            
            # Step 4: Create vegetation cubes
            self.logger.info("4. Creating vegetation ST-cubes...")
            vegetation_cubes = self.create_vegetation_cubes(clusters, data, spatial_coords)
            
            # Step 5: Generate visualizations if requested
            if create_visualizations and vegetation_cubes:
                self.logger.info("5. Creating visualizations...")
                self.create_visualizations(
                    vegetation_cubes, data, output_dir, municipality_name
                )
            
            self.logger.info(f"=== Segmentation completed: {len(vegetation_cubes)} clusters ===")
            return vegetation_cubes
            
        except Exception as e:
            self.logger.error(f"Error during vegetation segmentation: {str(e)}")
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
            self.logger.info(f"Loaded dataset with shape: {dict(data.dims)}")
            
            # Validate required variables
            required_vars = ['ndvi']
            missing_vars = [var for var in required_vars if var not in data.variables]
            if missing_vars:
                self.logger.error(f"Missing required variables: {missing_vars}")
                return None, None, None
            
            # Filter by municipality with better error handling
            if 'municipality' in data.dims:
                available_munis = list(data.municipality.values)
                if municipality_name not in available_munis:
                    self.logger.warning(f"Municipality '{municipality_name}' not found.")
                    self.logger.info(f"Available: {available_munis[:5]}...")  # Show first 5
                    municipality_name = available_munis[0]
                    self.logger.info(f"Using: {municipality_name}")
                
                data = data.sel(municipality=municipality_name)
            
            # Create valid mask with memory-efficient operations
            ndvi_data = data['ndvi']
            
            # Process in chunks to avoid memory issues
            valid_mask = self.create_valid_mask_chunked(ndvi_data)
            
            n_valid = int(valid_mask.sum())
            n_total = valid_mask.size
            
            self.logger.info(f"Valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            
            # Extract spatial coordinates
            spatial_coords = self.extract_spatial_coordinates(data)
            
            # Validate sufficient data
            if n_valid < self.params.min_cube_size:
                self.logger.error(f"Insufficient valid pixels: {n_valid}")
                return None, None, None
            
            return data, valid_mask, spatial_coords
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
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
    
    def extract_vegetation_pixels(self, data: xr.Dataset, 
                                valid_mask: np.ndarray, 
                                spatial_coords: Dict) -> Tuple[np.ndarray, np.ndarray]:
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
        
        self.logger.info(f"Found {len(vegetation_coords)} vegetation pixels")
        self.logger.info(f"Mean NDVI range: {mean_ndvi[vegetation_mask].min():.3f} - {mean_ndvi[vegetation_mask].max():.3f}")
        
        return vegetation_pixels.T, vegetation_coords  # Transpose for sklearn compatibility
    
    def perform_spatially_constrained_clustering(self, vegetation_pixels: np.ndarray, 
                                               vegetation_coords: np.ndarray) -> List[Dict]:
        """Perform spatially-constrained clustering."""
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from scipy.spatial.distance import pdist, squareform
        except ImportError:
            self.logger.error("Required packages not available. Install scikit-learn and scipy.")
            return []
        
        n_pixels = len(vegetation_pixels)
        self.logger.info(f"Clustering {n_pixels} vegetation pixels...")
        
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
        
        # Apply spatial constraints - filter clusters based on spatial distance
        clusters = self.apply_spatial_constraints(
            cluster_labels, vegetation_coords, vegetation_pixels
        )
        
        self.logger.info(f"Created {len(clusters)} spatially-constrained clusters")
        
        return clusters
    
    def apply_spatial_constraints(self, cluster_labels: np.ndarray, 
                                coords: np.ndarray, 
                                pixels: np.ndarray) -> List[Dict]:
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
            
            # Check spatial coherence - remove outliers
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
    
    def create_vegetation_cubes(self, clusters: List[Dict], 
                              data: xr.Dataset, 
                              spatial_coords: Dict) -> List[Dict]:
        """Create vegetation ST-cubes from clusters."""
        
        vegetation_cubes = []
        
        for cluster in tqdm(clusters, desc="Creating ST-cubes"):
            try:
                # Calculate additional statistics
                ndvi_profiles = cluster['ndvi_profiles']
                
                cube = {
                    **cluster,  # Include all cluster info
                    'area': cluster['size'],
                    'mean_temporal_profile': np.mean(ndvi_profiles, axis=0),
                    'std_temporal_profile': np.std(ndvi_profiles, axis=0),
                    'seasonality_score': self.calculate_seasonality_score(ndvi_profiles),
                    'trend_score': self.calculate_trend_score(ndvi_profiles),
                    'vegetation_type': self.classify_vegetation_type(cluster)
                }
                
                vegetation_cubes.append(cube)
                
            except Exception as e:
                self.logger.warning(f"Error creating cube for cluster {cluster['id']}: {e}")
                continue
        
        return vegetation_cubes
    
    def calculate_seasonality_score(self, ndvi_profiles: np.ndarray) -> float:
        """Calculate seasonality score based on NDVI variation patterns."""
        
        # Simple seasonality metric based on autocorrelation
        mean_profile = np.mean(ndvi_profiles, axis=0)
        if len(mean_profile) < 4:
            return 0.0
        
        # Calculate coefficient of variation as proxy for seasonality
        return np.std(mean_profile) / (np.mean(mean_profile) + 1e-6)
    
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
        
        try:
            # Import visualization modules
            try:
                from .interactive_visualization import InteractiveVisualization
                from .static_visualization import StaticVisualization
            except ImportError:
                # Try direct imports if relative imports fail
                import sys
                import os
                
                # Add current directory to path
                current_dir = Path(__file__).parent
                sys.path.insert(0, str(current_dir))
                
                from interactive_visualization import InteractiveVisualization
                from static_visualization import StaticVisualization
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create all visualizations
            visualizations_created = {}
            
            # 1. Interactive HTML visualizations
            self.logger.info("Creating interactive visualizations...")
            interactive_viz = InteractiveVisualization(output_directory=str(output_path / "interactive"))
            interactive_files = interactive_viz.create_all_visualizations(
                cubes=vegetation_cubes,
                data=data,
                municipality_name=municipality_name
            )
            visualizations_created.update(interactive_files)
            
            # 2. Static publication-ready visualizations
            self.logger.info("Creating static visualizations...")
            static_viz = StaticVisualization(output_directory=str(output_path / "static"))
            static_files = static_viz.create_all_static_visualizations(
                cubes=vegetation_cubes,
                data=data,
                municipality_name=municipality_name
            )
            visualizations_created.update(static_files)
            
            # 3. Legacy summary plots (for backwards compatibility)
            self.logger.info("Creating summary plots...")
            self.create_summary_plots(vegetation_cubes, output_path, municipality_name)
            
            self.logger.info(f"All visualizations created successfully!")
            self.logger.info(f"Interactive visualizations: {output_path / 'interactive'}")
            self.logger.info(f"Static visualizations: {output_path / 'static'}")
            self.logger.info(f"Total files created: {len(visualizations_created)}")
            
            return visualizations_created
            
        except ImportError as e:
            self.logger.warning(f"Visualization libraries not available: {str(e)}. Using basic plots.")
            # Fallback to basic visualization
            self.create_summary_plots(vegetation_cubes, Path(output_dir), municipality_name)
        except Exception as e:
            self.logger.warning(f"Failed to create comprehensive visualizations: {str(e)}")
            self.logger.info("Falling back to basic summary plots...")
            try:
                self.create_summary_plots(vegetation_cubes, Path(output_dir), municipality_name)
            except Exception as fallback_error:
                self.logger.error(f"Even basic visualizations failed: {str(fallback_error)}")
            import traceback
            traceback.print_exc()
    
    def create_summary_plots(self, vegetation_cubes: List[Dict], 
                           output_path: Path, 
                           municipality_name: str):
        """Create summary plots for vegetation analysis."""
        
        import matplotlib.pyplot as plt
        
        # Plot 1: Cluster size distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        sizes = [cube['area'] for cube in vegetation_cubes]
        ax1.hist(sizes, bins=20, alpha=0.7)
        ax1.set_xlabel('Cluster Size (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Vegetation Cluster Size Distribution')
        
        # Plot 2: NDVI distribution
        mean_ndvis = [cube['mean_ndvi'] for cube in vegetation_cubes]
        ax2.hist(mean_ndvis, bins=20, alpha=0.7, color='green')
        ax2.set_xlabel('Mean NDVI')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Mean NDVI Distribution')
        
        # Plot 3: Seasonality vs Mean NDVI
        seasonality = [cube['seasonality_score'] for cube in vegetation_cubes]
        ax3.scatter(mean_ndvis, seasonality, alpha=0.6)
        ax3.set_xlabel('Mean NDVI')
        ax3.set_ylabel('Seasonality Score')
        ax3.set_title('Seasonality vs NDVI')
        
        # Plot 4: Vegetation type distribution
        veg_types = [cube['vegetation_type'] for cube in vegetation_cubes]
        type_counts = pd.Series(veg_types).value_counts()
        ax4.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax4.set_title('Vegetation Type Distribution')
        
        plt.tight_layout()
        plt.suptitle(f'Vegetation Analysis Summary - {municipality_name}', y=0.98)
        
        output_file = output_path / f'vegetation_summary_{municipality_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary plots saved to: {output_file}")


def segment_vegetation(netcdf_path: str, 
                               parameters: Optional[VegetationSegmentationParameters] = None,
                               municipality_name: str = "Sant Martí",
                               create_visualizations: bool = True,
                               output_dir: str = "outputs/vegetation_clustering") -> List[Dict]:
    """
    Vegetation segmentation function with improved performance and memory usage.
    """
    
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
    
    # Test the version
    logger.info("Testing vegetation segmentation...")
    
    data_path = "D:/Uni/TFG/data/processed/landsat_multidimensional_Sant_Marti_Ciutat_Vella.nc"
    municipality = "Sant Martí"
    
    # Create parameters
    params = VegetationSegmentationParameters(
        max_spatial_distance=20,
        min_vegetation_ndvi=0.4,
        min_cube_size=10,
        ndvi_variance_threshold=0.015,
        n_clusters=10,
        temporal_weight=0.8
    )
    
    # Run segmentation
    vegetation_cubes = segment_vegetation(
        netcdf_path=data_path,
        parameters=params,
        municipality_name=municipality,
        create_visualizations=True,
        output_dir="outputs/vegetation_clustering"
    )
    
    logger.success(f"Completed! Found {len(vegetation_cubes)} vegetation clusters.")
    
    # Print detailed results
    if vegetation_cubes:
        logger.info("=== Cluster Summary ===")
        for i, cube in enumerate(vegetation_cubes[:5]):  # Show first 5
            logger.info(f"Cluster {i+1}: {cube['size']} pixels, "
                       f"NDVI={cube['mean_ndvi']:.3f}, "
                       f"Type={cube['vegetation_type']}")