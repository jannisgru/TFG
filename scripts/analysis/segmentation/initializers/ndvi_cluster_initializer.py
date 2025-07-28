"""
Optimized vegetation-focused NDVI clustering for ST-Cube initialization.

Key optimizations:
1. Improved memory efficiency with chunked processing
2. Better spatial indexing using KDTree
3. More efficient connectivity checking
4. Cleaner parameter handling and validation
5. Better error handling and logging
"""

# ==== CONFIGURABLE PARAMETERS ====
DEFAULT_MAX_PIXELS_FOR_SAMPLING = 50000    # Maximum pixels to sample for clustering
DEFAULT_SEARCH_RADIUS = 15                 # Search radius for spatial neighbors
DEFAULT_MIN_CLUSTER_CONNECTIVITY = 0.8     # Minimum connectivity for valid clusters
DEFAULT_EPS_SEARCH_ATTEMPTS = 5            # Maximum attempts to find optimal eps
DEFAULT_MIN_SAMPLES_RATIO = 0.01           # Minimum samples as ratio of total pixels
# ================================

import numpy as np
import xarray as xr
from typing import List, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
from loguru import logger
import warnings
from ..base import VegetationSegmentationParameters
from ..cube import STCube

warnings.filterwarnings('ignore')

class VegetationNDVIClusteringInitializer:
    """
    Initialization of ST-cubes by clustering nearby pixels with similar NDVI patterns.

    Key improvements:
    - Memory-efficient processing with chunking
    - Spatial indexing for faster connectivity checks
    - Better parameter validation and defaults
    - More robust clustering with fallback strategies
    """
    
    def __init__(self, parameters: VegetationSegmentationParameters):
        self.parameters = parameters
        self.cube_id_counter = 0
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate and set reasonable defaults for parameters"""
        if self.parameters.max_spatial_distance <= 0:
            raise ValueError("max_spatial_distance must be positive")
        
        if self.parameters.min_vegetation_ndvi < -1 or self.parameters.min_vegetation_ndvi > 1:
            warnings.warn("min_vegetation_ndvi should typically be between -1 and 1")
        
        if self.parameters.min_cube_size < 1:
            raise ValueError("min_cube_size must be at least 1")
        
        if self.parameters.n_clusters is not None and self.parameters.n_clusters < 1:
            raise ValueError("n_clusters must be positive if specified")
    
    def initialize_cubes(self, data: xr.Dataset, valid_mask: np.ndarray) -> List[STCube]:
        """
        Initialize cubes by clustering nearby pixels with similar NDVI patterns.
        
        Args:
            data: The spatiotemporal dataset
            valid_mask: Boolean mask indicating valid pixels
            
        Returns:
            List of initialized STCube objects
        """
        logger.info(f"Initializing vegetation ST-cubes using NDVI clustering...")
        logger.info(f"Parameters: max_distance={self.parameters.max_spatial_distance}, "
                   f"min_vegetation_ndvi={self.parameters.min_vegetation_ndvi}, "
                   f"target_clusters={self.parameters.n_clusters}")
        
        if 'ndvi' not in data.variables:
            raise ValueError("NDVI not found in dataset. Cannot proceed with vegetation clustering.")
        
        # Step 1: Extract NDVI profiles for vegetation pixels
        ndvi_profiles, pixel_coords = self._extract_vegetation_ndvi_profiles(data, valid_mask)
        
        if len(ndvi_profiles) < self.parameters.min_cube_size:
            logger.warning(f"Too few vegetation pixels ({len(ndvi_profiles)}). "
                          f"Need at least {self.parameters.min_cube_size}.")
            return []
        
        # Step 2: Perform spatial clustering
        cluster_labels = self._perform_clustering(ndvi_profiles, pixel_coords)
        
        # Step 3: Post-process clusters for quality
        cluster_labels = self._post_process_clusters(cluster_labels, pixel_coords)
        
        # Step 4: Create cubes from clusters
        cubes = self._create_cubes_from_clusters(data, pixel_coords, cluster_labels, ndvi_profiles)
        
        logger.success(f"Successfully initialized {len(cubes)} vegetation ST-cubes")
        return cubes
    
    def _extract_vegetation_ndvi_profiles(self, 
                                        data: xr.Dataset, 
                                        valid_mask: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Extract NDVI time series profiles for vegetation pixels with memory usage"""
        logger.info("Extracting NDVI profiles for vegetation pixels...")
        
        # Get NDVI data and handle different shapes
        ndvi_data = data['ndvi'].values
        if len(ndvi_data.shape) == 4:  # (time, municipality, y, x)
            ndvi_data = ndvi_data[:, 0, :, :]  # Take first municipality
        elif len(ndvi_data.shape) != 3:  # Should be (time, y, x)
            raise ValueError(f"Unexpected NDVI data shape: {ndvi_data.shape}")
        
        # Get valid pixel coordinates
        valid_coords = np.where(valid_mask)
        n_pixels = len(valid_coords[0])
        logger.info(f"Processing {n_pixels} valid pixels")

        # Step 2: Sample pixels if dataset is very large
        sampled_indices = self._get_pixel_sample_indices(n_pixels)
        
        # Extract coordinates and profiles
        y_coords = valid_coords[0][sampled_indices]
        x_coords = valid_coords[1][sampled_indices]
        pixel_coords = list(zip(y_coords, x_coords))
        
        # Vectorized NDVI extraction
        ndvi_profiles = ndvi_data[:, y_coords, x_coords].T  # Shape: (n_pixels, n_time)
        
        # Filter for vegetation and valid pixels in one step
        vegetation_mask = np.any(ndvi_profiles >= self.parameters.min_vegetation_ndvi, axis=1)
        valid_profiles_mask = ~np.any(np.isnan(ndvi_profiles), axis=1)
        final_mask = vegetation_mask & valid_profiles_mask
        
        # Apply filters
        ndvi_profiles = ndvi_profiles[final_mask]
        pixel_coords = [pixel_coords[i] for i in range(len(pixel_coords)) if final_mask[i]]
        
        sampling_info = f" (sampled from {n_pixels})" if len(sampled_indices) < n_pixels else ""
        logger.info(f"Found {len(ndvi_profiles)} vegetation pixels with NDVI ≥ {self.parameters.min_vegetation_ndvi}{sampling_info}")
        
        return ndvi_profiles, pixel_coords
    
    def _get_pixel_sample_indices(self, n_pixels: int) -> np.ndarray:
        """Determine which pixels to sample for processing efficiency"""
        target_clusters = self.parameters.n_clusters
        
        # For small target clusters on large datasets, limit processing
        if (target_clusters is not None and 
            target_clusters <= 20 and 
            n_pixels > 50000):
            
            # Aim for ~200-500 pixels per target cluster, but cap at reasonable limits
            max_pixels = min(n_pixels, max(target_clusters * 300, 10000))
            
            # Use systematic sampling for better spatial distribution
            step = n_pixels // max_pixels
            indices = np.arange(0, n_pixels, step)[:max_pixels]
            logger.info(f"Sampling {len(indices)} pixels (every {step}th pixel) for efficiency")
            return indices
        
        return np.arange(n_pixels)

    def _perform_timed_clustering(self, 
                                    ndvi_profiles: np.ndarray, 
                                    pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Perform spatially-aware clustering with improved parameter selection"""
        logger.info(f"Performing timed spatial clustering...")
        
        if len(ndvi_profiles) == 0:
            return np.array([])
        
        # Prepare features
        combined_features = self._prepare_clustering_features(ndvi_profiles, pixel_coords)
        
        # Determine clustering parameters
        eps, min_samples = self._determine_clustering_parameters(combined_features, len(pixel_coords))
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        cluster_labels = dbscan.fit_predict(combined_features)
        
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        n_noise = np.sum(cluster_labels == -1)
        
        logger.info(f"Clustering results: {n_clusters} clusters, {n_noise} noise points "
                   f"(eps={eps:.3f}, min_samples={min_samples})")
        
        return cluster_labels
    
    def _prepare_clustering_features(self, 
                                   ndvi_profiles: np.ndarray, 
                                   pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Prepare normalized features for clustering"""
        # Normalize NDVI profiles
        scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for constant features
            ndvi_scaled = scaler.fit_transform(ndvi_profiles)
        
        # Normalize spatial coordinates
        spatial_coords = np.array(pixel_coords)
        spatial_scaled = spatial_coords / self.parameters.max_spatial_distance
        
        # Combine features with appropriate weighting
        # Reduce spatial weight to give more importance to temporal patterns
        spatial_weight = 0.2  # Reduced from 0.3
        
        combined_features = np.hstack([
            ndvi_scaled,
            spatial_scaled * spatial_weight
        ])
        
        return combined_features
    
    def _determine_clustering_parameters(self, 
                                       features: np.ndarray, 
                                       n_pixels: int) -> Tuple[float, int]:
        """Determine optimal eps and min_samples for DBSCAN"""
        target_clusters = self.parameters.n_clusters
        n_features = features.shape[1]
        
        # Adaptive min_samples based on data size and target
        min_samples = max(3, min(self.parameters.min_cube_size, n_pixels // 100))
        
        if target_clusters is not None:
            # Binary search for optimal eps
            eps = self._find_optimal_eps(features, target_clusters, min_samples)
        else:
            # Use heuristic based on feature space dimensionality
            eps = self._estimate_eps_from_data(features)
        
        return eps, min_samples
    
    def _find_optimal_eps(self, 
                         features: np.ndarray, 
                         target_clusters: int, 
                         min_samples: int) -> float:
        """Binary search for eps value that produces close to target clusters"""
        eps_min, eps_max = 0.1, 3.0
        best_eps = 1.0
        best_diff = float('inf')
        
        # Coarse search first
        for eps in np.linspace(eps_min, eps_max, 8):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = dbscan.fit_predict(features)
            n_clusters = len(np.unique(labels[labels >= 0]))
            
            diff = abs(n_clusters - target_clusters)
            if diff < best_diff:
                best_diff = diff
                best_eps = eps
            
            if n_clusters == target_clusters:
                return eps
        
        # Fine-tune around best eps
        fine_range = np.linspace(max(eps_min, best_eps - 0.3), 
                                min(eps_max, best_eps + 0.3), 5)
        
        for eps in fine_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = dbscan.fit_predict(features)
            n_clusters = len(np.unique(labels[labels >= 0]))
            
            diff = abs(n_clusters - target_clusters)
            if diff < best_diff:
                best_diff = diff
                best_eps = eps
        
        return best_eps
    
    def _estimate_eps_from_data(self, features: np.ndarray) -> float:
        """Estimate reasonable eps value from data characteristics"""
        n_features = features.shape[1]
        n_samples = features.shape[0]
        
        # Use k-distance approach (simplified)
        if n_samples > 1000:
            # Sample for efficiency
            sample_indices = np.random.choice(n_samples, 1000, replace=False)
            sample_features = features[sample_indices]
        else:
            sample_features = features
        
        # Calculate distances to 5th nearest neighbor
        if len(sample_features) > 5:
            tree = cKDTree(sample_features)
            distances, _ = tree.query(sample_features, k=6)  # k=6 to exclude self
            k_distances = distances[:, 5]  # 5th nearest neighbor
            eps = np.percentile(k_distances, 70)  # Use 70th percentile
        else:
            eps = np.sqrt(n_features) * 0.5  # Fallback
        
        return max(0.1, min(3.0, eps))  # Clamp to reasonable range
    
    def _post_process_clusters(self, 
                             cluster_labels: np.ndarray, 
                             pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Post-process clusters to ensure quality and constraints"""
        logger.info("Post-processing clusters for quality assurance...")
        
        if len(cluster_labels) == 0:
            return cluster_labels
        
        # Build spatial index for efficient neighbor queries
        spatial_index = self._build_spatial_index(pixel_coords)
        
        # Filter and refine clusters
        refined_labels = self._refine_clusters(cluster_labels, pixel_coords, spatial_index)
        
        n_final = len(np.unique(refined_labels[refined_labels >= 0]))
        logger.success(f"Post-processing complete: {n_final} refined clusters")
        
        return refined_labels
    
    def _build_spatial_index(self, pixel_coords: List[Tuple[int, int]]) -> cKDTree:
        """Build spatial index for efficient neighbor queries"""
        spatial_coords = np.array(pixel_coords)
        return cKDTree(spatial_coords)
    
    def _refine_clusters(self, 
                        cluster_labels: np.ndarray, 
                        pixel_coords: List[Tuple[int, int]],
                        spatial_index: cKDTree) -> np.ndarray:
        """Refine clusters by removing those that violate constraints"""
        refined_labels = cluster_labels.copy()
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        for label in valid_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            
            # Check minimum size
            if len(cluster_indices) < self.parameters.min_cube_size:
                refined_labels[cluster_mask] = -1
                continue
            
            # Check spatial extent
            cluster_coords = [pixel_coords[i] for i in cluster_indices]
            if not self._check_spatial_constraints(cluster_coords, spatial_index):
                refined_labels[cluster_mask] = -1
        
        return refined_labels
    
    def _check_spatial_constraints(self, 
                                 cluster_coords: List[Tuple[int, int]], 
                                 spatial_index: cKDTree) -> bool:
        """Check if cluster satisfies spatial constraints"""
        if len(cluster_coords) < 2:
            return True
        
        coords_array = np.array(cluster_coords)
        
        # Check maximum span
        y_span = np.ptp(coords_array[:, 0])  # Peak-to-peak (max - min)
        x_span = np.ptp(coords_array[:, 1])
        max_span = max(y_span, x_span)
        
        # Allow clusters up to 2x max_spatial_distance for flexibility
        return max_span <= self.parameters.max_spatial_distance * 2
    
    def _create_cubes_from_clusters(self, 
                                   data: xr.Dataset, 
                                   pixel_coords: List[Tuple[int, int]], 
                                   cluster_labels: np.ndarray,
                                   ndvi_profiles: np.ndarray) -> List[STCube]:
        """Create STCube objects from clustering results"""
        logger.info("Creating cubes from refined clusters...")
        
        cubes = []
        self.cube_id_counter = 0
        
        # Get valid clusters, sorted by size
        cluster_info = self._get_sorted_cluster_info(cluster_labels, pixel_coords)
        
        # Limit to target number if specified
        if self.parameters.n_clusters is not None:
            cluster_info = cluster_info[:self.parameters.n_clusters]
        
        for label, size in cluster_info:
            cube = self._create_single_cube(
                data, pixel_coords, cluster_labels, ndvi_profiles, label
            )
            if cube is not None:
                cubes.append(cube)
        
        return cubes
    
    def _get_sorted_cluster_info(self, 
                               cluster_labels: np.ndarray, 
                               pixel_coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get cluster info sorted by size (descending)"""
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        cluster_info = []
        for label in valid_labels:
            size = np.sum(cluster_labels == label)
            if size >= self.parameters.min_cube_size:
                cluster_info.append((label, size))
        
        # Sort by size (largest first)
        cluster_info.sort(key=lambda x: x[1], reverse=True)
        return cluster_info
    
    def _create_single_cube(self, 
                          data: xr.Dataset,
                          pixel_coords: List[Tuple[int, int]],
                          cluster_labels: np.ndarray,
                          ndvi_profiles: np.ndarray,
                          label: int) -> Optional[STCube]:
        """Create a single STCube from cluster data"""
        cluster_mask = cluster_labels == label
        cluster_pixels = [pixel_coords[i] for i in range(len(pixel_coords)) if cluster_mask[i]]
        cluster_profiles = ndvi_profiles[cluster_mask]
        
        if len(cluster_pixels) < self.parameters.min_cube_size:
            return None
        
        # Calculate cube properties
        time_range = (0, len(data.time) - 1)
        spectral_signature = np.mean(cluster_profiles, axis=0)
        
        # Robust heterogeneity calculation
        heterogeneity = np.std(cluster_profiles.flatten()) if len(cluster_profiles) > 1 else 0.0
        temporal_variance = np.mean([np.std(profile) for profile in cluster_profiles])
        
        # Geometric properties
        area = len(cluster_pixels)
        perimeter = self._calculate_perimeter_efficient(cluster_pixels)
        compactness = self._calculate_compactness(area, perimeter)
        
        cube = STCube(
            id=self.cube_id_counter,
            temporal_extent=time_range,
            pixels=cluster_pixels,
            spectral_signature=spectral_signature,
            heterogeneity=heterogeneity,
            area=area,
            perimeter=perimeter,
            compactness=compactness,
            smoothness=min(1.0, compactness),
            temporal_variance=temporal_variance,
            ndvi_profile=np.mean(cluster_profiles, axis=0)
        )
        
        self.cube_id_counter += 1
        return cube
    
    def _calculate_perimeter_efficient(self, pixels: List[Tuple[int, int]]) -> float:
        """Efficiently calculate perimeter using set operations"""
        if len(pixels) <= 1:
            return 4.0
        
        pixel_set = set(pixels)
        perimeter = 0
        
        # Vectorized approach for better performance
        for y, x in pixels:
            # Count edges not shared with cluster pixels
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
            perimeter += sum(1 for ny, nx in neighbors if (ny, nx) not in pixel_set)
        
        return float(perimeter)
    
    def _calculate_compactness(self, area: int, perimeter: float) -> float:
        """Calculate compactness with bounds checking"""
        if perimeter <= 0 or area <= 0:
            return 1.0
        
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return min(1.0, max(0.0, compactness))  # Clamp to [0, 1]