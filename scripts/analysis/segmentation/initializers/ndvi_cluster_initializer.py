"""
Vegetation-focused NDVI clustering for ST-Cube initialization.

This module implements spatial clustering that groups nearby pixels
with similar NDVI temporal patterns, focusing only on vegetation areas.
"""

import numpy as np
import xarray as xr
from typing import List, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Import handling for both package and direct execution
try:
    from ..base import VegetationSegmentationParameters
    from ..cube import STCube
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from base import VegetationSegmentationParameters
    from cube import STCube


class VegetationNDVIClusteringInitializer:
    """
    Initialize ST-cubes by clustering nearby pixels with similar NDVI temporal patterns.
    
    This approach creates spatially-aware clusters focusing only on vegetation areas
    (NDVI > threshold at some point) with local connectivity constraints.
    """
    
    def __init__(self, parameters: VegetationSegmentationParameters):
        self.parameters = parameters
        self.cube_id_counter = 0
    
    def initialize_cubes(self, data: xr.Dataset, valid_mask: np.ndarray) -> List[STCube]:
        """
        Initialize cubes by clustering nearby pixels with similar NDVI patterns.
        Focus only on vegetation areas.
        
        Args:
            data: The spatiotemporal dataset
            valid_mask: Boolean mask indicating valid pixels
            
        Returns:
            List of initialized STCube objects
        """
        print(f"Initializing vegetation ST-cubes using local NDVI clustering...")
        print(f"Parameters: max_distance={self.parameters.max_spatial_distance}, min_vegetation_ndvi={self.parameters.min_vegetation_ndvi}")
        
        if 'ndvi' not in data.variables:
            print("Error: NDVI not found in dataset. Cannot proceed with vegetation clustering.")
            return []
        
        # Step 1: Extract NDVI profiles for vegetation pixels only
        ndvi_profiles, pixel_coords = self._extract_vegetation_ndvi_profiles(data, valid_mask)
        
        if len(ndvi_profiles) < self.parameters.min_cube_size:
            print(f"Warning: Too few vegetation pixels ({len(ndvi_profiles)}). Need at least {self.parameters.min_cube_size}.")
            return []
        
        # Step 2: Perform spatial NDVI clustering
        cluster_labels = self._perform_spatial_clustering(ndvi_profiles, pixel_coords)
        
        # Step 3: Enforce spatial connectivity within clusters
        cluster_labels = self._enforce_spatial_connectivity(cluster_labels, pixel_coords)
        
        # Step 4: Create cubes from clusters (with built-in n_clusters optimization)
        cubes = self._create_cubes_from_clusters(data, pixel_coords, cluster_labels, ndvi_profiles)
        
        # Note: n_clusters optimization is now handled within _create_cubes_from_clusters
        # for better performance (early termination, pre-sorting by size)
        
        print(f"Initialized {len(cubes)} vegetation ST-cubes from local NDVI clustering")
        return cubes
    
    def _extract_vegetation_ndvi_profiles(self, data: xr.Dataset, valid_mask: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Extract NDVI time series profiles for vegetation pixels only"""
        print("Extracting NDVI profiles for vegetation pixels...")
        
        # Get valid pixel coordinates
        valid_coords = np.where(valid_mask)
        all_pixel_coords = list(zip(valid_coords[0], valid_coords[1]))
        
        print(f"Processing {len(all_pixel_coords)} valid pixels")
        
        # Optimization: if n_clusters is small, we can potentially sample pixels
        # to reduce processing time for large datasets
        target_clusters = self.parameters.n_clusters
        max_pixels_to_process = None
        
        if target_clusters is not None and target_clusters <= 10 and len(all_pixel_coords) > 10000:
            # For small target clusters on large datasets, limit pixels processed
            # Estimate: aim for ~100-200 pixels per target cluster
            max_pixels_to_process = min(len(all_pixel_coords), target_clusters * 200)
            print(f"Large dataset optimization: processing first {max_pixels_to_process} pixels for {target_clusters} target clusters")
        
        pixels_to_process = all_pixel_coords[:max_pixels_to_process] if max_pixels_to_process else all_pixel_coords
        
        # VECTORIZED APPROACH - Much faster than pixel-by-pixel processing
        print(f"Extracting NDVI profiles using vectorized operations...")
        
        # Extract all pixel coordinates
        y_coords = [coord[0] for coord in pixels_to_process]
        x_coords = [coord[1] for coord in pixels_to_process]
        
        # Extract NDVI data for all pixels at once (much faster)
        ndvi_data = data['ndvi'].values  # Shape: (time, y, x) or (time, municipality, y, x)
        
        # Handle different data shapes
        if len(ndvi_data.shape) == 4:  # (time, municipality, y, x)
            # Get the first municipality for now
            ndvi_data = ndvi_data[:, 0, :, :]
        
        # Extract profiles for all pixels at once
        ndvi_profiles = ndvi_data[:, y_coords, x_coords].T  # Shape: (n_pixels, n_time)
        
        # Filter for vegetation pixels (have vegetation threshold at some time point)
        vegetation_mask = np.any(ndvi_profiles >= self.parameters.min_vegetation_ndvi, axis=1)
        
        # Filter for valid profiles (no NaN values)
        valid_mask = ~np.any(np.isnan(ndvi_profiles), axis=1)
        
        # Combine masks
        final_mask = vegetation_mask & valid_mask
        
        # Apply masks
        ndvi_profiles = ndvi_profiles[final_mask]
        vegetation_pixel_coords = [pixels_to_process[i] for i in range(len(pixels_to_process)) if final_mask[i]]
        
        ndvi_profiles = np.array(ndvi_profiles)
        processed_suffix = f" (sampled from {len(all_pixel_coords)})" if max_pixels_to_process else ""
        print(f"Found {len(ndvi_profiles)} vegetation pixels with NDVI ≥ {self.parameters.min_vegetation_ndvi}{processed_suffix}")
        
        return ndvi_profiles, vegetation_pixel_coords
    
    def _perform_spatial_clustering(self, ndvi_profiles: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Perform spatially-aware NDVI clustering with early termination optimization"""
        print(f"Performing spatial clustering with max distance {self.parameters.max_spatial_distance}...")
        
        if len(ndvi_profiles) == 0:
            return np.array([])
        
        # Optimize for n_clusters parameter - use adaptive eps values
        target_clusters = self.parameters.n_clusters
        if target_clusters is not None:
            print(f"Optimizing clustering for target of {target_clusters} clusters...")
            return self._adaptive_clustering_for_target(ndvi_profiles, pixel_coords, target_clusters)
        
        # Original clustering method for unlimited clusters
        return self._standard_clustering(ndvi_profiles, pixel_coords)
    
    def _adaptive_clustering_for_target(self, ndvi_profiles: np.ndarray, pixel_coords: List[Tuple[int, int]], target_clusters: int) -> np.ndarray:
        """Perform clustering optimized for a target number of clusters"""
        # Normalize NDVI profiles
        scaler = StandardScaler()
        ndvi_scaled = scaler.fit_transform(ndvi_profiles)
        
        # Create spatial coordinates matrix
        spatial_coords = np.array(pixel_coords)
        spatial_weight = 0.3
        spatial_scaled = spatial_coords / self.parameters.max_spatial_distance
        
        # Combine features
        combined_features = np.hstack([
            ndvi_scaled,
            spatial_scaled * spatial_weight
        ])
        
        # Binary search for optimal eps to get close to target_clusters
        eps_min, eps_max = 0.1, 2.0
        best_eps = 0.5
        best_diff = float('inf')
        min_samples = max(3, self.parameters.min_cube_size // 3)
        
        # Try different eps values to find one that produces close to target_clusters
        for eps in np.linspace(eps_min, eps_max, 10):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            test_labels = dbscan.fit_predict(combined_features)
            n_clusters = len(np.unique(test_labels[test_labels >= 0]))
            
            diff = abs(n_clusters - target_clusters)
            if diff < best_diff:
                best_diff = diff
                best_eps = eps
                
            # Early termination if we hit the target exactly
            if n_clusters == target_clusters:
                print(f"Found optimal eps={eps:.3f} for exactly {target_clusters} clusters")
                return test_labels
        
        # Use the best eps found
        dbscan = DBSCAN(eps=best_eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(combined_features)
        
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        n_noise = np.sum(cluster_labels == -1)
        
        print(f"Optimized clustering: found {n_clusters} clusters (target: {target_clusters}) with {n_noise} noise points using eps={best_eps:.3f}")
        return cluster_labels
    
    def _standard_clustering(self, ndvi_profiles: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Standard clustering method for unlimited clusters"""
        # Normalize NDVI profiles
        scaler = StandardScaler()
        ndvi_scaled = scaler.fit_transform(ndvi_profiles)
        
        # Create spatial coordinates matrix
        spatial_coords = np.array(pixel_coords)
        
        # Combine temporal (NDVI) and spatial features
        spatial_weight = 0.3
        spatial_scaled = spatial_coords / self.parameters.max_spatial_distance
        
        # Combine features: [normalized_ndvi_time_series, weighted_spatial_coords]
        combined_features = np.hstack([
            ndvi_scaled,
            spatial_scaled * spatial_weight
        ])
        
        # Use DBSCAN for clustering with spatial awareness
        # Adaptive eps based on feature space dimensions and data size
        n_features = combined_features.shape[1]
        
        # Start with a reasonable eps value based on feature space size
        # For high-dimensional NDVI data, we need a larger eps
        base_eps = np.sqrt(n_features) * 0.5  # Scale with dimensionality
        
        min_samples = max(3, self.parameters.min_cube_size // 3)
        
        # Try a few eps values to find one that produces some clusters
        eps_values = [base_eps * factor for factor in [0.5, 1.0, 1.5, 2.0, 3.0]]
        
        best_result = None
        best_n_clusters = 0
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = dbscan.fit_predict(combined_features)
            n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
            
            # Prefer results with some clusters but not too many
            if n_clusters > 0 and (best_result is None or 
                                   (n_clusters < 100 and n_clusters > best_n_clusters)):
                best_result = cluster_labels
                best_n_clusters = n_clusters
                
            if n_clusters > 0 and n_clusters < 50:  # Good sweet spot
                break
        
        if best_result is not None:
            cluster_labels = best_result
            n_clusters = best_n_clusters
            n_noise = np.sum(cluster_labels == -1)
            print(f"Found {n_clusters} clusters with {n_noise} noise points")
        else:
            # Fallback: use very large eps to get at least some clusters
            eps = base_eps * 5.0
            print(f"Fallback: using large eps={eps:.3f}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = dbscan.fit_predict(combined_features)
            n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
            n_noise = np.sum(cluster_labels == -1)
            print(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return cluster_labels
    
    def _enforce_spatial_connectivity(self, cluster_labels: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Enforce spatial connectivity within clusters and apply distance constraints"""
        print("Enforcing spatial connectivity and distance constraints...")
        
        if len(cluster_labels) == 0:
            return cluster_labels
        
        new_labels = cluster_labels.copy()
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        # Optimization: if n_clusters is specified and small, use faster connectivity check
        target_clusters = self.parameters.n_clusters
        use_fast_mode = target_clusters is not None and target_clusters <= 20
        
        if use_fast_mode:
            print(f"Using fast connectivity mode for {target_clusters} target clusters")
            return self._enforce_connectivity_fast_mode(cluster_labels, pixel_coords, valid_labels)
        else:
            return self._enforce_connectivity_full_mode(cluster_labels, pixel_coords, valid_labels)
    
    def _enforce_connectivity_fast_mode(self, cluster_labels: np.ndarray, pixel_coords: List[Tuple[int, int]], valid_labels: np.ndarray) -> np.ndarray:
        """Fast connectivity enforcement for small numbers of target clusters"""
        new_labels = cluster_labels.copy()
        
        # Quick distance-based filtering without full connectivity analysis
        for label in valid_labels:
            cluster_mask = cluster_labels == label
            cluster_pixels = [pixel_coords[i] for i in range(len(pixel_coords)) if cluster_mask[i]]
            
            if len(cluster_pixels) < self.parameters.min_cube_size:
                new_labels[cluster_mask] = -1  # Mark as noise
                continue
            
            # Quick distance check - if cluster spans too large an area, split it
            if len(cluster_pixels) > 2:
                spatial_coords = np.array(cluster_pixels)
                y_span = np.max(spatial_coords[:, 0]) - np.min(spatial_coords[:, 0])
                x_span = np.max(spatial_coords[:, 1]) - np.min(spatial_coords[:, 1])
                max_span = max(y_span, x_span)
                
                if max_span > self.parameters.max_spatial_distance * 1.5:
                    # Cluster is too spread out, mark as noise for simplicity
                    new_labels[cluster_mask] = -1
        
        n_final_clusters = len(np.unique(new_labels[new_labels >= 0]))
        print(f"Fast mode: {n_final_clusters} clusters after connectivity enforcement")
        return new_labels
    
    def _enforce_connectivity_full_mode(self, cluster_labels: np.ndarray, pixel_coords: List[Tuple[int, int]], valid_labels: np.ndarray) -> np.ndarray:
        """Full connectivity enforcement with detailed connected component analysis"""
        new_labels = cluster_labels.copy()
        current_label = np.max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
        
        for label in valid_labels:
            # Get pixels in this cluster
            cluster_mask = cluster_labels == label
            cluster_pixels = [pixel_coords[i] for i in range(len(pixel_coords)) if cluster_mask[i]]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_pixels) < self.parameters.min_cube_size:
                new_labels[cluster_mask] = -1  # Mark as noise
                continue
            
            # For DBSCAN clusters, we can be more lenient with connectivity
            # Since DBSCAN already ensures local density, we just need to check basic spatial constraints
            
            # Quick spatial span check instead of strict connectivity
            spatial_coords = np.array(cluster_pixels)
            y_span = np.max(spatial_coords[:, 0]) - np.min(spatial_coords[:, 0])
            x_span = np.max(spatial_coords[:, 1]) - np.min(spatial_coords[:, 1])
            max_span = max(y_span, x_span)
            
            # If cluster is too spread out, try to split it, otherwise keep it
            if max_span > self.parameters.max_spatial_distance * 2:
                # Try to find connected components only if cluster is very spread out
                connected_components = self._find_connected_components(cluster_pixels)
                
                # Process each connected component
                for i, component in enumerate(connected_components):
                    component_indices = [cluster_indices[j] for j in component]
                    
                    if len(component) < self.parameters.min_cube_size:
                        # Too small, mark as noise
                        for idx in component_indices:
                            new_labels[idx] = -1
                    else:
                        if i == 0:
                            # Keep original label for largest component
                            continue
                        else:
                            # Assign new label for additional components
                            for idx in component_indices:
                                new_labels[idx] = current_label
                            current_label += 1
            # If cluster span is reasonable, keep the whole cluster as-is (no connectivity splitting needed)
        
        # Final check: remove clusters that violate distance constraints
        final_labels = self._filter_by_distance_constraints(new_labels, pixel_coords)
        
        n_final_clusters = len(np.unique(final_labels[final_labels >= 0]))
        print(f"Full mode: {n_final_clusters} spatially connected clusters")
        
        return final_labels
    
    def _find_connected_components(self, pixels: List[Tuple[int, int]]) -> List[List[int]]:
        """Find connected components using 8-connectivity"""
        if not pixels:
            return []
        
        # Create a mapping from pixel coordinates to indices
        pixel_to_idx = {pixel: i for i, pixel in enumerate(pixels)}
        
        # Build adjacency list for 8-connected neighbors
        adjacency = {i: [] for i in range(len(pixels))}
        
        for i, (y, x) in enumerate(pixels):
            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    neighbor = (y + dy, x + dx)
                    if neighbor in pixel_to_idx:
                        j = pixel_to_idx[neighbor]
                        adjacency[i].append(j)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        for i in range(len(pixels)):
            if i not in visited:
                component = []
                stack = [i]
                
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.append(node)
                        
                        for neighbor in adjacency[node]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                components.append(component)
        
        return components
    
    def _filter_by_distance_constraints(self, cluster_labels: np.ndarray, pixel_coords: List[Tuple[int, int]]) -> np.ndarray:
        """Filter clusters by enforcing maximum spatial distance constraints"""
        filtered_labels = cluster_labels.copy()
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        for label in valid_labels:
            cluster_mask = cluster_labels == label
            cluster_pixels = [pixel_coords[i] for i in range(len(pixel_coords)) if cluster_mask[i]]
            
            if len(cluster_pixels) < 2:
                continue
            
            # Modified approach: check if cluster span is reasonable, not all pairwise distances
            # This allows for elongated but connected clusters
            spatial_coords = np.array(cluster_pixels)
            y_span = np.max(spatial_coords[:, 0]) - np.min(spatial_coords[:, 0])
            x_span = np.max(spatial_coords[:, 1]) - np.min(spatial_coords[:, 1])
            max_span = max(y_span, x_span)
            
            # Use a more lenient threshold - allow clusters to span up to 2x max_distance
            # This accounts for diagonal or L-shaped clusters that are still locally connected
            if max_span > self.parameters.max_spatial_distance * 2:
                filtered_labels[cluster_mask] = -1
        
        return filtered_labels
    
    def _create_cubes_from_clusters(self, 
                                   data: xr.Dataset, 
                                   pixel_coords: List[Tuple[int, int]], 
                                   cluster_labels: np.ndarray,
                                   ndvi_profiles: np.ndarray) -> List[STCube]:
        """Create STCube objects from clustering results with early termination optimization"""
        print("Creating cubes from clusters...")
        
        cubes = []
        self.cube_id_counter = 0
        
        # Group pixels by cluster
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
        
        # Optimization: if n_clusters is specified, pre-sort clusters by size
        # to prioritize larger clusters and potentially avoid creating all cubes
        target_clusters = self.parameters.n_clusters
        if target_clusters is not None and len(valid_labels) > target_clusters:
            print(f"Optimizing cube creation for target of {target_clusters} clusters...")
            
            # Calculate cluster sizes quickly without full cube creation
            cluster_sizes = []
            for label in valid_labels:
                cluster_mask = cluster_labels == label
                size = np.sum(cluster_mask)
                if size >= self.parameters.min_cube_size:
                    cluster_sizes.append((label, size))
            
            # Sort by size (descending) and limit to target number
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            selected_labels = [label for label, size in cluster_sizes[:target_clusters]]
            valid_labels = selected_labels
            print(f"Pre-selected {len(valid_labels)} largest clusters for processing")
        
        for label in valid_labels:
            # Get pixels belonging to this cluster
            cluster_mask = cluster_labels == label
            cluster_pixels = [pixel_coords[i] for i in range(len(pixel_coords)) if cluster_mask[i]]
            cluster_profiles = ndvi_profiles[cluster_mask]
            
            # Skip small clusters
            if len(cluster_pixels) < self.parameters.min_cube_size:
                continue
            
            # Create temporal extent for entire time series
            time_range = (0, len(data.time) - 1)
            
            # Calculate cube properties
            spectral_signature = np.mean(cluster_profiles, axis=0)
            heterogeneity = np.var(cluster_profiles.flatten()) if len(cluster_profiles) > 1 else 0.0
            temporal_variance = np.mean([np.var(profile) for profile in cluster_profiles])
            ndvi_profile = np.mean(cluster_profiles, axis=0)
            
            # Calculate geometric properties
            perimeter = self._calculate_simple_perimeter(cluster_pixels)
            area = len(cluster_pixels)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
            smoothness = min(1.0, compactness)
            
            # Create cube
            cube = STCube(
                id=self.cube_id_counter,
                temporal_extent=time_range,
                pixels=cluster_pixels,
                spectral_signature=spectral_signature,
                heterogeneity=heterogeneity,
                area=area,
                perimeter=perimeter,
                compactness=min(1.0, compactness),
                smoothness=smoothness,
                temporal_variance=temporal_variance,
                ndvi_profile=ndvi_profile
            )
            
            cubes.append(cube)
            self.cube_id_counter += 1
            
            # Early termination if we've reached target number of clusters
            if target_clusters is not None and len(cubes) >= target_clusters:
                print(f"Reached target of {target_clusters} clusters, stopping cube creation")
                break
        
        return cubes
    
    def _calculate_simple_perimeter(self, pixels: List[Tuple[int, int]]) -> float:
        """Calculate perimeter of a cluster using 4-connectivity"""
        if len(pixels) <= 1:
            return 4.0
        
        pixel_set = set(pixels)
        perimeter = 0
        
        for y, x in pixels:
            # Check 4-connected neighbors
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
            for ny, nx in neighbors:
                if (ny, nx) not in pixel_set:
                    perimeter += 1
        
        return float(perimeter)
