#!/usr/bin/env python3
"""
Static Matplotlib Visualization for Vegetation ST-Cube Segmentation Results

This script creates static visualizations using Matplotlib for
vegetation-focused ST-cube segmentation results.
"""

# ==== CONFIGURABLE PARAMETERS ====
DEFAULT_OUTPUT_DIRECTORY = "outputs/static_vegetation"    # Default output directory
DEFAULT_FIGURE_SIZE = (18, 12)                           # Default figure size
DEFAULT_DPI = 300                                        # Default DPI for saved figures
DEFAULT_COLOR_MAP = "Set3"                               # Default colormap
DEFAULT_GRID_ALPHA = 0.3                                # Default grid transparency
# ================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import warnings
from loguru import logger

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')


class StaticVisualization:
    """
    Static visualization generator for vegetation ST-cube segmentation results.
    """
    
    def __init__(self, output_directory: str = DEFAULT_OUTPUT_DIRECTORY):
        """Initialize the static visualization generator."""
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def _get_pixels_safely(self, cube: Dict) -> List[Tuple[int, int]]:
        """Safely extract pixels from cube data, handling different formats."""
        pixels = cube.get('pixels', [])
        if pixels is None:
            return []
        
        if isinstance(pixels, np.ndarray):
            if pixels.size == 0:
                return []
            # Convert numpy array to list of tuples
            if pixels.ndim == 1 and len(pixels) == 2:
                return [tuple(pixels.tolist())]
            elif pixels.ndim == 2:
                return [tuple(row) for row in pixels.tolist()]
            else:
                return pixels.tolist()
        elif isinstance(pixels, list):
            return pixels
        else:
            return []
    
    def _has_valid_ndvi_profile(self, cube: Dict) -> bool:
        """Check if cube has a valid NDVI profile."""
        ndvi_profile = cube.get('ndvi_profile')
        if ndvi_profile is None:
            return False
        
        # Handle numpy arrays
        if isinstance(ndvi_profile, np.ndarray):
            return ndvi_profile.size > 0
        
        # Handle lists
        if isinstance(ndvi_profile, list):
            return len(ndvi_profile) > 0
        
        # For other types, try to check length
        try:
            return len(ndvi_profile) > 0
        except (TypeError, AttributeError):
            return False

    def _has_valid_pixels(self, cube: Dict) -> bool:
        """Check if cube has valid pixel data."""
        return len(self._get_pixels_safely(cube)) > 0
        
    def create_all_static_visualizations(self, 
                                       cubes: List[Dict], 
                                       data: Any, 
                                       municipality_name: str = "Unknown") -> Dict[str, str]:
        """Create all static visualizations for vegetation clusters."""
        logger.info(f"Creating static visualizations for {municipality_name}...")
        
        visualizations = {}
        
        try:
            # 1. Comprehensive summary plot
            summary_file = f"comprehensive_summary_{municipality_name.replace(' ', '_')}.png"
            self.create_comprehensive_summary(cubes, summary_file, municipality_name)
            visualizations["comprehensive_summary"] = str(self.output_dir / summary_file)
            
            # 2. Spatial distribution map
            spatial_file = f"spatial_distribution_{municipality_name.replace(' ', '_')}.png"
            self.create_spatial_distribution_map(cubes, spatial_file, municipality_name)
            visualizations["spatial_distribution"] = str(self.output_dir / spatial_file)
            
            # 3. NDVI temporal analysis
            temporal_file = f"temporal_analysis_{municipality_name.replace(' ', '_')}.png"
            self.create_temporal_analysis(cubes, data, temporal_file, municipality_name)
            visualizations["temporal_analysis"] = str(self.output_dir / temporal_file)
            
            logger.success(f"All static visualizations created successfully in: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating static visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return visualizations
    
    def create_comprehensive_summary(self, cubes: List[Dict], filename: str, municipality_name: str):
        """Create a comprehensive summary plot with multiple panels."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Vegetation Clustering Analysis - {municipality_name}', fontsize=16, fontweight='bold')
        
        # Panel 1: Cluster size distribution
        sizes = [cube['area'] for cube in cubes if cube['area'] > 0]
        axes[0, 0].hist(sizes, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_xlabel('Cluster Size (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: NDVI distribution
        ndvis = [cube['mean_ndvi'] for cube in cubes if not np.isnan(cube['mean_ndvi'])]
        axes[0, 1].hist(ndvis, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Mean NDVI')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('NDVI Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Vegetation type pie chart
        veg_types = [cube.get('vegetation_type', 'Unknown') for cube in cubes]
        type_counts = pd.Series(veg_types).value_counts()
        axes[0, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Vegetation Types')
        
        # Panel 4: NDVI vs Area scatter
        areas = [cube['area'] for cube in cubes]
        ndvis = [cube['mean_ndvi'] for cube in cubes]
        seasonality_colors = [cube.get('seasonality_score', 0) for cube in cubes]
        
        scatter = axes[1, 0].scatter(areas, ndvis, c=seasonality_colors, cmap='viridis', 
                                   alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        axes[1, 0].set_xlabel('Area (pixels)')
        axes[1, 0].set_ylabel('Mean NDVI')
        axes[1, 0].set_title('NDVI vs Area (colored by seasonality)')
        axes[1, 0].grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=axes[1, 0], shrink=0.8)
        cbar.set_label('Seasonality Score')
        
        # Panel 5: Time series sample
        valid_cubes = [cube for cube in cubes if self._has_valid_ndvi_profile(cube)]
        
        if valid_cubes:
            top_cubes = sorted(valid_cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:5]
            
            for i, cube in enumerate(top_cubes):
                time_coords = list(range(len(cube['ndvi_profile'])))
                axes[1, 1].plot(time_coords, cube['ndvi_profile'], 
                              linewidth=2, alpha=0.8, label=f"Cluster {cube['id']}")
            
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('NDVI')
            axes[1, 1].set_title('NDVI Evolution - Most Seasonal Clusters')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Panel 6: Summary statistics
        axes[1, 2].axis('off')
        
        total_area = sum(cube['area'] for cube in cubes)
        mean_ndvi = np.mean([cube['mean_ndvi'] for cube in cubes])
        mean_seasonality = np.mean([cube.get('seasonality_score', 0) for cube in cubes])
        
        stats_text = f"""Summary Statistics:

Total Clusters: {len(cubes)}
Total Area: {total_area:,} pixels
Mean Cluster Size: {total_area/len(cubes):.1f} pixels

Overall Mean NDVI: {mean_ndvi:.3f}
Mean Seasonality Score: {mean_seasonality:.3f}

Largest Cluster: {max(cubes, key=lambda x: x['area'])['area']:,} pixels
Highest NDVI: {max(cubes, key=lambda x: x['mean_ndvi'])['mean_ndvi']:.3f}"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Comprehensive summary saved to: {output_file}")
    
    def create_spatial_distribution_map(self, cubes: List[Dict], filename: str, municipality_name: str):
        """Create a detailed spatial distribution map."""
        
        # Check if cubes have spatial data
        valid_cubes = []
        for cube in cubes:
            pixels = cube.get('pixels', [])
            if pixels is not None and len(pixels) > 0:
                valid_cubes.append(cube)
        
        if not valid_cubes:
            print("Warning: No spatial data available for mapping")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Spatial Distribution Analysis - {municipality_name}', fontsize=14, fontweight='bold')
        
        # Get spatial extent
        all_pixels = []
        for cube in cubes:
            pixels = self._get_pixels_safely(cube)
            if pixels:
                all_pixels.extend(pixels)
        
        if not all_pixels:
            print("Warning: No valid pixels found")
            return
        
        y_coords = [p[0] for p in all_pixels]
        x_coords = [p[1] for p in all_pixels]
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        
        # Create maps
        cluster_map = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        ndvi_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        area_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        seasonality_map = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan, dtype=float)
        
        # Fill maps
        for i, cube in enumerate(cubes):
            pixels = self._get_pixels_safely(cube)
            if not pixels:
                continue
            
            for y, x in pixels:
                if y_min <= y <= y_max and x_min <= x <= x_max:
                    map_y, map_x = y - y_min, x - x_min
                    cluster_map[map_y, map_x] = i
                    ndvi_map[map_y, map_x] = cube['mean_ndvi']
                    area_map[map_y, map_x] = cube['area']
                    seasonality_map[map_y, map_x] = cube.get('seasonality_score', 0)
        
        # Plot 1: Cluster boundaries
        im1 = axes[0, 0].imshow(cluster_map, cmap='tab20', aspect='equal', origin='lower')
        axes[0, 0].set_title('Cluster Boundaries')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        
        # Plot 2: NDVI distribution
        im2 = axes[0, 1].imshow(ndvi_map, cmap='RdYlGn', aspect='equal', origin='lower', vmin=0, vmax=1)
        axes[0, 1].set_title('NDVI Distribution')
        axes[0, 1].set_xlabel('X Coordinate')
        axes[0, 1].set_ylabel('Y Coordinate')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Mean NDVI')
        
        # Plot 3: Cluster size
        im3 = axes[1, 0].imshow(np.log1p(area_map), cmap='plasma', aspect='equal', origin='lower')
        axes[1, 0].set_title('Cluster Size (log scale)')
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        cbar3.set_label('Log(Area + 1)')
        
        # Plot 4: Seasonality
        im4 = axes[1, 1].imshow(seasonality_map, cmap='viridis', aspect='equal', origin='lower')
        axes[1, 1].set_title('Seasonality Score')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Y Coordinate')
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        cbar4.set_label('Seasonality Score')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Spatial distribution map saved to: {output_file}")
    
    def create_temporal_analysis(self, cubes: List[Dict], data: Any, filename: str, municipality_name: str):
        """Create temporal analysis plots."""
        
        valid_cubes = [cube for cube in cubes if self._has_valid_ndvi_profile(cube)]
        
        if not valid_cubes:
            print("Warning: No temporal data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Temporal Analysis - {municipality_name}', fontsize=14, fontweight='bold')
        
        # Get time coordinates
        n_times = len(valid_cubes[0]['ndvi_profile'])
        time_coords = list(range(n_times))
        
        # Plot 1: Time series of top clusters
        interesting_cubes = sorted(valid_cubes, key=lambda x: x.get('seasonality_score', 0), reverse=True)[:8]
        
        for i, cube in enumerate(interesting_cubes):
            color = self.colors[i % len(self.colors)]
            axes[0, 0].plot(time_coords, cube['ndvi_profile'], 
                          color=color, linewidth=2, alpha=0.8,
                          label=f"Cluster {cube['id']} (Area: {cube['area']})")
        
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('NDVI')
        axes[0, 0].set_title('NDVI Evolution - Most Seasonal Clusters')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average NDVI evolution by vegetation type
        veg_types = set(cube.get('vegetation_type', 'Unknown') for cube in valid_cubes)
        
        for veg_type in veg_types:
            type_cubes = [cube for cube in valid_cubes if cube.get('vegetation_type') == veg_type]
            if type_cubes:
                profiles = np.array([cube['ndvi_profile'] for cube in type_cubes])
                mean_profile = np.mean(profiles, axis=0)
                std_profile = np.std(profiles, axis=0)
                
                axes[0, 1].plot(time_coords, mean_profile, linewidth=3, label=veg_type)
                axes[0, 1].fill_between(time_coords, 
                                      mean_profile - std_profile, 
                                      mean_profile + std_profile, 
                                      alpha=0.2)
        
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Mean NDVI')
        axes[0, 1].set_title('Average NDVI by Vegetation Type')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: NDVI heatmap
        sample_cubes = valid_cubes[::max(1, len(valid_cubes)//20)]
        ndvi_matrix = np.array([cube['ndvi_profile'] for cube in sample_cubes])
        
        im3 = axes[1, 0].imshow(ndvi_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Cluster Index')
        axes[1, 0].set_title('NDVI Evolution Heatmap')
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        cbar3.set_label('NDVI')
        
        # Plot 4: Temporal statistics distribution
        temporal_means = [np.mean(cube['ndvi_profile']) for cube in valid_cubes]
        temporal_stds = [np.std(cube['ndvi_profile']) for cube in valid_cubes]
        
        axes[1, 1].hist(temporal_means, bins=20, alpha=0.7, label='Mean NDVI', color='green')
        ax_twin = axes[1, 1].twinx()
        ax_twin.hist(temporal_stds, bins=20, alpha=0.7, label='NDVI Std Dev', color='orange')
        
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency (Mean NDVI)', color='green')
        ax_twin.set_ylabel('Frequency (Std Dev)', color='orange')
        axes[1, 1].set_title('Temporal Statistics Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Temporal analysis saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    print("Static Visualization for Vegetation ST-Cube Segmentation")
    print("This module provides publication-ready static visualizations.")
    print("Use this module by importing StaticVisualization class.")
