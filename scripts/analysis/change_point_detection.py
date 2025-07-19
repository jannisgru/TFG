"""
Change Point Detection and Segmented Regression Analysis for NDVI Time Series
Detects statistically significant breaks in vegetation trends at pixel and municipality levels.

This script implements:
1. PELT (Pruned Exact Linear Time) algorithm for change point detection
2. Piecewise linear regression for trend segmentation
3. Spatial mapping of change points and their magnitudes
4. Statistical analysis of change patterns across municipalities

References:
- Lawton et al., Remote Sensing, 2021 (MDPI)
- Pandey et al., Geocarto International, 2018
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
import ruptures as rpt
from datetime import datetime

warnings.filterwarnings('ignore')


def load_data_with_municipalities(file_path):
    """Load NetCDF data with municipality information."""
    print(f"Loading data from: {file_path}")
    ds = xr.open_dataset(file_path)
    
    # Load municipality mapping
    municipality_mapping = None
    mapping_file = Path(file_path).parent / "municipality_mapping.csv"
    if mapping_file.exists():
        municipality_mapping = pd.read_csv(mapping_file)
        print(f"Municipality mapping loaded: {len(municipality_mapping)} municipalities")
    else:
        print("No municipality mapping found")
    
    return ds, municipality_mapping


def detect_change_points_pelt(time_series, penalty=10, min_size=3):
    """
    Detect change points using PELT algorithm.
    
    Parameters:
    - time_series: 1D array of NDVI values
    - penalty: Penalty parameter for PELT (higher = fewer change points)
    - min_size: Minimum segment size
    
    Returns:
    - change_points: List of indices where changes occur
    - n_changes: Number of change points detected
    """
    # Remove NaN values and track valid indices
    valid_mask = ~np.isnan(time_series)
    valid_data = time_series[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_data) < min_size * 2:
        return [], 0
    
    try:
        # Apply PELT algorithm
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(valid_data)
        change_points_valid = algo.predict(pen=penalty)
        
        # Convert back to original indices
        change_points = []
        for cp in change_points_valid[:-1]:  # Remove last point (end of series)
            if cp < len(valid_indices):
                change_points.append(valid_indices[cp])
        
        return change_points, len(change_points)
    
    except Exception as e:
        print(f"Error in PELT detection: {e}")
        return [], 0


def detect_change_points_sliding_window(time_series, window_size=5, threshold=0.02):
    """
    Alternative change point detection using sliding window approach.
    Detects points where the trend changes significantly.
    
    Parameters:
    - time_series: 1D array of NDVI values
    - window_size: Size of the sliding window
    - threshold: Threshold for significant change in slope
    
    Returns:
    - change_points: List of indices where changes occur
    - n_changes: Number of change points detected
    """
    valid_mask = ~np.isnan(time_series)
    if np.sum(valid_mask) < window_size * 2:
        return [], 0
    
    # Calculate rolling slopes
    change_points = []
    
    for i in range(window_size, len(time_series) - window_size):
        if not valid_mask[i]:
            continue
            
        # Calculate slope before and after point i
        left_indices = np.arange(max(0, i - window_size), i)
        right_indices = np.arange(i, min(len(time_series), i + window_size))
        
        # Filter valid points
        left_valid = left_indices[valid_mask[left_indices]]
        right_valid = right_indices[valid_mask[right_indices]]
        
        if len(left_valid) < 2 or len(right_valid) < 2:
            continue
        
        # Calculate slopes
        left_slope = np.polyfit(left_valid, time_series[left_valid], 1)[0]
        right_slope = np.polyfit(right_valid, time_series[right_valid], 1)[0]
        
        # Check if slope change is significant
        if abs(right_slope - left_slope) > threshold:
            change_points.append(i)
    
    return change_points, len(change_points)


def piecewise_linear_regression(x, y, change_points):
    """
    Fit piecewise linear regression given change points.
    
    Parameters:
    - x: Time indices
    - y: NDVI values
    - change_points: List of change point indices
    
    Returns:
    - segments: List of dictionaries with segment information
    - fitted_values: Fitted values for the entire series
    - r2_score: R-squared of the piecewise fit
    """
    segments = []
    fitted_values = np.full_like(y, np.nan)
    
    # Create segments based on change points
    segment_bounds = [0] + change_points + [len(y)]
    
    for i in range(len(segment_bounds) - 1):
        start_idx = segment_bounds[i]
        end_idx = segment_bounds[i + 1]
        
        # Get segment data
        x_seg = x[start_idx:end_idx]
        y_seg = y[start_idx:end_idx]
        
        # Remove NaN values
        valid_mask = ~np.isnan(y_seg)
        if np.sum(valid_mask) < 2:
            continue
        
        x_seg_valid = x_seg[valid_mask]
        y_seg_valid = y_seg[valid_mask]
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(x_seg_valid.reshape(-1, 1), y_seg_valid)
        
        # Calculate statistics
        y_pred = reg.predict(x_seg_valid.reshape(-1, 1))
        r2 = r2_score(y_seg_valid, y_pred)
        
        # Calculate trend (slope per year)
        trend = reg.coef_[0]
        
        # Store segment information
        segments.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_year': x[start_idx],
            'end_year': x[end_idx - 1],
            'duration': end_idx - start_idx,
            'slope': trend,
            'intercept': reg.intercept_,
            'r2': r2,
            'mean_ndvi': np.nanmean(y_seg),
            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
            'trend_magnitude': abs(trend)
        })
        
        # Store fitted values
        fitted_values[start_idx:end_idx][valid_mask] = reg.predict(x_seg_valid.reshape(-1, 1))
    
    # Calculate overall R-squared
    valid_mask = ~np.isnan(fitted_values) & ~np.isnan(y)
    if np.sum(valid_mask) > 0:
        overall_r2 = r2_score(y[valid_mask], fitted_values[valid_mask])
    else:
        overall_r2 = 0
    
    return segments, fitted_values, overall_r2


def detect_change_magnitude(segments):
    """
    Calculate the magnitude of changes between segments.
    
    Parameters:
    - segments: List of segment dictionaries
    
    Returns:
    - changes: List of change dictionaries
    """
    changes = []
    
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        # Calculate NDVI at the change point
        change_year = next_seg['start_year']
        current_end_ndvi = current_seg['slope'] * (current_seg['end_year'] - current_seg['start_year']) + current_seg['intercept']
        next_start_ndvi = next_seg['intercept']
        
        # Calculate change magnitude
        change_magnitude = next_start_ndvi - current_end_ndvi
        change_percent = (change_magnitude / current_end_ndvi) * 100 if current_end_ndvi != 0 else 0
        
        # Determine change type
        if abs(change_magnitude) > 0.1:  # Threshold for significant change
            change_type = 'major_greening' if change_magnitude > 0 else 'major_browning'
        elif abs(change_magnitude) > 0.05:
            change_type = 'moderate_greening' if change_magnitude > 0 else 'moderate_browning'
        else:
            change_type = 'minor_change'
        
        changes.append({
            'change_year': change_year,
            'change_magnitude': change_magnitude,
            'change_percent': change_percent,
            'change_type': change_type,
            'from_trend': current_seg['trend_direction'],
            'to_trend': next_seg['trend_direction'],
            'slope_change': next_seg['slope'] - current_seg['slope']
        })
    
    return changes


def analyze_pixel_change_points(ds, municipality_mapping, output_dir="outputs/change_point_analysis"):
    """
    Analyze change points for each pixel in the dataset.
    
    Parameters:
    - ds: xarray Dataset with NDVI data
    - municipality_mapping: DataFrame with municipality information
    - output_dir: Output directory for results
    
    Returns:
    - change_point_summary: Summary statistics
    - pixel_results: Detailed pixel-level results
    """
    print("Analyzing change points at pixel level...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get time coordinates as years
    time_coords = pd.to_datetime(ds.time.values).year.values
    
    # Initialize result arrays
    n_pixels = ds.dims['x'] * ds.dims['y']
    
    # Results storage
    change_point_results = {
        'n_change_points': np.full((ds.dims['y'], ds.dims['x']), np.nan),
        'first_change_year': np.full((ds.dims['y'], ds.dims['x']), np.nan),
        'last_change_year': np.full((ds.dims['y'], ds.dims['x']), np.nan),
        'max_change_magnitude': np.full((ds.dims['y'], ds.dims['x']), np.nan),
        'overall_trend': np.full((ds.dims['y'], ds.dims['x']), np.nan),
        'stability_index': np.full((ds.dims['y'], ds.dims['x']), np.nan),
        'dominant_change_type': np.full((ds.dims['y'], ds.dims['x']), np.nan, dtype=object)
    }
    
    pixel_details = []
    
    # Process each pixel
    print("Processing pixels...")
    total_pixels = ds.dims['y'] * ds.dims['x']
    processed_pixels = 0
    
    for y_idx in range(ds.dims['y']):
        for x_idx in range(ds.dims['x']):
            processed_pixels += 1
            
            if processed_pixels % 1000 == 0:
                print(f"Processed {processed_pixels}/{total_pixels} pixels ({processed_pixels/total_pixels*100:.1f}%)")
            
            # Get pixel time series
            pixel_ndvi = ds['ndvi'][:, y_idx, x_idx].values
            
            # Skip if mostly NaN
            if np.sum(~np.isnan(pixel_ndvi)) < 5:
                continue
            
            # Detect change points with more sensitive parameters
            change_points, n_changes = detect_change_points_pelt(pixel_ndvi, penalty=5, min_size=2)
            
            if n_changes == 0:
                # No change points - fit simple linear trend
                valid_mask = ~np.isnan(pixel_ndvi)
                if np.sum(valid_mask) >= 2:
                    reg = LinearRegression()
                    reg.fit(time_coords[valid_mask].reshape(-1, 1), pixel_ndvi[valid_mask])
                    overall_trend = reg.coef_[0]
                    stability_index = 1.0  # Very stable
                else:
                    overall_trend = 0
                    stability_index = 0
                
                change_point_results['n_change_points'][y_idx, x_idx] = 0
                change_point_results['overall_trend'][y_idx, x_idx] = overall_trend
                change_point_results['stability_index'][y_idx, x_idx] = stability_index
                change_point_results['dominant_change_type'][y_idx, x_idx] = 'stable'
            
            else:
                # Fit piecewise regression
                segments, fitted_values, r2 = piecewise_linear_regression(
                    time_coords, pixel_ndvi, change_points
                )
                
                if len(segments) > 0:
                    # Calculate change magnitudes
                    changes = detect_change_magnitude(segments)
                    
                    # Store results
                    change_point_results['n_change_points'][y_idx, x_idx] = n_changes
                    
                    if changes:
                        change_years = [c['change_year'] for c in changes]
                        change_magnitudes = [abs(c['change_magnitude']) for c in changes]
                        change_types = [c['change_type'] for c in changes]
                        
                        change_point_results['first_change_year'][y_idx, x_idx] = min(change_years)
                        change_point_results['last_change_year'][y_idx, x_idx] = max(change_years)
                        change_point_results['max_change_magnitude'][y_idx, x_idx] = max(change_magnitudes)
                        
                        # Determine dominant change type
                        change_type_counts = pd.Series(change_types).value_counts()
                        dominant_type = change_type_counts.index[0]
                        change_point_results['dominant_change_type'][y_idx, x_idx] = dominant_type
                    
                    # Calculate overall trend (average of all segments weighted by duration)
                    total_duration = sum(seg['duration'] for seg in segments)
                    overall_trend = sum(seg['slope'] * seg['duration'] for seg in segments) / total_duration
                    change_point_results['overall_trend'][y_idx, x_idx] = overall_trend
                    
                    # Calculate stability index (1 - normalized number of change points)
                    stability_index = 1.0 - (n_changes / len(time_coords))
                    change_point_results['stability_index'][y_idx, x_idx] = stability_index
                    
                    # Store detailed results for analysis
                    if len(ds.x) * len(ds.y) < 10000:  # Only store details for smaller datasets
                        pixel_details.append({
                            'x_idx': x_idx,
                            'y_idx': y_idx,
                            'longitude': float(ds.x.values[x_idx]),
                            'latitude': float(ds.y.values[y_idx]),
                            'n_change_points': n_changes,
                            'change_years': change_years if 'change_years' in locals() else [],
                            'segments': segments,
                            'changes': changes,
                            'r2': r2,
                            'overall_trend': overall_trend,
                            'stability_index': stability_index
                        })
    
    print(f"Completed processing {processed_pixels} pixels")
    
    # Create summary statistics
    summary_stats = {}
    for key, array in change_point_results.items():
        if key != 'dominant_change_type':
            valid_data = array[~np.isnan(array)]
            if len(valid_data) > 0:
                summary_stats[key] = {
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data),
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'median': np.median(valid_data),
                    'count': len(valid_data)
                }
    
    return change_point_results, pixel_details, summary_stats


def analyze_municipality_change_points(ds, municipality_mapping, output_dir="outputs/change_point_analysis"):
    """
    Analyze change points aggregated by municipality.
    
    Parameters:
    - ds: xarray Dataset with NDVI data
    - municipality_mapping: DataFrame with municipality information
    - output_dir: Output directory for results
    
    Returns:
    - municipality_results: Change point analysis by municipality
    """
    print("Analyzing change points by municipality...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get time coordinates as years
    time_coords = pd.to_datetime(ds.time.values).year.values
    
    municipality_results = []
    
    for _, row in municipality_mapping.iterrows():
        municipality_id = row['municipality_id']
        municipality_name = row['municipality_name']
        
        print(f"Processing municipality: {municipality_name}")
        
        # Get municipality mask
        if 'municipality_id' in ds.variables:
            if len(ds['municipality_id'].dims) == 3:
                mask = ds['municipality_id'].isel(time=0) == municipality_id
            else:
                mask = ds['municipality_id'] == municipality_id
        else:
            print(f"Warning: No municipality_id variable found")
            continue
        
        # Get NDVI data for municipality
        municipality_ndvi = ds['ndvi'].where(mask)
        
        # Calculate mean NDVI time series for the municipality
        mean_ndvi = municipality_ndvi.mean(dim=['x', 'y']).values
        
        # Skip if mostly NaN
        if np.sum(~np.isnan(mean_ndvi)) < 5:
            continue
        
        # Try multiple change point detection methods
        # Method 1: PELT with sensitive parameters
        change_points_pelt, n_changes_pelt = detect_change_points_pelt(mean_ndvi, penalty=3, min_size=2)
        
        # Method 2: Sliding window method
        change_points_sliding, n_changes_sliding = detect_change_points_sliding_window(mean_ndvi, window_size=5, threshold=0.01)
        
        # Choose the method that finds more change points (but not too many)
        if n_changes_pelt > 0 and n_changes_pelt <= 8:  # Reasonable number of change points
            change_points, n_changes = change_points_pelt, n_changes_pelt
            detection_method = "PELT"
        elif n_changes_sliding > 0 and n_changes_sliding <= 8:
            change_points, n_changes = change_points_sliding, n_changes_sliding
            detection_method = "Sliding Window"
        elif n_changes_pelt > 0:  # Prefer PELT even if many change points
            change_points, n_changes = change_points_pelt, n_changes_pelt
            detection_method = "PELT"
        else:  # Use sliding window as fallback
            change_points, n_changes = change_points_sliding, n_changes_sliding
            detection_method = "Sliding Window"
        
        print(f"  - {detection_method} method found {n_changes} change points")
        
        if n_changes == 0:
            # No change points - simple linear trend
            valid_mask = ~np.isnan(mean_ndvi)
            if np.sum(valid_mask) >= 2:
                reg = LinearRegression()
                reg.fit(time_coords[valid_mask].reshape(-1, 1), mean_ndvi[valid_mask])
                overall_trend = reg.coef_[0]
                r2 = r2_score(mean_ndvi[valid_mask], reg.predict(time_coords[valid_mask].reshape(-1, 1)))
            else:
                overall_trend = 0
                r2 = 0
            
            municipality_results.append({
                'municipality_id': municipality_id,
                'municipality_name': municipality_name,
                'n_change_points': 0,
                'change_years': [],
                'segments': [],
                'changes': [],
                'overall_trend': overall_trend,
                'stability_index': 1.0,
                'r2': r2,
                'mean_ndvi': np.nanmean(mean_ndvi),
                'ndvi_range': np.nanmax(mean_ndvi) - np.nanmin(mean_ndvi),
                'dominant_change_type': 'stable',
                'detection_method': 'Linear Trend'
            })
        
        else:
            # Fit piecewise regression
            segments, fitted_values, r2 = piecewise_linear_regression(
                time_coords, mean_ndvi, change_points
            )
            
            if len(segments) > 0:
                # Calculate change magnitudes
                changes = detect_change_magnitude(segments)
                
                # Determine dominant change type
                if changes:
                    change_types = [c['change_type'] for c in changes]
                    change_type_counts = pd.Series(change_types).value_counts()
                    dominant_type = change_type_counts.index[0]
                    change_years = [c['change_year'] for c in changes]
                else:
                    dominant_type = 'stable'
                    change_years = []
                
                # Calculate overall trend
                total_duration = sum(seg['duration'] for seg in segments)
                overall_trend = sum(seg['slope'] * seg['duration'] for seg in segments) / total_duration
                
                # Calculate stability index
                stability_index = 1.0 - (n_changes / len(time_coords))
                
                municipality_results.append({
                    'municipality_id': municipality_id,
                    'municipality_name': municipality_name,
                    'n_change_points': n_changes,
                    'change_years': change_years,
                    'segments': segments,
                    'changes': changes,
                    'overall_trend': overall_trend,
                    'stability_index': stability_index,
                    'r2': r2,
                    'mean_ndvi': np.nanmean(mean_ndvi),
                    'ndvi_range': np.nanmax(mean_ndvi) - np.nanmin(mean_ndvi),
                    'dominant_change_type': dominant_type,
                    'detection_method': detection_method
                })
    
    return municipality_results


def create_change_point_maps(ds, change_point_results, output_dir="outputs/change_point_analysis"):
    """
    Create spatial maps showing change point analysis results.
    
    Parameters:
    - ds: xarray Dataset
    - change_point_results: Dictionary with change point analysis results
    - output_dir: Output directory
    """
    print("Creating change point maps...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Change Point Detection Analysis - Spatial Maps', fontsize=16, fontweight='bold')
    
    # 1. Number of change points
    ax = axes[0, 0]
    im1 = ax.imshow(change_point_results['n_change_points'], 
                   cmap='viridis', interpolation='nearest',
                   extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()])
    ax.set_title('Number of Change Points')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax, label='Number of Change Points')
    
    # 2. First change year
    ax = axes[0, 1]
    im2 = ax.imshow(change_point_results['first_change_year'], 
                   cmap='plasma', interpolation='nearest',
                   extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()])
    ax.set_title('First Change Year')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax, label='Year')
    
    # 3. Maximum change magnitude
    ax = axes[0, 2]
    im3 = ax.imshow(change_point_results['max_change_magnitude'], 
                   cmap='RdBu_r', interpolation='nearest',
                   extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()],
                   vmin=-0.5, vmax=0.5)
    ax.set_title('Maximum Change Magnitude')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im3, ax=ax, label='NDVI Change')
    
    # 4. Overall trend
    ax = axes[1, 0]
    im4 = ax.imshow(change_point_results['overall_trend'], 
                   cmap='RdYlGn', interpolation='nearest',
                   extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()],
                   vmin=-0.01, vmax=0.01)
    ax.set_title('Overall Trend (NDVI/year)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im4, ax=ax, label='NDVI/year')
    
    # 5. Stability index
    ax = axes[1, 1]
    im5 = ax.imshow(change_point_results['stability_index'], 
                   cmap='RdYlBu', interpolation='nearest',
                   extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()],
                   vmin=0, vmax=1)
    ax.set_title('Stability Index')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im5, ax=ax, label='Stability (0-1)')
    
    # 6. Dominant change type (categorical)
    ax = axes[1, 2]
    # Create numeric representation of change types
    change_type_map = {
        'stable': 0,
        'minor_change': 1,
        'moderate_greening': 2,
        'moderate_browning': 3,
        'major_greening': 4,
        'major_browning': 5
    }
    
    change_type_numeric = np.full_like(change_point_results['n_change_points'], np.nan)
    for i in range(change_point_results['dominant_change_type'].shape[0]):
        for j in range(change_point_results['dominant_change_type'].shape[1]):
            change_type = change_point_results['dominant_change_type'][i, j]
            if change_type in change_type_map:
                change_type_numeric[i, j] = change_type_map[change_type]
    
    im6 = ax.imshow(change_type_numeric, 
                   cmap='Set3', interpolation='nearest',
                   extent=[ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()],
                   vmin=0, vmax=5)
    ax.set_title('Dominant Change Type')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Create custom colorbar for change types
    cbar = plt.colorbar(im6, ax=ax, label='Change Type')
    cbar.set_ticks(range(6))
    cbar.set_ticklabels(['Stable', 'Minor', 'Mod. Green', 'Mod. Brown', 'Major Green', 'Major Brown'])
    
    plt.tight_layout()
    
    # Save the figure
    output_file = output_path / "change_point_spatial_maps.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Change point maps saved: {output_file}")


def create_municipality_change_summary(municipality_results, output_dir="outputs/change_point_analysis"):
    """
    Create summary visualizations for municipality-level change point analysis.
    
    Parameters:
    - municipality_results: List of municipality analysis results
    - output_dir: Output directory
    """
    print("Creating municipality change summary...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(municipality_results)
    
    # Create interactive dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Change Points by Municipality', 'Stability vs Trend',
                       'Change Type Distribution', 'Timeline of Changes'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Bar chart of change points by municipality
    fig.add_trace(
        go.Bar(
            x=df['municipality_name'],
            y=df['n_change_points'],
            name='Number of Change Points',
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    # 2. Scatter plot: Stability vs Trend
    fig.add_trace(
        go.Scatter(
            x=df['stability_index'],
            y=df['overall_trend'],
            mode='markers+text',
            text=df['municipality_name'],
            textposition='top center',
            marker=dict(
                size=10,
                color=df['n_change_points'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Change Points")
            ),
            name='Municipalities'
        ),
        row=1, col=2
    )
    
    # 3. Change type distribution (bar chart instead of pie)
    change_type_counts = df['dominant_change_type'].value_counts()
    fig.add_trace(
        go.Bar(
            x=change_type_counts.index,
            y=change_type_counts.values,
            name='Change Type Count',
            marker_color='coral'
        ),
        row=2, col=1
    )
    
    # 4. Timeline of changes
    # Flatten all change years with municipality names
    timeline_data = []
    for _, row in df.iterrows():
        for year in row['change_years']:
            timeline_data.append({
                'municipality': row['municipality_name'],
                'year': year,
                'change_type': row['dominant_change_type']
            })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        year_counts = timeline_df['year'].value_counts().sort_index()
        
        fig.add_trace(
            go.Scatter(
                x=year_counts.index,
                y=year_counts.values,
                mode='lines+markers',
                name='Changes per Year',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Municipality Change Point Analysis Dashboard',
        height=800,
        showlegend=True
    )
    
    # Update x-axis labels for municipality names
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    # Save interactive plot
    output_file = output_path / "municipality_change_dashboard.html"
    fig.write_html(output_file)
    print(f"Municipality dashboard saved: {output_file}")
    
    # Create summary table
    summary_table = df[['municipality_name', 'n_change_points', 'dominant_change_type', 
                       'overall_trend', 'stability_index', 'r2', 'detection_method']].round(4)
    summary_table.columns = ['Municipality', 'Change Points', 'Dominant Change Type', 
                           'Overall Trend', 'Stability Index', 'R²', 'Detection Method']
    
    # Save summary table
    table_file = output_path / "municipality_change_summary.csv"
    summary_table.to_csv(table_file, index=False)
    print(f"Summary table saved: {table_file}")
    
    return summary_table


def create_change_point_time_series(municipality_results, output_dir="outputs/change_point_analysis"):
    """
    Create time series plots showing segmented regression results.
    
    Parameters:
    - municipality_results: List of municipality analysis results
    - output_dir: Output directory
    """
    print("Creating change point time series plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select top municipalities with most change points
    df = pd.DataFrame(municipality_results)
    top_municipalities = df.nlargest(6, 'n_change_points')
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Segmented Regression Analysis - Top 6 Municipalities', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_municipalities.iterrows()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        municipality_name = row['municipality_name']
        segments = row['segments']
        changes = row['changes']
        
        # Plot segments
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        
        for seg_idx, segment in enumerate(segments):
            start_year = segment['start_year']
            end_year = segment['end_year']
            
            # Generate points for the segment
            x_seg = np.linspace(start_year, end_year, 10)
            y_seg = segment['slope'] * (x_seg - start_year) + segment['intercept']
            
            ax.plot(x_seg, y_seg, color=colors[seg_idx % len(colors)], 
                   linewidth=2, label=f'Segment {seg_idx + 1}')
        
        # Mark change points
        for change in changes:
            ax.axvline(x=change['change_year'], color='red', linestyle='--', alpha=0.7)
            ax.text(change['change_year'], ax.get_ylim()[1] * 0.9, 
                   f"{change['change_year']}\n{change['change_type']}", 
                   rotation=90, ha='center', va='top', fontsize=8)
        
        ax.set_title(f'{municipality_name}\n{row["n_change_points"]} change points, R² = {row["r2"]:.3f}')
        ax.set_xlabel('Year')
        ax.set_ylabel('NDVI')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = output_path / "segmented_regression_time_series.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series plots saved: {output_file}")


def main(netcdf_path="data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc"):
    """
    Main function to perform comprehensive change point detection analysis.
    
    Parameters:
    - netcdf_path: Path to the NetCDF file with NDVI data
    """
    print("="*60)
    print("CHANGE POINT DETECTION AND SEGMENTED REGRESSION ANALYSIS")
    print("="*60)
    
    # Load data
    ds, municipality_mapping = load_data_with_municipalities(netcdf_path)
    
    # Print dataset information
    print(f"\n📊 Dataset Information:")
    print(f"- Spatial dimensions: {ds.dims['y']} x {ds.dims['x']}")
    print(f"- Time range: {ds.time.min().values} - {ds.time.max().values}")
    print(f"- Variables: {list(ds.data_vars.keys())}")
    
    if municipality_mapping is not None:
        print(f"- Municipalities: {len(municipality_mapping)}")
    
    # Set output directory
    output_dir = "outputs/change_point_analysis"
    
    print(f"\n{'='*40}")
    print("MUNICIPALITY-LEVEL ANALYSIS")
    print("="*40)
    
    # Analyze change points by municipality
    municipality_results = analyze_municipality_change_points(ds, municipality_mapping, output_dir)
    
    # Create municipality summary
    summary_table = create_municipality_change_summary(municipality_results, output_dir)
    
    # Create time series plots
    create_change_point_time_series(municipality_results, output_dir)
    
    # Print summary statistics
    print(f"\n📈 MUNICIPALITY ANALYSIS SUMMARY:")
    print(f"- Total municipalities analyzed: {len(municipality_results)}")
    
    df = pd.DataFrame(municipality_results)
    print(f"- Average change points per municipality: {df['n_change_points'].mean():.2f}")
    print(f"- Municipalities with no change points: {sum(df['n_change_points'] == 0)}")
    print(f"- Municipalities with >2 change points: {sum(df['n_change_points'] > 2)}")
    
    # Most common change types
    change_type_counts = df['dominant_change_type'].value_counts()
    print(f"\n🔄 DOMINANT CHANGE TYPES:")
    for change_type, count in change_type_counts.head().items():
        print(f"- {change_type}: {count} municipalities")
    
    # Years with most changes
    all_change_years = []
    for result in municipality_results:
        all_change_years.extend(result['change_years'])
    
    if all_change_years:
        change_year_counts = pd.Series(all_change_years).value_counts().head(5)
        print(f"\n📅 YEARS WITH MOST CHANGES:")
        for year, count in change_year_counts.items():
            print(f"- {year}: {count} municipalities")
    
    # Pixel-level analysis (only for smaller datasets)
    if ds.dims['x'] * ds.dims['y'] < 50000:  # Limit to prevent memory issues
        print(f"\n{'='*40}")
        print("PIXEL-LEVEL ANALYSIS")
        print("="*40)
        
        # Analyze change points at pixel level
        pixel_results, pixel_details, summary_stats = analyze_pixel_change_points(
            ds, municipality_mapping, output_dir
        )
        
        # Create spatial maps
        create_change_point_maps(ds, pixel_results, output_dir)
        
        print(f"\n📊 PIXEL-LEVEL SUMMARY:")
        for key, stats in summary_stats.items():
            if key == 'n_change_points':
                print(f"- Average change points per pixel: {stats['mean']:.2f}")
                print(f"- Max change points in any pixel: {int(stats['max'])}")
    
    else:
        print(f"\n⚠️  Pixel-level analysis skipped (dataset too large: {ds.dims['x']} x {ds.dims['y']} pixels)")
        print("   Increase the threshold in the code if you want to process larger datasets.")
    
    print(f"\n✅ CHANGE POINT ANALYSIS COMPLETED!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📈 Key outputs:")
    print(f"   - municipality_change_dashboard.html (interactive dashboard)")
    print(f"   - municipality_change_summary.csv (summary table)")
    print(f"   - segmented_regression_time_series.png (time series plots)")
    if ds.dims['x'] * ds.dims['y'] < 50000:
        print(f"   - change_point_spatial_maps.png (spatial maps)")
    
    return municipality_results, summary_table


if __name__ == "__main__":
    # Run the analysis
    netcdf_file = "data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc"
    municipality_results, summary_table = main(netcdf_file)
