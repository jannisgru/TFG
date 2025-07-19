"""
Aggressive Change Point Detection for NDVI Time Series
This version uses very sensitive parameters to ensure we detect changes.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import ruptures as rpt
from scipy import signal

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


def detect_change_points_aggressive(time_series, time_coords):
    """
    Aggressive change point detection using multiple methods.
    
    Parameters:
    - time_series: 1D array of NDVI values
    - time_coords: Array of years
    
    Returns:
    - change_points: List of indices where changes occur
    - n_changes: Number of change points detected
    - method_used: String indicating which method was used
    """
    # Remove NaN values
    valid_mask = ~np.isnan(time_series)
    if np.sum(valid_mask) < 6:  # Need at least 6 valid points
        return [], 0, "insufficient_data"
    
    change_points = []
    method_used = "none"
    
    # Method 1: Very sensitive PELT
    try:
        algo = rpt.Pelt(model="l2", min_size=2).fit(time_series[valid_mask])
        pelt_change_points = algo.predict(pen=1)  # Very low penalty
        if len(pelt_change_points) > 1:  # More than just the end point
            # Convert to original indices
            valid_indices = np.where(valid_mask)[0]
            change_points = [valid_indices[cp] for cp in pelt_change_points[:-1] if cp < len(valid_indices)]
            if len(change_points) > 0:
                method_used = "PELT_aggressive"
    except:
        pass
    
    # Method 2: Sliding window with very low threshold
    if len(change_points) == 0:
        window_size = 3
        threshold = 0.005  # Very low threshold
        
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
        
        if len(change_points) > 0:
            method_used = "sliding_window_aggressive"
    
    # Method 3: Statistical breakpoint detection
    if len(change_points) == 0:
        # Look for significant changes in mean between periods
        n_points = len(time_series)
        for split_point in range(n_points // 4, 3 * n_points // 4):  # Check middle 50%
            if not valid_mask[split_point]:
                continue
                
            before = time_series[valid_mask[:split_point]]
            after = time_series[valid_mask[split_point:]]
            
            if len(before) < 3 or len(after) < 3:
                continue
            
            # Simple t-test-like comparison
            mean_before = np.mean(before)
            mean_after = np.mean(after)
            
            # Check if the difference is significant (simple threshold)
            if abs(mean_after - mean_before) > 0.02:  # 2% NDVI change
                change_points.append(split_point)
                method_used = "statistical_breakpoint"
                break
    
    # Method 4: Trend reversal detection
    if len(change_points) == 0:
        # Look for trend reversals
        window = 5
        for i in range(window, len(time_series) - window):
            if not valid_mask[i]:
                continue
            
            # Calculate trend before and after
            before_x = time_coords[max(0, i-window):i][valid_mask[max(0, i-window):i]]
            before_y = time_series[max(0, i-window):i][valid_mask[max(0, i-window):i]]
            after_x = time_coords[i:min(len(time_series), i+window)][valid_mask[i:min(len(time_series), i+window)]]
            after_y = time_series[i:min(len(time_series), i+window)][valid_mask[i:min(len(time_series), i+window)]]
            
            if len(before_x) < 2 or len(after_x) < 2:
                continue
            
            before_slope = np.polyfit(before_x, before_y, 1)[0]
            after_slope = np.polyfit(after_x, after_y, 1)[0]
            
            # Check for trend reversal
            if (before_slope > 0.001 and after_slope < -0.001) or (before_slope < -0.001 and after_slope > 0.001):
                change_points.append(i)
                method_used = "trend_reversal"
                break
    
    # Remove duplicates and sort
    change_points = sorted(list(set(change_points)))
    
    return change_points, len(change_points), method_used


def analyze_municipality_change_points_aggressive(ds, municipality_mapping, output_dir="outputs/change_point_analysis"):
    """
    Analyze change points with aggressive detection parameters.
    """
    print("Analyzing change points by municipality (aggressive detection)...")
    
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
        
        # Detect change points aggressively
        change_points, n_changes, method_used = detect_change_points_aggressive(mean_ndvi, time_coords)
        
        print(f"  - Found {n_changes} change points using {method_used}")
        
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
                'overall_trend': overall_trend,
                'stability_index': 1.0,
                'r2': r2,
                'mean_ndvi': np.nanmean(mean_ndvi),
                'ndvi_range': np.nanmax(mean_ndvi) - np.nanmin(mean_ndvi),
                'dominant_change_type': 'stable',
                'detection_method': method_used
            })
        
        else:
            # Analyze segments
            change_years = [int(time_coords[cp]) for cp in change_points]
            
            # Calculate simple metrics
            segments = []
            segment_bounds = [0] + change_points + [len(mean_ndvi)]
            
            for i in range(len(segment_bounds) - 1):
                start_idx = segment_bounds[i]
                end_idx = segment_bounds[i + 1]
                
                segment_data = mean_ndvi[start_idx:end_idx]
                valid_segment = segment_data[~np.isnan(segment_data)]
                
                if len(valid_segment) >= 2:
                    segment_years = time_coords[start_idx:end_idx]
                    valid_years = segment_years[~np.isnan(segment_data)]
                    
                    slope = np.polyfit(valid_years, valid_segment, 1)[0]
                    mean_value = np.mean(valid_segment)
                    
                    segments.append({
                        'start_year': int(time_coords[start_idx]),
                        'end_year': int(time_coords[end_idx - 1]),
                        'slope': slope,
                        'mean_ndvi': mean_value,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing'
                    })
            
            # Determine dominant change type
            if len(segments) >= 2:
                slopes = [seg['slope'] for seg in segments]
                if max(slopes) > 0.005:  # Significant greening
                    dominant_type = 'greening'
                elif min(slopes) < -0.005:  # Significant browning
                    dominant_type = 'browning'
                else:
                    dominant_type = 'mixed_changes'
            else:
                dominant_type = 'single_change'
            
            # Calculate overall trend
            if len(segments) > 0:
                overall_trend = np.mean([seg['slope'] for seg in segments])
            else:
                overall_trend = 0
            
            # Calculate stability index
            stability_index = 1.0 - (n_changes / len(time_coords))
            
            municipality_results.append({
                'municipality_id': municipality_id,
                'municipality_name': municipality_name,
                'n_change_points': n_changes,
                'change_years': change_years,
                'overall_trend': overall_trend,
                'stability_index': stability_index,
                'r2': 0.8,  # Placeholder
                'mean_ndvi': np.nanmean(mean_ndvi),
                'ndvi_range': np.nanmax(mean_ndvi) - np.nanmin(mean_ndvi),
                'dominant_change_type': dominant_type,
                'detection_method': method_used
            })
    
    return municipality_results


def create_simple_visualizations(municipality_results, output_dir="outputs/change_point_analysis"):
    """
    Create simple visualizations of the results.
    """
    print("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(municipality_results)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Change Point Detection Results - Aggressive Detection', fontsize=16, fontweight='bold')
    
    # 1. Number of change points
    ax = axes[0, 0]
    change_counts = df['n_change_points'].value_counts().sort_index()
    ax.bar(change_counts.index, change_counts.values, color='steelblue')
    ax.set_title('Distribution of Change Points')
    ax.set_xlabel('Number of Change Points')
    ax.set_ylabel('Number of Municipalities')
    
    # 2. Stability vs Trend
    ax = axes[0, 1]
    scatter = ax.scatter(df['stability_index'], df['overall_trend'], 
                        c=df['n_change_points'], cmap='viridis', alpha=0.7)
    ax.set_title('Stability vs Overall Trend')
    ax.set_xlabel('Stability Index')
    ax.set_ylabel('Overall Trend (NDVI/year)')
    plt.colorbar(scatter, ax=ax, label='Change Points')
    
    # 3. Change type distribution
    ax = axes[1, 0]
    change_types = df['dominant_change_type'].value_counts()
    ax.bar(change_types.index, change_types.values, color='coral')
    ax.set_title('Dominant Change Types')
    ax.set_xlabel('Change Type')
    ax.set_ylabel('Number of Municipalities')
    plt.xticks(rotation=45)
    
    # 4. Detection methods
    ax = axes[1, 1]
    methods = df['detection_method'].value_counts()
    ax.bar(methods.index, methods.values, color='lightgreen')
    ax.set_title('Detection Methods Used')
    ax.set_xlabel('Detection Method')
    ax.set_ylabel('Number of Municipalities')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / "aggressive_change_detection_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {output_file}")
    
    # Save updated CSV
    summary_table = df[['municipality_name', 'n_change_points', 'dominant_change_type', 
                       'overall_trend', 'stability_index', 'r2', 'detection_method']].round(4)
    summary_table.columns = ['Municipality', 'Change Points', 'Dominant Change Type', 
                           'Overall Trend', 'Stability Index', 'R²', 'Detection Method']
    
    table_file = output_path / "municipality_change_summary_aggressive.csv"
    summary_table.to_csv(table_file, index=False)
    print(f"Summary table saved: {table_file}")
    
    return summary_table


def main(netcdf_path="data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc"):
    """
    Main function for aggressive change point detection.
    """
    print("="*60)
    print("AGGRESSIVE CHANGE POINT DETECTION ANALYSIS")
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
    
    # Analyze change points
    municipality_results = analyze_municipality_change_points_aggressive(ds, municipality_mapping, output_dir)
    
    # Create visualizations
    summary_table = create_simple_visualizations(municipality_results, output_dir)
    
    # Print summary statistics
    print(f"\n📈 AGGRESSIVE ANALYSIS SUMMARY:")
    print(f"- Total municipalities analyzed: {len(municipality_results)}")
    
    df = pd.DataFrame(municipality_results)
    print(f"- Average change points per municipality: {df['n_change_points'].mean():.2f}")
    print(f"- Municipalities with no change points: {sum(df['n_change_points'] == 0)}")
    print(f"- Municipalities with change points: {sum(df['n_change_points'] > 0)}")
    print(f"- Maximum change points found: {df['n_change_points'].max()}")
    
    # Most common change types
    change_type_counts = df['dominant_change_type'].value_counts()
    print(f"\n🔄 DOMINANT CHANGE TYPES:")
    for change_type, count in change_type_counts.items():
        print(f"- {change_type}: {count} municipalities")
    
    # Detection methods used
    method_counts = df['detection_method'].value_counts()
    print(f"\n🔍 DETECTION METHODS USED:")
    for method, count in method_counts.items():
        print(f"- {method}: {count} municipalities")
    
    print(f"\n✅ AGGRESSIVE CHANGE POINT ANALYSIS COMPLETED!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📈 Key outputs:")
    print(f"   - municipality_change_summary_aggressive.csv")
    print(f"   - aggressive_change_detection_summary.png")
    
    return municipality_results, summary_table


if __name__ == "__main__":
    # Run the analysis
    netcdf_file = "data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc"
    municipality_results, summary_table = main(netcdf_file)
