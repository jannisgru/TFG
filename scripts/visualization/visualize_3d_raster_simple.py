"""
3D Visualization of Multidimensional Raster Data
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

logger.remove()  # Remove default handler

def format_log_message(record):
    """Custom formatter"""
    time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S")
    level = record["level"].name
    message = record["message"]
    
    if level == "SUCCESS":
        return f"\033[92m{time_str} | {level} | {message}\033[0m"  # Green
    else:
        return f"{time_str} | {level} | {message}"

logger.add(lambda msg: print(msg), format=format_log_message, level="INFO")


def load_and_downsample_data(file_path, downsample_factor=5):
    """Load and heavily downsample the multidimensional NetCDF data."""
    logger.info(f"Loading NetCDF data from {Path(file_path).name}")
    
    ds = xr.open_dataset(file_path)
    
    # Downsample by selecting every nth pixel
    x_indices = np.arange(0, ds.dims['x'], downsample_factor)
    y_indices = np.arange(0, ds.dims['y'], downsample_factor)
    ds_downsampled = ds.isel(x=x_indices, y=y_indices)
    
    logger.info(f"Data downsampled by factor {downsample_factor} → {ds_downsampled.dims['y']} x {ds_downsampled.dims['x']} pixels")
    
    return ds_downsampled


def create_3d_raster_8_layers(ds, output_dir="outputs/3d_raster"):
    """Create 3D raster visualization"""
    logger.info("Creating 3D visualization with 8 time layers (1985-2020)")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select data variable and visualization parameters
    if 'ndvi' in ds.variables:
        data_array = ds['ndvi']
        colorscale = 'RdYlGn'
        title_suffix = 'NDVI'
        zmin, zmax = -0.5, 1.0
    else:
        data_var = list(ds.data_vars.keys())[0]
        data_array = ds[data_var]
        colorscale = 'Viridis'
        title_suffix = data_var
        zmin, zmax = None, None
    
    # Define target years in 5-year steps from 1985 to 2020
    target_years = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
    
    # Find closest time indices for each target year
    all_years = [pd.to_datetime(time_val).year for time_val in ds.time.values]
    time_indices = []
    
    for target_year in target_years:
        closest_idx = min(range(len(all_years)), 
                         key=lambda i: abs(all_years[i] - target_year))
        time_indices.append(closest_idx)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Get coordinate grids
    x_coords = ds.x.values
    y_coords = ds.y.values
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Add each time step as a surface layer
    for i, time_idx in enumerate(time_indices):
        time_data = data_array.isel(time=time_idx).values
        valid_mask = ~np.isnan(time_data)
        
        target_year = target_years[i]
        
        # Set Z coordinate (height) using the target year
        Z = np.full_like(X, target_year)
        Z_masked = np.where(valid_mask, Z, np.nan)
        time_data_masked = np.where(valid_mask, time_data, np.nan)
        
        # Add surface
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z_masked,
            surfacecolor=time_data_masked,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            opacity=0.8,
            showscale=False,
            name=f'Year {target_year}',
            hovertemplate=(f'Year: {target_year}<br>X: %{{x:.4f}}<br>'
                          f'Y: %{{y:.4f}}<br>{title_suffix}: %{{surfacecolor:.3f}}<extra></extra>')
        ))
    
    # Update layout with proper axis labels and explicit Z-axis ticks
    fig.update_layout(
        title=dict(
            text=f'Visualization of Multidimensional Raster Data<br>{title_suffix} Time Series',
            font=dict(size=24, family='Arial Black')
        ),        
        scene=dict(
            xaxis=dict(
                title=dict(text='Longitude', font=dict(size=20, color='black', family='Arial Black')),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='Latitude', font=dict(size=20, color='black', family='Arial Black')),
                tickfont=dict(size=14)
            ),
            zaxis=dict(
                title=dict(text='    Year    ', font=dict(size=20, color='black', family='Arial Black')),
                tickmode='array',
                tickvals=target_years,
                ticktext=[str(year) for year in target_years],
                tickfont=dict(size=14)
            ),
            camera=dict(eye=dict(x=1.2, y=-1.2, z=0.8)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4)  
        ),
        height=1000,
        width=1400,
        showlegend=True
    )
    
    # Save the visualization
    output_file = output_path / "3d_raster_8_layers.html"
    fig.write_html(output_file)
    logger.success(f"3D visualization saved to {output_file}")
    
    return fig


def main(netcdf_path="data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc",
         downsample_factor=5):
    """Main function to create 3D raster visualization."""
    
    # Load and downsample data
    ds = load_and_downsample_data(netcdf_path, downsample_factor)
    
    # Create the visualization
    fig = create_3d_raster_8_layers(ds)
    
    return ds, fig


if __name__ == "__main__":
    main(downsample_factor=5)