"""
3D Visualization of Multidimensional Raster Data
"""

# ==== CONFIGURABLE PARAMETERS ====
NETCDF_PATH = "data/processed/landsat_mdim_all_muni.nc"
DOWNSAMPLE_FACTOR = 2
OUTPUT_DIR = "outputs/3d_raster"
OUTPUT_NAME = "3d_raster_barcelona_test.html"
DISTRICTS_2_VISUALIZE = [
    "Ciutat Vella", "L'Eixample", "Gràcia", "Horta - Guinardó", "Les Corts",
    "Nou Barris", "Sant Andreu", "Sant Martí", "Sants-Montjuic", "Sarrià - Sant Gervasi"
] # Leave empty to visualize all municipalities
TARGET_YEARS = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
# ================================

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


def create_3d_raster_layers(ds, output_dir=OUTPUT_DIR, output_name=OUTPUT_NAME, target_years=TARGET_YEARS):
    """Create 3D raster visualization with a layer for each year in target_years."""
    logger.info(f"Creating 3D visualization with {len(target_years)} time layers ({', '.join(map(str, target_years))})")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / output_name

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
    
    # Find closest time indices for each target year
    all_years = [pd.to_datetime(time_val).year for time_val in ds.time.values]
    time_indices = []
    for target_year in target_years:
        closest_idx = min(range(len(all_years)), key=lambda i: abs(all_years[i] - target_year))
        time_indices.append(closest_idx)
    
    # Create 3D plot
    fig = go.Figure()
    x_coords = ds.x.values
    y_coords = ds.y.values
    X, Y = np.meshgrid(x_coords, y_coords)
    
    for i, time_idx in enumerate(time_indices):
        time_data = data_array.isel(time=time_idx).values
        valid_mask = ~np.isnan(time_data)
        year = target_years[i]
        Z = np.full_like(X, year)
        Z_masked = np.where(valid_mask, Z, np.nan)
        time_data_masked = np.where(valid_mask, time_data, np.nan)
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z_masked,
            surfacecolor=time_data_masked,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            showscale=False,
            name=f'Year {year}',
            hovertemplate=(f'Year: {year}<br>X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br><extra></extra>')
        ))
    
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
    
    fig.write_html(output_file)
    logger.success(f"3D visualization saved to {output_file}")
    return fig


def filter_by_municipality(ds, municipality_names):
    """
    Filter the dataset to include only pixels belonging to the specified municipalities.
    If municipality_names is empty, return the dataset unchanged.
    """
    if not municipality_names:
        logger.info("No municipality filter applied; using all data.")
        return ds
    else:
        logger.info(f"Filtering dataset for municipalities...")
    
    # Check if municipality_id and municipality_names are present
    if 'municipality_id' not in ds or 'municipality_names' not in ds.attrs:
        raise ValueError("Dataset does not contain municipality information.")

    # Get mapping from id to name
    id_to_name = {i+1: name for i, name in enumerate(ds.attrs['municipality_names'].split(', '))}
    # Find municipality IDs for the requested names
    selected_ids = [mid for mid, name in id_to_name.items() if name in municipality_names]
    if not selected_ids:
        raise ValueError("No matching municipality IDs found for the given names.")

    # Create mask for selected municipalities
    muni_mask = np.isin(ds['municipality_id'].values, selected_ids)
    # Mask all variables
    ds_filtered = ds.copy()
    for var in ds.data_vars:
        if ds[var].dims[-2:] == ('y', 'x'):
            ds_filtered[var] = ds[var].where(muni_mask)
    ds_filtered['municipality_id'] = ds['municipality_id'].where(muni_mask)
    # Update attrs
    ds_filtered.attrs['municipality_names'] = ', '.join([id_to_name[mid] for mid in selected_ids])
    ds_filtered.attrs['filtered_municipality_ids'] = selected_ids
    return ds_filtered


def main(netcdf_path=NETCDF_PATH, downsample_factor=DOWNSAMPLE_FACTOR,
         output_dir=OUTPUT_DIR, output_name=OUTPUT_NAME, target_years=TARGET_YEARS):
    """Main function to create 3D raster visualization for selected districts and years."""
    ds = load_and_downsample_data(netcdf_path, downsample_factor)
    ds_barcelona = filter_by_municipality(ds, DISTRICTS_2_VISUALIZE)
    fig = create_3d_raster_layers(ds_barcelona, output_dir, output_name, target_years)
    return ds_barcelona, fig

if __name__ == "__main__":
    main()