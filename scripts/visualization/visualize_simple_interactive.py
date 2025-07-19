"""
Simple interactive visualization script for ALL AMB municipalities NetCDF data.
Memory-efficient version without complex features.
"""
import xarray as xr
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_data_with_municipalities(file_path):
    """Load NetCDF data with municipality information."""
    print(f"Loading data from: {file_path}")
    ds = xr.open_dataset(file_path)
    print(f"Data loaded: {dict(ds.dims)}")
    
    # Load municipality mapping if available
    municipality_mapping = None
    mapping_file = Path(file_path).parent / "municipality_mapping.csv"
    if mapping_file.exists():
        municipality_mapping = pd.read_csv(mapping_file)
        print(f"Municipality mapping loaded: {len(municipality_mapping)} municipalities")
    else:
        print("Municipality mapping not found - overall analysis only")
    
    return ds, municipality_mapping


def create_simple_time_series(ds, output_dir="outputs/interactive"):
    """Create simple time series plot."""
    print("Creating simple time series...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate overall statistics
    ndvi_mean = ds['ndvi'].mean(dim=['x', 'y'])
    ndvi_std = ds['ndvi'].std(dim=['x', 'y'])
    
    # Create figure
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=ds.time.values,
        y=ndvi_mean.values,
        mode='lines+markers',
        name='Mean NDVI',
        line=dict(color='blue', width=3),
        hovertemplate='Year: %{x}<br>NDVI: %{y:.3f}<extra></extra>'
    ))
    
    # Add confidence band
    fig.add_trace(go.Scatter(
        x=ds.time.values,
        y=ndvi_mean.values + ndvi_std.values,
        mode='lines',
        line=dict(color='rgba(0,0,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=ds.time.values,
        y=ndvi_mean.values - ndvi_std.values,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(0,0,255,0)'),
        name='±1 Std Dev',
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title='Interactive NDVI Time Series - All AMB Municipalities',
        xaxis_title='Year',
        yaxis_title='NDVI',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    # Save
    output_file = output_path / "simple_time_series.html"
    fig.write_html(output_file)
    print(f"Simple time series saved: {output_file}")
    return fig


def create_simple_spatial_map(ds, output_dir="outputs/interactive"):
    """Create simple spatial map."""
    print("Creating simple spatial map...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select middle year
    middle_year_idx = len(ds.time) // 2
    year_data = ds['ndvi'].isel(time=middle_year_idx)
    year = int(ds.time.isel(time=middle_year_idx).values)
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=year_data.values,
        x=ds.x.values,
        y=ds.y.values,
        colorscale='RdYlGn',
        zmin=-0.5,
        zmax=1.0,
        colorbar=dict(title="NDVI"),
        hovertemplate='Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>NDVI: %{z:.3f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'NDVI Spatial Map - All AMB Municipalities ({year})',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        template='plotly_white',
        height=600
    )
    
    # Save
    output_file = output_path / "simple_spatial_map.html"
    fig.write_html(output_file)
    print(f"Simple spatial map saved: {output_file}")
    return fig


def main(netcdf_path="data/processed/landsat_multidimensional_ALL_AMB_municipalities.nc"):
    """Main function to generate simple visualizations."""
    print("="*50)
    print("SIMPLE INTERACTIVE VISUALIZATIONS")
    print("="*50)
    
    # Load data
    ds, municipality_mapping = load_data_with_municipalities(netcdf_path)
    
    # Print dataset information
    print(f"\n📊 Dataset Information:")
    print(f"- Total pixels: {ds.dims['x']} x {ds.dims['y']}")
    print(f"- Time range: {ds.time.min().values} - {ds.time.max().values}")
    print(f"- Variables: {list(ds.data_vars.keys())}")
    print(f"- NDVI range: {float(ds.ndvi.min()):.3f} to {float(ds.ndvi.max()):.3f}")
    
    if municipality_mapping is not None:
        print(f"- Municipalities: {len(municipality_mapping)}")
    
    # Set output directory
    output_dir = "outputs/interactive"
    
    print(f"\n{'='*30}")
    print("GENERATING VISUALIZATIONS")
    print("="*30)
    
    try:
        # 1. Simple time series
        create_simple_time_series(ds, output_dir)
        
        # 2. Simple spatial map
        create_simple_spatial_map(ds, output_dir)
        
        print(f"\n✅ SIMPLE VISUALIZATIONS COMPLETED!")
        print(f"📁 Check the '{output_dir}' folder for HTML files")
        print(f"🌐 Open the HTML files in your web browser")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    netcdf_file = "data/processed/landsat_mdim_all_muni.nc"
    main(netcdf_file)
