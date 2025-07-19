"""
Simple interactive visualization script for ALL AMB municipalities NetCDF data.
Memory-efficient version without complex features.
"""

# ==== CONFIGURABLE PARAMETERS ====
NETCDF_PATH = "data/processed/landsat_mdim_all_muni.nc"
OUTPUT_DIR = "outputs/interactive"
TIME_SERIES_NAME = "simple_time_series.html"
SPATIAL_MAP_NAME = "simple_spatial_map.html"
LOG_PATH = "logs/visualization_{time:YYYY-MM-DD}.log"
# ================================

import xarray as xr
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# Configure loguru
logger.add(
    LOG_PATH,
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def load_data_with_municipalities(file_path):
    """Load NetCDF data with municipality information."""
    logger.info(f"Loading data from: {file_path}")
    ds = xr.open_dataset(file_path)
    mapping_file = Path(file_path).parent / "municipality_mapping.csv"
    municipality_mapping = pd.read_csv(mapping_file) if mapping_file.exists() else None
    logger.info(f"Data loaded: {dict(ds.dims)}")
    if municipality_mapping is not None:
        logger.info(f"Municipality mapping loaded: {len(municipality_mapping)} municipalities")
    else:
        logger.info("Municipality mapping not found - overall analysis only")
    return ds


def create_simple_time_series(ds, output_dir=OUTPUT_DIR, output_name=TIME_SERIES_NAME):
    """Create simple time series plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ndvi_mean = ds['ndvi'].mean(dim=['x', 'y'])
    ndvi_p25 = ds['ndvi'].quantile(0.2, dim=['x', 'y'])
    ndvi_p75 = ds['ndvi'].quantile(0.8, dim=['x', 'y'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ds.time.values,
        y=ndvi_mean.values,
        mode='lines+markers',
        name='Mean NDVI',
        line=dict(color='blue', width=3),
        hovertemplate='Year: %{x}<br>NDVI: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=ds.time.values,
        y=ndvi_p75.values,
        mode='lines',
        line=dict(color='rgba(0,0,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=ds.time.values,
        y=ndvi_p25.values,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(0,0,255,0)'),
        name='25th-75th Percentile Range',
        hoverinfo='skip'
    ))
    fig.update_layout(
        title='Interactive NDVI Time Series - All AMB Municipalities',
        xaxis_title='Year',
        yaxis_title='NDVI',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        yaxis=dict(range=[0, 1])
    )
    output_file = output_path / output_name
    fig.write_html(output_file, include_plotlyjs='cdn', auto_play=False)
    logger.success(f"Time series saved: {output_file}")
    return fig


def create_simple_spatial_map(ds, output_dir=OUTPUT_DIR, output_name=SPATIAL_MAP_NAME):
    """Create memory-efficient spatial map with time slider using animation frames."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Downsample NDVI and coordinates by factor of 2 (60m pixels)
    ndvi_ds = ds['ndvi'][:, ::2, ::2]
    x_ds = ds.x.values[::2]
    y_ds = ds.y.values[::2]

    # Convert time values to years
    years = pd.to_datetime(ds.time.values).year

    # Initial frame (first year)
    first_year_data = ndvi_ds.isel(time=0)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=first_year_data.values,
                x=x_ds,
                y=y_ds,
                colorscale='RdYlGn',
                zmin=-0.5,
                zmax=1.0,
                colorbar=dict(title="NDVI"),
                hovertemplate=f'Year: {years[0]}<br>Lon: %{{x:.4f}}<br>Lat: %{{y:.4f}}<br>NDVI: %{{z:.3f}}<extra></extra>',
            )
        ],
        frames=[
            go.Frame(
                data=[
                    go.Heatmap(
                        z=ndvi_ds.isel(time=i).values,
                        x=x_ds,
                        y=y_ds,
                        colorscale='RdYlGn',
                        zmin=-0.5,
                        zmax=1.0,
                        colorbar=dict(title="NDVI"),
                        hovertemplate=f'Year: {years[i]}<br>Lon: %{{x:.4f}}<br>Lat: %{{y:.4f}}<br>NDVI: %{{z:.3f}}<extra></extra>',
                    )
                ],
                name=str(years[i])
            )
            for i in range(len(years))
        ]
    )

    # Add slider and play/stop buttons
    fig.update_layout(
        title='NDVI Spatial Map - All AMB Municipalities (Interactive Time Slider, 60m pixels)',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        template='plotly_white',
        height=900,
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.03,
            y=-0.1,
            showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                dict(label="Stop", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}])
            ]
        )],
        sliders=[dict(
            steps=[
                dict(method="animate", args=[[str(years[i])], {"mode": "immediate"}], label=str(years[i]))
                for i in range(len(years))
            ],
            active=0,
            currentvalue={"prefix": "Year: "},
            pad={"t": 80},  # Move slider further down to make space for buttons
            x=0.05,
            y=0.05,
        )]
    )
    # Make pixels appear square
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    output_file = output_path / output_name
    fig.write_html(output_file, include_plotlyjs='cdn', auto_play=False)
    logger.success(f"Spatial map with time slider saved: {output_file}")
    return fig


def main(netcdf_path=NETCDF_PATH, output_dir=OUTPUT_DIR):
    try:
        ds = load_data_with_municipalities(netcdf_path)
        create_simple_time_series(ds, output_dir)
        create_simple_spatial_map(ds, output_dir)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
