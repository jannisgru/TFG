"""
Compare average NDVI value for each municipality per year.
Creates an interactive line plot with buttons to switch between comarca groups.
"""

# ==== CONFIGURABLE PARAMETERS ====
NETCDF_PATH = "data/processed/landsat_mdim_all_muni.nc"
OUTPUT_DIR = "outputs/avg_ndvi_per_municipality"
OUTPUT_NAME = "avg_ndvi_per_municipality.html"
LOG_PATH = "logs/avg_ndvi_muni_{time:YYYY-MM-DD}.log"
# =================================

import xarray as xr
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from loguru import logger
import warnings
import numpy as np
import plotly.colors as pc

warnings.filterwarnings('ignore')

# Configure loguru
logger.add(
    LOG_PATH,
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def get_comarca_grouping():
    """Define municipality groupings by comarca for Barcelona Metropolitan Area."""
    return {
        'Baix Llobregat South': [
            'Begues', 'Gavà', 'Castelldefels', 'Viladecans', 'El Prat de Llobregat',
            'Sant Boi de Llobregat', 'Sant Climent de Llobregat', 'Torrelles de Llobregat',
            'Santa Coloma de Cervelló', 'Sant Joan Despí', 'Cornellà de Llobregat'
        ],
        'Baix Llobregat North': [
            'Esplugues de Llobregat', 'Sant Just Desvern', 'Sant Feliu de Llobregat',
            'Sant Vicenç dels Horts', 'Molins de Rei', 'El Papiol', 'Sant Andreu de la Barca',
            'Corbera de Llobregat', 'La Palma de Cervelló', 'Cervelló', 'Pallejà'
        ],
        'Barcelonès': [
            'Sarrià - Sant Gervasi', 'Sants-Montjuic', 'Sant Martí', 'Les Corts',
            'Ciutat Vella', "L'Eixample", 'Gràcia', 'Horta - Guinardó', 'Nou Barris',
            'Sant Andreu', 'Hospitalet de Llobregat', 'Badalona',
            'Santa Coloma de Gramenet', 'Sant Adrià del Besòs'
        ],
        'Vallès + Maresme': [
            'Cerdanyola del Vallès', 'Barberà del Vallès', 'Ripollet', 'Badia del Vallès',
            'Montcada i Reixac', 'Sant Cugat del Vallès', 'Castellbisbal',
            'Tiana', 'Montgat'
        ]
    }


def compute_avg_ndvi(ds):
    """Compute average NDVI for each municipality per year using pure numpy operations.
    Args:
        ds (xarray.Dataset): The loaded NetCDF dataset.
    Returns:
        pd.DataFrame: DataFrame with columns ['municipality', 'year', 'mean_ndvi'].
    """
    logger.info("Computing average NDVI per municipality per year...")

    # Get municipality names and IDs from NetCDF attributes
    if 'municipality_names' in ds.attrs and 'municipality_id' in ds:
        muni_names = ds.attrs['municipality_names'].split(', ')
        muni_ids = np.arange(1, len(muni_names) + 1)
    else:
        logger.error("municipality_names attribute or municipality_id variable not found in NetCDF dataset.")
        raise ValueError("Municipality information not found in NetCDF dataset.")

    years = pd.to_datetime(ds.time.values).year  # Extract years from time dimension
    ndvi_data = ds['ndvi'].values                # NDVI values (time, x, y)
    muni_id_data = ds['municipality_id'].values  # Municipality ID mask (x, y) or (1, x, y)

    # Ensure municipality ID mask is 2D
    if muni_id_data.ndim == 3:
        muni_id_2d = muni_id_data[0]
    else:
        muni_id_2d = muni_id_data

    # Create boolean masks for each municipality
    muni_masks = {muni_id: muni_id_2d == muni_id for muni_id in muni_ids}
    df_data = []

    # Loop over each year and municipality to compute mean NDVI
    if ndvi_data.ndim == 3:
        for t_idx, year in enumerate(years):
            ndvi_slice = ndvi_data[t_idx]
            for muni_idx, muni_id in enumerate(muni_ids):
                mask = muni_masks[muni_id]
                if np.any(mask):  # Only compute if mask has valid pixels
                    mean_ndvi = np.nanmean(ndvi_slice[mask])
                    df_data.append({
                        'municipality': muni_names[muni_idx],
                        'year': year,
                        'mean_ndvi': float(mean_ndvi)
                    })
    else:
        raise ValueError(f"Unexpected NDVI array shape: {ndvi_data.shape}. Expected 3D.")

    df = pd.DataFrame(df_data)
    logger.info(f"Computed NDVI for {len(muni_ids)} municipalities over {len(years)} years.")
    return df


def create_ndvi_traces(df, comarca_groups):
    """Generate Plotly traces for each municipality in each group.
    Args:
        df (pd.DataFrame): DataFrame with NDVI data.
        comarca_groups (dict): Grouping of municipalities.
    Returns:
        list: List of (group_name, trace) tuples.
    """
    # Combine several color palettes for variety
    color_palettes = [
        pc.qualitative.Plotly, pc.qualitative.D3, pc.qualitative.Set1,
        pc.qualitative.Set2, pc.qualitative.Set3, pc.qualitative.Dark24, pc.qualitative.Light24
    ]
    all_colors = [color for palette in color_palettes for color in palette]
    traces = []
    color_index = 0

    # Iterate over groups and municipalities to create traces
    for group_idx, (group_name, municipalities) in enumerate(comarca_groups.items()):
        available_munis = [muni for muni in municipalities if muni in df['municipality'].unique()]
        for muni_idx, muni in enumerate(sorted(available_munis)):
            group_data = df[df['municipality'] == muni].sort_values('year')
            if not group_data.empty:
                muni_color = all_colors[color_index % len(all_colors)]
                color_index += 1
                line_styles = ['solid', 'dash', 'dot', 'dashdot']
                line_style = line_styles[muni_idx % len(line_styles)]
                line_width = 2 + (muni_idx % 3) * 0.5
                trace = go.Scatter(
                    x=group_data['year'],
                    y=group_data['mean_ndvi'],
                    mode='lines+markers',
                    name=muni,
                    line=dict(color=muni_color, dash=line_style, width=line_width),
                    marker=dict(color=muni_color, size=6, symbol='circle'),
                    hovertemplate='Municipality: %{text}<br>Year: %{x}<br>NDVI: %{y:.3f}<extra></extra>',
                    text=[muni]*len(group_data),
                    visible=(group_idx == 0)  # Only first group visible by default
                )
                traces.append((group_name, trace))
    return traces


def create_comarca_buttons(traces, comarca_groups):
    """Create button list for each comarca group for interactive filtering.
    Args:
        traces (list): List of (group_name, trace) tuples.
        comarca_groups (dict): Grouping of municipalities.
    Returns:
        list: List of button dicts for Plotly.
    """
    buttons = []
    for group_idx, (group_name, _) in enumerate(comarca_groups.items()):
        visibility = [trace_group_name == group_name for trace_group_name, _ in traces]
        buttons.append(dict(label=group_name, method="update", args=[{"visible": visibility}]))
    buttons.append(dict(label="Show All Groups", method="update", args=[{"visible": [True] * len(traces)}]))
    return buttons


def plot_avg_ndvi(df, output_dir=OUTPUT_DIR, output_name=OUTPUT_NAME):
    """Create interactive line plot and save, logging results.
    Args:
        df (pd.DataFrame): DataFrame with NDVI data.
        output_dir (str): Directory to save the plot.
        output_name (str): Output HTML file name.
    Returns:
        plotly.graph_objects.Figure: The generated figure.
    """
    logger.info("Creating interactive plot with comarca group buttons...")
    comarca_groups = get_comarca_grouping()
    traces = create_ndvi_traces(df, comarca_groups)
    fig = go.Figure()
    for _, trace in traces:
        fig.add_trace(trace)
    buttons = create_comarca_buttons(traces, comarca_groups)

    # Configure layout, legend, buttons, and annotations
    fig.update_layout(
        title="Average NDVI per Municipality per Year - By Comarca Groups",
        xaxis_title="Year",
        yaxis_title="Mean NDVI",
        hovermode="x unified",
        template="plotly_white",
        height=700,
        yaxis=dict(range=[0, 1]),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=10)
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        annotations=[
            dict(
                text="Select Comarca Group:",
                showarrow=False,
                x=0.01,
                y=1.18,
                xref="paper",
                yref="paper",
                align="left",
                font=dict(size=12)
            )
        ]
    )

    # Save plot as HTML
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / output_name
    fig.write_html(output_file, include_plotlyjs='cdn', auto_play=False)
    logger.success(f"Plot saved: {output_file}")

    return fig


def main(netcdf_path=NETCDF_PATH, output_dir=OUTPUT_DIR):
    try:
        logger.info(f"Loading dataset from {netcdf_path}")
        ds = xr.open_dataset(netcdf_path)
        df = compute_avg_ndvi(ds)
        plot_avg_ndvi(df, output_dir)
        ds.close()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()