"""
Municipality and Natural Park NDVI Category Bar Charts Generator

Creates stacked bar charts showing NDVI category distributions over time for:
- Each municipality (general statistics)
- Each natural park (PEIN and XPN) with aggregated data

The charts exclude the "Water" category and stack all other NDVI categories vertically.
"""

# ==== CONFIGURABLE PARAMETERS ====
MUNICIPALITY_STATS_JSON = "outputs/municipality_statistics/municipality_ndvi_statistics.json"
OUTPUT_DIR = "outputs/ndvi_category_charts"
FIGURE_SIZE = (14, 8)
DPI = 300
# =================================

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict

# Configure logger
logger.add(
    f"logs/ndvi_category_charts_{datetime.now().strftime('%Y-%m-%d')}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# NDVI categories (excluding Water)
NDVI_CATEGORIES = [
    "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"
]

# Color palette for NDVI categories (light to dark green)
CATEGORY_COLORS = [
    '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476',
    '#41ab5d', '#238b45', '#006d2c', '#00441b', '#002d14'
]


def load_municipality_stats(json_path):
    """Load municipality NDVI statistics from JSON file."""
    logger.info(f"Loading municipality statistics from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_category_data_by_year(yearly_data):
    """Extract category counts organized by year, excluding Water."""
    if not yearly_data:
        logger.warning("No yearly data provided to extract_category_data_by_year")
        return [], {}
    
    years = sorted([int(year) for year in yearly_data.keys()])
    category_data = {category: [] for category in NDVI_CATEGORIES}
    
    logger.info(f"Extracting data for years: {years}")
    logger.info(f"Yearly data keys available: {list(yearly_data.keys())}")
    
    # Debug: Check structure of first year's data
    if yearly_data:
        first_year = next(iter(yearly_data.keys()))
        first_year_data = yearly_data[first_year]
        logger.info(f"First year ({first_year}) data structure: {type(first_year_data)}")
        if isinstance(first_year_data, dict):
            logger.info(f"First year ({first_year}) data keys: {list(first_year_data.keys())}")
            if 'category_counts' in first_year_data:
                logger.info(f"First year ({first_year}) category_counts: {first_year_data['category_counts']}")
        else:
            logger.info(f"First year ({first_year}) data content: {first_year_data}")
    
    for year in years:
        year_str = str(year)
        if year_str in yearly_data:
            year_data_item = yearly_data[year_str]
            logger.debug(f"Processing year {year}, data type: {type(year_data_item)}")
            
            # Handle both direct category_counts dict and nested structure
            if isinstance(year_data_item, dict) and 'category_counts' in year_data_item:
                category_counts = year_data_item['category_counts']
            elif isinstance(year_data_item, dict):
                # Maybe the yearly_data is already the category_counts dict
                category_counts = year_data_item
            else:
                logger.warning(f"Unexpected data structure for year {year}: {type(year_data_item)}")
                category_counts = {}
            
            year_total = 0
            for category in NDVI_CATEGORIES:
                count = category_counts.get(category, 0)
                category_data[category].append(count)
                year_total += count
            logger.debug(f"Year {year}: {year_total} total pixels, categories: {[category_counts.get(cat, 0) for cat in NDVI_CATEGORIES[:3]]}...")
        else:
            # Missing year data
            for category in NDVI_CATEGORIES:
                category_data[category].append(0)
            logger.warning(f"Missing data for year {year}")
    
    # Log summary
    total_pixels = sum(sum(category_data[cat]) for cat in NDVI_CATEGORIES)
    logger.info(f"Total pixels across all years and categories: {total_pixels}")
    
    return years, category_data


def create_stacked_bar_chart(years, category_data, title, output_path):
    """Create a stacked bar chart for NDVI categories over time."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Convert to numpy arrays for stacking
    bottoms = np.zeros(len(years))
    
    # Create stacked bars
    for i, category in enumerate(NDVI_CATEGORIES):
        values = np.array(category_data[category])
        ax.bar(years, values, bottom=bottoms, 
               label=category, color=CATEGORY_COLORS[i], 
               edgecolor='white', linewidth=0.5)
        bottoms += values
    
    # Customize the plot
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Pixel Count', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(np.arange(min(years), max(years)+1, 5))
    ax.set_xlim(min(years)-0.5, max(years)+0.5)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              title='NDVI Categories', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis
    ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved chart: {output_path}")


def create_municipality_charts(data, output_dir):
    """Create bar charts for all municipalities."""
    logger.info("Creating municipality charts...")
    
    municipalities = data.get('municipalities', {})
    municipality_dir = output_dir / "municipalities"
    municipality_dir.mkdir(parents=True, exist_ok=True)
    
    for municipality_name, municipality_data in municipalities.items():
        logger.info(f"Processing municipality: {municipality_name}")
        
        # Extract general yearly data
        general_yearly = municipality_data.get('general', {}).get('yearly', {})
        
        if not general_yearly:
            logger.warning(f"No yearly data found for {municipality_name}")
            continue
        
        # Extract category data
        years, category_data = extract_category_data_by_year(general_yearly)
        
        # Create chart
        title = f"NDVI Category Distribution - {municipality_name}"
        filename = f"{municipality_name.replace(' ', '_').replace('/', '_')}_ndvi_categories.png"
        output_path = municipality_dir / filename
        
        create_stacked_bar_chart(years, category_data, title, output_path)


def aggregate_park_data(municipalities_data, park_type):
    """Aggregate data for natural parks across municipalities."""
    logger.info(f"Aggregating {park_type.upper()} park data...")
    
    park_aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_parks_found = 0
    municipalities_with_parks = 0
    
    for municipality_name, municipality_data in municipalities_data.items():
        parks_data = municipality_data.get(park_type, {})
        
        if parks_data:
            municipalities_with_parks += 1
            logger.info(f"Municipality {municipality_name} has {len(parks_data)} {park_type.upper()} parks: {list(parks_data.keys())}")
            
            for park_name, park_data in parks_data.items():
                total_parks_found += 1
                
                # Check if park_data is empty or has no yearly data
                if not park_data or not isinstance(park_data, dict):
                    logger.warning(f"Empty park data for {park_type.upper()} park '{park_name}' in {municipality_name}")
                    continue
                    
                yearly_data = park_data.get('yearly', {})
                
                if not yearly_data:
                    logger.warning(f"No yearly data found for {park_type.upper()} park '{park_name}' in {municipality_name}")
                    continue
                
                logger.info(f"Processing {park_type.upper()} park '{park_name}' with {len(yearly_data)} years of data")
                
                # Debug: Print sample of yearly data structure
                if yearly_data:
                    sample_year = next(iter(yearly_data.keys()))
                    sample_data = yearly_data[sample_year]
                    logger.info(f"Sample year {sample_year} data structure: {list(sample_data.keys())}")
                    if 'category_counts' in sample_data:
                        category_counts = sample_data['category_counts']
                        total_year_pixels = sum(category_counts.values())
                        logger.info(f"Sample year {sample_year} has {total_year_pixels} total pixels: {category_counts}")
                
                for year, year_data in yearly_data.items():
                    category_counts = year_data.get('category_counts', {})
                    
                    if not category_counts:
                        logger.warning(f"No category counts for {park_type.upper()} park '{park_name}' in {municipality_name} for year {year}")
                        continue
                    
                    year_total = sum(category_counts.values())
                    logger.debug(f"Park '{park_name}' year {year}: {year_total} pixels")
                    
                    for category, count in category_counts.items():
                        park_aggregated[park_name][year][category] += count
        else:
            logger.debug(f"Municipality {municipality_name} has no {park_type.upper()} parks")
    
    logger.info(f"Found {total_parks_found} {park_type.upper()} park instances across {municipalities_with_parks} municipalities")
    logger.info(f"Unique {park_type.upper()} parks: {list(park_aggregated.keys())}")
    
    # Log aggregated data summary
    parks_with_data = 0
    for park_name, park_yearly_data in park_aggregated.items():
        years_count = len(park_yearly_data)
        total_pixels = sum(sum(category_counts.values()) for category_counts in park_yearly_data.values())
        if total_pixels > 0:
            parks_with_data += 1
            logger.info(f"{park_type.upper()} park '{park_name}': {years_count} years, {total_pixels} total pixels")
        else:
            logger.warning(f"{park_type.upper()} park '{park_name}': {years_count} years, but 0 total pixels (empty data)")
    
    logger.info(f"Summary: {parks_with_data} out of {len(park_aggregated)} {park_type.upper()} parks have actual data")
    
    return park_aggregated


def create_park_charts(data, output_dir):
    """Create bar charts for all natural parks (PEIN and XPN)."""
    logger.info("Creating natural park charts...")
    
    municipalities = data.get('municipalities', {})
    
    # Process PEIN parks
    pein_dir = output_dir / "pein_parks"
    pein_dir.mkdir(parents=True, exist_ok=True)
    
    pein_aggregated = aggregate_park_data(municipalities, 'pein')
    
    for park_name, park_yearly_data in pein_aggregated.items():
        logger.info(f"Processing PEIN park: {park_name}")
        
        # Check if park has any actual data
        if not park_yearly_data:
            logger.warning(f"No aggregated data found for PEIN park: {park_name}, skipping")
            continue
        
        # Extract category data
        years, category_data = extract_category_data_by_year(park_yearly_data)
        
        if not years:
            logger.warning(f"No yearly data found for PEIN park: {park_name}")
            continue
            
        # Check if there's any actual pixel data
        total_pixels = sum(sum(category_data[cat]) for cat in NDVI_CATEGORIES)
        if total_pixels == 0:
            logger.warning(f"No pixel data found for PEIN park: {park_name}, skipping chart creation")
            continue
        
        # Create chart
        title = f"NDVI Category Distribution - PEIN: {park_name}"
        filename = f"PEIN_{park_name.replace(' ', '_').replace('/', '_')}_ndvi_categories.png"
        output_path = pein_dir / filename
        
        create_stacked_bar_chart(years, category_data, title, output_path)
    
    # Process XPN parks
    xpn_dir = output_dir / "xpn_parks"
    xpn_dir.mkdir(parents=True, exist_ok=True)
    
    xpn_aggregated = aggregate_park_data(municipalities, 'xpn')
    
    for park_name, park_yearly_data in xpn_aggregated.items():
        logger.info(f"Processing XPN park: {park_name}")
        
        # Check if park has any actual data
        if not park_yearly_data:
            logger.warning(f"No aggregated data found for XPN park: {park_name}, skipping")
            continue
        
        # Extract category data
        years, category_data = extract_category_data_by_year(park_yearly_data)
        
        if not years:
            logger.warning(f"No yearly data found for XPN park: {park_name}")
            continue
            
        # Check if there's any actual pixel data
        total_pixels = sum(sum(category_data[cat]) for cat in NDVI_CATEGORIES)
        if total_pixels == 0:
            logger.warning(f"No pixel data found for XPN park: {park_name}, skipping chart creation")
            continue
        
        # Create chart
        title = f"NDVI Category Distribution - XPN: {park_name}"
        filename = f"XPN_{park_name.replace(' ', '_').replace('/', '_')}_ndvi_categories.png"
        output_path = xpn_dir / filename
        
        create_stacked_bar_chart(years, category_data, title, output_path)


def main():
    """Main function to generate all NDVI category bar charts."""
    logger.info("Starting NDVI category charts generation")
    
    # Load data
    try:
        data = load_municipality_stats(MUNICIPALITY_STATS_JSON)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create municipality charts
    try:
        create_municipality_charts(data, output_dir)
        logger.success("Municipality charts completed")
    except Exception as e:
        logger.error(f"Error creating municipality charts: {e}")
    
    # Create natural park charts
    try:
        create_park_charts(data, output_dir)
        logger.success("Natural park charts completed")
    except Exception as e:
        logger.error(f"Error creating park charts: {e}")
    
    # Print summary
    municipality_count = len(data.get('municipalities', {}))
    print(f"\n=== NDVI CATEGORY CHARTS SUMMARY ===")
    print(f"Charts generated for {municipality_count} municipalities")
    print(f"Charts generated for PEIN and XPN natural parks")
    print(f"Output directory: {output_dir}")
    print(f"Chart resolution: {DPI} DPI")
    
    logger.success("NDVI category charts generation completed")


if __name__ == "__main__":
    main()
