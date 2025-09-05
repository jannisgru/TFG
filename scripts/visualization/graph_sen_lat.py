import matplotlib.pyplot as plt

# NDVI data for Landsat and Sentinel 2 from 2017 to 2025
years = list(range(2017, 2026))

# Sant Martí data (2017-2025)
ndvi_landsat_sant_marti = [0.333, 0.330, 0.335, 0.349, 0.325, 0.337, 0.329, 0.333, 0.312]
ndvi_sentinel_sant_marti = [0.365, 0.350, 0.373, 0.382, 0.390, 0.343, 0.357, 0.365, 0.320]

# Torrelles de Llobregat data (2017-2025)
ndvi_landsat_torrelles = [0.739, 0.734, 0.735, 0.754, 0.721, 0.706, 0.691, 0.733, 0.684]
ndvi_sentinel_torrelles = [0.840, 0.833, 0.786, 0.793, 0.794, 0.770, 0.792, 0.774, 0.766]

# Create the plot
plt.figure(figsize=(10, 5))

# Sant Martí data
plt.plot(years, ndvi_landsat_sant_marti, marker='o', label='Sant Martí - Landsat 8 NDVI', color='darkgreen', linewidth=2)
plt.plot(years, ndvi_sentinel_sant_marti, marker='s', label='Sant Martí - Sentinel 2 NDVI', color='darkred', linewidth=2)

# Torrelles de Llobregat data
plt.plot(years, ndvi_landsat_torrelles, marker='o', label='Torrelles de Llobregat - Landsat 8 NDVI', color='green', linewidth=2)
plt.plot(years, ndvi_sentinel_torrelles, marker='s', label='Torrelles de Llobregat - Sentinel 2 NDVI', color='red', linewidth=2)

# Adding titles and labels
plt.title('Landsat 8 and Sentinel 2 NDVI Comparison (2017-2025)', fontsize=16, weight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('NDVI', fontsize=14)
plt.xticks(years)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Adding a legend below the graph
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2, fontsize=14)

# Save the plot as a PNG file
plt.savefig('ndvi_comparison.png', dpi=300, bbox_inches='tight')
plt.close()