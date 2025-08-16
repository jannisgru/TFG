/*
Google Earth Engine Script for AMB Sentinel-2 Data Acquisition
==============================================================

This script creates annual Sentinel-2 composites for the Barcelona Metropolitan Area (AMB)
from 2017-2025 using quality mosaics based on NDVI values.

Output: Annual GeoTIFF files with Blue, Green, Red, and NIR bands
Projection: EPSG:4326 (WGS84)
Resolution: 10m
*/

// Define the area of interest (AOI)
var aoi = ee.FeatureCollection("projects/ee-jannisgruber/assets/AMB_Municipalities").geometry(); // Upload the shapefile and change the location

// Configuration
var CLOUD_COVER_THRESHOLD = 50;
var BLACKLISTED_DATES = ['2022-07-11']; // Add any specific dates to exclude if needed
var START_YEAR = 2017;
var END_YEAR = 2025;

// Sentinel-2 collection
var SENTINEL2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED';

// Process bands and calculate NDVI
function processBands(image) {
  // Sentinel-2 band selection: B2=Blue, B3=Green, B4=Red, B8=NIR
  var bands = ['B2', 'B3', 'B4', 'B8'];
  
  // Scale factor for Sentinel-2 Surface Reflectance is 0.0001
  var proc = image.select(bands, ['BLUE', 'GREEN', 'RED', 'NIR']).multiply(0.0001);
  
  // Calculate NDVI
  var ndvi = proc.select('NIR')
            .subtract(proc.select('RED'))
            .divide(proc.select('NIR')
            .add(proc.select('RED')))
            .rename('NDVI');
  
  // Apply cloud mask using SCL (Scene Classification Layer)
  var scl = image.select('SCL');
  var mask = scl.neq(3)
            .and(scl.neq(8))
            .and(scl.neq(9))
            .and(scl.neq(10))
            .and(scl.neq(11))
            .and(scl.neq(1))
            .and(scl.neq(0));

  return proc.addBands(ndvi).updateMask(mask);
}

// Get filtered collection
function getCollection(year) {
  var collection = ee.ImageCollection(SENTINEL2_COLLECTION)
    .filterBounds(aoi)
    .filterDate(year + '-01-01', (parseInt(year) + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_THRESHOLD));
    
  // Apply blacklist if any dates are specified
  BLACKLISTED_DATES.forEach(function(dateStr) {
    var dateObj = ee.Date(dateStr);
    var yearStr = dateObj.format('YYYY').getInfo();
    if (yearStr === year) {
      var start = dateObj;
      var end = start.advance(1, 'day');
      collection = collection.filter(
        ee.Filter.or(
          ee.Filter.lt('system:time_start', start.millis()),
          ee.Filter.gte('system:time_start', end.millis())
        )
      );
    }
  });
  return collection;
}

// Create composite for each year
function createComposite(year) {
  var images = getCollection(year);
  var proc = images.map(processBands);
  var composite = proc.qualityMosaic('NDVI').select(['BLUE', 'GREEN', 'RED', 'NIR']).clip(aoi);
  
  Export.image.toDrive({
    image: composite,
    description: year,
    folder: 'Sentinel2_Composites',
    fileNamePrefix: year,
    crs: 'EPSG:4326',
    region: aoi,
    scale: 10,
    maxPixels: 1e13,
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    }
  });
}

// Process all years
for (var year = START_YEAR; year <= END_YEAR; year++) {
  createComposite(year.toString());
}

/*
Usage Instructions:
==================

1. Copy this code into Google Earth Engine Code Editor
2. Update the AOI file location if needed
3. Run the script - it will export all years automatically
4. Download the exported files from Google Drive
5. Place the downloaded .tif files in the data/raw/ folder
6. Files should be named: 2017.tif, 2018.tif, ..., 2025.tif

Expected Output:
- Files: 2017.tif through 2025.tif
- Bands: BLUE, GREEN, RED, NIR
- Format: Cloud-optimized GeoTIFF
- NoData: -9999
- Resolution: 10m
- CRS: EPSG:4326 (WGS84)
*/