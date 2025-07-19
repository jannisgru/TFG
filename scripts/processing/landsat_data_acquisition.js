/*
Google Earth Engine Script for AMB Landsat Data Acquisition
==========================================================

This script creates annual Landsat composites for the Barcelona Metropolitan Area (AMB)
from 1984-2025 using quality mosaics based on NDVI values.

Output: Annual GeoTIFF files with Blue, Green, Red, and NIR bands
Projection: EPSG:4326 (WGS84)
Resolution: 30m
*/

// Define the area of interest (AOI)
var aoi = ee.FeatureCollection("projects/ee-jannisgruber/assets/AMB_Municipalities").geometry(); // Upload the shapefile and change the location

// Configuration
var CLOUD_COVER_THRESHOLD = 50;
var BLACKLISTED_DATES = ['1991-06-23'];
var START_YEAR = 1984;
var END_YEAR = 2025;

// Landsat collection mapping
var LANDSAT_COLLECTIONS = {
  'LT05': 'LANDSAT/LT05/C02/T1_L2',
  'LE07': 'LANDSAT/LE07/C02/T1_L2', 
  'LC08': 'LANDSAT/LC08/C02/T1_L2'
};

// Process bands and calculate NDVI
function processBands(image) {
  var sensor = image.get('SPACECRAFT_ID');
  var isL5L7 = ee.List(['LANDSAT_5','LANDSAT_7']).contains(sensor);
  
  // Band selection - now including green band
  var bands = ee.Algorithms.If(isL5L7, 
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'], // L5/L7: Blue, Green, Red, NIR
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']  // L8: Blue, Green, Red, NIR
  );
  
  // Convert digital values to reflectance and scale
  var proc = image.select(bands, ['BLUE', 'GREEN', 'RED', 'NIR']).multiply(0.0000275).add(-0.2);
  
  // Calculate NDVI
  var ndvi = proc.select('NIR')
            .subtract(proc.select('RED'))
            .divide(proc.select('NIR')
            .add(proc.select('RED')))
            .rename('NDVI');
  
  // Apply cloud mask
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
            .and(qa.bitwiseAnd(1 << 4).eq(0))
            .and(qa.bitwiseAnd(1 << 2).eq(0));
  
  return proc.addBands(ndvi).updateMask(mask);
}

// Get sensors for each year
function getSensors(year) {
  var y = parseInt(year);
  var sensors = [];
  if (y >= 1984 && y <= 2012) sensors.push('LT05');
  if (y === 2012) sensors.push('LE07');
  if (y >= 2013) sensors.push('LC08');
  return sensors;
}

// Get filtered collection
function getCollection(sensor, year) {
  var collection = ee.ImageCollection(LANDSAT_COLLECTIONS[sensor])
    .filterBounds(aoi)
    .filterDate(year + '-01-01', (parseInt(year) + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_THRESHOLD));
    
  // Apply blacklist
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
  var sensors = getSensors(year);
  var images = ee.ImageCollection([]);
  
  sensors.forEach(function(sensor) {
    images = images.merge(getCollection(sensor, year));
  });
  
  var proc = images.map(processBands);
  var composite = proc.qualityMosaic('NDVI').select(['BLUE', 'GREEN', 'RED', 'NIR']).clip(aoi);
  
  Export.image.toDrive({
    image: composite,
    description: year,
    folder: 'Landsat_Composites',
    fileNamePrefix: year,
    crs: 'EPSG:4326',
    region: aoi,
    scale: 30,
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
2. Update the AOI file location
3. Run the script - it will export all years automatically
4. Download the exported files from Google Drive
5. Place the downloaded .tif files in the data/raw/ folder
6. Files should be named: 1998.tif, 1999.tif, ..., 2025.tif

Expected Output:
- Files: 1984.tif through 2025.tif
- Bands: BLUE, GREEN, RED, NIR
- Format: Cloud-optimized GeoTIFF
- NoData: -9999
- Resolution: 30m
- CRS: EPSG:4326 (WGS84)
*/
