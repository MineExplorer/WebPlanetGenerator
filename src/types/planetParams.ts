export interface PlanetParams {
  radius: number;
  segments: number;
  seed: number;
  // Continent noise layer
  continentScale: number;
  continentOctaves: number;
  // Biome noise layers (temperature + moisture)
  biomeScale: number;
  // Climate additive offsets applied after latitude function + noise  [-1 .. 1]
  tempOffset: number;
  moistureOffset: number;
  // High-frequency detail layer
  detailScale: number;
  detailStrength: number;
  // Displacement amount for land above sea level
  elevation: number;
  // Sea level: fraction of normalised height [0..1] below which is ocean
  seaLevel: number;
  // Width of the beach / shore strip above sea level [0..1 height units]
  shoreWidth: number;
  // Debug: show greyscale height + flat-blue sea instead of biomes
  topoMap: boolean;
  // Debug: show moisture map with conventional colours
  moistureView: boolean;
  // Debug: show temperature map with conventional colours
  tempView: boolean;
  // Canvas texture resolution (independent of geometry segments)
  textureWidth: number;
  textureHeight: number;
  // Cloud layer
  cloudScale: number;
  cloudDensity: number;
  cloudOpacity: number;
  cloudHeight: number;
  showClouds: boolean;
  // Ocean
  fancyOcean: boolean;
  // Globe grid overlay
  showGrid: boolean;
  // Elevation-based moisture drying
  elevationMoisture: boolean;
  // Large-scale (continent-size) moisture map
  moistureMapScale: number;
  moistureMapStrength: number;
  // How strongly latitude drives temperature and moisture (0 = none, 1 = normal, 2 = extreme)
  latitudeFactor: number;
}
