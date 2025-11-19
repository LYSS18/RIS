"""
Network deployment module for base stations and RIS
"""
import pandas as pd
import math
import os
from config.settings import *

class NetworkDeployment:
    """Handles deployment of base stations and RIS in the network"""
    
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.grid = []
        self.ris_deployed = []
        self._initialize_coverage_area()
        self._deploy_base_stations()
        self._deploy_ris()
    
    def _initialize_coverage_area(self):
        """Initialize coverage area based on trajectory data"""
        latitude_max, longitude_max = float('-inf'), float('-inf')
        latitude_min, longitude_min = float('inf'), float('inf')
        
        folder_path = 'Geolife Trajectories 1.3\\Data\\000\\Trajectory'
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.plt'):
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path, sep=',', skiprows=6).iloc[:, 0:2].values
                latitude_max = max(latitude_max, data[:, 0].max())
                longitude_max = max(longitude_max, data[:, 1].max())
                latitude_min = min(latitude_min, data[:, 0].min())
                longitude_min = min(longitude_min, data[:, 1].min())
        
        self.coverage_area = {
            'lat_max': latitude_max,
            'lat_min': latitude_min,
            'lon_max': longitude_max,
            'lon_min': longitude_min
        }
        
        print(f"Coverage area: {latitude_max}, {longitude_max}, {latitude_min}, {longitude_min}")
    
    def _deploy_base_stations(self):
        """Deploy base stations in hexagonal grid pattern"""
        lat_min = self.coverage_area['lat_min']
        lat_max = self.coverage_area['lat_max']
        lon_min = self.coverage_area['lon_min']
        lon_max = self.coverage_area['lon_max']
        
        delta_latitude = BASE_STATION_DISTANCE / 111320
        latitude_reference = (lat_min + lat_max) / 2
        delta_longitude = BASE_STATION_DISTANCE / (111320 * math.cos(math.radians(latitude_reference)))
        
        point_id = 1
        lat = lat_min
        row_index = 0
        
        while lat <= lat_max:
            lon = lon_min
            if row_index % 2 != 0:
                lon += delta_longitude / 2
            
            while lon <= lon_max:
                self.grid.append((point_id, lat, lon))
                point_id += 1
                lon += delta_longitude
            
            lat += delta_latitude * math.sqrt(3) / 2
            row_index += 1
    
    def _deploy_ris(self):
        """Deploy RIS around each base station"""
        self.ris_deployed = self.deploy_ris_around_base_stations(
            self.grid, RIS_RADIUS, NUM_RIS_PER_BS
        )
    
    def deploy_ris_around_base_stations(self, grid, radius=RIS_RADIUS, num_ris=NUM_RIS_PER_BS):
        """Deploy RIS in circular pattern around base stations"""
        ris_list = []
        ris_id = 1
        
        for base_station_id, base_lat, base_lon in grid:
            delta_lat = radius / 111320
            delta_lon = radius / (111320 * math.cos(math.radians(base_lat)))
            
            for i in range(num_ris):
                angle = (2 * math.pi / num_ris) * i
                ris_lat = base_lat + delta_lat * math.sin(angle)
                ris_lon = base_lon + delta_lon * math.cos(angle)
                
                ris_list.append({
                    "ris_id": ris_id,
                    "base_station_id": base_station_id,
                    "ris_latitude": ris_lat,
                    "ris_longitude": ris_lon
                })
                ris_id += 1
        
        return ris_list
    
    def get_base_stations(self):
        """Get list of deployed base stations"""
        return self.grid
    
    def get_ris_list(self):
        """Get list of deployed RIS"""
        return self.ris_deployed
