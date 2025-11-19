"""
Network element selection module for finding nearest base stations and RIS
"""
from utils.distance_calculator import haversine
from src.core.network_deployment import NetworkDeployment

class NetworkSelector:
    """Handles selection of nearest base stations and RIS for users"""
    
    def __init__(self):
        self.deployment = NetworkDeployment()
        self.grid = self.deployment.get_base_stations()
        self.ris_list = self.deployment.get_ris_list()
    
    def find_nearest_base_station(self, query_latitude, query_longitude):
        """
        Find the nearest base station to a given location
        
        Args:
            query_latitude: User latitude
            query_longitude: User longitude
        
        Returns:
            Dictionary with nearest base station information
        """
        min_distance = float('inf')
        closest_point_id = None
        closest_lat = None
        closest_lon = None
        
        for point_id, lat, lon in self.grid:
            distance = haversine(query_latitude, query_longitude, lat, lon)
            if distance < min_distance:
                min_distance = distance
                closest_point_id = point_id
                closest_lat = lat
                closest_lon = lon
        
        return {
            "query_latitude": query_latitude,
            "query_longitude": query_longitude,
            "closest_point_id": closest_point_id,
            "closest_lat": closest_lat,
            "closest_lon": closest_lon,
            "min_distance": min_distance
        }
    
    def find_nearest_ris(self, query_latitude, query_longitude, num=10):
        """
        Find the nearest RIS to a given location
        
        Args:
            query_latitude: User latitude
            query_longitude: User longitude
            num: Number of RIS to consider (for compatibility)
        
        Returns:
            Dictionary with nearest RIS information
        """
        min_distance = float('inf')
        closest_ris_id = None
        closest_base_station_id = None
        closest_lat = None
        closest_lon = None
        
        for ris in self.ris_list:
            ris_id = ris["ris_id"]
            base_station_id = ris["base_station_id"]
            ris_lat = ris["ris_latitude"]
            ris_lon = ris["ris_longitude"]
            distance = haversine(query_latitude, query_longitude, ris_lat, ris_lon)
            
            if distance < min_distance:
                min_distance = distance
                closest_ris_id = ris_id
                closest_base_station_id = base_station_id
                closest_lat = ris_lat
                closest_lon = ris_lon
        
        return {
            "query_latitude": query_latitude,
            "query_longitude": query_longitude,
            "closest_ris_id": closest_ris_id,
            "closest_base_station_id": closest_base_station_id,
            "closest_lat": closest_lat,
            "closest_lon": closest_lon,
            "min_distance": min_distance
        }
    
    def get_nearest_station_and_ris_for_points(self, points_in_station, num):
        """
        Get nearest base stations and RIS for a list of trajectory points - Optimized

        Args:
            points_in_station: List of (latitude, longitude) tuples
            num: Number of RIS to consider

        Returns:
            List containing [base_station_coords, ris_coords]
        """
        import numpy as np
        from utils.distance_calculator import haversine_vectorized

        # Convert points to numpy arrays for vectorized operations
        points_array = np.array(points_in_station)
        query_lats = points_array[:, 0]
        query_lons = points_array[:, 1]

        # Convert grid and RIS to numpy arrays for faster computation
        grid_array = np.array(self.grid)
        grid_lats = grid_array[:, 1].astype(float)
        grid_lons = grid_array[:, 2].astype(float)

        # Vectorized distance calculation for base stations
        base_station_coords = []
        for i, (query_lat, query_lon) in enumerate(points_in_station):
            # Use simple distance calculation for better performance
            distances = haversine_vectorized(query_lat, query_lon, grid_lats, grid_lons)
            min_idx = np.argmin(distances)
            base_station_coords.append((grid_lats[min_idx], grid_lons[min_idx]))

        # Vectorized distance calculation for RIS
        ris_coords = []
        ris_lats = np.array([ris["ris_latitude"] for ris in self.ris_list])
        ris_lons = np.array([ris["ris_longitude"] for ris in self.ris_list])

        for i, (query_lat, query_lon) in enumerate(points_in_station):
            distances = haversine_vectorized(query_lat, query_lon, ris_lats, ris_lons)
            min_idx = np.argmin(distances)
            ris_coords.append((ris_lats[min_idx], ris_lons[min_idx]))

        return [base_station_coords, ris_coords]
