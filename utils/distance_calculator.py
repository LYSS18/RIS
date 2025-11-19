"""
Distance calculation utilities
"""
from math import radians, sin, cos, sqrt, atan2
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized version of haversine distance calculation
    
    Args:
        lat1, lon1: Arrays of latitude and longitude of first points
        lat2, lon2: Arrays of latitude and longitude of second points
    
    Returns:
        Array of distances in meters
    """
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_angle_from_cosine_law(dist1, dist2, dist3):
    """
    Calculate angle using cosine law
    
    Args:
        dist1, dist2, dist3: Three sides of triangle
    
    Returns:
        Angle in radians
    """
    cos_theta = (dist1**2 + dist3**2 - dist2**2) / (2 * dist1 * dist3)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))
