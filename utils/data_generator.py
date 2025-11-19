"""
Data generation utilities for simulation
"""
import numpy as np
import pandas as pd
import os
from config.settings import *

def generate_random_points(center, radius, num_points):
    """
    Generate random points around a center location within a given radius
    
    Args:
        center: Tuple of (latitude, longitude) for center point
        radius: Radius in meters
        num_points: Number of points to generate
    
    Returns:
        List of (latitude, longitude) tuples
    """
    EARTH_RADIUS = 6371000
    radius_in_degrees = radius / EARTH_RADIUS * (180 / np.pi)
    random_points = []
    
    for _ in range(num_points):
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.random.uniform(0, radius_in_degrees)
        delta_lat = offset * np.cos(angle)
        delta_lon = offset * np.sin(angle) / np.cos(np.radians(center[0]))
        random_lat = center[0] + delta_lat
        random_lon = center[1] + delta_lon
        random_points.append((random_lat, random_lon))
    
    return random_points

def load_trajectory_data(data_path, num_points=10):
    """
    Load trajectory data from PLT files
    
    Args:
        data_path: Path to trajectory data directory
        num_points: Number of points to load
    
    Returns:
        List of (latitude, longitude) tuples
    """
    plt_files = sorted([f for f in os.listdir(data_path) if f.endswith(".plt")])
    if plt_files:
        first_plt = os.path.join(data_path, plt_files[0])
        df = pd.read_csv(
            first_plt, 
            skiprows=6, 
            header=None, 
            usecols=[0, 1], 
            names=['latitude', 'longitude'], 
            encoding='utf-8'
        ).head(num_points)
        return list(df.itertuples(index=False, name=None))
    return []

def generate_interference_users(base_station_location, radius, num_users, num_points):
    """
    Generate multiple interfering users with trajectories
    
    Args:
        base_station_location: Base station coordinates
        radius: Coverage radius
        num_users: Number of interfering users
        num_points: Number of trajectory points per user
    
    Returns:
        List of user trajectories
    """
    interference_users = []
    for _ in range(num_users):
        user_trajectory = generate_random_points(base_station_location, radius, num_points)
        interference_users.append(user_trajectory)
    return interference_users
