"""
RIS control module for intelligent switching and SINR calculation
"""
import numpy as np
import math
from config.settings import *
from utils.distance_calculator import haversine, calculate_angle_from_cosine_law

class RISController:
    """Handles RIS switching decisions and SINR calculations"""
    
    def __init__(self):
        self.pi = PI
        self.P_transmit = P_TRANSMIT
        self.P_noise = P_NOISE
        self.fc = FC
        self.c = C
        self.wavelength = WAVELENGTH
        self.C_L = C_L
        self.alpha = ALPHA
        self.num_elements = NUM_ELEMENTS
    
    def calculate_interference_factor(self, angle):
        """Calculate interference factor based on angle"""
        return 10/angle
    
    def should_calculate_interference(self, angle, threshold=PI / 4):
        """Determine if interference should be calculated based on angle"""
        return angle < threshold
    
    def rayleigh_fading(self):
        """Generate Rayleigh fading coefficient"""
        return 1.5
    
    def calculate_signal_power(self, distance, fading=None):
        """Calculate signal power using free space propagation model"""
        fading = fading if fading is not None else self.rayleigh_fading()
        return (self.P_transmit * self.C_L * fading) / (distance ** self.alpha)
    
    def calculate_reflected_signal_power(self, distance_to_ris, distance_from_ris_to_bs, 
                                       fading_ris=None, elements=None):
        """Calculate signal power through RIS reflection"""
        if elements is None:
            elements = self.num_elements
        fading_ris = fading_ris if fading_ris is not None else self.rayleigh_fading()
        total_distance = distance_to_ris * distance_from_ris_to_bs
        return elements * self.calculate_signal_power(total_distance, fading_ris) * 2
    
    def calculate_interference_power(self, interfering_distances, ris_distances, ris_distances_bs, 
                                   angles=None, ris_reflection=True, angle_threshold=PI / 2, 
                                   ris_factor=1.0, elements=None):
        """Calculate total interference power"""
        if elements is None:
            elements = self.num_elements
        
        interference_power = 0
        
        if ris_reflection and ris_distances and angles:
            for dist, ris_distance_user, ris_distance_bs, angle in zip(
                interfering_distances, ris_distances, ris_distances_bs, angles
            ):
                if self.should_calculate_interference(angle, angle_threshold):
                    interference_factor = self.calculate_interference_factor(angle)
                    interference_power += (elements/100 * 
                                         self.calculate_reflected_signal_power(
                                             ris_distance_user, ris_distance_bs, elements=elements
                                         ) * interference_factor * ris_factor)
                interference_power += self.calculate_signal_power(dist)
        else:
            for dist in interfering_distances:
                interference_power += self.calculate_signal_power(dist)
        
        return interference_power
    
    def calculate_SINR(self, signal_power, interference_power, noise_power):
        """Calculate Signal-to-Interference-plus-Noise Ratio"""
        return signal_power / (interference_power + noise_power)
    
    def determine_RIS_switch(self, distance_to_base_station, distance_to_ris, ris_to_bs, 
                           interfering_distances, ris_distances, ris_distances_bs, angles, 
                           angle_threshold=PI / 4, elements=None):
        """Determine optimal RIS switch state"""
        if elements is None:
            elements = self.num_elements
        
        lambda_factor = 3
        ris_factor = 10 * (lambda_factor * abs(distance_to_ris - ris_to_bs) / 
                          (distance_to_ris + ris_to_bs))
        
        fading_direct = self.rayleigh_fading()
        fading_ris = self.rayleigh_fading()
        
        # SINR without RIS
        signal_power_no_ris = self.calculate_signal_power(distance_to_base_station, fading_direct)
        interference_power_no_ris = self.calculate_interference_power(
            interfering_distances, ris_distances=None, ris_distances_bs=None, 
            ris_reflection=False, elements=elements
        )
        sinr_no_ris = self.calculate_SINR(signal_power_no_ris, interference_power_no_ris, self.P_noise)
        
        # SINR with RIS
        signal_power_with_ris = (self.calculate_reflected_signal_power(
            distance_to_ris, ris_to_bs, fading_ris, elements
        ) + signal_power_no_ris)
        interference_power_with_ris = self.calculate_interference_power(
            interfering_distances, ris_distances, ris_distances_bs, angles, 
            ris_reflection=True, angle_threshold=angle_threshold, 
            ris_factor=ris_factor, elements=elements
        )
        sinr_with_ris = self.calculate_SINR(signal_power_with_ris, interference_power_with_ris, self.P_noise)
        
        return 1 if sinr_with_ris > sinr_no_ris else 0

    def calculate_sinr_seq(self, distance_to_base_station, distance_to_ris, ris_to_bs,
                          interfering_distances, ris_distances, ris_distances_bs, angles,
                          angle_threshold=PI / 2, always_on=0, elements=None):
        """Calculate SINR for different RIS operation modes"""
        if elements is None:
            elements = self.num_elements

        lambda_factor = 3
        ris_factor = 10 * np.exp(-lambda_factor * abs(distance_to_ris - ris_to_bs) /
                                (distance_to_ris + ris_to_bs))
        ris_factor = 10 * (lambda_factor * abs(distance_to_ris - ris_to_bs) /
                          (distance_to_ris + ris_to_bs))

        fading_direct = self.rayleigh_fading()
        fading_ris = self.rayleigh_fading()

        if always_on == 1:  # RIS always on
            signal_power_with_ris = (self.calculate_reflected_signal_power(
                distance_to_ris, ris_to_bs, fading_ris, elements
            ) + self.calculate_signal_power(distance_to_base_station, fading_ris))
            interference_power_with_ris = self.calculate_interference_power(
                interfering_distances, ris_distances, ris_distances_bs, angles,
                ris_reflection=True, angle_threshold=angle_threshold,
                ris_factor=ris_factor, elements=elements
            )
            sinr_with_ris = self.calculate_SINR(signal_power_with_ris, interference_power_with_ris, self.P_noise)
            return sinr_with_ris

        elif always_on == -1:  # RIS always off
            signal_power_no_ris = self.calculate_signal_power(distance_to_base_station, fading_direct)
            interference_power_no_ris = self.calculate_interference_power(
                interfering_distances, ris_distances=None, ris_distances_bs=None,
                ris_reflection=False, elements=elements
            )
            sinr_no_ris = self.calculate_SINR(signal_power_no_ris, interference_power_no_ris, self.P_noise)
            return sinr_no_ris

        # Dynamic switching
        signal_power_no_ris = self.calculate_signal_power(distance_to_base_station, fading_direct)
        interference_power_no_ris = self.calculate_interference_power(
            interfering_distances, ris_distances=None, ris_distances_bs=None,
            ris_reflection=False, elements=elements
        )
        sinr_no_ris = self.calculate_SINR(signal_power_no_ris, interference_power_no_ris, self.P_noise)

        signal_power_with_ris = (self.calculate_reflected_signal_power(
            distance_to_ris, ris_to_bs, fading_ris, elements
        ) + signal_power_no_ris)
        interference_power_with_ris = self.calculate_interference_power(
            interfering_distances, ris_distances, ris_distances_bs, angles,
            ris_reflection=True, angle_threshold=angle_threshold,
            ris_factor=ris_factor, elements=elements
        )
        sinr_with_ris = self.calculate_SINR(signal_power_with_ris, interference_power_with_ris, self.P_noise)

        return sinr_with_ris if sinr_with_ris > sinr_no_ris else sinr_no_ris

    def calculate_ris_switch(self, user_coords, base_station_coords, ris_coords,
                           interfering_user_coords, always_on=0, sinr=0, elements=None):
        """Main function to calculate RIS switch state or SINR"""
        if elements is None:
            elements = self.num_elements

        # Calculate distances
        distance_to_base_station = haversine(
            user_coords[0], user_coords[1], base_station_coords[0], base_station_coords[1]
        )
        distance_to_ris = haversine(
            user_coords[0], user_coords[1], ris_coords[0], ris_coords[1]
        )
        ris_to_bs = haversine(
            ris_coords[0], ris_coords[1], base_station_coords[0], base_station_coords[1]
        )

        # Calculate interference distances
        interfering_distances = [
            haversine(iu[0], iu[1], base_station_coords[0], base_station_coords[1])
            for iu in interfering_user_coords
        ]
        interfering_distances_user = [
            haversine(iu[0], iu[1], user_coords[0], user_coords[1])
            for iu in interfering_user_coords
        ]
        ris_distances = [
            haversine(iu[0], iu[1], ris_coords[0], ris_coords[1])
            for iu in interfering_user_coords
        ]
        ris_distances_bs = [
            haversine(ris_coords[0], ris_coords[1], base_station_coords[0], base_station_coords[1])
            for iu in interfering_user_coords
        ]

        # Calculate angles
        angles = []
        for dist1, dist2, dist3 in zip(ris_distances, interfering_distances_user,
                                      [distance_to_ris] * len(ris_distances)):
            angle = calculate_angle_from_cosine_law(dist1, dist2, dist3)
            angles.append(angle)

        if sinr:
            return self.calculate_sinr_seq(
                distance_to_base_station, distance_to_ris, ris_to_bs,
                interfering_distances, ris_distances, ris_distances_bs, angles,
                PI / 4, always_on, elements=elements
            )

        return self.determine_RIS_switch(
            distance_to_base_station, distance_to_ris, ris_to_bs,
            interfering_distances, ris_distances, ris_distances_bs, angles, elements=elements
        )
