"""
Configuration settings for RIS-assisted wireless communication system
"""
import math

# Physical constants
PI = math.pi
C = 3 * 10**8  # Speed of light (m/s)

# Communication parameters
FC = 1e9  # Carrier frequency (1 GHz)
WAVELENGTH = C / FC  # Wavelength
C_L = (WAVELENGTH / (4 * PI))**2  # NLOS intercept
ALPHA = 3  # Path loss exponent

# Power settings
P_TRANSMIT = 1  # Base station transmit power (W)
P_NOISE = 1e-12  # Noise power (W), 1pW

# RIS parameters
NUM_ELEMENTS = 600  # Number of RIS reflecting elements
BASE_STATION_DISTANCE = 300  # Distance between base stations (m)
RIS_RADIUS = 10  # RIS deployment radius around base station (m)
NUM_RIS_PER_BS = 10  # Number of RIS per base station

# Trajectory prediction parameters
TRAIN_NUM = 8  # Number of previous points for LSTM training
PREDICT_STEPS = 1  # Number of steps to predict
LSTM_UNITS = 120  # LSTM hidden units
EPOCHS = 100  # Training epochs
BATCH_SIZE = 64  # Training batch size

# CNN parameters for interference classification
CNN_SEQUENCE_LENGTH = 32  # I/Q signal window length
CNN_NUM_FEATURES = 2  # I/Q channels
CNN_NUM_CLASSES = 4  # Number of interference levels
CNN_FILTERS_1 = 64  # First CNN layer filters
CNN_FILTERS_2 = 128  # Second CNN layer filters
CNN_DENSE_UNITS = 128  # Dense layer units

# Simulation parameters
NUM_INTERFERENCE_USERS = 5  # Number of interfering users
NUM_TRAJECTORY_POINTS = 10  # Number of trajectory points
SIMULATION_RADIUS = 300  # Simulation area radius (m)

# File paths
DATA_PATH = "Geolife Trajectories 1.3/Data/001/Trajectory/"
MODEL_PATH_LSTM = "./models/traj_model_120.h5"
MODEL_PATH_CNN = "./models/cnn_ris_model.h5"
NORMALIZATION_PATH = "./models/traj_model_trueNorm.npy"

# Analysis parameters
POWER_LEVELS = [0.050, 0.112, 0.223, 0.334, 0.445, 0.556, 0.667, 0.778, 0.889, 1.0]
ELEMENT_COUNTS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Visualization settings
FIGURE_SIZE = (18/2.54, 12/2.54)  # Figure size in inches (18cm x 12cm)
DPI = 300  # Figure DPI for high quality output
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10
