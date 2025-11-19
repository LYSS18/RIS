"""
LSTM-based trajectory prediction model
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from operator import concat
from config.settings import *

class TrajectoryPredictor:
    """LSTM model for trajectory prediction"""
    
    def __init__(self):
        self.model = None
        self.normalize = None
        self.train_num = TRAIN_NUM
        self.lstm_units = LSTM_UNITS
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
    
    def create_dataset(self, data, n_predictions, n_next):
        """Process data for LSTM training"""
        dim = data.shape[1]
        train_X, train_Y = [], []
        
        for i in range(data.shape[0] - n_predictions - n_next - 1):
            a = data[i:(i + n_predictions), :]
            train_X.append(a)
            tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
            b = []
            for j in range(len(tempb)):
                for k in range(dim):
                    b.append(tempb[j, k])
            train_Y.append(b)
        
        train_X = np.array(train_X, dtype='float64')
        train_Y = np.array(train_Y, dtype='float64')
        
        test_X, test_Y = [], []
        i = data.shape[0] - n_predictions - n_next - 1
        a = data[i:(i + n_predictions), :]
        test_X.append(a)
        tempb = data[(i + n_predictions):(i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        test_Y.append(b)
        test_X = np.array(test_X, dtype='float64')
        test_Y = np.array(test_Y, dtype='float64')
        
        return train_X, train_Y, test_X, test_Y
    
    def normalize_mult(self, data, set_range):
        """Normalize data for training"""
        normalize = np.arange(2 * data.shape[1], dtype='float64')
        normalize = normalize.reshape(data.shape[1], 2)
        
        for i in range(0, data.shape[1]):
            if set_range == True:
                list_data = data[:, i]
                listlow, listhigh = np.percentile(list_data, [0, 100])
            else:
                if i == 0:
                    listlow = -90
                    listhigh = 90
                else:
                    listlow = -180
                    listhigh = 180
            
            normalize[i, 0] = listlow
            normalize[i, 1] = listhigh
            
            delta = listhigh - listlow
            if delta != 0:
                for j in range(0, data.shape[0]):
                    data[j, i] = (data[j, i] - listlow) / delta
        
        return data, normalize
    
    def denormalize_mult(self, data, normalize):
        """Denormalize data after prediction"""
        data = np.array(data, dtype='float64')
        for i in range(0, data.shape[1]):
            listlow = normalize[i, 0]
            listhigh = normalize[i, 1]
            delta = listhigh - listlow
            if delta != 0:
                for j in range(0, data.shape[0]):
                    data[j, i] = data[j, i] * delta + listlow
        return data
    
    def build_model(self, train_X, train_Y):
        """Build and train LSTM model"""
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(self.lstm_units, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(train_Y.shape[1]))
        model.add(Activation("relu"))
        model.compile(loss='mse', optimizer='adam', metrics=['acc'])
        model.fit(train_X, train_Y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        model.summary()
        return model
    
    def train(self, data_path=None, save_model=True):
        """Train the trajectory prediction model"""
        if data_path is None:
            data_path = 'Geolife Trajectories 1.3\\Data\\001\\Trajectory\\20081024234405.plt'
        
        # Load and prepare data
        data = pd.read_csv(data_path, sep=',', skiprows=6).iloc[:, 0:2].values
        print(f"Sample count: {data.shape[0]}, Dimensions: {data.shape[1]}")
        
        # Visualize original data
        plt.plot(data[:, 1], data[:, 0], c='r', label='Original trajectory')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()
        
        # Normalize data
        data, self.normalize = self.normalize_mult(data, set_range=True)
        
        # Create training dataset
        train_X, train_Y, test_X, test_Y = self.create_dataset(data, self.train_num, 1)
        print(f"Training X shape: {train_X.shape}")
        print(f"Training Y shape: {train_Y.shape}")
        
        # Train model
        self.model = self.build_model(train_X, train_Y)
        loss, acc = self.model.evaluate(train_X, train_Y, verbose=2)
        print(f'Loss: {loss}, Accuracy: {acc * 100}')
        
        # Save model and normalization parameters
        if save_model:
            np.save(NORMALIZATION_PATH, self.normalize)
            self.model.save(MODEL_PATH_LSTM)
        
        return self.model, self.normalize
    
    def predict_trajectory(self, previous_trajectory, steps_to_predict, 
                          model_path=MODEL_PATH_LSTM, normalization_path=NORMALIZATION_PATH):
        """Predict future trajectory points"""
        # Load model and normalization parameters
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        normalize = np.load(normalization_path)
        
        previous_trajectory = np.array(previous_trajectory, dtype='float64')
        
        # Normalize input trajectory
        normalized_trajectory = self._normalize_data(previous_trajectory.copy(), normalize)
        
        # Predict trajectory points
        predicted_trajectory = []
        current_input = normalized_trajectory[-8:].reshape(1, 8, 2)
        
        for _ in range(steps_to_predict):
            prediction = model.predict(current_input)
            predicted_point = prediction.reshape(-1, 2)
            predicted_trajectory.append(predicted_point[0])
            current_input = np.append(current_input[:, 1:, :], predicted_point.reshape(1, 1, 2), axis=1)
        
        # Denormalize predictions
        predicted_trajectory = np.array(predicted_trajectory, dtype='float64')
        predicted_trajectory = self._denormalize_data(predicted_trajectory, normalize)
        
        # Combine original and predicted trajectories
        final_trajectory = np.vstack([previous_trajectory, predicted_trajectory])
        return [tuple(item) for item in final_trajectory]
    
    def _normalize_data(self, data, normalize):
        """Helper function to normalize data"""
        for i in range(data.shape[1]):
            listlow = normalize[i, 0]
            listhigh = normalize[i, 1]
            delta = listhigh - listlow
            if delta != 0:
                data[:, i] = (data[:, i] - listlow) / delta
        return data
    
    def _denormalize_data(self, data, normalize):
        """Helper function to denormalize data"""
        for i in range(data.shape[1]):
            listlow = normalize[i, 0]
            listhigh = normalize[i, 1]
            delta = listhigh - listlow
            if delta != 0:
                data[:, i] = data[:, i] * delta + listlow
        return data
