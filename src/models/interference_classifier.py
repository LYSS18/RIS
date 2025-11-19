"""
CNN-based interference classification model
"""
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from config.settings import *

class InterferenceClassifier:
    """CNN model for interference user classification"""
    
    def __init__(self):
        self.model = None
        self.sequence_length = CNN_SEQUENCE_LENGTH
        self.num_features = CNN_NUM_FEATURES
        self.num_classes = CNN_NUM_CLASSES
        self.filters_1 = CNN_FILTERS_1
        self.filters_2 = CNN_FILTERS_2
        self.dense_units = CNN_DENSE_UNITS
    
    def build_cnn(self, input_shape, num_classes):
        """Build CNN architecture for interference classification"""
        model = Sequential([
            Conv1D(filters=self.filters_1, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=self.filters_2, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(self.dense_units, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def generate_training_data(self, num_samples=10000):
        """Generate synthetic I/Q training data"""
        X_train = np.random.rand(num_samples, self.sequence_length, self.num_features)
        y_train = np.random.randint(0, self.num_classes, size=(num_samples,))
        y_train = np.eye(self.num_classes)[y_train]  # One-hot encoding
        return X_train, y_train
    
    def train(self, num_samples=10000, epochs=10, batch_size=32, save_model=True):
        """Train the interference classification model"""
        # Generate training data
        X_train, y_train = self.generate_training_data(num_samples)
        
        # Build and train model
        self.model = self.build_cnn((self.sequence_length, self.num_features), self.num_classes)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        # Save model
        if save_model:
            self.model.save(MODEL_PATH_CNN)
        
        return self.model
    
    def predict_interfering_users(self, iq_samples, model_path=MODEL_PATH_CNN):
        """Predict interference user category from I/Q samples"""
        model = load_model(model_path)
        iq_samples = np.array(iq_samples).reshape(1, self.sequence_length, self.num_features)
        predictions = model.predict(iq_samples)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return predicted_class
    
    def load_model(self, model_path=MODEL_PATH_CNN):
        """Load pre-trained model"""
        self.model = load_model(model_path)
        return self.model
