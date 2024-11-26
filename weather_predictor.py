import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Tuple, Union

class WeatherPredictor:
    """
    A class to build and train an LSTM model for weather forecasting.

    Attributes:
        model: keras.Model
            The LSTM-based model used for predictions.
        history: keras.callbacks.History
            The training history of the model.
        train_data: pd.DataFrame
            Normalized training data.
        val_data: pd.DataFrame
            Normalized validation data.
        temp_mean: float
            Mean of the temperature column for de-normalization.
        temp_std: float
            Standard deviation of the temperature column for de-normalization.
    """

    def __init__(self, data_path: str):
        """
        Initializes the WeatherPredictor with the dataset.

        Parameters:
            data_path (str): Path to the CSV file containing the dataset.
        """
        self.data_frame = pd.read_csv(data_path)
        self.temp_mean = None
        self.temp_std = None
        self.model = None
        self.history = None
        self.train_data = None
        self.val_data = None
        self._prepare_data()

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the dataset by subtracting the mean and dividing by the standard deviation.

        Parameters:
            data (pd.DataFrame): The data to normalize.

        Returns:
            pd.DataFrame: Normalized data.
        """
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        return (data - data_mean) / data_std

    def _prepare_data(self):
        """
        Prepares and normalizes the dataset, splitting it into training and validation data.
        """
        time = self.data_frame['Date Time']
        temperature = self.data_frame['T (degC)']
        pressure = self.data_frame['p (mbar)']
        relative_humidity = self.data_frame['rh (%)']
        vapor_pressure = self.data_frame['VPact (mbar)']
        wind_speed = self.data_frame['wv (m/s)']
        airtight = self.data_frame['rho (g/m**3)']

        features = pd.concat(
            [temperature, pressure, relative_humidity, vapor_pressure, wind_speed, airtight], axis=1
        )
        features.index = time

        # Store temperature mean and std for de-normalization
        self.temp_mean = temperature.mean()
        self.temp_std = temperature.std()

        normalized_features = self._normalize(features)

        training_size = int(0.8 * normalized_features.shape[0])
        self.train_data = normalized_features.iloc[:training_size]
        self.val_data = normalized_features.iloc[training_size:]

    def _create_dataset(self, data: pd.DataFrame, sequence_length: int, start_offset: int = 432 + 36) -> Tuple:
        """
        Creates a dataset for training or validation.

        Parameters:
            data (pd.DataFrame): The data to use for dataset creation.
            sequence_length (int): The length of each sequence.
            start_offset (int): The offset to start generating targets.

        Returns:
            Tuple: A tuple containing the input and target datasets.
        """
        x_data = data.iloc[:-start_offset][[i for i in range(6)]].values
        y_data = data.iloc[start_offset:][[0]]
        
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            x_data, y_data, sequence_length=sequence_length, batch_size=64
        )
        return dataset

    def build_model(self, sequence_length: int):
        """
        Builds the LSTM model for weather prediction.

        Parameters:
            sequence_length (int): The length of the input sequences.
        """
        inputs = keras.layers.Input(shape=(sequence_length, 6))
        lstm_out = keras.layers.LSTM(32)(inputs)
        outputs = keras.layers.Dense(1)(lstm_out)

        self.model = keras.Model(name="Weather_forecaster", inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    def train(self, epochs: int = 15):
        """
        Trains the LSTM model.

        Parameters:
            epochs (int): The number of epochs to train for.
        """
        sequence_length = int(432 / 6)
        dataset_train = self._create_dataset(self.train_data, sequence_length)
        dataset_val = self._create_dataset(self.val_data, sequence_length)

        self.history = self.model.fit(
            dataset_train, epochs=epochs, validation_data=dataset_val
        )

    def plot_loss(self):
        """
        Plots the training loss over epochs.
        """
        loss = self.history.history["loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def predict_latest(self) -> Tuple[float, float]:
        """
        Predicts the temperature for the latest time step based on the last 3 days of data.

        Returns:
            Tuple[float, float]: A tuple containing the true temperature and the predicted temperature (de-normalized).
        """
        sequence_length = int(432 / 6)
        recent_data = self.val_data.iloc[-sequence_length:][[i for i in range(6)]].values
        recent_data = np.expand_dims(recent_data, axis=0)

        true_value = self.val_data.iloc[-1][0] * self.temp_std + self.temp_mean
        prediction = self.model.predict(recent_data)[0][0] * self.temp_std + self.temp_mean

        return true_value, prediction

predictor = WeatherPredictor("data/jena_1_1_2024_30_6_2024.csv")
predictor.build_model(sequence_length=int(432/6))
predictor.train(epochs=15)
predictor.plot_loss()
true_value, predicted_value = predictor.predict_latest()
print(f"True Temperature: {true_value}, Predicted Temperature: {predicted_value}")
