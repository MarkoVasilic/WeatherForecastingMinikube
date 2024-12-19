import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataset(Dataset):
    """
    Dataset class for weather data.

    Attributes:
        data (np.ndarray): The normalized weather data.
        sequence_length (int): The length of the sequences to be used for training.
    """
    def __init__(self, data: np.ndarray, sequence_length: int):
        """
        Initialize the WeatherDataset.

        Args:
            data (np.ndarray): Normalized weather data.
            sequence_length (int): Length of the input sequence.
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Input sequence (x) and target value (y).
        """
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class WeatherLSTM(nn.Module):
    """
    LSTM-based neural network for weather prediction.

    Attributes:
        lstm1 (nn.LSTM): First LSTM layer.
        dropout1 (nn.Dropout): Dropout layer after first LSTM.
        lstm2 (nn.LSTM): Second LSTM layer.
        dropout2 (nn.Dropout): Dropout layer after second LSTM.
        fc1 (nn.Linear): First fully connected layer.
        dropout3 (nn.Dropout): Dropout layer after first fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        dropout4 (nn.Dropout): Dropout layer after second fully connected layer.
        fc3 (nn.Linear): Output layer.
    """
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int, dropout_rate: float):
        """
        Initialize the WeatherLSTM model.

        Args:
            input_size (int): Number of input features.
            hidden_size1 (int): Number of units in the first LSTM layer.
            hidden_size2 (int): Number of units in the second LSTM layer.
            output_size (int): Number of output features.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(WeatherLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, 8)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(8, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc1(out[:, -1, :])  # Fully connected layer on last time step
        out = self.dropout3(out)
        out = self.fc2(out)
        out = self.dropout4(out)
        out = self.fc3(out)
        return out

class WeatherPredictor:
    """
    High-level class for weather prediction using the WeatherLSTM model.

    Attributes:
        input_size (int): Number of input features.
        sequence_length (int): Length of input sequences.
        model (WeatherLSTM): Instance of the WeatherLSTM model.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        criterion (nn.Module): Loss function.
    """
    def __init__(self, input_size: int = 6, hidden_size1: int = 64, hidden_size2: int = 32, sequence_length: int = 3, learning_rate: float = 0.001, dropout_rate: float = 0.3):
        """
        Initialize the WeatherPredictor.

        Args:
            input_size (int): Number of input features.
            hidden_size1 (int): Number of units in the first LSTM layer.
            hidden_size2 (int): Number of units in the second LSTM layer.
            sequence_length (int): Length of input sequences.
            learning_rate (float): Learning rate for the optimizer.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.model = WeatherLSTM(input_size, hidden_size1, hidden_size2, output_size=1, dropout_rate=dropout_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize the input data.

        Args:
            data (np.ndarray): Raw data.

        Returns:
            np.ndarray: Normalized data.
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        return (data - self.mean) / self.std

    def inverse_transform(self, data: float, column_idx: int = 0) -> float:
        """
        Reverse the normalization process for a single value.

        Args:
            data (float): Normalized value.
            column_idx (int): Column index to use for mean and std deviation.

        Returns:
            float: Original value.
        """
        return data * self.std[column_idx] + self.mean[column_idx]

    def prepare_data(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Prepare the input data for training and prediction.

        Args:
            dataframe (pd.DataFrame): Raw weather data.

        Returns:
            np.ndarray: Processed and normalized data sampled every hour.
        """
        features = dataframe[["T", "p", "rh", "Vpact", "wv", "rho"]]
        normalized_features = self.normalize(features.values)
        return normalized_features[::6]

    def train(self, dataframe: pd.DataFrame, epochs: int = 30, batch_size: int = 64):
        """
        Train the WeatherLSTM model.

        Args:
            dataframe (pd.DataFrame): Raw weather data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        """
        sampled_data = self.prepare_data(dataframe)
        dataset = WeatherDataset(sampled_data, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(x_batch)
                loss = self.criterion(predictions.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    def predict_next_hour(self, dataframe: pd.DataFrame) -> float:
        """
        Predict the temperature for the next hour.

        Args:
            dataframe (pd.DataFrame): Raw weather data.

        Returns:
            float: Predicted temperature.
        """
        sampled_data = self.prepare_data(dataframe)
        input_data = sampled_data[-self.sequence_length:]
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(input_tensor).item()
        return self.inverse_transform(prediction, column_idx=0)

    def predict_next_24_hours(self, dataframe: pd.DataFrame) -> list:
        """
        Predict the temperatures for the next 24 hours.

        Args:
            dataframe (pd.DataFrame): Raw weather data.

        Returns:
            list: List of predicted temperatures for the next 24 hours.
        """
        sampled_data = self.prepare_data(dataframe)
        input_data = sampled_data[-self.sequence_length:]
        predictions = []
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            for _ in range(24):
                prediction = self.model(input_tensor).item()
                predictions.append(self.inverse_transform(prediction, column_idx=0))
                next_input = np.expand_dims(np.concatenate(([prediction], input_tensor[0, -1, 1:].numpy())), axis=0)
                input_tensor = torch.cat((input_tensor[:, 1:, :], torch.tensor(next_input, dtype=torch.float32).unsqueeze(0)), dim=1)
        return predictions
    
    def predict_next_7_days(self, dataframe: pd.DataFrame) -> list:
        """
        Predicts the temperature for the next 7 days, providing one prediction per day.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe.
        
        Returns:
            list: The predicted temperatures for the next 7 days.
        """
        logger.info("Prediction for next 7 days started.")
        sampled_data = self.prepare_data(dataframe)
        input_data = sampled_data[-self.sequence_length:]
        predictions = []
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            for _ in range(7):
                prediction = self.model(input_tensor).item()
                predictions.append(self.inverse_transform(prediction, column_idx=0))
                next_input = np.expand_dims(np.concatenate(([prediction], input_tensor[0, -1, 1:].numpy())), axis=0)
                input_tensor = torch.cat((input_tensor[:, 4:, :], torch.tensor(next_input, dtype=torch.float32).unsqueeze(0)), dim=1)
        logger.info(f"Predictions for next 7 days: {predictions}")
        return predictions
