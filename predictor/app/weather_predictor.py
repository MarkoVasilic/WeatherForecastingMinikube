import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        """
        Initializes the dataset with input data and sequence length.
        
        Args:
            data (np.ndarray): The normalized input data.
            sequence_length (int): The number of previous data points to use for each prediction.
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: The number of samples.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): The index of the sample.
        
        Returns:
            tuple: Input features (x) and target value (y).
        """
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class WeatherLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the LSTM model with the specified parameters.
        
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of units in the hidden layer.
            output_size (int): The number of output features.
        """
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Predicted output.
        """
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

class WeatherPredictor:
    def __init__(self, input_size: int = 6, hidden_size: int = 32, sequence_length: int = 3, learning_rate: float = 0.001):
        """
        Initializes the weather predictor with the specified parameters.
        
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of units in the hidden layer.
            sequence_length (int): The number of previous data points to use for each prediction.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.model = WeatherLSTM(input_size, hidden_size, output_size=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes the input data.
        
        Args:
            data (np.ndarray): The input data.
        
        Returns:
            np.ndarray: The normalized data.
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        logger.debug("Data normalized with mean and std: %s, %s", self.mean, self.std)
        return (data - self.mean) / self.std

    def inverse_transform(self, data: float, column_idx: int = 0) -> float:
        """
        Applies inverse normalization to the data.
        
        Args:
            data (float): The normalized data.
            column_idx (int): The column index for inverse transformation.
        
        Returns:
            float: The denormalized data.
        """
        return data * self.std[column_idx] + self.mean[column_idx]

    def prepare_data(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Prepares the input data by selecting features, normalizing, and sampling.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe.
        
        Returns:
            np.ndarray: The prepared data.
        """
        features = dataframe[["T", "p", "rh", "Vpact", "wv", "rho"]]
        normalized_features = self.normalize(features.values)
        logger.info(f"Data prepared with shape: {normalized_features.shape}")
        return normalized_features[::6]

    def train(self, dataframe: pd.DataFrame, epochs: int = 30, batch_size: int = 64) -> None:
        """
        Trains the LSTM model on the given dataframe.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
        """
        logger.info("Training started.")
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
        Predicts the temperature for the next hour.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe.
        
        Returns:
            float: The predicted temperature.
        """
        logger.info("Prediction for next hour started.")
        sampled_data = self.prepare_data(dataframe)
        input_data = sampled_data[-self.sequence_length:]
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(input_tensor).item()
        logger.info(f"Prediction for next hour: {prediction}")
        return self.inverse_transform(prediction, column_idx=0)

    def predict_next_24_hours(self, dataframe: pd.DataFrame) -> List[float]:
        """
        Predicts the temperature for the next 24 hours.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe.
        
        Returns:
            List[float]: The predicted temperatures for the next 24 hours.
        """
        logger.info("Prediction for next 24 hours started.")
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
        logger.info(f"Predictions for next 24 hours: {predictions}")
        return predictions

    def predict_next_7_days(self, dataframe: pd.DataFrame) -> List[float]:
        """
        Predicts the temperature for the next 7 days, providing one prediction per day.
        
        Args:
            dataframe (pd.DataFrame): The input dataframe.
        
        Returns:
            List[float]: The predicted temperatures for the next 7 days.
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