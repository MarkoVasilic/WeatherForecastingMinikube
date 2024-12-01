from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import logging
from app.weather_predictor import WeatherPredictor

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = WeatherPredictor(sequence_length=3)

class TrainingResponse(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    predictions: List[float]

def json_to_dataframe(data: List[dict]) -> pd.DataFrame:
    try:
        df = pd.DataFrame(data)
        logger.debug(f"Converted JSON to DataFrame with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error converting JSON to DataFrame: {e}")
        raise HTTPException(status_code=400, detail=f"Error converting JSON to DataFrame: {e}")

@app.post("/train", response_model=TrainingResponse)
async def train_model(data: List[dict]):
    """
    Endpoint to train the weather predictor model.
    
    Args:
        data (List[dict]): List of dictionaries containing the training data.
    
    Returns:
        TrainingResponse: A message indicating successful training.
    """
    logger.info("Training the model started.")
    data_frame = json_to_dataframe(data)
    
    if data_frame.empty:
        logger.warning("Input DataFrame is empty.")
        raise HTTPException(status_code=400, detail="Input DataFrame is empty.")
    
    predictor.train(data_frame)
    logger.info("Model trained successfully.")
    return {"message": "Model trained successfully."}

@app.post("/predict/next_hour", response_model=PredictionResponse)
async def predict_next_hour(data: List[dict]):
    """
    Endpoint to predict the temperature for the next hour.
    
    Args:
        data (List[dict]): List of dictionaries containing the input data.
    
    Returns:
        PredictionResponse: The predicted temperature for the next hour.
    """
    logger.info("Prediction for next hour started.")
    data_frame = json_to_dataframe(data)
    
    if data_frame.empty:
        logger.warning("Input DataFrame is empty.")
        raise HTTPException(status_code=400, detail="Input DataFrame is empty.")
    
    prediction = predictor.predict_next_hour(data_frame)
    logger.info(f"Prediction for next hour: {prediction}")
    return {"predictions": [prediction]}

@app.post("/predict/next_24_hours", response_model=PredictionResponse)
async def predict_next_24_hours(data: List[dict]):
    """
    Endpoint to predict the temperature for the next 24 hours.
    
    Args:
        data (List[dict]): List of dictionaries containing the input data.
    
    Returns:
        PredictionResponse: A list of predicted temperatures for the next 24 hours.
    """
    logger.info("Prediction for next 24 hours started.")
    data_frame = json_to_dataframe(data)
    
    if data_frame.empty:
        logger.warning("Input DataFrame is empty.")
        raise HTTPException(status_code=400, detail="Input DataFrame is empty.")
    
    predictions = predictor.predict_next_24_hours(data_frame)
    logger.info(f"Predictions for next 24 hours: {predictions}")
    return {"predictions": predictions}

@app.post("/predict/next_7_days", response_model=PredictionResponse)
async def predict_next_7_days(data: List[dict]):
    """
    Endpoint to predict the temperature for the next 7 days.
    
    Args:
        data (List[dict]): List of dictionaries containing the input data.
    
    Returns:
        PredictionResponse: A list of predicted temperatures for the next 7 days.
    """
    logger.info("Prediction for next 7 days started.")
    data_frame = json_to_dataframe(data)
    
    if data_frame.empty:
        logger.warning("Input DataFrame is empty.")
        raise HTTPException(status_code=400, detail="Input DataFrame is empty.")
    
    predictions = predictor.predict_next_7_days(data_frame)
    logger.info(f"Predictions for next 7 days: {predictions}")
    return {"predictions": predictions}
