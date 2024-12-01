from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import json
from io import StringIO
from . import database, crud
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload-csv/", status_code=201)
async def upload_csv(file: UploadFile, db: Session = Depends(get_db)):
    """
    Upload a CSV file and save its data to the PostgreSQL database.
    """
    if not file.filename.endswith('.csv'):
        logging.warning(f"Invalid file format: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    try:
        contents = await file.read()
        dataframe = pd.read_csv(StringIO(contents.decode("utf-8")))
        success = crud.save_csv_to_postgres(dataframe, "weather_data")
        if not success:
            logging.error("Failed to save data to the database.")
            raise HTTPException(status_code=500, detail="Failed to save data to the database.")

        logging.info(f"CSV data successfully uploaded: {file.filename}")
        return {"message": "CSV data successfully uploaded and saved to the database."}
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

def send_prediction_request(endpoint: str, db: Session):
    data = crud.get_last_432_rows(db)
    if data is None:
        logging.error("Failed to retrieve data for prediction.")
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")
    
    data_dicts = [vars(item) for item in data]
    for record in data_dicts:
        record.pop("_sa_instance_state", None)
    
    data_frame = pd.DataFrame(data_dicts)
    json_payload = data_frame.to_dict(orient="records")
    model_api_url = f"http://predictor:8006/{endpoint}"
    response = requests.post(model_api_url, json=json_payload)
    if response.status_code != 200:
        logging.error(f"Prediction request failed: {response.status_code}, {response.content}")
        raise HTTPException(status_code=500, detail="Prediction request failed.")
    
    return json.loads(response.content.decode('utf-8'))

@app.post("/predictor/train/")
def train_model(db: Session = Depends(get_db)):
    logging.info("Starting model training.")
    data = crud.get_all_data(db)
    if data is None:
        logging.error("Failed to retrieve data for training.")
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")
    
    data_dicts = [vars(item) for item in data]
    for record in data_dicts:
        record.pop("_sa_instance_state", None)
    
    data_frame = pd.DataFrame(data_dicts)
    json_payload = data_frame.to_dict(orient="records")
    model_api_url = f"http://predictor:8006/train/"
    response = requests.post(model_api_url, json=json_payload)
    if response.status_code != 200:
        logging.error(f"Prediction request failed: {response.status_code}, {response.content}")
        raise HTTPException(status_code=500, detail="Training request failed.")
    
    return json.loads(response.content.decode('utf-8'))

@app.post("/predictor/predict/next_hour/")
def predict_next_hour(db: Session = Depends(get_db)):
    logging.info("Predicting next hour.")
    response = send_prediction_request("predict/next_hour", db)
    logging.info("Next hour prediction completed.")
    return response

@app.post("/predictor/predict/next_24_hours/")
def predict_next_24_hours(db: Session = Depends(get_db)):
    logging.info("Predicting next 24 hours.")
    response = send_prediction_request("predict/next_24_hours", db)
    logging.info("Next 24 hours prediction completed.")
    return response

@app.post("/predictor/predict/next_7_days/")
def predict_next_7_days(db: Session = Depends(get_db)):
    logging.info("Predicting next 7 days.")
    response = send_prediction_request("predict/next_7_days", db)
    logging.info("Next 7 days prediction completed.")
    return response
