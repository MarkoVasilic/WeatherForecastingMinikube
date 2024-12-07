from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import json
from io import StringIO
from . import database, crud, schema
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


@app.post("/add-row/", status_code=201)
def add_row(row: schema.DataCreate, db: Session = Depends(get_db)):
    """
    Add a new row to the weather_data table.
    """
    success = crud.insert_new_row(db, row.model_dump())
    if not success:
        logging.error("Failed to insert new row.")
        raise HTTPException(status_code=500, detail="Failed to insert new row.")
    logging.info("New row successfully inserted.")
    return {"message": "New row successfully inserted."}


@app.delete("/delete-all/", status_code=200)
def delete_all_data(db: Session = Depends(get_db)):
    """
    Delete all rows from the weather_data table.
    """
    success = crud.delete_all_data(db)
    if not success:
        logging.error("Failed to delete all rows.")
        raise HTTPException(status_code=500, detail="Failed to delete all rows.")
    logging.info("All data successfully deleted.")
    return {"message": "All data successfully deleted from the database."}


@app.delete("/delete-last-row/", status_code=200)
def delete_last_row(db: Session = Depends(get_db)):
    """
    Delete the last row from the weather_data table.
    """
    success = crud.delete_last_row(db)
    if not success:
        logging.error("Failed to delete the last row.")
        raise HTTPException(status_code=500, detail="Failed to delete the last row.")
    logging.info("Last row successfully deleted.")
    return {"message": "Last row successfully deleted from the database."}


def send_prediction_request(endpoint: str, db: Session):
    data = crud.get_last_432_rows(db)
    if data is None:
        logging.error("Failed to retrieve data for prediction.")
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")
    
    data_dicts = [vars(item) for item in data]
    for record in data_dicts:
        record.pop("_sa_instance_state", None)
    
    data_frame = pd.DataFrame(data_dicts)
    data_frame = pd.DataFrame(data_dicts)
    if 'date_time' in data_frame.columns:
        data_frame = data_frame.drop(columns=['date_time'])

    json_payload = data_frame.to_dict(orient="records")
    model_api_url = f"http://predictor:8006/{endpoint}"
    response = requests.post(model_api_url, json=json_payload)
    if response.status_code != 200:
        logging.error(f"Prediction request failed: {response.status_code}, {response.content}")
        raise HTTPException(status_code=500, detail="Prediction request failed.")
    
    return json.loads(response.content.decode('utf-8'))


@app.get("/predictor/train/")
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
    if 'date_time' in data_frame.columns:
        data_frame = data_frame.drop(columns=['date_time'])

    json_payload = data_frame.to_dict(orient="records")
    model_api_url = f"http://predictor:8006/train/"
    response = requests.post(model_api_url, json=json_payload)
    if response.status_code != 200:
        logging.error(f"Prediction request failed: {response.status_code}, {response.content}")
        raise HTTPException(status_code=500, detail="Training request failed.")
    
    return json.loads(response.content.decode('utf-8'))


@app.get("/predictor/predict/next_hour/")
def predict_next_hour(db: Session = Depends(get_db)):
    logging.info("Predicting next hour.")
    response = send_prediction_request("predict/next_hour", db)
    logging.info("Next hour prediction completed.")
    return response


@app.get("/predictor/predict/next_24_hours/")
def predict_next_24_hours(db: Session = Depends(get_db)):
    logging.info("Predicting next 24 hours.")
    response = send_prediction_request("predict/next_24_hours", db)
    logging.info("Next 24 hours prediction completed.")
    return response


@app.get("/predictor/predict/next_7_days/")
def predict_next_7_days(db: Session = Depends(get_db)):
    logging.info("Predicting next 7 days.")
    response = send_prediction_request("predict/next_7_days", db)
    logging.info("Next 7 days prediction completed.")
    return response
