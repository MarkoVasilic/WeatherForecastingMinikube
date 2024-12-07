import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

connect_host = f"connector:8003"

try:
    train_url= f"http://{connect_host}/predictor/train"
    response = requests.get(train_url)
    if response.status_code == 200:
        logging.info("Predictor trained!")
    else:
        logging.error("Training failed!")
except Exception as e:
    logging.error(f"Error happened: {e}")