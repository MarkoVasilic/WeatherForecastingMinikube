import pandas as pd
import logging
from datetime import datetime
from sqlalchemy import desc
from sqlalchemy.orm import Session
from . import database
from . import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_csv_to_postgres(dataframe: pd.DataFrame, table_name: str) -> bool:
    """
    Save a Pandas DataFrame to a PostgreSQL database table.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        table_name (str): The name of the table in the database.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        if 'date_time' in dataframe.columns:
            dataframe['date_time'] = pd.to_datetime(
                dataframe['date_time'],
                format="%d.%m.%Y %H:%M:%S",
                errors='coerce'
            )

            if dataframe['date_time'].isna().any():
                logging.warning("Some 'date_time' values could not be parsed and were set to NaT.")
        dataframe['id'] = range(1, len(dataframe) + 1)
        dataframe.to_sql(table_name, database.engine, if_exists='replace', index=False)
        logging.info(f"Data successfully saved to table '{table_name}'.")
        return True
    except Exception as e:
        logging.error(f"Error saving data to table '{table_name}': {e}")
        return False


def get_all_data(session: Session) -> list[models.Data] | None:
    """
    Retrieve all rows from the 'weather_data' table.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        list[models.Data] | None: List of all rows as `models.Data` objects, or None if an error occurs.
    """
    try:
        data = session.query(models.Data).all()
        logging.info(f"Retrieved {len(data)} rows from the database.")
        return data
    except Exception as e:
        logging.error(f"Error retrieving all data: {e}")
        return None


def get_last_432_rows(session: Session) -> list[models.Data] | None:
    """
    Retrieve the last 432 rows from the 'weather_data' table, ordered by the 'date_time' column in descending order.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        list[models.Data] | None: List of the last 432 rows as `models.Data` objects, or None if an error occurs.
    """
    try:
        data = session.query(models.Data).order_by(desc(models.Data.date_time)).limit(432).all()
        logging.info(f"Retrieved the last {len(data)} rows from the database.")
        return data
    except Exception as e:
        logging.error(f"Error retrieving the last 432 rows: {e}")
        return None


def insert_new_row(session: Session, data: dict) -> bool:
    """
    Insert a new row into the 'weather_data' table.

    Args:
        session (Session): SQLAlchemy session object.
        data (dict): Dictionary containing column names as keys and their corresponding values.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        if 'date_time' in data:
            try:
                data['date_time'] = datetime.strptime(data['date_time'], "%d.%m.%Y %H:%M:%S")
            except ValueError as e:
                logging.error(f"Invalid date_time format: {data['date_time']}. Error: {e}")
                return False
            
        current_row_count = session.query(models.Data).count()
        data['id'] = current_row_count + 1
        
        new_row = models.Data(**data)
        session.add(new_row)
        session.commit()
        
        logging.info("New row successfully inserted into the database.")
        return True
    except Exception as e:
        logging.error(f"Error inserting new row: {e}")
        session.rollback()
        return False


def delete_all_data(session: Session) -> bool:
    """
    Delete all rows from the 'weather_data' table.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        session.query(models.Data).delete()
        session.commit()
        logging.info("All data successfully deleted from the database.")
        return True
    except Exception as e:
        logging.error(f"Error deleting all data: {e}")
        session.rollback()
        return False


def delete_last_row(session: Session) -> bool:
    """
    Delete the last row from the 'weather_data' table, ordered by the 'date_time' column in descending order.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        last_row = session.query(models.Data).order_by(desc(models.Data.date_time)).first()
        if last_row:
            session.delete(last_row)
            session.commit()
            logging.info("Last row successfully deleted from the database.")
            return True
        else:
            logging.warning("No rows to delete.")
            return False
    except Exception as e:
        logging.error(f"Error deleting the last row: {e}")
        session.rollback()
        return False
