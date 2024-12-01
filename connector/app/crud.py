import pandas as pd
import logging
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
