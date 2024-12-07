FROM python:3.10-slim
WORKDIR /app
COPY ./app ./app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
ENV STREAMLIT_HOME=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "app/app.py"]