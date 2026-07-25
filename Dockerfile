FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cpu torch==2.4.0

RUN pip install --default-timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]