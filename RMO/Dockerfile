FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

COPY Predict_Docker.py .
COPY yolov11_customdata_trained.pt .

RUN mkdir -p /app/input_dir /app/output_dir

CMD ["python", "Predict_Docker.py"]