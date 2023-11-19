FROM python:3.8.16

RUN pip install --upgrade pip

RUN pip install torch==1.11.0
RUN pip install torchvision==0.12.0
RUN pip install torchaudio==0.11.0
RUN pip install rembg
RUN pip install onnx==1.13.0
RUN pip install opencv-python
RUN pip install python-multipart
RUN pip install pandas
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install onnxruntime
RUN pip install ultralytics
RUN pip install gcloud
RUN pip install aiofile
RUN pip install gcloud-aio-storage
RUN pip install google-cloud-storage
RUN pip install python-dotenv
RUN pip install pyyaml-include
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install python-multipart
RUN pip install werkzeug



RUN mkdir -p /app
WORKDIR /app

COPY . /app
RUN apt-get update && apt-get install python3-opencv -y

RUN ls -la /app

USER root
CMD ["./run_server.sh"]
