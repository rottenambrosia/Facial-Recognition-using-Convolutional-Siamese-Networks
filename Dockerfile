FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update && apt-get install -y libgl1 cmake
RUN pip install opencv-python matplotlib