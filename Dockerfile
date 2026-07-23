FROM tensorflow/tensorflow:2.15.0-gpu-jupyter
RUN apt-get update && apt-get install -y libgl1 cmake
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir opencv-python==4.10.0.84