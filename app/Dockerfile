FROM continuumio/miniconda3


RUN conda update --all
RUN conda install pandas numpy joblib scikit-learn
RUN conda install pytorch torchvision -c pytorch
RUN conda install -c conda-forge xgboost
RUN conda install -c conda-forge openmpi
RUN conda install -c conda-forge skorch
RUN conda install -c conda-forge catboost
RUN apt-get update && apt-get install -y libssl-dev libusb-1.0-0 python-dev swig g++
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN pip install pybind11
RUN pip install scikit-learn==0.22
#RUN conda install -c conda-forge compilers
RUN pip install hnswlib
# Move to the app folder
RUN bash -c 'mkdir -p /{app, infer, model, output, scratch, train}'
WORKDIR /app/

# Copy our python program for training and inference
COPY . /app/

# Add executable permission to Bash scripts
RUN chmod +x /app/train.sh /app/infer.sh