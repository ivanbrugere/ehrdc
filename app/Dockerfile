FROM continuumio/miniconda3


RUN conda update --all
RUN conda install pandas numpy joblib scikit-learn
RUN conda install pytorch torchvision -c pytorch
RUN conda install -c conda-forge xgboost
RUN conda install -c conda-forge openmpi
RUN conda install -c conda-forge skorch
RUN pip install scikit-learn==0.22
# Move to the app folder
RUN bash -c 'mkdir -p /{app, infer, model, output, scratch, train}'
WORKDIR /app/

# Copy our python program for training and inference
COPY . /app/

# Add executable permission to Bash scripts
RUN chmod +x /app/train.sh /app/infer.sh