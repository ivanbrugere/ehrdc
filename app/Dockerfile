FROM python:3.7

# Install dependencies
RUN pip install --upgrade pip
RUN pip install pandas numpy sklearn datetime joblib xgboost intel-openmp

RUN apt-get update && apt-get install -y \
        libssl-dev \
        libusb-1.0-0 \
        python-dev \
        swig \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
RUN pip install auto-sklearn

# Move to the app folder
RUN bash -c 'mkdir -p /{app, infer, model, output, scratch, train}'
WORKDIR /app/

# Copy our python program for training and inference
COPY . /app/

# Add executable permission to Bash scripts
RUN chmod +x /app/train.sh /app/infer.sh