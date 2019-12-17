FROM python:3.7

# Install dependencies
RUN pip install --upgrade pip
RUN pip install pandas numpy datetime joblib xgboost intel-openmp torch torchvision skorch
RUN pip install -U scikit-learn

# Move to the app folder
RUN bash -c 'mkdir -p /{app, infer, model, output, scratch, train}'
WORKDIR /app/

# Copy our python program for training and inference
COPY . /app/

# Add executable permission to Bash scripts
RUN chmod +x /app/train.sh /app/infer.sh