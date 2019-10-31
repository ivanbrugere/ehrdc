FROM python:3.7

# Install dependencies
RUN pip install --upgrade pip
RUN pip install pandas numpy sklearn datetime joblib xgboost intel-openmp

# Move to the app folder
RUN bash -c 'mkdir -p /{app, infer, model, output, scratch, train}'
WORKDIR /app/

# Copy our python program for training and inference
COPY . /app/

# Add executable permission to Bash scripts
RUN chmod +x /app/train.sh /app/infer.sh