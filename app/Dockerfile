FROM python:3.7

# Install dependencies
RUN pip install pandas \
  numpy \
  sklearn \
  datetime \
  joblib

# Move to the app folder
RUN bash -c 'mkdir -p /{app, infer, model, output, scratch, train}'
WORKDIR /

# Copy our python program for training and inference
COPY . /app/

# Add executable permission to Bash scripts
RUN chmod +x /app/train.sh /app/infer.sh