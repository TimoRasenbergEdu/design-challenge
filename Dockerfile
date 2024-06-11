FROM tensorflow/tensorflow:latest-gpu

RUN apt update --upgrade
run apt install -y libgl1

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
