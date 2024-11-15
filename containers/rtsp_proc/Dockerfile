FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  && apt-get upgrade -y
RUN apt-get install  -y python3.9 
RUN apt-get install  -y python3-pip 
RUN apt-get install  -y libgstreamer1.0-0 
RUN apt-get install  -y gstreamer1.0-plugins-base 
RUN apt-get install  -y gstreamer1.0-plugins-good 
RUN apt-get install  -y gstreamer1.0-plugins-bad 
RUN apt-get install  -y gstreamer1.0-plugins-ugly 
RUN apt-get install  -y gstreamer1.0-libav 
RUN apt-get install  -y gstreamer1.0-tools 
RUN apt-get install  -y libgstreamer-plugins-base1.0-dev 
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV RTSP_URL="rtsp://192.168.1.100:8554/stream"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]