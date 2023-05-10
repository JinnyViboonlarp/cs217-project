FROM python:3.9.7

ENV LISTEN_PORT=5000
EXPOSE 5000

WORKDIR ./app

RUN python3 -m pip install --upgrade pip setuptools wheel 

COPY requirements.txt /tmp/                                                                                                                                                                                               
RUN python3 -m pip install -r /tmp/requirements.txt  

COPY ./ ./

CMD ["python", "app.py"]