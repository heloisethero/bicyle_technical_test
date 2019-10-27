FROM python:3.6.5

VOLUME ["/app"]
WORKDIR /app

COPY ./requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt

CMD [ "./main.sh" ]
