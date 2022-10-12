FROM python:3.8

WORKDIR /ws

COPY ./app .

RUN make install

ENTRYPOINT ["python3"]

CMD ["main.py"]