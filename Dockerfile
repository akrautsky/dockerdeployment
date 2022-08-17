FROM python:3.8.10

COPY . /app

WORKDIR /app

RUN pip install -r Requirement.txt

EXPOSE $PORT

CMD ["python:3.8.10", "./RegressionModel.ipynb"]
