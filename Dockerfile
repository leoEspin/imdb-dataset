FROM gcr.io/deeplearning-platform-release/tf2-gpu

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt  --ignore-installed

RUN mkdir -p  /trainer
COPY trainer/. /trainer/

ENTRYPOINT ["python", "trainer/trainer.py"]