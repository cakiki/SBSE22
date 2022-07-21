FROM python:3.10

# Set the working directory to /app
WORKDIR /app

COPY submission/requirements.txt submission/requirements.txt

RUN python3 -m pip install -r submission/requirements.txt

COPY . .

ENTRYPOINT exec python3 -m unittest -v tests1-simple.py tests2-extended.py