FROM python:3.10

# Set the working directory to /app
WORKDIR /app

COPY . .

RUN python3 -m pip install -r submission/requirements.txt

ENTRYPOINT exec python3 -m unittest tests1-simple.py tests2-extended.py