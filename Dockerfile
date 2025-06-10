FROM python:3.11-slim

# working directory
WORKDIR /app
#COPY SORCE CODE

COPY . /app/

#Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

#RUN THE APP
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
# To build the Docker image, run:
# docker build -t "my-python-app" .
# To run the Docker container, use:
# docker run -p 8080:8080 my-python-app
# docker  run -p 8080:8080 -d --name height_predictor height_predictor
# To run the Docker container in detached mode, use:
# docker run -d -p 8080:8080 my-python-app
# To run the Docker container with a specific name, use:
# docker run --name my-container -p 8080:8080 my-python-app
# To run the Docker container with an interactive terminal, use:
# docker run -it -p 8080:8080 my-python-app
# To remove the Docker container after it stops, use:
# docker run --rm -p 8080:8080 my-python-app