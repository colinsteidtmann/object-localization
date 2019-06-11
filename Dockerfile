# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y python3-pip \
    wget
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt


# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# # Run app.py when the container launches
CMD ["python3", "train.py"]
