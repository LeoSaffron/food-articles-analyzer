# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any necessary dependencies
RUN pip install --no-cache-dir fastapi pymongo requests uvicorn python-dotenv


# Expose the port FastAPI runs on
EXPOSE 8002

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]