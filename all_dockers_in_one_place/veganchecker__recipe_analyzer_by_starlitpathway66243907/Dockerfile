# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to avoid .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR .
# WORKDIR /app

# Copy the dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# COPY app/ .
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# CMD ["uvicorn", "extractor_from_url_api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "extractor_from_url_api.py"]
