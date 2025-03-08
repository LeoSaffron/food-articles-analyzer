# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8086
EXPOSE 8086

# Command to run the application
CMD ["python", "recipe_predict_if_plant_based_by_url_with_dockers.py"]
