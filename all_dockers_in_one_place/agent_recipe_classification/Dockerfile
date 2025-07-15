FROM python:3.9

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
#COPY . /app

COPY requirements.txt .
#RUN pip install -r requirements.txt

# Install system dependencies required by Chromium
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libxshmfence1 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libgtk-3-0 \
    libdrm2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (including Playwright)
RUN pip install --no-cache-dir \
    fastapi \
    pymongo \
    requests \
    uvicorn \
    python-dotenv \
    beautifulsoup4 \
    ollama \
    unidecode \
    playwright \
    recipe-scrapers

# Install Playwright browser binaries
RUN playwright install --with-deps

# Expose the port FastAPI runs on
EXPOSE 8002

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "debug"]
