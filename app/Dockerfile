# Use official Python image as base
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application using Uvicorn with Gunicorn for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
