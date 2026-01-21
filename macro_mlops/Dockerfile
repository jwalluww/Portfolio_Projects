# Base image with python pre-installed (slim=smaller size)
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from your local directory to the container's /app folder
COPY . .

# Install all python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI that my app will run on
EXPOSE 8000

# Run the FastAPI app using Uvicorn (production server)
CMD ["uvicorn", "src.api.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]