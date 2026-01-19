# Use slim python image
FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl
RUN curl https://www.google.com


# # Set working directory
# WORKDIR /app

# # Copy files
# COPY . .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose port for FastAPI
# EXPOSE 8000

# # Run the app
# CMD ["uvicorn", "macro_mlops.src.api.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]