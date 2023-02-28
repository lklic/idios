FROM python:3.10

# Set the working directory
WORKDIR /api

# Copy the requirements file
COPY requirements.txt requirements-dev.txt .

# Install the dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-dev.txt

# Copy the application code
COPY api .

# Expose the port for the application
EXPOSE 4213

# Start the application
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "4213"]
