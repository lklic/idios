FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port for the application
EXPOSE 4213

# Start the application
CMD ["python", "app.py"]
