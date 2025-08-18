# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for pyodbc and ODBC drivers
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    odbcinst \
    curl \
    gnupg2 \
    libpq-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server (if using MSSQL)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    echo "deb [arch=amd64] https://packages.microsoft.com/debian/10/prod buster main" > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y \
    msodbcsql18 \
    mssql-tools18 && \
    rm -rf /var/lib/apt/lists/*

# Set PATH for SQL Server tools
ENV PATH="/opt/mssql-tools18/bin:${PATH}"

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]

