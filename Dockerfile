# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Noninteractive apt
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# System deps for pyodbc + build, and tools for key handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gnupg \
    ca-certificates \
    unixodbc \
    unixodbc-dev \
    odbcinst \
    libpq-dev \
    libsqlite3-dev \
 && rm -rf /var/lib/apt/lists/*

# Add Microsoft repo (Debian 12/bookworm) using keyring, not apt-key
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
      msodbcsql18 \
      mssql-tools18 \
    && rm -rf /var/lib/apt/lists/*

# Put sqlcmd/bcp on PATH
ENV PATH="/opt/mssql-tools18/bin:${PATH}"

# Copy your code
COPY . .

# Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose and run with Gunicorn
EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
