# Docker
!!! abstract "Learning Objectives"
    - Understand when to use Docker vs. virtual environments
    - Learn basic Docker concepts and commands
    - Create Dockerfiles for data analysis projects
    - Develop best practices for Docker in scientific computing

## Why Docker?

While virtual environments (venv, conda, renv) handle package dependencies, Docker goes further by providing:

1. **Complete Environment**: OS, system libraries, and all dependencies
2. **Perfect Reproducibility**: Identical environment across any platform
3. **System Dependencies**: Handle complex system-level requirements
4. **Cross-Platform**: Works the same on Linux, Mac, and Windows
5. **Production Ready**: Easy transition from development to deployment

### When to Use Docker vs. Virtual Environments

Use Docker when you need:

- System-level dependencies (e.g., CUDA, system libraries)
- Cross-platform compatibility
- Multiple language environments (e.g., R + Python)
- Production deployment
- Continuous Integration/Deployment (CI/CD)

Use virtual environments when:

- Working with single-language projects
- Need lighter-weight solutions
- Quick local development
- Limited system resources

## Docker Basics

### Key Concepts

1. **Image**: Blueprint for your environment
2. **Container**: Running instance of an image
3. **Dockerfile**: Instructions to build an image
4. **Registry**: Storage for Docker images (e.g., Docker Hub)

### Essential Commands

```bash
# Build image from Dockerfile
docker build -t myproject .

# Run container
docker run -it myproject

# List containers
docker ps

# Stop container
docker stop container_id

# Remove container
docker rm container_id
```

## Dockerfile for Data Analysis

### Basic Structure

```dockerfile
# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Command to run
CMD ["python", "analysis.py"]
```

### R + Python Environment

```dockerfile
# Use rocker/tidyverse as base
FROM rocker/tidyverse:4.2.0

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install R packages
RUN R -e "install.packages(c('reticulate', 'renv'))"

# Copy project files
COPY . .

# Set up renv
RUN R -e "renv::restore()"
```

## Best Practices

### 1. Layer Optimization

```dockerfile
# Good: Combine commands to reduce layers
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*

# Bad: Multiple RUN commands
RUN apt-get update
RUN apt-get install package1
RUN apt-get install package2
```

### 2. Use .dockerignore

```plaintext
# .dockerignore
.git
__pycache__
*.pyc
env/
venv/
data/
```

### 3. Multi-stage Builds

```dockerfile
# Build stage
FROM python:3.9 AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY . .
```

## Package Development Workflow

### 1. Development Container

```dockerfile
# Dockerfile.dev
FROM python:3.9

WORKDIR /app

# Install dev tools
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Mount source code as volume
VOLUME /app

# Keep container running
CMD ["bash"]
```

### 2. Testing Container

```dockerfile
# Dockerfile.test
FROM python:3.9

WORKDIR /app

COPY . .
RUN pip install -e ".[test]"

CMD ["pytest"]
```

### 3. Production Container

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . .
RUN pip install .

CMD ["python", "-m", "mypackage"]
```

## Recommended Workflow

1. **Initial Setup**:
```bash
# Create project structure
mkdir myproject && cd myproject

# Create Dockerfile and .dockerignore
touch Dockerfile .dockerignore

# Create docker-compose for development
touch docker-compose.yml
```

2. **Development Workflow**:
```bash
# Build development container
docker-compose build

# Start development environment
docker-compose up -d

# Run tests in container
docker-compose exec app pytest

# Install new dependency
docker-compose exec app pip install newpackage
```

3. **Testing Workflow**:
```bash
# Build test container
docker build -f Dockerfile.test -t myproject-test .

# Run tests
docker run myproject-test
```

4. **Release Workflow**:
```bash
# Build production image
docker build -t myproject:v1.0.0 .

# Test production image
docker run myproject:v1.0.0

# Push to registry
docker push myproject:v1.0.0
```

### Example docker-compose.yml

```yaml
version: '3'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
    ports:
      - "8888:8888"  # For Jupyter
    environment:
      - PYTHONPATH=/app
```

## Common Issues and Solutions

1. **Large Image Sizes**:
```dockerfile
# Use slim base images
FROM python:3.9-slim

# Clean up after installations
RUN pip install --no-cache-dir -r requirements.txt
```

2. **Slow Builds**:
```dockerfile
# Cache dependencies layer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code later
COPY . .
```

3. **Permission Issues**:
```dockerfile
# Create non-root user
RUN useradd -m myuser
USER myuser
```

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Rocker Project](https://www.rocker-project.org/)
- [Docker for Data Science](https://jupyter-docker-stacks.readthedocs.io/)
