# .devcontainer/Dockerfile
FROM python:3.11-slim

# Dependencias del sistema necesarias para algunas libs (statsmodels, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar requirements en build para instalarlas
COPY requirements.txt /workspace/requirements.txt

# Instalar requirements
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Puerto por defecto de Streamlit
EXPOSE 8501

# Comando por defecto (no lo ejecutamos aquí; Codespaces usará postCreateCommand)
CMD ["bash"]
